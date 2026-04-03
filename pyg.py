import math
import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="outdated")

import torch
import torch.nn.functional as F

# Add to safe globals before loading
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# from logger import Logger

class Stats:
    def __init__(self):
        self.values = []

    def add(self, time):
        self.values.append(time)

    def min(self):
        return min(self.values)

    def max(self):
        return max(self.values)

    def mean(self):
        return sum(self.values) / len(self.values)

    def count(self):
        return len(self.values)

    def std(self):
        mean = self.mean()
        s = sum(map(lambda x: (x - mean)**2, self.values))
        return math.sqrt(s / (self.count() - 1))

losses = Stats()
sageconv_time = Stats()
relu_time = Stats()
normalize_time = Stats()
log_softmax_time = Stats()
inference_time = Stats()
gradient_time = Stats()

zero_grad_time = Stats()
inference_time = Stats()
gradient_time = Stats()
update_weights = Stats()

def print_stats():
    stats = [
        ("zero_grad",    zero_grad_time),
        ("inference",    inference_time),
        ("  sageconv",   sageconv_time),
        ("  relu",       relu_time),
        ("  normalize",  normalize_time),
        ("  log_softmax", log_softmax_time),
        ("gradient",     gradient_time),
        ("update_weights", update_weights),
        ("loss",         losses),
    ]

    headers = ["name", "avg", "std", "total", "min", "max", "calls"]
    widths  = [20, 12, 12, 12, 12, 12, 8]

    header_line = "".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    for name, s in stats:
        if s.count() == 0:
            continue
        print(f"{name:<20}"
              f"{s.mean():<12.6f}"
              f"{s.std():<12.6f}"
              f"{sum(s.values):<12.6f}"
              f"{s.min():<12.6f}"
              f"{s.max():<12.6f}"
              f"{s.count():<8d}")

def print_csv():
    stats = [
        ("zero_grad",    "",          zero_grad_time),
        ("inference",    "",          inference_time),
        ("sageconv",     "inference", sageconv_time),
        ("relu",         "inference", relu_time),
        ("normalize",    "inference", normalize_time),
        ("log_softmax",  "inference", log_softmax_time),
        ("gradient",     "",          gradient_time),
        ("update_weights", "",        update_weights),
        ("loss",         "",          losses),
    ]

    print("\n--- CSV_OUTPUT_BEGIN ---")
    print("name,parent,avg(s),std,total(s),min(s),max(s),calls")

    for name, parent, s in stats:
        if s.count() == 0:
            continue
        print(f"{name},{parent},{s.mean():.6f},{s.std():.6f},"
              f"{sum(s.values):.6f},{s.min():.6f},{s.max():.6f},{s.count()}")

    print("--- CSV_OUTPUT_END ---")

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        self.convs = torch.nn.ModuleList([
            SAGEConv(channels[i], channels[i+1], normalize=False)
            for i in range(num_layers)
        ])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            t = time.perf_counter()
            x = conv(x, adj_t)
            sageconv_time.add(time.perf_counter() - t)

            t = time.perf_counter()
            x = F.relu(x)
            relu_time.add(time.perf_counter() - t)

            t = time.perf_counter()
            x = F.normalize(x, p=2., dim=-1)
            normalize_time.add(time.perf_counter() - t)

        t = time.perf_counter()
        x = self.convs[-1](x, adj_t)
        sageconv_time.add(time.perf_counter() - t)

        t = time.perf_counter()
        result = x.log_softmax(dim=-1)
        log_softmax_time.add(time.perf_counter() - t)

        return result

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main(args):
    torch.manual_seed(0)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.data_dir, transform=T.ToSparseTensor())

    data = dataset[0]
    if args.dataset == "arxiv":
        data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = SAGE(data.num_features, args.hidden_channels,
                 dataset.num_classes, args.num_layers).to(device)
    print(model)

    evaluator = Evaluator(name='ogbn-arxiv')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, 1 + args.epochs):
        # Reset gradients
        t = time.perf_counter()
        optimizer.zero_grad()
        zero_grad_time.add(time.perf_counter() - t)

        # Inference
        t = time.perf_counter()
        out = model(data.x, data.adj_t)[train_idx]
        inference_time.add(time.perf_counter() - t)

        # Loss and accuracy
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx], reduction="mean")
        y_pred = out.argmax(dim=-1, keepdim=True)
        acc = evaluator.eval({'y_true': data.y[train_idx], 'y_pred': y_pred})['acc']

        # Gradient descent
        t = time.perf_counter()
        loss.backward()
        gradient_time.add(time.perf_counter() - t)

        # Update weights
        t = time.perf_counter()
        optimizer.step()
        update_weights.add(time.perf_counter() - t)

        losses.add(loss)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {100 * acc:.4f}')

    train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
    print(f'Run: {run + 1:02d}, '
          f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')


    print_stats()
    print_csv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--stdout', action='store_true')
    parser.add_argument('--dataset', type=str, default="arxiv")
    parser.add_argument('--data_dir', type=str, default="~/D1/pyg-dataset")


    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    print(args)

    fname = (f"layers{args.num_layers}_"
             f"hidden{args.hidden_channels}_"
             f"lr{args.lr}_"
             f"epochs{args.epochs}.log")

    if not args.stdout:
        f = open(fname, 'w')
        sys.stdout = f

    main(args)

    if not args.stdout:
        sys.stdout = sys.__stdout__
        f.close()
