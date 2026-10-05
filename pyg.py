import os
import sys
import time
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module="outdated")
import torch
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})
import torch.nn.functional as F
# Add to safe globals before loading
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

class Logger():
    def __init__(self):
        self.loss_hist = []
        self.train_hist = []
        self.valid_hist = []
        self.test_hist = []
        self.epochs = 0

    def add(self, loss, train, valid, test):
        self.loss_hist.append(loss)
        self.train_hist.append(train)
        self.valid_hist.append(valid)
        self.test_hist.append(test)
        self.epochs += 1

    def export_csv(self):
        print("\n--- CSV_OUTPUT_BEGIN ---")
        print("epoch,loss,train,valid,test")
        for ep in range(1, self.epochs+1):
            print(f"{ep},{self.loss_hist[ep-1]},{100*self.train_hist[ep-1]},{100*self.valid_hist[ep-1]},{100*self.test_hist[ep-1]}")
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
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.normalize(x, p=2., dim=-1)
        x = self.convs[-1](x, adj_t)
        result = x.log_softmax(dim=-1)
        return result

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

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


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "gpu":
        if not torch.cuda.is_available():
            print("Error: --device gpu was requested but CUDA is not available.", file=sys.stderr)
            print("Available devices: cpu", file=sys.stderr)
            sys.exit(1)
        return torch.device("cuda:0")

    print(f"Error: Unknown device '{requested}'. Use 'cpu' or 'gpu'.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"],
                        help="Device to run on: 'cpu' or 'gpu'")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of layers")
    parser.add_argument('--hidden_channels', type=int, default=256, help="Hidden channels")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=1000, help="Training epochs")
    parser.add_argument('--benchmark', action='store_true', help="Run benchmark")
    parser.add_argument('--losstrack', action='store_true', help="Track loss")
    parser.add_argument('--dataset', type=str, default="ogbn-arxiv", choices=["ogbn-arxiv","ogbn-products","ogbn-papers100M"], help="Dataset to use")
    parser.add_argument('--root', type=str, default="~/D1/pyg-dataset", help="Dataset root directory")

    args = parser.parse_args()
    args.root = os.path.expanduser(args.root)
    print(args)

    torch.manual_seed(0)

    device = get_device(args.device)
    print(f"Using device: {device}")
    device = torch.device(device)
    to_symmetric = {"ogbn-arxiv": True, "ogbn-products": False, "ogbn-papers100M": True}
    dataset = PygNodePropPredDataset(name=args.dataset, root=args.root, transform=T.ToSparseTensor())
    data = dataset[0]
    if to_symmetric[args.dataset]:
        data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = SAGE(data.num_features, args.hidden_channels,
                 dataset.num_classes, args.num_layers).to(device)
    print(model)

    evaluator = Evaluator(name=args.dataset)
    logger = Logger()

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_hist = []
    time_hist = []
    model.train()

    if args.benchmark:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        epoch_times = []
        for ep in range(0, 10):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            loss = train(model, data, train_idx, optimizer)
            train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            epoch_times.append(end_time - start_time)

        epoch_times = np.array(epoch_times)
        print(f"Total Time:  {np.sum(epoch_times):.4f} s")
        print(f"Avg Time:    {np.mean(epoch_times):.4f} s")
        print(f"Std Dev:     {np.std(epoch_times):.4f} s")
        print(f"Min Time:    {np.min(epoch_times):.4f} s")
        print(f"Max Time:    {np.max(epoch_times):.4f} s")
        print(f"P95 Time:    {np.percentile(epoch_times, 95):.4f} s")
        print(f"P99 Time:    {np.percentile(epoch_times, 99):.4f} s")
    else:
        for ep in range(1, args.epochs+1):
            loss = train(model, data, train_idx, optimizer)
            train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
            logger.add(loss, train_acc, valid_acc, test_acc)
            if ep % 10 == 0:
                print(f'Epoch: {ep}/{args.epochs}, '
                      f'Loss: {loss:.6f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
        logger.export_csv()
