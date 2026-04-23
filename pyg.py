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

def train(model, data, train_idx, optimizer, evaluator):
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx], reduction="mean")
    y_pred = out.argmax(dim=-1, keepdim=True)
    acc = evaluator.eval({'y_true': data.y[train_idx], 'y_pred': y_pred})['acc']
    loss.backward()
    optimizer.step()
    return loss, acc

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

DATASETS = {
    "arxiv":      {"name": "ogbn-arxiv",     "symmetric": True},
    "papers100M": {"name": "ogbn-papers100M", "symmetric": True},
    "products":   {"name": "ogbn-products",  "symmetric": False},
}

def load_dataset(dataset_key, data_dir):
    cfg = DATASETS[dataset_key]
    dataset = PygNodePropPredDataset(
        name=cfg["name"], root=data_dir, transform=T.ToSparseTensor()
    )
    data = dataset[0]
    if cfg["symmetric"]:
        data.adj_t = data.adj_t.to_symmetric()
    return dataset, data

def main(args):
    torch.manual_seed(0)

    device = get_device(args.device)
    print(f"Using device: {device}")
    device = torch.device(device)

    dataset, data = load_dataset(args.dataset, args.datadir)
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = SAGE(data.num_features, args.hidden_channels,
                 dataset.num_classes, args.num_layers).to(device)
    print(model)

    evaluator = Evaluator(name=DATASETS[args.dataset]["name"])

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_hist = []
    time_hist = []
    model.train()

    warmup = 10
    for epoch in range(warmup):
        loss, acc = train(model, data, train_idx, optimizer, evaluator)
        if (args.losstrack):
            loss_hist.append(loss.item())
        if args.log:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {100 * acc:.4f}, Avg Epochs/s: warming up')

    for epoch in range(1, 1 + args.epochs - warmup):
        t = time.perf_counter()
        loss, acc = train(model, data, train_idx, optimizer, evaluator)
        time_hist.append(time.perf_counter() - t)
        if (args.losstrack):
            loss_hist.append(loss.item())
        if args.log:
            avg_eps = epoch / sum(time_hist)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {100 * acc:.4f}, Avg Epochs/s: {avg_eps:.2f}')

    train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
    print(f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')

    times = np.array(time_hist)
    print(f"Time: min={times.min():.4f}s, max={times.max():.4f}s, "
          f"avg={times.mean():.4f}s, std={times.std():.4f}s, "
          f"total={times.sum():.4f}s")

    if (args.losstrack):
        print(f"Loss history:\n{loss_hist}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"],
                        help="Device to run on: 'cpu' or 'gpu' (default: gpu)")
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--losstrack', action='store_true')
    parser.add_argument('--dataset', type=str, default="arxiv", choices=DATASETS.keys())
    parser.add_argument('--datadir', type=str, default="~/D1/pyg-dataset")

    args = parser.parse_args()
    args.datadir = os.path.expanduser(args.datadir)
    print(args)

    main(args)
