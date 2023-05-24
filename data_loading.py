from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

def get_data_loader(args):
    train_data = UPFD(root=".", name="gossipcop", feature=args.feature, split="train")
    val_data = UPFD(root=".", name="gossipcop", feature=args.feature, split="val")
    test_data = UPFD(root=".", name="gossipcop", feature=args.feature, split="test")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data

