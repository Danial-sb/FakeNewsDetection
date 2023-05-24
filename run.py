import argparse
from models import *
from data_loading import *
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['gcn', 'mlp', 'sage', 'gat', 'gin'], default='gcn', help='which model to use (default: gcn)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--seed', type=int, default=1234, help='seed for reproducibility')
parser.add_argument('--hidden_channels', type=int, default=128, help='hidden channles of the layers')
parser.add_argument('--feature', type=str, default='bert', help='feature type: [profile, spacy, bert, content]')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--min_delta', type=float, default=0.001, help='min_delta in early stopping')
parser.add_argument('--patience', type=int, default=10, help='patience in early stopping')
parser.add_argument('--dropout_ratio', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs')
args = parser.parse_args()

train_loader, val_loader, test_loader, train_data = get_data_loader(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(args):
    if args.model == 'gcn':
        model = GCN(train_data.num_features, args.hidden_channels, 1, 1).to(device)
        model.reset_parameters()
    elif args.model == 'sage':
        model = GraphSage(train_data.num_features, args.hidden_channels, 1).to(device)
        model.reset_parameters()
    elif args.model == 'gat':
        model = GAT(train_data.num_features, args.hidden_channels, 1).to(device)
        model.reset_parameters()
    elif args.model == 'gin':
        model = Graph_Isomorphism_Network(train_data.num_features, args.hidden_channels, 1, 3).to(device)
        model.reset_parameters()
    else:
        model = MLP(train_data.num_features, args.hidden_channels, 1).to(device)
        model.reset_parameters()
        
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    loss_fnc = torch.nn.BCELoss()
    
    return optimizer, loss_fnc, model

optimizer, loss_fnc, model = run(args)

torch.manual_seed(args.seed)

def train():
    model.train()
    num_correct = num_examples = total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if args.model in ['gcn', 'sage', 'gat', 'gin']:
          out = model(data.x, data.edge_index, data.batch)
        else:
          out = model(data.x, data.batch)
        loss = loss_fnc(out.view(-1), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        preds = torch.round(out.view(-1)).long()
        num_correct += (preds == data.y.view(-1).long()).sum().item()
        num_examples += data.num_graphs

    accuracy = num_correct / num_examples
    return total_loss / len(train_loader.dataset), accuracy

@torch.no_grad()
def compute_test(loader):
    model.eval()
    num_correct = num_examples = total_loss = 0
    all_preds = []
    all_targets = []
    for data in loader:
        data = data.to(device)
        if args.model in ['gcn', 'sage', 'gat', 'gin']:
          out = model(data.x, data.edge_index, data.batch)
        else:
          out = model(data.x, data.batch)  
        loss = loss_fnc(out.view(-1), data.y.float())
        total_loss += loss.item() * data.num_graphs
        preds = torch.round(out.view(-1)).long()
        all_preds.append(preds.cpu())
        all_targets.append(data.y.view(-1).cpu())
        num_correct += (preds == data.y.view(-1).long()).sum().item()
        num_examples += data.num_graphs
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # Calculate Metrics
    accuracy = num_correct / num_examples
    f1 = f1_score(all_targets, all_preds, average='macro')

    return total_loss / len(loader.dataset), accuracy, f1


results_file = 'results.csv'
try:
    results_df = pd.read_csv(results_file)
except FileNotFoundError:
    results_df = pd.DataFrame(columns=["Model", "Test_Accuracy", "Acc_std", "Test_F1_Score", "F1_std", "Trainable_Parameters"])

def main_experiment():
    best_acc = 0
    best_model = None
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    counter = 0

    for epoch in range(1, 1 + args.epochs):
        train_loss, train_acc = train()
        val_loss, val_acc, val_f1 = compute_test(val_loader)
        print(f'Epoch: {epoch:02d} |  TrainLoss: {train_loss:.2f} | TrainAcc: {train_acc:.2f} '
              f'ValLoss: {val_loss:.2f} | ValAcc: {val_acc:.2f} | ValF1: {val_f1:.2f}')
        
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        current_acc = val_acc

        if current_acc > best_acc + args.min_delta:
            best_acc = current_acc
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= args.patience:
            print(f"Validation performance did not improve by at least {args.min_delta:.3f} for {args.patience} epochs. Stopping training...")
            print(f"Best validation accuracy: {best_acc:.2f}")
            break
    print("========================================================================")
    print()
    model.load_state_dict(best_model)
    _, acc_test, f1_test = compute_test(test_loader)
    print(f'Test Accuracy: {acc_test:.2f}, Test F1 Score: {f1_test:.2f}, Model: {args.model}')

    return acc_test, f1_test, train_loss_history, train_acc_history, val_loss_history, val_acc_history

if __name__ == '__main__':
    num_runs = 5
    test_accuracies = []
    f1_scores = []
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        model.reset_parameters()
        test_acc, f1_sco, _, _, _, _ = main_experiment()
        test_accuracies.append(test_acc)
        f1_scores.append(f1_sco)

    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    f1_score_mean = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print(f"Mean test accuracy over {num_runs} runs: {mean_acc:.3f} +- {std_acc:.3f}\n Mean F1 Score over {num_runs} runs: {f1_score_mean:.3f} +- {std_f1:.3f}\n No. of parameters: {count_parameters(model)}")
    result = {"Model": args.model, "Test_Accuracy": mean_acc, "Acc_std":std_acc, "Test_F1_Score": f1_score_mean, "F1_std": std_f1, "Trainable_Parameters": count_parameters(model)}
    results_df = results_df.append(result, ignore_index=True)
    results_df.to_csv(results_file, index=False)  