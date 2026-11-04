import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "transactions.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "step",
]
SEED = 42


class FraudGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.25):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.classifier(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-accounts", type=int, default=100_000)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    print(f"Reading up to {args.max_rows:,} transactions...")
    # Preserve all fraud rows within the selected chronological prefix.
    frame = pd.read_csv(RAW_FILE, nrows=args.max_rows)
    required = set(FEATURES + ["nameOrig", "nameDest", "isFraud", "type"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    frame = frame.dropna(subset=["nameOrig", "nameDest", "isFraud"]).copy()
    frame["isFraud"] = frame["isFraud"].astype("int64")

    # Bound graph size by retaining rows whose endpoints fit the account cap.
    account_ids = {}
    kept_rows = []
    for idx, row in frame.iterrows():
        src = row["nameOrig"]
        dst = row["nameDest"]
        new_accounts = int(src not in account_ids) + int(dst not in account_ids)
        if len(account_ids) + new_accounts > args.max_accounts:
            continue
        if src not in account_ids:
            account_ids[src] = len(account_ids)
        if dst not in account_ids:
            account_ids[dst] = len(account_ids)
        kept_rows.append(idx)

    frame = frame.loc[kept_rows].reset_index(drop=True)
    if len(frame) < 100:
        raise ValueError("Too few transactions remain for training.")

    print(f"Transactions used: {len(frame):,}")
    print(f"Accounts (nodes): {len(account_ids):,}")
    print(f"Fraud transactions: {int(frame['isFraud'].sum()):,}")

    # Account node features: average outgoing/incoming amounts and transaction counts.
    numeric = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    node_features = np.zeros((len(account_ids), 6), dtype=np.float32)
    node_counts = np.zeros((len(account_ids), 2), dtype=np.float32)
    node_sums = np.zeros((len(account_ids), 2), dtype=np.float64)

    src_ids = frame["nameOrig"].map(account_ids).to_numpy(dtype=np.int64)
    dst_ids = frame["nameDest"].map(account_ids).to_numpy(dtype=np.int64)
    amounts = frame["amount"].to_numpy(dtype=np.float64)

    np.add.at(node_counts[:, 0], src_ids, 1)
    np.add.at(node_counts[:, 1], dst_ids, 1)
    np.add.at(node_sums[:, 0], src_ids, amounts)
    np.add.at(node_sums[:, 1], dst_ids, amounts)

    # Aggregate per-account balances from the first occurrence in the sample.
    for col_idx, col in enumerate(FEATURES):
        values = numeric[col].to_numpy(dtype=np.float32)
        node_features[:, col_idx] = 0.0
        np.add.at(node_features[:, col_idx], src_ids, values)
    outgoing_den = np.maximum(node_counts[:, 0], 1)
    node_features[:, 0] = (node_sums[:, 0] / outgoing_den).astype(np.float32)
    node_features[:, 1] = node_counts[:, 0]
    node_features[:, 2] = (node_sums[:, 1] / np.maximum(node_counts[:, 1], 1)).astype(np.float32)
    node_features[:, 3] = node_counts[:, 1]
    node_features[:, 4] = np.log1p(node_sums[:, 0]).astype(np.float32)
    node_features[:, 5] = np.log1p(node_sums[:, 1]).astype(np.float32)

    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features).astype(np.float32)

    # One directed edge per transaction, with reverse connectivity for message passing.
    edge_index = torch.tensor(
        np.vstack([
            np.concatenate([src_ids, dst_ids]),
            np.concatenate([dst_ids, src_ids]),
        ]),
        dtype=torch.long,
    )
    x = torch.tensor(node_features, dtype=torch.float32)

    # Transaction labels are mapped to sender-account nodes. Account label is positive
    # if any outgoing transaction from that account in this sample is fraudulent.
    y = np.zeros(len(account_ids), dtype=np.float32)
    fraud_senders = frame.loc[frame["isFraud"] == 1, "nameOrig"].map(account_ids).to_numpy()
    y[fraud_senders] = 1.0

    labeled_nodes = np.arange(len(account_ids))
    train_nodes, test_nodes = train_test_split(
        labeled_nodes,
        test_size=0.2,
        random_state=SEED,
        stratify=y if len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2 else None,
    )

    data = Data(x=x, edge_index=edge_index)
    model = FraudGraphSAGE(input_dim=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    positives = max(float(y[train_nodes].sum()), 1.0)
    negatives = max(float(len(train_nodes) - y[train_nodes].sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    train_idx = torch.tensor(train_nodes, dtype=torch.long)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Training GraphSAGE...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_idx], y_tensor[train_idx], pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch:02d}/{args.epochs} - loss: {loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(data.x, data.edge_index)).numpy()

    truth = y[test_nodes].astype(int)
    scores = probabilities[test_nodes]
    predictions = (scores >= 0.5).astype(int)

    metrics = {
        "model": "GraphSAGE",
        "target": "account-level label: account has at least one fraudulent outgoing transaction in sampled data",
        "sampled_transactions": int(len(frame)),
        "accounts": int(len(account_ids)),
        "fraud_transactions_in_sample": int(frame["isFraud"].sum()),
        "fraud_labeled_accounts": int(y.sum()),
        "epochs": int(args.epochs),
        "threshold": 0.5,
        "average_precision": float(average_precision_score(truth, scores)) if len(np.unique(truth)) > 1 else None,
        "roc_auc": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) > 1 else None,
        "classification_report": classification_report(
            truth, predictions, zero_division=0, output_dict=True
        ),
        "note": "Sampled-data prototype; account-level labels are not transaction-level fraud predictions.",
    }

    model_path = MODEL_DIR / "fraud_gnn.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dim": 32,
        "dropout": 0.25,
        "account_ids": account_ids,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }, model_path)
    joblib.dump(scaler, MODEL_DIR / "fraud_gnn_scaler.joblib")
    metrics_path = REPORT_DIR / "gnn_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nGNN training complete.")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Average precision: {metrics['average_precision']}")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print("\nClassification report:")
    print(classification_report(truth, predictions, zero_division=0))


if __name__ == "__main__":
    main()
