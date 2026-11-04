import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "transactions.csv"
MODEL_FILE = ROOT / "models" / "fraud_gnn.pt"
SCALER_FILE = ROOT / "models" / "fraud_gnn_scaler.joblib"

FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "step",
]


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


def build_graph(frame, max_accounts):
    account_ids = {}
    kept_rows = []

    for idx, row in frame.iterrows():
        src, dst = row["nameOrig"], row["nameDest"]
        new_accounts = int(src not in account_ids) + int(dst not in account_ids)

        if len(account_ids) + new_accounts > max_accounts:
            continue

        if src not in account_ids:
            account_ids[src] = len(account_ids)
        if dst not in account_ids:
            account_ids[dst] = len(account_ids)

        kept_rows.append(idx)

    frame = frame.loc[kept_rows].reset_index(drop=True)
    src_ids = frame["nameOrig"].map(account_ids).to_numpy(dtype=np.int64)
    dst_ids = frame["nameDest"].map(account_ids).to_numpy(dtype=np.int64)
    amounts = frame["amount"].to_numpy(dtype=np.float64)

    count_out = np.zeros(len(account_ids), dtype=np.float32)
    count_in = np.zeros(len(account_ids), dtype=np.float32)
    sum_out = np.zeros(len(account_ids), dtype=np.float64)
    sum_in = np.zeros(len(account_ids), dtype=np.float64)

    np.add.at(count_out, src_ids, 1)
    np.add.at(count_in, dst_ids, 1)
    np.add.at(sum_out, src_ids, amounts)
    np.add.at(sum_in, dst_ids, amounts)

    features = np.column_stack([
        sum_out / np.maximum(count_out, 1),
        count_out,
        sum_in / np.maximum(count_in, 1),
        count_in,
        np.log1p(sum_out),
        np.log1p(sum_in),
    ]).astype(np.float32)

    edge_index = torch.tensor(
        np.vstack([
            np.concatenate([src_ids, dst_ids]),
            np.concatenate([dst_ids, src_ids]),
        ]),
        dtype=torch.long,
    )

    return account_ids, features, edge_index, len(frame)


class GNNAccountRiskModel:
    def __init__(self):
        if not RAW_FILE.exists():
            raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")
        if not MODEL_FILE.exists() or not SCALER_FILE.exists():
            raise FileNotFoundError(
                "GNN model or scaler missing. Run src\\train_gnn.py first."
            )

        self.checkpoint = torch.load(
            MODEL_FILE, map_location="cpu", weights_only=False
        )
        self.account_ids = self.checkpoint["account_ids"]
        self.scaler = joblib.load(SCALER_FILE)

        print("Rebuilding the sampled account graph...")
        frame = pd.read_csv(RAW_FILE, nrows=200_000)

        required = set(
            FEATURES + ["nameOrig", "nameDest", "isFraud", "type"]
        )
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {sorted(missing)}")

        frame = frame.dropna(
            subset=["nameOrig", "nameDest", "isFraud"]
        ).copy()

        rebuilt_ids, features, edge_index, self.transaction_count = build_graph(
            frame, max_accounts=len(self.account_ids)
        )

        if rebuilt_ids != self.account_ids:
            raise ValueError(
                "Rebuilt graph does not match the trained model's account mapping."
            )

        features = self.scaler.transform(features).astype(np.float32)
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
        )

        self.model = FraudGraphSAGE(
            input_dim=self.checkpoint["input_dim"],
            hidden_dim=self.checkpoint["hidden_dim"],
            dropout=self.checkpoint["dropout"],
        )
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()

        with torch.no_grad():
            self.scores = torch.sigmoid(
                self.model(data.x, data.edge_index)
            ).numpy()

        print("GNN account-risk inference ready.")
        print(f"Transactions in graph: {self.transaction_count:,}")
        print(f"Accounts in graph: {len(self.account_ids):,}")

    def get_account_risk(self, account_id):
        node_id = self.account_ids.get(account_id)
        if node_id is None:
            return None
        return float(self.scores[node_id])

    def top_accounts(self, limit=10):
        reverse_ids = [None] * len(self.account_ids)
        for account, node_id in self.account_ids.items():
            reverse_ids[node_id] = account

        ranked = np.argsort(self.scores)[::-1][:limit]
        return [
            {
                "account_id": reverse_ids[int(node_id)],
                "risk_score": float(self.scores[node_id]),
            }
            for node_id in ranked
        ]


def main():
    parser = argparse.ArgumentParser(
        description="GraphSAGE account-level fraud-risk inference"
    )
    parser.add_argument(
        "--account-id",
        help="Account ID from the sampled PaySim graph",
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    detector = GNNAccountRiskModel()

    if args.account_id:
        score = detector.get_account_risk(args.account_id)
        if score is None:
            print("Account not found in the sampled graph.")
        else:
            print(f"Account: {args.account_id}")
            print(f"Account-level risk score: {score:.6f}")
    else:
        print("\nHighest-scoring accounts in the sampled graph:")
        for item in detector.top_accounts(max(1, args.top)):
            print(
                f"{item['account_id']}  "
                f"score={item['risk_score']:.6f}"
            )

    print(
        "\nNote: These are account-level prototype scores from the sampled "
        "graph, not transaction-level fraud probabilities or confirmed fraud."
    )


if __name__ == "__main__":
    main()

