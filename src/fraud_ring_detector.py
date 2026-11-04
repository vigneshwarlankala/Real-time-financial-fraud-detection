
import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "transactions.csv"
OUTPUT_DIR = ROOT / "reports" / "fraud_rings"

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
}


def load_transactions(
    limit=None,
    start_step=None,
    end_step=None,
):
    """
    Load transactions from the requested time range.

    Fraud labels are retained for evaluation only. They are not
    used to generate or rank candidate groups.
    """
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(
        RAW_FILE,
        nrows=limit,
        usecols=lambda col: col in REQUIRED_COLUMNS,
    )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()

    for col in ["step", "amount", "isFraud"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step", "amount", "isFraud"])

    df = df[
        (df["amount"] >= 0)
        & df["isFraud"].isin([0, 1])
    ].copy()

    df["step"] = df["step"].astype(int)
    df["isFraud"] = df["isFraud"].astype(int)
    df["nameOrig"] = df["nameOrig"].astype(str)
    df["nameDest"] = df["nameDest"].astype(str)
    df["type"] = df["type"].astype(str)

    # Apply the requested time window before detection.
    if start_step is not None:
        df = df[df["step"] >= start_step].copy()

    if end_step is not None:
        df = df[df["step"] <= end_step].copy()

    print(f"Transactions loaded: {len(df):,}")

    if not df.empty:
        print(
            f"Selected step range: "
            f"{df['step'].min()}–{df['step'].max()}"
        )
    else:
        print("WARNING: No transactions found in this time range.")

    # Diagnostic only; this count does not affect detection.
    print(
        "Fraud labels reserved for evaluation:",
        int(df["isFraud"].sum()),
    )

    return df


def detect_shared_recipient_groups(
    df,
    min_senders=5,
    max_senders=50,
    min_transactions=5,
):
    """
    Generate and rank shared-recipient candidate groups.

    No fraud labels are used in candidate generation or scoring.
    Scores are heuristic screening scores, not fraud probabilities.
    """
    candidates = []

    for recipient, group in df.groupby("nameDest", sort=False):
        sender_stats = group.groupby("nameOrig").agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
        )

        sender_count = len(sender_stats)
        transaction_count = len(group)

        if not min_senders <= sender_count <= max_senders:
            continue

        if transaction_count < min_transactions:
            continue

        first_step = int(group["step"].min())
        last_step = int(group["step"].max())
        step_span = last_step - first_step

        # Inclusive number of time steps represented by the group.
        inclusive_step_span = step_span + 1
        transactions_per_step = (
            transaction_count / inclusive_step_span
        )

        transaction_types = sorted(
            group["type"].unique().tolist()
        )
        total_amount = float(group["amount"].sum())

        active_senders = int(
            (sender_stats["transaction_count"] >= 2).sum()
        )

        score = 0
        reasons = []

        # 1. Number of distinct senders.
        if sender_count >= 20:
            score += 2
            reasons.append(
                "very high number of distinct senders"
            )
        elif sender_count >= 10:
            score += 1
            reasons.append("high number of distinct senders")
        else:
            score += 1
            reasons.append("multiple distinct senders")

        # 2. Repeat activity among senders.
        if active_senders >= 2:
            score += 1
            reasons.append(
                "multiple senders have repeat activity"
            )

        # 3. CASH_OUT and TRANSFER share.
        high_interest_types = {"CASH_OUT", "TRANSFER"}

        high_interest_count = int(
            group["type"].isin(high_interest_types).sum()
        )

        high_interest_share = (
            high_interest_count / transaction_count
        )

        if high_interest_share >= 0.80:
            score += 2
            reasons.append(
                "at least 80% CASH_OUT or TRANSFER transactions"
            )
        elif high_interest_share >= 0.50:
            score += 1
            reasons.append(
                "at least 50% CASH_OUT or TRANSFER transactions"
            )

        # 4. Transaction-type diversity.
        if len(transaction_types) >= 3:
            score += 1
            reasons.append(
                "three or more transaction types"
            )

        # 5. Temporal persistence bonus.
        # This is exploratory, not proof of fraud.
        if step_span >= 11:
            score += 2
            reasons.append(
                "activity spans 11 or more step intervals"
            )
        elif step_span >= 6:
            score += 1
            reasons.append(
                "activity spans 6 to 10 step intervals"
            )

        candidates.append({
            "group_id": "",
            "recipient_id": recipient,
            "account_count": sender_count + 1,
            "sender_count": sender_count,
            "transaction_count": transaction_count,
            "total_transaction_amount": round(
                total_amount, 2
            ),
            "senders_with_repeat_activity": active_senders,
            "transaction_types": "|".join(transaction_types),
            "first_step": first_step,
            "last_step": last_step,
            "step_span": step_span,
            "transactions_per_step": round(
                transactions_per_step, 4
            ),
            "screening_score": score,
            "high_interest_transaction_share": round(
                high_interest_share, 4
            ),
            "flag_reason": "; ".join(reasons),
            "account_ids": sorted(
                set(sender_stats.index.tolist()) | {recipient}
            ),
            "sender_ids": sorted(
                sender_stats.index.tolist()
            ),
        })

    # Rank only on non-label features.
    candidates.sort(
        key=lambda item: (
            item["screening_score"],
            item["sender_count"],
            item["transaction_count"],
            item["total_transaction_amount"],
        ),
        reverse=True,
    )

    for index, candidate in enumerate(candidates, start=1):
        candidate["group_id"] = f"CANDIDATE-{index:04d}"

    return candidates


def evaluate_candidates(candidates, df):
    """
    Evaluate ranked candidates using fraud labels after detection.

    Matching fraud-labeled edges is not proof of a fraud ring.
    """
    fraud_df = df[df["isFraud"] == 1]

    fraud_pairs = set(
        zip(fraud_df["nameOrig"], fraud_df["nameDest"])
    )
    fraud_recipients = set(fraud_df["nameDest"])

    for candidate in candidates:
        recipient = candidate["recipient_id"]
        senders = set(candidate["sender_ids"])

        matched_edges = sum(
            (sender, recipient) in fraud_pairs
            for sender in senders
        )

        candidate["fraud_labeled_edges_to_recipient"] = int(
            matched_edges
        )
        candidate[
            "recipient_has_fraud_labeled_incoming_edge"
        ] = recipient in fraud_recipients

    groups_with_fraud_edges = sum(
        candidate["fraud_labeled_edges_to_recipient"] > 0
        for candidate in candidates
    )

    groups_with_fraud_recipient = sum(
        candidate[
            "recipient_has_fraud_labeled_incoming_edge"
        ]
        for candidate in candidates
    )

    return {
        "selected_step_start": (
            int(df["step"].min()) if not df.empty else None
        ),
        "selected_step_end": (
            int(df["step"].max()) if not df.empty else None
        ),
        "transactions_in_sample": int(len(df)),
        "fraud_labeled_transactions_in_sample": int(
            df["isFraud"].sum()
        ),
        "candidate_groups": len(candidates),
        "groups_with_fraud_labeled_edges_to_recipient": int(
            groups_with_fraud_edges
        ),
        "groups_whose_recipient_has_fraud_labeled_incoming_edge": int(
            groups_with_fraud_recipient
        ),
        "note": (
            "Fraud labels are used only after candidate generation "
            "and ranking. Shared recipients and label overlap do "
            "not prove a fraud ring. Screening scores are not "
            "calibrated probabilities. Temporal persistence is "
            "an exploratory heuristic."
        ),
    }


def save_results(candidates, evaluation, report_tag=""):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{report_tag}" if report_tag else ""

    summary_rows = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"account_ids", "sender_ids"}
        }
        for candidate in candidates
    ]

    membership_rows = []

    for candidate in candidates:
        for account_id in candidate["account_ids"]:
            membership_rows.append({
                "group_id": candidate["group_id"],
                "recipient_id": candidate["recipient_id"],
                "account_id": account_id,
                "role": (
                    "recipient"
                    if account_id == candidate["recipient_id"]
                    else "sender"
                ),
            })

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / f"fraud_ring_summary{suffix}.csv",
        index=False,
    )

    pd.DataFrame(
        membership_rows,
        columns=[
            "group_id",
            "recipient_id",
            "account_id",
            "role",
        ],
    ).to_csv(
        OUTPUT_DIR / f"fraud_ring_members{suffix}.csv",
        index=False,
    )

    with (
        OUTPUT_DIR / f"fraud_ring_results{suffix}.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "evaluation": evaluation,
                "candidate_groups": candidates,
            },
            file,
            indent=2,
        )

    with (
        OUTPUT_DIR / f"fraud_ring_evaluation{suffix}.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2)

    print("\nSaved reports to:", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect and evaluate shared-recipient candidate groups."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CSV rows to read.",
    )
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--min-senders", type=int, default=5)
    parser.add_argument("--max-senders", type=int, default=50)
    parser.add_argument("--min-transactions", type=int, default=5)
    parser.add_argument(
        "--report-tag",
        type=str,
        default="",
        help="Optional suffix for reports to avoid overwriting them.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    if (
        args.start_step is not None
        and args.start_step < 1
    ):
        parser.error("--start-step must be at least 1.")

    if (
        args.end_step is not None
        and args.end_step < 1
    ):
        parser.error("--end-step must be at least 1.")

    if (
        args.start_step is not None
        and args.end_step is not None
        and args.start_step > args.end_step
    ):
        parser.error("--start-step must be <= --end-step.")

    if args.min_senders < 2:
        parser.error("--min-senders must be at least 2.")

    if args.max_senders < args.min_senders:
        parser.error("--max-senders must be >= --min-senders.")

    if args.min_transactions < 1:
        parser.error("--min-transactions must be positive.")

    if (
        args.report_tag
        and not all(
            char.isalnum() or char in "-_"
            for char in args.report_tag
        )
    ):
        parser.error(
            "--report-tag may contain only letters, numbers, "
            "hyphens, and underscores."
        )

    df = load_transactions(
        limit=args.limit,
        start_step=args.start_step,
        end_step=args.end_step,
    )

    if df.empty:
        parser.error(
            "No valid transactions in the selected range."
        )

    # Detection and ranking do not use fraud labels.
    candidates = detect_shared_recipient_groups(
        df,
        min_senders=args.min_senders,
        max_senders=args.max_senders,
        min_transactions=args.min_transactions,
    )

    # Labels are consulted only after candidate generation/ranking.
    evaluation = evaluate_candidates(candidates, df)

    print("\nCandidate groups found:", len(candidates))
    print("\nEvaluation:")
    print(json.dumps(evaluation, indent=2))

    print("\nTop 10 candidate groups:")
    for candidate in candidates[:10]:
        print(
            f"{candidate['group_id']}: "
            f"recipient={candidate['recipient_id']}, "
            f"senders={candidate['sender_count']}, "
            f"transactions={candidate['transaction_count']}, "
            f"step_span={candidate['step_span']}, "
            f"score={candidate['screening_score']}, "
            f"amount={candidate['total_transaction_amount']:.2f}"
        )

    save_results(
        candidates,
        evaluation,
        report_tag=args.report_tag,
    )


if __name__ == "__main__":
    main()
