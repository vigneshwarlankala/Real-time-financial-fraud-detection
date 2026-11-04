import pandas as pd
from feast import FeatureStore


def main():
    store = FeatureStore(repo_path=".", fs_yaml_file="feature_store_test.yaml")

    entity_df = pd.DataFrame({
        "event_timestamp": [
            pd.Timestamp("2020-01-01 01:00:00", tz="UTC")
        ],
        "nameOrig": ["C1231006815"],
    })

    features = [
        "origin_transaction_features:amount",
        "origin_transaction_features:oldbalanceOrg",
        "origin_transaction_features:newbalanceOrig",
        "origin_transaction_features:origin_balance_change",
        "origin_transaction_features:step",
    ]

    print("Requesting historical features from Feast...")
    result = store.get_historical_features(
        entity_df=entity_df,
        features=features,
    ).to_df()

    print("\nHistorical feature result:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()


