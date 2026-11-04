from feast import FeatureStore


def main():
    store = FeatureStore(repo_path=".")
    result = store.get_online_features(
        features=[
            "origin_transaction_features:amount",
            "origin_transaction_features:oldbalanceOrg",
        ],
        entity_rows=[{"nameOrig": "C1231006815"}],
    ).to_dict()

    print("Online feature result:", result)

    assert result["amount"][0] == 9839.64, "Unexpected amount"
    assert result["oldbalanceOrg"][0] == 170136.0, "Unexpected origin balance"
    print("PASS: Feast online retrieval returned expected values.")


if __name__ == "__main__":
    main()
