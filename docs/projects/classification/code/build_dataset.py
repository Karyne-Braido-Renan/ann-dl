"""Constrói o dataset analítico da classificação diretamente dos JSONs do EDA."""

import json
from json import JSONDecoder
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = {"Conservadora", "Moderada", "Sofisticada"}
BUCKETS = {
    "stocks": "pct_stocks",
    "fixedIncome": "pct_fixed_income",
    "repo": "pct_repo",
    "investimentFund": "pct_investment_fund",
    "balance": "pct_cash",
    "options": "pct_options",
    "treasure": "pct_treasure",
    "future": "pct_future",
    "privatePension": "pct_private_pension",
    "term": "pct_term",
    "coe": "pct_coe",
    "realEstate": "pct_real_estate",
    "structruedProduct": "pct_structured_product",
}


def stream_json_array(path, chunk_size=1024 * 1024):
    """Lê a wallet, que é uma lista JSON grande, sem carregar tudo de uma vez."""
    decoder = JSONDecoder()
    with open(path, encoding="utf-8") as file:
        buffer = ""
        started = False
        finished = False

        while True:
            if not finished and len(buffer) < chunk_size:
                chunk = file.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    finished = True

            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    if finished:
                        return
                    continue
                if buffer[0] != "[":
                    raise ValueError("A wallet deve ter uma lista no nível superior.")
                buffer = buffer[1:]
                started = True
                continue

            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            if not buffer:
                if finished:
                    return
                continue

            try:
                item, end = decoder.raw_decode(buffer)
            except ValueError:
                if finished:
                    raise
                buffer += file.read(chunk_size)
                continue

            buffer = buffer[end:]
            yield item


def bucket_percent(bucket):
    if not isinstance(bucket, dict):
        return 0.0
    value = bucket.get("percent", bucket.get("percentDetail"))
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def load_wallet(path):
    rows = []
    for item in stream_json_array(path):
        response = item.get("response") or {}
        output = response.get("output") or {}
        success = bool(response.get("success", False))
        row = {
            "accountCode": str(item.get("client")),
            "wallet_ok": success,
            "wallet_updated_at": output.get("updatedAt"),
        }
        values = []
        for source, column in BUCKETS.items():
            value = bucket_percent(output.get(source)) if success else np.nan
            row[column] = value
            values.append(value)
        row["num_active_buckets"] = int(sum(value > 0 for value in values)) if success else np.nan
        row["max_bucket_pct"] = max(values) if success else np.nan
        row["pct_sum_selected"] = sum(values) if success else np.nan
        row["empty_wallet"] = bool(all(value == 0 for value in values)) if success else False
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(data_dir):
    data_dir = Path(data_dir)
    wallet = load_wallet(data_dir / "clientes_wallet_distribution.json")

    with open(data_dir / "clientes_investment_policy.json", encoding="utf-8") as file:
        policy_json = json.load(file)
    policy = pd.DataFrame([
        {
            "accountCode": str(account_code),
            "investmentPolicy": ((obj or {}).get("output") or {}).get("adherence", {}).get("investmentPolicy"),
        }
        for account_code, obj in policy_json.items()
    ])
    policy = policy[policy["investmentPolicy"].isin(CLASSES)].drop_duplicates("accountCode")

    with open(data_dir / "account.json", encoding="utf-8") as file:
        account = pd.DataFrame(json.load(file).get("data", []))
    account["accountCode"] = account["accountCode"].astype(str)
    if "currentRegisterIndicator" in account:
        current = account["currentRegisterIndicator"] == 1
        if current.any():
            account = account[current]
    account = account.drop_duplicates("accountCode", keep="last")

    result = (
        wallet[wallet["wallet_ok"]]
        .merge(policy, on="accountCode", how="inner")
        .merge(account[["accountCode", "personType"]], on="accountCode", how="left")
    )
    return result


if __name__ == "__main__":
    data = build_dataset(Path(__file__).parents[1] / "Dados")
    print(data.shape)
    print(data["investmentPolicy"].value_counts())
