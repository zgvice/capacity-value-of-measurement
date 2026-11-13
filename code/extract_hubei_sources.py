"""Extract event and settlement readings from the anonymized Hubei source sheets."""

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "inputs" / "hubei_sources"
OUTPUT = ROOT / "results" / "hubei"


def event_records():
    records = []
    workbook = load_workbook(SOURCES / "events.xlsx", read_only=True, data_only=True)
    try:
        for sheet in workbook:
            rows = list(sheet.values)
            customer, date, source, source_sheet, priority = (row[1] for row in rows[:5])
            for column, values in enumerate(zip(*rows[5:], strict=True), start=1):
                clock, baseline, nomination, target, actual, reported_response = values
                if column == 1 or clock is None or None in (baseline, target, actual):
                    continue
                hour, minute = map(int, clock.split(":"))
                derived = nomination is None
                if derived:
                    nomination = target - baseline
                if nomination <= 0:
                    continue
                response = actual - baseline
                error = actual - target
                records.append(dict(
                    user=f"LSE {customer}", event=date[5:], minute=60 * hour + minute,
                    time=clock, baseline_kW=baseline, nomination_kW=nomination,
                    target_kW=target, actual_kW=actual, response_increase_kW=response,
                    target_error_kW=error, target_error_fraction=error / target,
                    response_ratio=response / nomination, nomination_derived=derived,
                    view="main", priority=priority, source=source, sheet=source_sheet,
                    column=column, target_identity_error_kW=target - baseline - nomination,
                    response_identity_error_kW=(
                        0.0 if reported_response is None else response - reported_response
                    ),
                    customer_code=customer,
                ))
    finally:
        workbook.close()
    return pd.DataFrame(records).sort_values(["user", "event", "minute"])


def settlement_rows(sheet):
    rows = iter(sheet.values)
    head = next(rows)
    minutes = [60 * int(h[1:3]) + int(h[3:]) for h in head[3:]]
    records = {}
    for date, customer, account, *values in rows:
        key = (date, account)
        if key in records:
            raise ValueError(f"Duplicate settlement row: {date}, {account}")
        records[key] = (customer, dict(zip(minutes, values, strict=True)))
    return records


def paired_readings(events):
    workbook = load_workbook(SOURCES / "settlement.xlsx", read_only=True, data_only=True)
    try:
        baseline = settlement_rows(workbook["Baseline"])
        meter = settlement_rows(workbook["Meter"])
    finally:
        workbook.close()
    if baseline.keys() != meter.keys():
        raise ValueError("Baseline and meter customer-days differ.")
    windows = events.groupby(["customer_code", "event"])["minute"].agg(["min", "max"])
    records = []
    for key, (customer, before_response) in baseline.items():
        date, account = key
        metered_customer, metered = meter[key]
        if customer != metered_customer or before_response.keys() != metered.keys():
            raise ValueError(f"Unmatched baseline and meter readings: {date}, {account}")
        start, end = windows.loc[customer, date[5:]]
        for minute, value in before_response.items():
            measured = metered[minute]
            segment = "调用前" if minute < start else "调用后" if minute > end else "调用中"
            records.append({
                "客户": f"LSE {customer}", "日期": date, "分钟": minute, "片段": segment,
                "基线_千瓦": value, "电表_千瓦": measured, "差值_千瓦": value - measured,
                "customer_code": customer,
            })
    return pd.DataFrame(records)


def main():
    events = event_records()
    pairs = paired_readings(events)
    diagnostics = OUTPUT / "diag"
    diagnostics.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT / "meter_series.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(diagnostics / "baseline_meter_pairs.csv", index=False, encoding="utf-8-sig")

    data = ROOT / "data"
    aliases = pd.read_csv(data / "Customer_Labels.csv").set_index("customer_code")["customer_alias"]
    events.assign(user=events["customer_code"].map(aliases)).to_csv(
        data / "D01.csv", index=False, encoding="utf-8-sig"
    )
    pairs.assign(客户=pairs["customer_code"].map(aliases)).to_csv(
        data / "D34.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
