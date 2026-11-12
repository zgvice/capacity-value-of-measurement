"""Pure readers for frozen official workbooks, retained from the verified parser."""
from pathlib import Path
import re,zipfile
import numpy as np
import pandas as pd
from lxml import etree
NS="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
EPOCH=np.datetime64("1899-12-30","D")
OUT_COLS=["TotalUnlimitedOutput","TotalVariableOutput","TotalLimitedDurationAndCombinationOutput","DROutput"]

class OfficialInputError(ValueError):
    pass

def _column(ref):
    letters = re.match(r"[A-Z]+", ref).group()
    n = 0
    for char in letters:
        n = 26 * n + ord(char) - 64
    return n - 1

def read_numeric_xlsx(path):
    """Stream the actual one-sheet numeric tables, preserving header positions.

    Dates remain original Excel serials here. Their known date columns are
    converted separately. No worksheet formulas are evaluated or authored.
    """
    with zipfile.ZipFile(path) as z:
        sheets = [n for n in z.namelist()
                  if re.fullmatch(r"xl/worksheets/sheet[0-9]+\.xml", n)]
        if sheets != ["xl/worksheets/sheet1.xml"]:
            raise OfficialInputError(f"Unexpected sheet layout: {sheets}")
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = etree.fromstring(z.read("xl/sharedStrings.xml"))
            strings = ["".join(si.itertext()) for si in tree]
        with z.open(sheets[0]) as f:
            prefix = f.read(2048)
        match = re.search(rb'<dimension ref="([A-Z]+)(\d+):([A-Z]+)(\d+)"', prefix)
        if not match:
            raise OfficialInputError("Missing or unsupported worksheet dimensions")
        nrow = int(match[4]); ncol = _column(match[3].decode()+"1") + 1
        data = np.full((nrow-1, ncol), np.nan, dtype=np.float64)
        header = [None] * ncol
        count = 0
        with z.open(sheets[0]) as f:
            for _, row in etree.iterparse(f, events=("end",), tag=NS+"row"):
                ri = int(row.attrib["r"]) - 1
                for cell in row:
                    ci = _column(cell.attrib["r"])
                    value = cell.findtext(NS+"v")
                    typ = cell.attrib.get("t", "n")
                    if ri == 0:
                        if typ == "s":
                            header[ci] = strings[int(value)]
                        elif typ == "inlineStr":
                            header[ci] = "".join(cell.itertext())
                        elif value is not None:
                            header[ci] = float(value)
                    elif value is not None:
                        if typ not in ("n", ""):
                            raise OfficialInputError("Non-numeric body in numeric workbook")
                        data[ri-1, ci] = float(value)
                if ri:
                    count += 1
                row.clear()
                while row.getprevious() is not None:
                    del row.getparent()[0]
        if count != nrow-1 or not np.isfinite(data).all():
            raise OfficialInputError("Incomplete numeric worksheet; no silent fill")
    return header, data

def profile_keys(names, expected_years=31, expected_rotations=13):
    keys = []
    for name in names:
        match = re.fullmatch(r"([A-M])([0-9]{4})", str(name))
        if not match:
            raise OfficialInputError(f"Unexpected profile label: {name}")
        keys.append((int(match[2]), ord(match[1])-ord("A")))
    if len(set(keys)) != len(keys):
        raise OfficialInputError("Duplicate profile labels")
    years = sorted({y for y, _ in keys})
    expected = {(y, r) for y in years for r in range(expected_rotations)}
    if (len(years) != expected_years or set(keys) != expected
            or years != list(range(years[0], years[0]+expected_years))):
        raise OfficialInputError("Profiles do not cover the full weather/rotation product")
    return keys, years

def validate_sparse(frame, atoms, hours):
    need = ["Year", "Rotation", "Replication", "HourBeginningInYear", "Weight",
            "DayOfDeliveryYear", "HourBeginning", "Margin", "LOLEFlag", "Load"] + OUT_COLS
    missing = set(need) - set(frame)
    if missing:
        raise OfficialInputError(f"Missing shortage columns: {sorted(missing)}")
    if frame[need].isna().any().any():
        raise OfficialInputError("Null primitive in shortage table")
    key = ["Year", "Rotation", "Replication", "HourBeginningInYear"]
    if frame.duplicated(key).any():
        raise OfficialInputError("Duplicate atom/hour in shortage table")
    allowed = set(map(tuple, atoms[["Year", "Rotation", "Replication"]].to_numpy()))
    actual = list(map(tuple, frame[["Year", "Rotation", "Replication"]].to_numpy()))
    if not all(k in allowed for k in actual):
        raise OfficialInputError("Shortage row outside full atom registry")
    hh = frame.HourBeginningInYear.to_numpy()
    if (not np.equal(hh, np.floor(hh)).all() or np.any(hh < 0)
            or np.any(hh >= hours)):
        raise OfficialInputError("Invalid model-hour index")
    if not np.array_equal(hh, 24*frame.DayOfDeliveryYear+frame.HourBeginning):
        raise OfficialInputError("Hour/day identities disagree")
    p = atoms.set_index(["Year", "Rotation", "Replication"])["p"]
    want = np.array([p.loc[k] for k in actual])
    if not np.allclose(frame.Weight, want, rtol=0, atol=1e-15):
        raise OfficialInputError("Source annual weights disagree; no renormalization permitted")
    balance = frame[OUT_COLS].sum(axis=1)-frame.Load
    residual = float(np.max(np.abs(balance-frame.Margin)))
    if residual > 1e-6:
        raise OfficialInputError(f"Margin is not output minus Load: {residual}")
    if not ((frame.Margin < 0) & (frame.LOLEFlag == 1)).all():
        raise OfficialInputError("Sparse table contains non-shortage or inconsistent flag")
    return residual

def reliability_metrics(frame):
    lost = frame.loc[frame.Margin < 0]
    return {
        "EUE_MWh": float((-lost.Margin*lost.Weight).sum()),
        "LOLH_hours": float(lost.Weight.sum()),
        "LOLE_days": float(lost.drop_duplicates(
            ["Year", "Rotation", "Replication", "DayOfDeliveryYear"]).Weight.sum())
    }

def _serial_day(series):
    return (pd.to_datetime(series).to_numpy(dtype="datetime64[D]")-EPOCH).astype(np.int32)

def _model_dates(start_year, hours):
    # This mapping is corroborated against every published Date/Day row below.
    # It is model chronology, not a claim that the underlying weather preprocessor
    # removed Feb 29, nor a UTC/EPT localization.
    return (np.datetime64(f"{start_year}-06-01", "D")
            + np.arange(hours)//24)
