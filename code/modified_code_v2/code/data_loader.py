import argparse
from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def load_prices(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists() and path_obj.name == "sample_synthetic_prices.csv":
        generate_synthetic_prices(path_obj)
    df = read_price_file(path_obj)
    df = normalize_price_columns(df)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close"])
    df = df[df["close"] > 0]
    df["volume"] = df["volume"].fillna(0)
    return df.sort_values(["date", "ticker"])[["date", "ticker", "open", "high", "low", "close", "volume"]]


def read_price_file(path: str | Path) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj, dtype={"Stkcd": str, "ticker": str})
    if suffix in {".xlsx", ".xlsm"}:
        return read_excel_prices(path_obj)
    raise ValueError(f"不支持的行情文件格式: {path_obj.suffix}")


def read_excel_prices(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, dtype={"Stkcd": str, "ticker": str}, skiprows=[1, 2])
    except ImportError:
        return read_xlsx_without_openpyxl(path, skip_rows={2, 3})


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    rename = {}
    normalized_columns = {column: normalize_column_name(column) for column in work.columns}
    aliases = {
        "date": {"date", "trddt", "tradedate", "tradingdate", "交易日期"},
        "ticker": {"ticker", "symbol", "stkcd", "secucode", "证券代码"},
        "open": {"open", "opnprc", "openprice", "日开盘价"},
        "high": {"high", "hiprc", "highprice", "日最高价"},
        "low": {"low", "loprc", "lowprice", "日最低价"},
        "close": {"close", "clsprc", "closeprice", "日收盘价"},
        "volume": {"volume", "vol", "dnshrtrd", "日个股交易股数"},
    }
    for target, names in aliases.items():
        for column, normalized in normalized_columns.items():
            if normalized in {normalize_column_name(name) for name in names}:
                rename[column] = target
                break
    work = work.rename(columns=rename)
    if "ticker" in work.columns:
        work["ticker"] = work["ticker"].map(format_ticker)
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    return work


def normalize_column_name(value) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "")).lower()


def format_ticker(value) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d+(?:\.0)?", text):
        text = text.split(".", 1)[0]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text.upper()


def read_xlsx_without_openpyxl(path: str | Path, skip_rows: set[int] | None = None) -> pd.DataFrame:
    skip_rows = skip_rows or set()
    with ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet_path = first_sheet_path(archive)
        rows = []
        for row_number, values in iter_xlsx_rows(archive, sheet_path, shared_strings):
            if row_number in skip_rows:
                continue
            rows.append(values)
    if not rows:
        return pd.DataFrame()
    header = [str(value) for value in rows[0]]
    data = rows[1:]
    width = len(header)
    normalized = [row + [""] * max(0, width - len(row)) for row in data]
    return pd.DataFrame([row[:width] for row in normalized], columns=header)


def read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    out = []
    for item in root.findall("a:si", ns):
        out.append("".join(node.text or "" for node in item.findall(".//a:t", ns)))
    return out


def first_sheet_path(archive: ZipFile) -> str:
    for name in archive.namelist():
        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
            return name
    raise ValueError("xlsx 文件中没有找到 worksheet")


def iter_xlsx_rows(archive: ZipFile, sheet_path: str, shared_strings: list[str]):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    for _, elem in ET.iterparse(archive.open(sheet_path), events=("end",)):
        if not elem.tag.endswith("}row"):
            continue
        row_number = int(elem.get("r", "0") or 0)
        values = []
        for cell in elem.findall("a:c", ns):
            index = column_index_from_cell_ref(cell.get("r", ""))
            while len(values) < index:
                values.append("")
            values.append(cell_value(cell, shared_strings, ns))
        yield row_number, values
        elem.clear()


def cell_value(cell, shared_strings: list[str], ns: dict) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", ns))
    value_node = cell.find("a:v", ns)
    value = "" if value_node is None else value_node.text or ""
    if cell_type == "s" and value:
        return shared_strings[int(value)]
    return value


def column_index_from_cell_ref(ref: str) -> int:
    letters = "".join(ch for ch in str(ref) if ch.isalpha())
    index = 0
    for letter in letters:
        index = index * 26 + (ord(letter.upper()) - ord("A") + 1)
    return max(1, index)


def generate_synthetic_prices(
    out_path: str | Path,
    tickers=None,
    start: str = "2023-01-02",
    periods: int = 360,
    seed: int = 2026,
) -> pd.DataFrame:
    """Generate a deterministic multi-regime market for demos and tests.

    The synthetic market deliberately contains trend, reversal, shock and recovery
    regimes so that cooperation/competition choices have visible consequences.
    """
    out_path = Path(out_path)
    tickers = tickers or ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA", "SIGMA"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)

    regimes = []
    for idx in range(periods):
        if idx < periods * 0.25:
            regimes.append((0.0009, 0.012, "trend"))
        elif idx < periods * 0.50:
            regimes.append((-0.0002, 0.018, "rotation"))
        elif idx < periods * 0.68:
            regimes.append((-0.0010, 0.030, "shock"))
        elif idx < periods * 0.84:
            regimes.append((0.0004, 0.022, "recovery"))
        else:
            regimes.append((0.0001, 0.014, "sideways"))

    rows = []
    for t_index, ticker in enumerate(tickers):
        price = 65 + 15 * t_index + rng.normal(0, 2)
        beta = 0.65 + 0.13 * t_index
        reversal_bias = (-1) ** t_index * 0.00035
        for date, (drift, vol, regime) in zip(dates, regimes):
            idio = rng.normal(0, vol * (0.75 + 0.08 * t_index))
            seasonal = 0.004 * np.sin((len(rows) + t_index * 11) / 17)
            if regime == "rotation" and t_index % 2 == 0:
                idio -= 0.006
            if regime == "shock" and ticker in {"GAMMA", "OMEGA"}:
                idio -= 0.012
            if regime == "recovery" and ticker in {"GAMMA", "OMEGA"}:
                idio += 0.010
            ret = drift * beta + idio + seasonal + reversal_bias
            prev = price
            price = max(3.0, price * (1 + ret))
            high = max(prev, price) * (1 + abs(rng.normal(0.003, 0.004)))
            low = min(prev, price) * (1 - abs(rng.normal(0.003, 0.004)))
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "open": round(prev, 4),
                    "high": round(high, 4),
                    "low": round(max(0.5, low), 4),
                    "close": round(price, 4),
                    "volume": int(rng.integers(500_000, 5_000_000)),
                }
            )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def download_with_yfinance(tickers, start, end, out_path):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("请先安装 yfinance：pip install yfinance") from exc

    frames = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            print(f"警告：{ticker} 没有下载到数据")
            continue
        data = data.reset_index()
        data["ticker"] = ticker
        data = data.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frames.append(data[["date", "ticker", "open", "high", "low", "close", "volume"]])
    if not frames:
        raise RuntimeError("没有下载到任何数据。请检查网络、ticker 或日期范围。")
    df = pd.concat(frames, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"])
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--out", default="data/market_prices.csv")
    parser.add_argument("--synthetic", action="store_true", help="生成离线合成行情，不访问网络")
    args = parser.parse_args()
    if args.synthetic:
        df = generate_synthetic_prices(args.out, tickers=args.tickers, start=args.start)
    else:
        df = download_with_yfinance(args.tickers, args.start, args.end, args.out)
    print(f"已保存 {len(df)} 行数据到 {args.out}")
