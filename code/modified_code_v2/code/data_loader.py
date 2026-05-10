import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists() and path_obj.name == "sample_synthetic_prices.csv":
        generate_synthetic_prices(path_obj)
    df = pd.read_csv(path_obj)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"])


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
