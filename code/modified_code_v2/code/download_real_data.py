import argparse
from pathlib import Path
import pandas as pd


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"])


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
        data = data.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
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
    args = parser.parse_args()
    df = download_with_yfinance(args.tickers, args.start, args.end, args.out)
    print(f"已保存 {len(df)} 行数据到 {args.out}")
