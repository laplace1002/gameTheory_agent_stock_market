# 数据说明

1. `sample_synthetic_prices.csv` 是离线演示数据，只用于保证项目在没有网络时也能跑通。
2. 正式实验应使用真实股票日行情数据。推荐运行：
   `python code/download_real_data.py --tickers AAPL MSFT GOOGL AMZN NVDA TSLA --start 2022-01-01 --end 2025-12-31 --out data/market_prices.csv`
3. 下载后再运行：
   `python code/run_experiment.py --prices data/market_prices.csv`
4. 本项目只做课程研究和纸面交易模拟，不构成任何投资建议。
