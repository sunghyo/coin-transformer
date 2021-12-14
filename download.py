import os
import time
import json
import datetime
import argparse
import random
import multiprocessing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import pyupbit


class DownloadUpbitData:
    def __init__(self):
        self.data_path = "./data"
        self.interval = "minute60"  # minute60, minute1
        self.tickers = pyupbit.get_tickers(fiat="KRW")
        print(self.tickers)

    def get_ohlcv(self, ticker="KRW-BTC"):
        """
        https://github.com/sharebook-kr/pyupbit/blob/master/pyupbit/quotation_api.py#L82
        """
        df = pyupbit.get_ohlcv(ticker, count=20000, interval=self.interval, period=0.10)
        return df

    def df_to_dict(self, df):
        data_dict = df.to_dict()
        new_data = {}
        for key in data_dict.keys():
            new_data[key] = list(data_dict[key].values())
            if key == "open":
                timestamps = list(data_dict[key].keys())
                new_data["timestamp"] = [str(x) for x in timestamps]
        return new_data

    def download(self):
        try:
            for ticker in self.tickers:
                print("Download", ticker)
                df = self.get_ohlcv(ticker)
                data_dict = self.df_to_dict(df)
                json_path = os.path.join(
                    self.data_path, f"{ticker}_{self.interval}.json"
                )
                with open(json_path, "w") as outfile:
                    json.dump(data_dict, outfile)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--num_workers", type=int, default=1, help="worker number")
    # args = parser.parse_args()
    data = DownloadUpbitData()
    data.download()