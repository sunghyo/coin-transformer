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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pyupbit

# import talib


class UpbitDataset:
    def __init__(self):
        self.tickers = pyupbit.get_tickers(fiat="KRW")
        # self.tickers = self.tickers[:2]

        self.interval = 3600  # seconds 3600, 900
        self.cache_path = "./data"

        self.start_date = datetime(2019, 10, 1, 9)
        self.end_date = datetime(2021, 11, 18, 8)
        # self.end_date = datetime(2021, 11, 18, 8, 45)

    def get_data(self):
        train_list = []
        val_list = []
        for ticker in self.tickers:
            try:
                _train_list, _val_list = self.get_ticker_data(ticker)
                print(f"{ticker} train: {len(_train_list)} val: {len(_val_list)}")
            except Exception as e:
                print("Skip", ticker, e)
                continue

            train_list.extend(_train_list)
            val_list.extend(_val_list)

        print("Num train", len(train_list))
        print("Num val", len(val_list))
        return train_list, val_list

    def get_ticker_data(self, ticker):
        data_dict = self.read_json(ticker)
        full_df_dict = self.gen_full_df(data_dict)
        full_df_dict = self.normalize(full_df_dict)
        day_list = self.split_and_label(full_df_dict)
        day_list = day_list[1:-1]  # remove first and last days

        num_train = int(len(day_list) * 0.9)
        train_list, val_list = day_list[:num_train], day_list[num_train:]
        return train_list, val_list

    def read_json(self, ticker):
        json_path = os.path.join(self.cache_path, f"{ticker}.json")
        # json_path = os.path.join(self.cache_path, f"{ticker}_minute15.json")

        with open(json_path) as json_file:
            data = json.load(json_file)

        data["timestamp"] = [
            datetime.strptime(x, "%Y-%m-%d %X") for x in data["timestamp"]
        ]
        df = pd.DataFrame.from_dict(data)
        df = df.set_index("timestamp")
        data_dict = df.to_dict()
        return data_dict

    def gen_full_df(self, data_dict):
        timestamp_list = []
        current_date = self.start_date

        while True:
            timestamp_list.append(current_date)
            current_date = current_date + timedelta(seconds=self.interval)
            if current_date > self.end_date:
                break

        feature_keys = ["open", "high", "low", "close", "volume", "value"]
        full_df_dict = {}
        for key in feature_keys + ["date", "missing"]:
            full_df_dict[key] = []

        for i, date in enumerate(timestamp_list):
            full_df_dict["date"].append(date)

            if date in data_dict["close"].keys():
                full_df_dict["missing"].append(False)
                for key in feature_keys:
                    full_df_dict[key].append(data_dict[key][date])
            else:
                if i == 0:
                    raise ValueError("No record")

                full_df_dict["missing"].append(True)
                for key in feature_keys:
                    full_df_dict[key].append(full_df_dict[key][-1])
        return full_df_dict

    def split_and_label(self, full_df_dict):
        assert len(full_df_dict["date"]) % 24 == 0
        num_days = len(full_df_dict["date"]) // 24
        print("num_days", num_days)
        for key in full_df_dict.keys():
            full_df_dict[key] = np.array_split(full_df_dict[key], num_days)

        day_list = []
        for i in range(len(full_df_dict["date"])):
            day_item = {}
            is_nan = False

            for key in full_df_dict.keys():
                day_item[key] = list(full_df_dict[key][i])
                # print(key, len(day_item[key]))
                assert len(list(full_df_dict[key][i])) == 24

                if pd.isnull(full_df_dict[key][i]).any():
                    is_nan = True
                    break
            if not is_nan:
                day_list.append(day_item)

        for i in range(len(day_list) - 1):
            close_price = day_list[i]["close"][-1]
            tomorrow_price = day_list[i + 1]["close"][-1]

            diff_perc = (tomorrow_price - close_price) / close_price * 100.0
            if diff_perc > 0:
                day_list[i]["label"] = 1
            else:
                day_list[i]["label"] = 0

        return day_list

    def normalize(self, full_df_dict):
        close = np.array(full_df_dict["close"])

        z_open = np.array(full_df_dict["open"]) / close - 1
        z_high = np.array(full_df_dict["high"]) / close - 1
        z_low = np.array(full_df_dict["low"]) / close - 1

        _z_close = (
            np.array(full_df_dict["close"])[1:] / np.array(full_df_dict["close"])[:-1]
            - 1
        )
        z_close = np.zeros(len(full_df_dict["close"]))
        z_close[1:] = _z_close

        _z_volume = (full_df_dict["volume"])[1:] / np.array(full_df_dict["volume"])[
            :-1
        ] - 1
        z_volume = np.zeros(len(full_df_dict["volume"]))
        z_volume[1:] = _z_volume

        full_df_dict["z_open"] = list(z_open)
        full_df_dict["z_high"] = list(z_high)
        full_df_dict["z_low"] = list(z_low)
        full_df_dict["z_close"] = list(z_close)
        full_df_dict["z_volume"] = list(z_volume)

        # sma_4 = talib.MA(np.array(full_df_dict["close"]), 4) / close - 1
        # sma_8 = talib.MA(np.array(full_df_dict["close"]), 8) / close - 1
        # sma_12 = talib.MA(np.array(full_df_dict["close"]), 12) / close - 1
        # sma_24 = talib.MA(np.array(full_df_dict["close"]), 24) / close - 1

        # full_df_dict["sma_4"] = list(sma_4)
        # full_df_dict["sma_8"] = list(sma_8)
        # full_df_dict["sma_12"] = list(sma_12)
        # full_df_dict["sma_24"] = list(sma_24)

        return full_df_dict


class CoinDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def to_tensor(self, day):
        x = torch.from_numpy(
            np.array(
                [
                    day["z_open"],
                    day["z_high"],
                    day["z_low"],
                    day["z_close"],
                    day["z_volume"],
                    # day["sma_4"],
                    # day["sma_8"],
                    # day["sma_12"],
                    # day["sma_24"],
                    # day["norm_close"],
                    # day["norm_volume"],
                ],
                dtype=np.float32,
            )
        )
        y = day["label"]
        return x, y

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        day = self.data_list[index]
        x, y = self.to_tensor(day)
        return x, y
