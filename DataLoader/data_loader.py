import pandas as pd
import numpy as np
import os

from typing import (Union)

from Configuration.config import Config


class DataLoader(Config):
    def __init__(self):
        config = Config()
        self._candles_path: str = config.CANDLES_PATH
        self._scores_path: str = config.SCORES_PATH
        self._candles: bool = config.CANDLES_DATA
        self._scores: bool = config.SCORES_DATA

        self._available_darwins_candles, self._available_darwins_scores = self._get_available_darwins()

        self._valid_darwins: list = config.VALID_DARWINS
        self.data_candles: Union[pd.DataFrame, None] = None
        self.data_scores: Union[pd.DataFrame, None] = None

    def _get_available_darwins(self):
        available_darwins_candles = []
        available_darwins_scores = []
        if self._candles:
            available_darwins_candles = [list({name.split("_")[-2]})[0] for name in os.listdir(self._candles_path)]
            print(f'You have {len(available_darwins_candles)} available darwins with CANDLE data: ',
                  available_darwins_candles)
        if self._scores:
            available_darwins_scores = [list({name.split("_")[-2]})[0] for name in os.listdir(self._scores_path)]
            print(f'You have {len(available_darwins_scores)} available darwins with SCORES data: ',
                  available_darwins_scores)
        if self._candles and self._scores:
            missing = [x for x in available_darwins_candles if x not in available_darwins_scores]
            if missing:
                print(f'You have missing data for {missing} SCORES')

        return available_darwins_candles, available_darwins_scores

    def _load_data(self, darwin: str):
        empty_data = []
        candles_directory = self._candles_path
        scores_directory = self._scores_path

        if self._candles:
            if darwin in self._valid_darwins and darwin in self._available_darwins_candles:
                self.data_candles = pd.read_csv(f'{candles_directory}/DARWINUniverseCandlesOHLC_{darwin}_train.csv',
                                                index_col='Unnamed: 0')
                # If price remains the same means the darwin is not doing trades
                if len(set(self.data_candles['close'])) <= 1:
                    print(f'The {darwin} is invalid since it did not made any trade!!')
                    empty_data.append(darwin)
                else:
                    self.data_candles = self.data_candles[self.data_candles['close'].notna()]

                    if self._scores:
                        try:
                            self.data_scores = pd.read_csv(f'{scores_directory}/scoresData_{darwin}_train.csv',
                                                           index_col='eod_ts')
                        except FileNotFoundError:
                            print('There are not evaluation metrics for ', darwin)
                    return self.data_candles, self.data_scores
                return self.data_candles, None
            else:
                print(f'There is not available data for {darwin}')
        else:
            if darwin in self._valid_darwins and darwin in self._available_darwins_scores:
                self.data_scores = pd.read_csv(f'{scores_directory}/scoresData_{darwin}_train.csv', index_col='eod_ts')
            return None, self.data_scores

    def merge_dfs(self, darwin: str):
        self.data_candles, self.data_scores = self._load_data(darwin)
        if self.data_candles is not None and self.data_scores is not None:
            data_scores = self.data_scores.reset_index().rename(columns={'eod_ts': 'index'})
            data_candles = self.data_candles.reset_index()
            data = pd.merge(data_candles, data_scores, on='index', how='left')
        else:
            data = pd.concat([self.data_candles, self.data_scores], axis=0)
        self._release_memory()
        return data

    def _release_memory(self):
        self.data_scores = None
        self.data_candles = None




