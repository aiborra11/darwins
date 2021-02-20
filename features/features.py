import empyrical as em
import pandas as pd
import numpy as np
import h2o
import talib
import time
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from matplotlib.figure import figaspect
from h2o.automl import H2OAutoML
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fbprophet.diagnostics import cross_validation, performance_metrics




class Features():
    def __init__(self, df):
        self.df = df

    # Finding out the session in which the DARWIN is operating
    def session_identifier(self):

        self.df['hour'] = self.df['hour'].str.split(':', 1).str[0]
        self.df['hour'] = self.df['hour'].astype(int)

        self.df.loc[(self.df['hour'] >= 1) & (self.df['hour'] <= 7), 'session'] = 1             # Tokyo
        self.df.loc[(self.df['hour'] >= 8) & (self.df['hour'] <= 15), 'session'] = 2            # London
        self.df.loc[(self.df['hour'] >= 15) & (self.df['hour'] <= 17), 'session'] = 3           # NY + London
        self.df.loc[(self.df['hour'] >= 17) & (self.df['hour'] <= 22), 'session'] = 4           # NY
        self.df.loc[(self.df['hour'] > 22) | (self.df['hour'] == 0), 'session'] = 5             # Others

        return self.df

    def technical_indicators(self):

        # df['SMA_3'] = talib.SMA(df['Close'].values, timeperiod=3)
        # df['SMA_6'] = talib.SMA(df['Close'].values, timeperiod=6)
        # df['SMA_9'] = talib.SMA(df['Close'].values, timeperiod=9)

        self.df['EMA_3'] = talib.EMA(self.df['close'].values, timeperiod=3)
        self.df['EMA_6'] = talib.EMA(self.df['close'].values, timeperiod=6)
        self.df['EMA_9'] = talib.EMA(self.df['close'].values, timeperiod=9)
        self.df['EMA_20'] = talib.EMA(self.df['close'].values, timeperiod=20)

        self.df['SAR'] = talib.SAR(self.df['high'].values, self.df['Low'].values, acceleration=0, maximum=0)

        self.df['RSI_5'] = talib.RSI(self.df['close'].values, timeperiod=5)
        self.df['RSI_15'] = talib.RSI(self.df['close'].values, timeperiod=15)
        self.df['RSI_20'] = talib.RSI(self.df['close'].values, timeperiod=20)

        # df['macd', 'macdsignal', 'macdhist'] = talib.MACD(df['Close'].values, 12, 26, 9)

        self.df['STOCH_fastk'], self.df['STOCH_fastd'] = talib.STOCHF(self.df['high'].values, self.df['Low'].values,
                                                                      self.df['close'].values,
                                                                      fastk_period=5, fastd_period=3, fastd_matype=3)
        # df['Delta_1'] = df['Close'].values - df['SMA_3']
        # df['Delta_2'] = df['Close'].values - df['SMA_6']
        # df['Delta_3'] = df['Close'].values - df['SMA_9']

        self.df['PricesMean'] = self.df[['high', 'Open', 'Low', 'close']].mean(axis=1)

        self.df['MOM_5'] = talib.MOM(self.df['close'].values, timeperiod=5)

        self.df['OH'] = self.df['Open'].values - self.df['high'].values
        self.df['OL'] = self.df['Open'].values - self.df['Low'].values

        self.df['OpO'] = self.df['Open'].values - self.df.Open.shift(1)
        self.df['OpC'] = self.df['Open'].values - self.df.close.shift(1)
        self.df['CpC'] = self.df['close'].values - self.df.close.shift(1)

        self.df['ROC'] = talib.ROC(self.df['close'].values, timeperiod=1)

        # indicators = Indicators(df)
        # indicators.awesome_oscillator(column_name='Aw_Os')
        # df['Aw_Os'] = indicators.df['Aw_Os']

        self.df['ADX_7'] = talib.ADX(self.df['high'].values, self.df['Low'].values, self.df['close'].values,
                                     timeperiod=7)
        self.df['ADX_14'] = talib.ADX(self.df['high'].values, self.df['Low'].values, self.df['close'].values,
                                      timeperiod=14)

        # df['AROON_7'] = talib.AROON(df['High'],df['Low'], timeperiod=7)
        # df['AROON_14'] = talib.AROON(df['High'],df['Low'], timeperiod=14)

        self.df['DX_7'] = talib.DX(self.df['high'].values, self.df['Low'].values, self.df['close'].values,
                                   timeperiod=7)
        self.df['DX_14'] = talib.DX(self.df['high'].values, self.df['Low'].values, self.df['close'].values,
                                    timeperiod=14)

        self.df['MOM_5'] = talib.MOM(self.df['close'].values, timeperiod=5)
        self.df['MOM_10'] = talib.MOM(self.df['close'].values, timeperiod=10)

        self.df['ULTOSC'] = talib.ULTOSC(self.df['high'].values, self.df['Low'].values, self.df['close'].values,
                                         timeperiod1=5, timeperiod2=10, timeperiod3=20)

        return self.df
