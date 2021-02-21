class Config:
    CANDLES_PATH = 'Data/darwinex/trainTimeSeries/trainTimeSeries/TrainCandles'
    SCORES_PATH = 'Data/darwinex/trainTimeSeries/trainTimeSeries/TrainScores'

    CANDLES_DATA = True
    SCORES_DATA = True

    VALID_DARWINS = ["REU", "VRT", "EEY", "JTL", "SEH", "BSX", "OJG", "UEI", "HQU", "ZXW", "LEN", "YEC",
                     "UYZ", "LWK", "ACY", "HEO", "FIR", "BGN"]

    TRAIN_SIZE = 0.8
    LOG_COLS = ['close', 'max', 'min', 'open']

    TIMEFRAME = '1H'

config=Config()
