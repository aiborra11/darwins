class Config:
    CANDLES_PATH: str = 'Data/darwinex/trainTimeSeries/trainTimeSeries/TrainCandles'
    SCORES_PATH: str = 'Data/darwinex/trainTimeSeries/trainTimeSeries/TrainScores'

    CANDLES_DATA: bool = True
    SCORES_DATA: bool = True

    VALID_DARWINS: list = ["REU", "VRT", "EEY", "JTL", "SEH", "BSX", "OJG", "UEI", "HQU", "ZXW", "LEN", "YEC",
                           "UYZ", "LWK", "ACY", "HEO", "FIR", "BGN"]

    TRAIN_SIZE: float = 0.8
    LOG_COLS: list = ['close', 'max', 'min', 'open']

    TIMEFRAME: str = '1H'

    OPERATING_MARKETS: bool = True
    TECHNICAL_INDICATORS: bool = False

config=Config()
