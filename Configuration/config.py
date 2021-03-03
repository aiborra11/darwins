class Config:
    CANDLES_PATH: str = 'Data/darwinex/trainTimeSeries/trainTimeSeries/TrainCandles'
    SCORES_PATH: str = 'Data/darwinex/trainTimeSeries/trainTimeSeries/TrainScores'

    CANDLES_DATA: bool = True
    SCORES_DATA: bool = False

    VALID_DARWINS: list = ["REU", "VRT", "EEY", "JTL", "SEH", "BSX", "OJG", "UEI", "HQU", "ZXW", "LEN", "YEC",
                           "UYZ", "LWK", "ACY", "HEO", "FIR", "BGN"]

    LOG_CANDLES: bool = True
    LOG_TECHNICAL_INDICATORS: bool = True

    TRAIN_SIZE: float = 0.8

    TIMEFRAME: str = '4H'

    OPERATING_MARKETS: bool = True
    TECHNICAL_INDICATORS: bool = True

config=Config()
