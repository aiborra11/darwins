from typing import (List)

# TODO Never touch else statements!!!!!

class Config:
    CANDLES_PATH: str = 'Data/darwinex/trainTimeSeries/trainTimeSeries/TrainCandles'
    SCORES_PATH: str = 'Data/darwinex/trainTimeSeries/trainTimeSeries/TrainScores'

    CANDLES_DATA: bool = True
    SCORES_DATA: bool = True
    TIMEFRAME: str = '4H'

    VALID_DARWINS: List[str] = ["REU", "VRT", "EEY", "JTL", "SEH", "BSX", "OJG", "UEI", "HQU", "ZXW", "LEN", "YEC",
                                "UYZ", "LWK", "ACY", "HEO", "FIR", "BGN"]

    OPERATING_MARKETS: bool = True
    if CANDLES_DATA:
        TECHNICAL_INDICATORS: bool = True
        NOT_STANDARDIZATION_COLS = ['Label', 'date', 'hour']

    else:
        TECHNICAL_INDICATORS: bool = False
        LOG_CANDLES: bool = False
        LOG_TECHNICAL_INDICATORS: bool = False

    TRAIN_SIZE: float = 0.8



config = Config()
