from DataLoader.data_loader import DataLoader
from Configuration.config import Config
from DataPreprocessing.data_preprocessing import DataPreprocessing

dataloader = DataLoader()
darwins = Config().VALID_DARWINS

for darwin in darwins[:1]:
    data = dataloader.merge_dfs(darwin)
    print(data.columns)
    # print(data)
    data_preprocessed = DataPreprocessing(data)
    # print(data_preprocessed)
    # data_timeframe = data_preprocessed.timeframe_conversor()
    # data_timeframe = data_indexed.timeframe_conversor()
    # print(data_timeframe)

