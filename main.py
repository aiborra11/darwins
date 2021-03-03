from src.DataLoader.data_loader import DataLoader
from Configuration.config import config
from src.DataPreprocessing.data_preprocessing import DataPreprocessing


dataloader = DataLoader()
darwins = config.VALID_DARWINS

for darwin in darwins[:1]:
    data = dataloader.merge_dfs(darwin)

    data_features = DataPreprocessing(data)
    train, test = data_features.train_test_split(data_features)
    print(darwin)
    # print(darwin.columns)
    # print(train)
    # print(test)
