from DataLoader.data_loader import DataLoader
from Configuration.config import Config
from DataPreprocessing.data_preprocessing import (DataPreprocessing, FeaturesCalculator)


dataloader = DataLoader()
darwins = Config().VALID_DARWINS

for darwin in darwins[:1]:
    data = dataloader.merge_dfs(darwin)

    data_features = FeaturesCalculator(data)
    train, test = data_features.train_test_split()
