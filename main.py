from src.DataLoader.data_loader import DataLoader
from Configuration.config import config
from src.DataPreprocessing.data_preprocessing import FeaturesCalculator


dataloader = DataLoader()
darwins = config.VALID_DARWINS

for darwin in darwins[:1]:
    data = dataloader.merge_dfs(darwin)

    data_features = FeaturesCalculator(data)
    train, test = data_features.train_test_split()
    print(darwin)
    print(train)
    print(test)
