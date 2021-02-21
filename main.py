from DataLoader.data_loader import DataLoader
from Configuration.config import Config
from DataPreprocessing.data_preprocessing import DataPreprocessing

dataloader = DataLoader()
darwins = Config().VALID_DARWINS

for darwin in darwins[:1]:
    data = dataloader.merge_dfs(darwin)
    data_preprocessed = DataPreprocessing(data)
    # data_preprocessed.get_logarithmic_data()
