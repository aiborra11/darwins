from DataLoader.data_loader import DataLoader
from Configuration.config import Config

dataloader = DataLoader()
darwins = Config().VALID_DARWINS

for darwin in darwins[:1]:
    data = dataloader.merge_dfs(darwin)


