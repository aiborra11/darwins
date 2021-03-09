from src.DataLoader.data_loader import DataLoader
from config import config
from src.DataPreprocessing.data_preprocessing import DataPreprocessing
from src.Predictors.predictors import ProphetPredictor


dataloader = DataLoader()
darwins = config.VALID_DARWINS

for darwin in darwins[1:2]:
    data = dataloader.merge_dfs(darwin)
    data_features = DataPreprocessing(data)

    train, test = data_features.train_test_split()

    train_standardized = data_features.scaler_standardizer(train, train_df=True)
    test_standardized = data_features.scaler_standardizer(test, train_df=False)

    # prophet_model = ProphetPredictor(train_standardized, test_standardized)
    # prophet_trained = prophet_model.execute_prophet()

