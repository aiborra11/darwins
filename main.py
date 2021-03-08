from src.DataLoader.data_loader import DataLoader
from config import config
from src.DataPreprocessing.data_preprocessing import DataPreprocessing
from src.Predictors.predictors import ProphetPredictor


dataloader = DataLoader()
darwins = config.VALID_DARWINS

for darwin in darwins[:1]:
    data = dataloader.merge_dfs(darwin)
    data_features = DataPreprocessing(data)
    # print(data_features.logarithmic_standardizer())

    print(data_features.scaler_standaradizer())
    train, test = data_features.train_test_split()

    # prophet_model = ProphetPredictor(train, test)
    # prophet_trained = prophet_model.execute_prophet()
    #
    #
    print(darwin)
    print(train)
    print(train.columns)

    print(test)
    # print(test.columns)
    # print('prophet_trained', prophet_trained)
    # print('prophet_trained', prophet_trained['yhat'])


