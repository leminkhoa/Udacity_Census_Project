import pytest
import pandas as pd
import warnings
from hydra import compose, initialize
from starter.ml.data import process_data
warnings.filterwarnings("ignore")


@pytest.fixture(scope='session')
def config():
    with initialize(config_path='../starter/experiments', version_base=None):
        cfg = compose(config_name="ml_config")
    return cfg


@pytest.fixture(scope='session')
def raw_data():
    raw_data = pd.read_csv('test/census_test_sample.csv', nrows=50)
    return raw_data


@pytest.fixture(scope='session')
def cleaned_data():
    raw_data = pd.read_csv('test/census_test_sample.csv', nrows=50)
    return raw_data


@pytest.fixture(scope='session')
def processed_data(cleaned_data, config):
    return process_data(
        cleaned_data,
        categorical_features=config.main.categorical_cols,
        label=config.main.label_col,
        training=True)
