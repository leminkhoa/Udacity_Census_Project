from omegaconf import ListConfig


def test__import_config_data(config):
    '''Test data config'''
    assert config.main.data == 'data/census_updated.csv'


def test__import_config_default_model(config):
    '''Test chosen model config'''
    assert config.main.model == 'random_forest'


def test__import_config_list(config):
    '''Test type of categories config'''
    assert isinstance(config.main.categorical_cols, ListConfig), \
        "Datatype imported from config is not of ListConfig type"
