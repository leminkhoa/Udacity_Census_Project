from omegaconf import ListConfig


def test__import_config_data(config):
    assert config.main.data == 'data/census_updated.csv'


def test__import_config_default_model(config):
    assert config.main.model == 'random_forest'


def test__import_config_list(config):
    assert isinstance(config.main.categorical_cols, ListConfig), \
                    "Datatype imported from config is not of ListConfig type"


def test__import_config_referenced_variable(config):
    model = config.main.model
    assert config.modeling.model_output_path == f'model/trained_{model}_model.joblib'
