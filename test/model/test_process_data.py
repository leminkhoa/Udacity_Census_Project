from sklearn.model_selection import train_test_split
from training.ml.data import process_data


def test__process_data(cleaned_data, config):
    train, test = train_test_split(cleaned_data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=config.main.categorical_cols, label='salary', training=True
    )
    assert X_train.shape[0] == train.shape[0]
    assert y_train.shape == (train.shape[0],)
    assert set(y_train) == {0, 1}
    assert len(encoder.categories_) == len(config.main.categorical_cols)
    assert list(lb.classes_) == ['<=50K', '>50K']

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=config.main.categorical_cols, label='salary', encoder=encoder, lb=lb, training=False
    )
    assert X_test.shape[0] == test.shape[0]
    assert y_test.shape == (test.shape[0],)
    assert set(y_test) == {0, 1}
