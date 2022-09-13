import importlib
import logging
from collections import defaultdict
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def infer_model(model_name: str, import_module: str, model_params={}):
    """Returns a scikit-learn model."""
    model_class = getattr(importlib.import_module(import_module), model_name)
    model = model_class(**model_params)  # Instantiates the model
    return model


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, model):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model:
        GridSearchCV or a sklearn classifier model
    Returns
    -------
    model
        Trained machine learning model.
    """

    model.fit(X_train, y_train)
    if isinstance(model, GridSearchCV):
        # Print out best params
        best_parameters = model.best_params_
        logging.info(f"Training completed! Best params: {best_parameters}")
    else:
        logging.info("Training completed! ")

    return model


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def slice_compute_model_metrics(test, preds, categorical_features, label, lb):
    result = defaultdict(lambda: defaultdict(dict))

    for category in categorical_features:
        for cls in test[category].unique():
            test_temp = test.loc[test[category]==cls, label]
            test_temp = lb.transform(test_temp.values).ravel()
            preds_temp = preds[test[category]==cls]
            precision, recall, fbeta = compute_model_metrics(test_temp, preds_temp)
            metrics = dict(
                precision=precision,
                recall=recall,
                fbeta=fbeta
            )
            result[category][cls] = metrics
    return result
