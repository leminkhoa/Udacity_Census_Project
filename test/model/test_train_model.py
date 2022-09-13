from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import is_classifier
from training.ml.model import infer_model, train_model, inference, compute_model_metrics


def test__infer_model():
    inferred_model = infer_model('RandomForestClassifier', 'sklearn.ensemble')
    assert isinstance(inferred_model, RandomForestClassifier)


def test__train_model_random_forest(processed_data):
    X, y, _, _ = processed_data
    clf = train_model(X, y, RandomForestClassifier(n_estimators=5))
    assert is_classifier(clf)


def test__train_model_svc(processed_data):
    X, y, _, _ = processed_data
    clf = train_model(X, y, SVC())
    assert is_classifier(clf)


def test__train_model_logistic(processed_data):
    X, y, _, _ = processed_data
    clf = train_model(X, y, LogisticRegression())
    assert is_classifier(clf)


def test__train_model_grid(processed_data):
    X, y, _, _ = processed_data
    clf = train_model(X, y, GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={},
        cv=3,
        n_jobs=-1,
    ))
    assert is_classifier(clf)


def test__inference(processed_data):
    X, y, _, _ = processed_data
    clf = train_model(X, y, LogisticRegression())
    pred = inference(clf, X)
    assert pred.shape[0] == X.shape[0]
    assert set(pred) == {0, 1}


def test__compute_model_metrics(processed_data):
    X, y, _, _ = processed_data
    clf = train_model(X, y, LogisticRegression())
    pred = inference(clf, X)
    precision, recall, fbeta = compute_model_metrics(y, pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
