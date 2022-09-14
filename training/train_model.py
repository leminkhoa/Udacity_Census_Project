# Script to train machine learning model.
import pandas as pd
import hydra
import joblib
import warnings
import logging
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_roc_curve, confusion_matrix, ConfusionMatrixDisplay
from ml.data import process_data
from ml.model import infer_model, train_model, inference, compute_model_metrics, slice_compute_model_metrics
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver(
        "module",
        lambda module_path, module_name: infer_model(
            model_name=module_name,
            import_module=module_path))


@hydra.main(config_path='experiments', config_name='ml_config')
def main(config: OmegaConf):
    # Load data
    logging.info("Load dataset...")
    data = pd.read_csv(config.main.data)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logging.info("Split train, test...")
    train, test = train_test_split(data, test_size=0.20, random_state=config.main.random_state)

    cat_features = config.main.categorical_cols
    label = config.main.label_col

    logging.info("Preprocessing on train and test datasets...")
    # Process the train data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )

    # Process the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label=label, encoder=encoder, lb=lb, training=False
    )

    # Define algorithm
    chosen_model = config.main.model
    clf = config.modeling.models[chosen_model].module
    params = dict(config.modeling.models[chosen_model].hyperparameters)
    gs = GridSearchCV(
            estimator=clf,
            param_grid=params,
            scoring=config.GridSearchCV.scoring,
            cv=config.GridSearchCV.cv,
            n_jobs=config.GridSearchCV.n_jobs,
    )
    logging.info('''
                f"GridSearch configurations: \n \t"
                f"Model: {chosen_model},\n \tParams: {params},\n \t"
                f"Scoring: {config.GridSearchCV.scoring}
                ''')

    # Train and save model.
    logging.info("Start training...")
    clf = train_model(X_train, y_train, gs)
    joblib.dump(clf, config.modeling.model_output_path)
    logging.info(f"Model saved to path '{config.modeling.model_output_path}'")

    # Save other transformers
    joblib.dump(lb, config.modeling.lb_output_path)
    logging.info(f"Label encoder saved to path '{config.modeling.lb_output_path}'")
    joblib.dump(encoder, config.modeling.encoder_output_path)
    logging.info(f"Feature encoder saved to path '{config.modeling.encoder_output_path}'")

    # Load the model
    clf = joblib.load(config.modeling.model_output_path)

    # predict
    preds = inference(clf, X_test)

    # ROC Curve
    logging.info("Generating ROC curve report...")
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    plot_roc_curve(clf, X_test, y_test).plot(ax=ax, alpha=0.8)
    plt.title("ROC Curve")
    plt.savefig(config.plots.roc_output_path)
    plt.close()
    logging.info(
        "Saved ROC curve result at %s !",
        config.plots.roc_output_path
    )

    # Confusion matrix
    logging.info("Generating Confusion Matrix...")
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    cm = confusion_matrix(y_test, preds, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(config.plots.cm_output_path)
    plt.close()
    logging.info(
        "Saved Confusion Matrix result at %s !",
        config.plots.cm_output_path)

    # Compute model metrics
    logging.info("Compute overall model metrics...")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    with open(config.metrics.overall_output_path, 'w') as f:
        f.write(json.dumps(dict(
            precision=precision,
            recall=recall,
            fbeta=fbeta
        ), indent=4))
        logging.info(f"Overall metrics results saved to path '{config.metrics.overall_output_path}'")

    # Compute model metrics per slice
    logging.info("Compute model metrics per slice...")
    slice_result = slice_compute_model_metrics(test, preds, categorical_features=cat_features, label=label, lb=lb)
    with open(config.metrics.slice_output_path, 'w') as f:
        f.write(json.dumps(slice_result, indent=4))
        logging.info(f"Slice metrics results saved to path '{config.metrics.slice_output_path}'")


if __name__ == "__main__":
    main()
