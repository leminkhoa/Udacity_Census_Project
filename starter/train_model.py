# Script to train machine learning model.
import pandas as pd
import hydra
import joblib
import warnings
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from ml.data import process_data
from ml.model import infer_model, train_model, inference, compute_model_metrics
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver(
        "module",
        lambda module_path, module_name: infer_model(
            model_name=module_name,
            import_module=module_path))

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path='experiments', config_name='ml_config')
def main(config: OmegaConf):
    # Load data
    logger.info("Load dataset...")
    data = pd.read_csv(config.main.data)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logger.info("Split train, test...")
    train, test = train_test_split(data, test_size=0.20, random_state=config.main.random_state)

    cat_features = config.main.categorical_cols
    label = config.main.label_col

    logger.info("Preprocessing on train and test datasets...")
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
    logger.info(f"GridSearch configurations: \n \t"
                f"Model: {chosen_model},\n \tParams: {params},\n \t"
                f"Scoring: {config.GridSearchCV.scoring}")

    # Train and save model.
    logger.info("Start training...")
    clf = train_model(X_train, y_train, gs)
    joblib.dump(clf, config.modeling.model_output_path)
    logger.info(f"Model saved to path '{config.modeling.model_output_path}'")

    # Save other transformers
    joblib.dump(lb, config.modeling.lb_output_path)
    logger.info(f"Label encoder saved to path '{config.modeling.lb_output_path}'")
    joblib.dump(encoder, config.modeling.encoder_output_path)
    logger.info(f"Feature encoder saved to path '{config.modeling.encoder_output_path}'")

    # Load the model
    clf = joblib.load(config.modeling.model_output_path)

    # predict
    preds = inference(clf, X_test)

    # Compute model metrics
    logger.info("Validating model...")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    with open(config.metrics.output_path, 'w') as f:
        f.write(f"Precision: {precision}\nRecall: {recall}\nFbeta: {fbeta}")
        logger.info(f"Metrics results saved to path '{config.metrics.output_path}'")


if __name__ == "__main__":
    main()
