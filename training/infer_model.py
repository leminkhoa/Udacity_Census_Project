import pandas as pd
import joblib
import logging
from hydra import compose, initialize
from training.ml.data import process_data
from training.ml.model import inference

# Initialize logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(filename)15s] - [%(name)s] - [%(levelname)s] - %(message)s")
logger = logging.getLogger()

# Import config
with initialize(version_base=None, config_path="experiments"):
    config = compose(config_name="ml_config")


def predict_salary(item):
    # load user inputs
    input_json = {k: [v] for k, v in dict(item).items()}
    logger.info(f"Load input \n {input_json}")
    input_df = pd.DataFrame.from_dict(input_json)
    logger.info("Converted input to dataframe")

    # load pipeline components
    encoder = joblib.load(config.modeling.encoder_output_path)
    lb = joblib.load(config.modeling.lb_output_path)
    cat_features = config.main.categorical_cols
    logger.info("Loaded pipeline components")

    # load trained model
    clf = joblib.load(config.modeling.model_output_path)
    logger.info("Loaded model")

    # Process data
    logger.info("Processing input")
    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, label=None, encoder=encoder, lb=lb, training=False
    )

    # predict
    logger.info("Predict output")
    preds = inference(clf, X)
    # Convert back
    salary = lb.inverse_transform(preds)[0]
    return salary
