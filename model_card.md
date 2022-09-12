# Model Card

## Model Details

- **Model last updated**: September 11th, 2022
- **Model version**: 1.0.0
- **Model type**: By default, `Grid Search` with `Random Forest Classifier` is used for training. Users can easily test other approaches by adjusting config file.

For any feedback on this mode, please send me an email through `leminkhoa@gmail.com`

## Intended Use
- The model is built to predict whether a U.S. citizen makes over 50K a year.
- The model is developed for learning purpose and is not limited to any person.
- May not be ideal to be used for citizon outside of U.S Territory.

## Training Data
- Classification model is trained from publicly available Census Bureau [data](https://archive.ics.uci.edu/ml/datasets/census+income), training data split

## Evaluation Data
- Test data split
- Chosen as a basic proof-of-concept

## Data Description
The below show variables within our dataset:

Features:
```
age: integer 
workclass: string (Ex: "State-gov", "Federal-gov", ...)
fnlgt: integer
education: string (Ex: "Bachelors", "Assoc-acdm", ...)
education_num: integer
marital_status: string (Ex: "Never-married", "Married-civ-spouse", ...)
occupation: string (Ex: "Adm-clerical", "Handlers-cleaners", ...)
relationship: string (Ex: "Not-in-family",
race: string (Ex: "White", "Black", ...)
sex: string (Ex: "Male", "Female", ...)
capital_gain: integer/float
capital_loss: integer/float
hours_per_week: integer
native_country: string ("United-States", "Ireland", ...)
```

Label:
```
salary: str - (Binary: "<=50K" or ">50K")
```
## Parameters
- Default Parameters Grid (`Random Forest Classifier`):
    - n_estimators: [10]
    - max_depth: [6]

## Metrics
- Evaluation Metrics (Defined in [model.py](starter/ml/model.py#compute_model_metrics)):
    - precision
    - recall
    - fbeta
- Current performance:
``` 
Precision: 0.8105263157894737
Recall: 0.49013367281986
Fbeta: 0.6108687028956764
```

## Analysis
