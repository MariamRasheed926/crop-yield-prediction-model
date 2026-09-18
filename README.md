# Crop Yield Prediction Model

This project uses machine learning to predict crop yield based on historical agricultural data.

A **Random Forest Regressor** is used to learn the relationship between crop type, year, and crop yield.

## Dataset

The dataset contains historical crop yield records, including:

* **Year**
* **Crop Type (Item)**
* **Yield Value (Value)**

The target variable is crop yield, measured in **hg/ha**.

## Model

The project uses a **Random Forest Regressor**.

The preprocessing steps include:

* One-hot encoding for crop type.
* Handling missing values.
* Feature standardization using `StandardScaler`.

## Performance

The model achieved a Mean Squared Error (MSE) of approximately:

**2.33 × 10⁹**

## Development Tools

Python
Pandas
Scikit-learn
Random Forest
Machine Learning

## Future Development

Future improvements may include adding more agricultural and environmental features, using a separate test dataset, and applying hyperparameter tuning to improve prediction performance.
