# ML Assignment 1: Polynomial Regression

**Name:** Akshaya Akula  
**Roll Number:** BT2024215  
**Course:** Machine Learning

## Overview

This project implements polynomial regression for two personalised datasets:

- **var1:** Steam Turbine Optimisation
- **var2:** Thermal Reservoir Mapping

The implementation uses polynomial feature expansion with **Lasso** and **Ridge** regularisation. Polynomial degree and regularisation strength are selected using a validation set based on Mean Squared Error (MSE).

## Dataset

The following files are required in the same directory as the Python program:

```text
BT2024215_train_var1.csv
BT2024215_test_var1.csv
BT2024215_train_var2.csv
BT2024215_test_var2.csv
```

The training files contain the target column `y`. The test files contain only the input features.

## Methodology

The implementation performs the following steps:

1. Load the training and test CSV files.
2. Split the training data into training and validation sets.
3. Standardise the original input features.
4. Generate polynomial features with total degree up to the selected degree.
5. Standardise the generated polynomial features.
6. Train both Lasso and Ridge models for candidate polynomial degrees and regularisation strengths.
7. Select the configuration with the lowest validation MSE.
8. Calculate the validation $R^2$ score for the selected model.
9. Retrain the selected model on the complete training dataset.
10. Generate predictions for the test dataset.
11. Save prediction CSV files and visualisation plots.

A validation fraction of **20%** and random seed **42** are used.

## Polynomial Feature Generation

Polynomial features are generated manually using combinations with replacement.

For a polynomial degree $d$, all terms whose total degree is at most $d$ are included.

For example, for two variables and degree 2:

```text
x1
x2
x1^2
x1*x2
x2^2
```

## Regularisation

### Lasso

Lasso uses an $L_1$ penalty:

$$
\min_{\beta}
\left[
\frac{1}{2n}\sum_i(y_i-\hat y_i)^2
+
\alpha\sum_j|\beta_j|
\right]
$$

The Lasso optimisation is implemented using a FISTA/proximal-gradient approach.

### Ridge

Ridge uses an $L_2$ penalty:

$$
\min_{\beta}
\left[
\frac{1}{2n}\sum_i(y_i-\hat y_i)^2
+
\alpha\sum_j\beta_j^2
\right]
$$

The Ridge implementation uses the dual formulation to avoid directly solving a large feature-by-feature system.

## Model Selection

The assignment allows polynomial degree up to 10 for var1 and up to 20 for var2.

The program evaluates both Lasso and Ridge across the allowed degrees and automatically selects the best configuration using validation MSE.

### Final Models

| Dataset | Selected Model | Degree | Alpha | Validation MSE | Validation R² |
|---|---|---:|---:|---:|---:|
| var1 | Lasso | 5 | 0.00942 | 0.4500 | 0.9665 |
| var2 | Ridge | 12 | 0.00170 | 0.2464 | 0.9950 |

## Output Files

After running the program, the following prediction files are generated:

```text
BT2024215_pred_var1.csv
BT2024215_pred_var2.csv
```

Each prediction file contains a column named:

```text
y
```

The program also generates four PNG plots:

```text
BT2024215_var1_mse.png
BT2024215_var1_actual_vs_pred.png
BT2024215_var2_mse.png
BT2024215_var2_actual_vs_pred.png
```

### Plot descriptions

- `var1_mse.png`: Validation MSE versus polynomial degree for Lasso and Ridge.
- `var1_actual_vs_pred.png`: Actual versus predicted validation values for the selected var1 model.
- `var2_mse.png`: Validation MSE versus polynomial degree for Lasso and Ridge.
- `var2_actual_vs_pred.png`: Actual versus predicted validation values for the selected var2 model.

## How to Run

Install the required Python packages:

```bash
pip install numpy pandas matplotlib
```

Place the Python file and all four CSV datasets in the same directory.

Run:

```bash
python a.py
```

The program will print the selected model, degree, alpha, validation MSE, and validation $R^2$, and will save the prediction files and plots.

## Project Structure

```text
ML-Assignment-1/
│
├── a.py
│
├── BT2024215_train_var1.csv
├── BT2024215_test_var1.csv
├── BT2024215_train_var2.csv
├── BT2024215_test_var2.csv
│
├── BT2024215_pred_var1.csv
├── BT2024215_pred_var2.csv
│
├── BT2024215_var1_mse.png
├── BT2024215_var1_actual_vs_pred.png
├── BT2024215_var2_mse.png
└── BT2024215_var2_actual_vs_pred.png
```

## Evaluation Metrics

### Mean Squared Error

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Lower MSE indicates better prediction accuracy.

### R² Score

$$
R^2 =
1 -
\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}
$$

Higher $R^2$ indicates that the model explains more of the variation in the target.

## Conclusion

The implementation uses polynomial regression with regularisation to model the nonlinear relationships in both datasets. Lasso with degree 5 was selected for var1, while Ridge with degree 12 was selected for var2 based on validation MSE.


