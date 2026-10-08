
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement


ROLL = "BT2024215"
VAL_SIZE = 0.20
SEED = 42


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def mse(y, pred):
    return np.mean((y - pred) ** 2)


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum(
        (y - y.mean()) ** 2
    )


def train_val_split(X, y):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))

    n_val = int(len(X) * VAL_SIZE)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return (
        X[train_idx],
        X[val_idx],
        y[train_idx],
        y[val_idx]
    )


# ============================================================
# POLYNOMIAL FEATURES
# total degree <= degree
# ============================================================

def polynomial_features(X, degree):

    n, p = X.shape
    features = []

    for d in range(1, degree + 1):

        for ind in combinations_with_replacement(
            range(p), d
        ):
            features.append(
                np.prod(X[:, ind], axis=1)
            )

    return np.column_stack(features)


# ============================================================
# STANDARDIZATION
# ============================================================

def scaler_fit(X):

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std == 0] = 1

    return mean, std


def scaler_transform(X, mean, std):

    return (X - mean) / std


# ============================================================
# LASSO
# FISTA / proximal gradient descent
# ============================================================

def lasso(X, y, alpha, max_iter=200):

    n, p = X.shape

    intercept = y.mean()
    yc = y - intercept

    beta = np.zeros(p)
    z = beta.copy()
    t = 1.0

    # Upper bound on the Lipschitz constant
    L = np.sum(X * X) / n

    for _ in range(max_iter):

        old_beta = beta.copy()

        gradient = X.T @ (X @ z - yc) / n

        temp = z - gradient / L

        beta = np.sign(temp) * np.maximum(
            np.abs(temp) - alpha / L,
            0
        )

        new_t = (
            1 + np.sqrt(1 + 4 * t * t)
        ) / 2

        z = beta + (
            (t - 1) / new_t
        ) * (beta - old_beta)

        t = new_t

        if np.linalg.norm(beta - old_beta) < 1e-6:
            break

    return beta, intercept


# ============================================================
# RIDGE
# ============================================================

def ridge(X, y, alpha):

    n = X.shape[0]

    intercept = y.mean()
    yc = y - intercept

    # Dual form avoids solving a huge p x p system
    K = X @ X.T / n

    beta_dual = np.linalg.solve(
        K + alpha * np.eye(n),
        yc
    )

    beta = X.T @ beta_dual / n

    return beta, intercept


# ============================================================
# PREPARE POLYNOMIAL DATA
# ============================================================

def prepare(X_train, X_val, degree):

    # Scale original features
    x_mean, x_std = scaler_fit(X_train)

    X_train = scaler_transform(
        X_train,
        x_mean,
        x_std
    )

    X_val = scaler_transform(
        X_val,
        x_mean,
        x_std
    )

    # Polynomial expansion
    X_train = polynomial_features(
        X_train,
        degree
    )

    X_val = polynomial_features(
        X_val,
        degree
    )

    # Scale polynomial features
    p_mean, p_std = scaler_fit(X_train)

    X_train = scaler_transform(
        X_train,
        p_mean,
        p_std
    )

    X_val = scaler_transform(
        X_val,
        p_mean,
        p_std
    )

    return X_train, X_val


# ============================================================
# SEARCH BEST MODEL
# ============================================================

def find_best(X, y, max_degree):

    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y
    )

    best_lasso = {
        "mse": np.inf
    }

    best_ridge = {
        "mse": np.inf
    }

    lasso_curve = []
    ridge_curve = []

    for degree in range(1, max_degree + 1):

        Xt, Xv = prepare(
            X_train,
            X_val,
            degree
        )

        n = len(y_train)
        yc = y_train - y_train.mean()

        # -----------------------------------------------
        # Alpha range
        # -----------------------------------------------

        alpha_max = np.max(
            np.abs(Xt.T @ yc)
        ) / n

        if alpha_max < 1e-8:
            alpha_max = 1e-8

        lasso_alphas = (
            alpha_max *
            np.logspace(0, -4, 10)
        )

        ridge_alphas = (
            alpha_max *
            np.logspace(-4, 2, 10)
        )

        # -----------------------------------------------
        # LASSO
        # -----------------------------------------------

        best_degree_lasso = np.inf

        for alpha in lasso_alphas:

            beta, intercept = lasso(
                Xt,
                y_train,
                alpha
            )

            pred = intercept + Xv @ beta

            score = mse(
                y_val,
                pred
            )

            best_degree_lasso = min(
                best_degree_lasso,
                score
            )

            if score < best_lasso["mse"]:

                best_lasso = {
                    "mse": score,
                    "degree": degree,
                    "alpha": alpha
                }

        lasso_curve.append(
            best_degree_lasso
        )

        # -----------------------------------------------
        # RIDGE
        # -----------------------------------------------

        best_degree_ridge = np.inf

        for alpha in ridge_alphas:

            beta, intercept = ridge(
                Xt,
                y_train,
                alpha
            )

            pred = intercept + Xv @ beta

            score = mse(
                y_val,
                pred
            )

            best_degree_ridge = min(
                best_degree_ridge,
                score
            )

            if score < best_ridge["mse"]:

                best_ridge = {
                    "mse": score,
                    "degree": degree,
                    "alpha": alpha
                }

        ridge_curve.append(
            best_degree_ridge
        )

    return (
        best_lasso,
        best_ridge,
        lasso_curve,
        ridge_curve,
        X_train,
        X_val,
        y_train,
        y_val
    )


# ============================================================
# FINAL SOLUTION FOR ONE VARIABLE
# ============================================================

def solve(var, max_degree):

    train = pd.read_csv(
        f"{ROLL}_train_{var}.csv"
    )

    test = pd.read_csv(
        f"{ROLL}_test_{var}.csv"
    )

    X = train.drop(
        columns=["y"]
    ).to_numpy(float)

    y = train["y"].to_numpy(float)

    X_test = test.to_numpy(float)

    (
        best_lasso,
        best_ridge,
        lasso_curve,
        ridge_curve,
        X_train,
        X_val,
        y_train,
        y_val
    ) = find_best(
        X,
        y,
        max_degree
    )

    # ========================================================
    # CHOOSE LASSO OR RIDGE
    # ========================================================

    if best_lasso["mse"] <= best_ridge["mse"]:

        best = best_lasso
        best["model"] = "Lasso"

    else:

        best = best_ridge
        best["model"] = "Ridge"


    # ========================================================
    # VALIDATION R2
    # ========================================================

    Xt, Xv = prepare(
        X_train,
        X_val,
        best["degree"]
    )

    if best["model"] == "Lasso":

        beta, intercept = lasso(
            Xt,
            y_train,
            best["alpha"],
            max_iter=400
        )

    else:

        beta, intercept = ridge(
            Xt,
            y_train,
            best["alpha"]
        )

    val_pred = (
        intercept +
        Xv @ beta
    )

    val_r2 = r2(
        y_val,
        val_pred
    )


    # ========================================================
    # TRAIN FINAL MODEL ON ALL TRAINING DATA
    # ========================================================

    x_mean, x_std = scaler_fit(X)

    X_scaled = scaler_transform(
        X,
        x_mean,
        x_std
    )

    X_test_scaled = scaler_transform(
        X_test,
        x_mean,
        x_std
    )

    X_poly = polynomial_features(
        X_scaled,
        best["degree"]
    )

    X_test_poly = polynomial_features(
        X_test_scaled,
        best["degree"]
    )

    p_mean, p_std = scaler_fit(X_poly)

    X_poly = scaler_transform(
        X_poly,
        p_mean,
        p_std
    )

    X_test_poly = scaler_transform(
        X_test_poly,
        p_mean,
        p_std
    )

    if best["model"] == "Lasso":

        beta, intercept = lasso(
            X_poly,
            y,
            best["alpha"],
            max_iter=400
        )

    else:

        beta, intercept = ridge(
            X_poly,
            y,
            best["alpha"]
        )

    test_pred = (
        intercept +
        X_test_poly @ beta
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    output = f"{ROLL}_pred_{var}.csv"

    pd.DataFrame({
        "y": test_pred
    }).to_csv(
        output,
        index=False
    )


    # ========================================================
    # PLOT 1: DEGREE VS MSE
    # ========================================================

    degrees = range(
        1,
        max_degree + 1
    )

    plt.figure()

    plt.plot(
        degrees,
        lasso_curve,
        marker="o",
        label="Lasso"
    )

    plt.plot(
        degrees,
        ridge_curve,
        marker="o",
        label="Ridge"
    )

    plt.xlabel("Polynomial Degree")
    plt.ylabel("Validation MSE")
    plt.title(f"{var}: Lasso vs Ridge")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{ROLL}_{var}_mse.png",
        dpi=200
    )

    plt.close()


    # ========================================================
    # PLOT 2: ACTUAL VS PREDICTED
    # ========================================================

    plt.figure()

    plt.scatter(
        y_val,
        val_pred,
        s=12
    )

    mn = min(
        y_val.min(),
        val_pred.min()
    )

    mx = max(
        y_val.max(),
        val_pred.max()
    )

    plt.plot(
        [mn, mx],
        [mn, mx]
    )

    plt.xlabel("Actual y")
    plt.ylabel("Predicted y")
    plt.title(
        f"{var}: {best['model']}"
    )

    plt.grid(True)

    plt.savefig(
        f"{ROLL}_{var}_actual_vs_pred.png",
        dpi=200
    )

    plt.close()


    # ========================================================
    # MINIMAL OUTPUT
    # ========================================================

    print(
        f"{var}: "
        f"Lasso d={best_lasso['degree']} "
        f"MSE={best_lasso['mse']:.4f} | "
        f"Ridge d={best_ridge['degree']} "
        f"MSE={best_ridge['mse']:.4f} | "
        f"BEST={best['model']} "
        f"d={best['degree']} "
        f"alpha={best['alpha']:.5f} "
        f"R2={val_r2:.4f}"
    )

    print(
        f"  saved {output} ({len(test_pred)} rows)"
    )

    print(
        f"  saved 2 plots for {var}"
    )


# ============================================================
# RUN
# ============================================================

solve("var1", 10)
solve("var2", 20)
