# from pycox.datasets import metabric
from lifelines.datasets import load_dd
import numpy as np
import pandas as pd
from xgbse import XGBSEStackedWeibull
import matplotlib.pyplot as plt
from xgbse.converters import convert_to_structured
from xgbse import XGBSEKaplanTree, XGBSEBootstrapEstimator
from xgbse.metrics import concordance_index, approx_brier_score, dist_calibration_score
from sklearn.model_selection import train_test_split

plt.style.use('bmh')


# to easily plot confidence intervals
def plot_ci(mean_, upper_ci_, lower_ci_, i=42, title='Probability of survival $P(T \\geq t)$'):
    # plotting mean and confidence intervals
    plt.figure(figsize=(12, 4), dpi=120)
    plt.plot(mean_.columns, mean_.iloc[i])
    plt.fill_between(mean_.columns, lower_ci_.iloc[i], upper_ci_.iloc[i], alpha=0.2)

    plt.title(title)
    plt.xlabel('Time [days]')
    plt.ylabel('Probability')
    plt.tight_layout()


df = load_dd()

# splitting to X, T, E format
X = df.drop(['duration', 'observed'], axis=1)
X = X.astype({c: "category" for c in df.columns if df[c].dtype.name == "object"})
feature_types = ["c" if X[c].dtype.name in ["object", "category"] else "q" for c in X.columns]
T = df['duration']
E = df['observed']
y = convert_to_structured(T, E)

# splitting between train, and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
TIME_BINS = np.arange(T.min(), T.max(), int((T.max() - T.min())/10 + 1))
# TIME_BINS
# ######################################################################################################################
bootstrap_estimator = XGBSEBootstrapEstimator(
    XGBSEStackedWeibull(),
    n_estimators=3
)

# fitting the meta estimator
bootstrap_estimator.fit(X_train, y_train,
                        time_bins=TIME_BINS,
                        enable_categorical=True,
                        validation_data=(X_test, y_test),
                        early_stopping_rounds=100,
                        verbose_eval=100,
                        feature_types=feature_types)

# predicting
mean_prob, upper_ci_prob, lower_ci_prob = bootstrap_estimator.predict(
    X_test,
    return_ci=True,
    enable_categorical=True,
    feature_types=feature_types
)

print(f"C-index XGBSEStackedWeibull bootstrap: {concordance_index(y_test, mean_prob)}")
print(f"Avg. Brier Score XGBSEStackedWeibull bootstrap: {approx_brier_score(y_test, mean_prob)}")

d_calib_weibull = dist_calibration_score(y_test, mean_prob, returns='all')
print(f"D-Calibration XGBSEStackedWeibull: {d_calib_weibull}")

# ######################################################################################################################


# xgboost parameters to fit our model
PARAMS_TREE = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'tree_method': 'hist',
    'max_depth': 10,
    'booster': 'dart',
    'subsample': 1.0,
    'min_child_weight': 50,
    'colsample_bynode': 1.0
}

# fitting xgbse model
xgbse_model = XGBSEKaplanTree(PARAMS_TREE)
xgbse_model.fit(X_train, y_train, time_bins=TIME_BINS)

# predicting
mean, upper_ci, lower_ci = xgbse_model.predict(X_test, return_ci=True)

# print metrics
print(f"C-index: {concordance_index(y_test, mean)}")
print(f"Avg. Brier Score: {approx_brier_score(y_test, mean)}")

# plotting CIs
plot_ci(mean, upper_ci, lower_ci)

#
# %%time
# ######################################################################################################################
# base model as XGBSEKaplanTree
base_model = XGBSEKaplanTree(PARAMS_TREE)

# bootstrap meta estimator
bootstrap_estimator = XGBSEBootstrapEstimator(base_model, n_estimators=100)

# fitting the meta estimator
bootstrap_estimator.fit(X_train, y_train, time_bins=TIME_BINS)

# predicting
mean, upper_ci, lower_ci = bootstrap_estimator.predict(X_test, return_ci=True)

# print metrics
print(f"C-index: {concordance_index(y_test, mean)}")
print(f"Avg. Brier Score: {approx_brier_score(y_test, mean)}")

# plotting CIs
plot_ci(mean, upper_ci, lower_ci)

# ######################################################################################################################
# ######################################################################################################################
print("End of script!")
