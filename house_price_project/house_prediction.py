# ==============================================================
# House Price Prediction - Regression + Exploratory Analysis
# ==============================================================

# Description: In this project, we analyze the 1990 California housing price dataset.
# We will perform an exploratory data analysis (EDA) and then build a regression model
# using XGBoost, to predict the median house price given information about the block where it is located.
#Afterwards, we will evaluate the model and review the test results.
# The main purpose is to study statistical analysis and machine learning techniques.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# =========================================
# Load Data
# =========================================

df = pd.read_csv("data/housing.csv")

# =========================================
# Feature Engineering
# =========================================

# In this step, we create new features to make the dataset
# more informative and better represent the underlying patterns.
# Absolute values are normalized by the number of households
# to capture density-related effects.

df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']
df['population_per_household'] = df['population'] / df['households']

# Drop original absolute features to avoid redundancy
df.drop(
    ['total_rooms', 'total_bedrooms', 'population'],
    axis=1,
    inplace=True
)

# =========================================
# Exploratory Data Analysis (EDA)
# =========================================

# The purpose of this exploratory analysis is to better understand
# the structure and relationships within the dataset.
# No information from this analysis is used in a way that could
# introduce data leakage into the model training process.

#Understanding data

print(df.head())
print(df.info())
print(df.isna().sum())

# Only bedrooms_per_household contains missing values


plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# Median income shows the strongest correlation with house prices.
# This is expected, as higher-income neighborhoods tend to have
# more expensive properties.

# Analyze the distribution of house prices to identify
# skewness and potential outliers

plt.figure(figsize=(15, 8))
plt.boxplot(df['median_house_value'])
plt.xticks([])   
plt.ylabel("Median House Value (USD)")
plt.title("Distribution of Median House Value")
plt.tight_layout()
plt.savefig(FIG_DIR /"boxplot_median_house_value.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['median_house_value'], 5, edgecolor='black')
plt.xlabel("Median House Value (USD)")
plt.ylabel("Frequency")
plt.title("Median House Value Distribution")
plt.tight_layout()
plt.savefig(FIG_DIR /"hist_median_house_value.png", dpi=200, bbox_inches="tight")
plt.show()

# Most houses have a median value below 300,000 USD,
# indicating a right-skewed distribution.

# Analyze the impact of ocean proximity on house prices

df_grouped = df.groupby('ocean_proximity')['median_house_value'].mean()
plt.figure(figsize=(10, 6))
plt.bar(df_grouped.index, df_grouped.values,color = ['salmon', 'skyblue', 'lightgreen', 'gold', 'plum'])
plt.xlabel("Ocean Proximity")
plt.ylabel("Average Median House Value (USD)")
plt.title("Average House Value per Ocean Proximity")
plt.tight_layout()
plt.savefig(FIG_DIR /"avg_median_house_value_per_ocean_proximity.png", dpi=200, bbox_inches="tight")
plt.show()

# Houses whose proximity to the ocean is classified as ISLAND have the highest average prices,
# while those classified as INLAND have the lowest average prices.


# =========================================
# Train / Test Split
# =========================================

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================================
# Preprocessing
# =========================================

# No scaling will be applied to numerical features,
# as XGBoost is a tree-based algorithm and does not rely on feature scale.

cat=['ocean_proximity']
num=['bedrooms_per_household']

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", SimpleImputer(strategy="median"), num) 
    ],
    remainder="passthrough"
)

# =========================================
# Model
# =========================================

xgb = XGBRegressor(
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb)
    ]
)

# =========================================
# Grid Search CV
# =========================================

# We perform hyperparameter tuning for the final model and apply cross-validation.
# The goal is to find the best combination of parameters that minimizes
# the mean absolute error (MAE). MAE was chosen because it is easy to interpret
# and more robust to outliers than the mean squared error. 
# Since we are dealing with house prices, which can have very high values,
# using the squared error could exaggerate the impact of large deviations,
# making the results harder to interpret.

param_grid = {
    "model__n_estimators": [300, 500],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.7,0.9],
    "model__colsample_bytree": [0.7,0.9]
}



grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# =========================================
# CV Results
# ========================================


best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_mae = -grid_search.best_score_


best_idx = grid_search.best_index_
n_folds = grid_search.cv

fold_scores = []
for i in range(n_folds):
    score = grid_search.cv_results_[f"split{i}_test_score"][best_idx]
    fold_scores.append(-score)

cv_matrix = pd.DataFrame(fold_scores, columns=["MAE"], index=[f"Fold {i+1}" for i in range(n_folds)])
cv_matrix.loc["Mean"] = cv_matrix.mean()

plt.figure(figsize=(6,6))
sns.heatmap(cv_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("CV MAE Scores (Final Model) ")
plt.tight_layout()
plt.savefig(FIG_DIR /"cv_scores.png", dpi=200, bbox_inches="tight")
plt.show()

# =========================================
# Feature Importances
# =========================================

preprocessor = best_model.named_steps["preprocessor"]
model = best_model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importances = model.feature_importances_
feature_names = [name.split("__")[-1] for name in feature_names]

fi = pd.DataFrame({"Feature": feature_names, "Importance": importances}) \
       .sort_values(by="Importance", ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(
    data=fi,
    x="Importance",
    y="Feature"
)
plt.tight_layout()
plt.savefig(FIG_DIR /"feature_importance.png", dpi=200, bbox_inches="tight")
plt.show()

# The exploratory data analysis indicated that median neighborhood income is the variable
# most strongly correlated with median house prices. Consistently, median income emerged
# as a highly relevant feature in the model, compared to others numerical features, contributing significantly
# to splits and error reduction.
# Within the categorical feature related to ocean proximity, the INLAND category is associated 
# with lower median house values on average. This characteristic may have contributed to its 
# importance in the model, as it participated in splits that helped reduce prediction error.

# =========================================
# Test Set Evaluation
# =========================================

y_pred =best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

test_results = pd.DataFrame(
    {"MAE": [mae], "MSE": [mse], "R²": [r2]},
    index=["Metrics"]
)

#Test Metrics
plt.figure(figsize=(6, 4))
sns.heatmap(test_results, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Test Set Performance")
plt.tight_layout()
plt.savefig(FIG_DIR /"test_set_performance.png", dpi=200, bbox_inches="tight")
plt.show()

#Actual X Predicted Values
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--"
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.tight_layout()
plt.savefig(FIG_DIR /"actual_vs_predicted_values.png", dpi=200, bbox_inches="tight")
plt.show()


#Residuals 
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(FIG_DIR /"residuals.png", dpi=200, bbox_inches="tight")
plt.show()

#Comparision between mean median price house value and the mean absolute error
mean_house_value = y_test.mean()

x_labels = ['Mean Median House Value', 'Mean Absolute Error']
y_values = [mean_house_value, mae]

plt.figure(figsize=(10,6))
bars = plt.bar(x_labels, y_values, color=['skyblue', 'salmon'])

plt.title('Comparison Between Mean Median House Value and Mean Absolute Error')
plt.ylabel('Value (USD)')
plt.yscale('log')

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:,.2f}',
        ha='center',
        va='bottom'
    )
plt.tight_layout()
plt.savefig(FIG_DIR /"comparsion.png", dpi=200, bbox_inches="tight")
plt.show()

# After first examining the overall MAE on the test set, we inspected the maximum house price (y_test.max())
# and defined three price ranges: low (0-200k), medium (200k-400k), and high (400k+)
# to evaluate how the prediction error (MAE) varies within each range.
#
# However, these values can be potentially misleading. When calculating the MAE, a small number 
# of large errors may disproportionately influence the result, while numerous smaller errors 
# could remain obscured.
# Additionally, the number of houses in each range influences the aggregated MAE.
#
# A naive comparison of the overall MAE to the mean median house value suggests an error 
# of approximately 14% of the actual house price.
# To take a more pessimistic and realistic perspective, we evaluate how many predictions 
# fall within a 20% error margin of the Actual value. We adopt a wider threshold because we do not expect 
# the model to achieve very high precision across all price ranges.

# Reset the index to ensure alignment between y_test and y_pred
y_test.reset_index(drop=True, inplace=True)

# -----------------------------
# Define house price ranges
# -----------------------------

range_low = y_test <= 200_000
range_medium = (y_test > 200_000) & (y_test <= 400_000)
range_high = y_test > 400_000

y_low = y_test[range_low]
y_medium = y_test[range_medium]
y_high = y_test[range_high]

# Select corresponding predictions
y_low_pred = y_pred[y_low.index]
y_medium_pred = y_pred[y_medium.index]
y_high_pred = y_pred[y_high.index]

# ---------------------------------------------------------
# Compute Mean Absolute Error (MAE) for each price range
# ---------------------------------------------------------

mae_low = mean_absolute_error(y_low, y_low_pred)
mae_medium = mean_absolute_error(y_medium, y_medium_pred)
mae_high = mean_absolute_error(y_high, y_high_pred)

# ------------------------------------------------------------------------
# Compute proportion of predictions with ≤20% of actual House Price error
# ------------------------------------------------------------------------

residual_low = np.abs(y_low - y_low_pred)
residual_medium = np.abs(y_medium - y_medium_pred)
residual_high = np.abs(y_high - y_high_pred)

prop_low = (residual_low <= 0.2 * y_low).sum() / y_low.count()
prop_medium = (residual_medium <= 0.2 * y_medium).sum() / y_medium.count()
prop_high = (residual_high <= 0.2 * y_high).sum() / y_high.count()

# -----------------------------
# Visualization:
# -----------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))  
labels = ['0-200k', '200k-400k', '400k+']
colors = ['skyblue', 'salmon', 'lightgreen']

#Number of houses per price range
counts = [y_low.count(), y_medium.count(), y_high.count()]
bars = axes[0].bar(labels, counts, color=colors)
axes[0].set_title('Number of Houses by Price Range')
axes[0].set_ylabel('Number of Houses')


for bar in bars:
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom'
    )

#MAE per price range
maes = [mae_low, mae_medium, mae_high]
bars = axes[1].bar(labels, maes, color=colors)
axes[1].set_title('Mean Absolute Error by Price Range')
axes[1].set_ylabel('MAE (USD)')

for bar in bars:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:,.2f}',
        ha='center',
        va='bottom'
    )

#Percentage of predictions within 20% actual house price error
percentages = [prop_low, prop_medium, prop_high]
bars = axes[2].bar(labels, percentages, color=colors)
axes[2].set_title('Predictions with ≤20% Actual House Price Error')
axes[2].set_ylabel('Proportion')
axes[2].set_ylim(0, 1)

for bar in bars:
    height = bar.get_height()
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height*100:.1f}%',
        ha='center',
        va='bottom'
    )
plt.tight_layout()
plt.savefig(FIG_DIR /"ranges_analysis.png", dpi=200, bbox_inches="tight")
plt.show()

# It can be observed that, across all price ranges, in at least 70% of the cases 
# the prediction error did not exceed 20% of the actual house value. 
# The range with the highest proportion of accurate predictions was the medium range, 
# where approximately 80% of the cases achieved this level of accuracy.
#
# The absolute mean error in the high-price range is larger, although this range has the 
# fewest representatives. This indicates that the model made larger mistakes in this range, 
# although it rarely committed extremely disproportionate errors. 
#This observation is consistent with the scatter plot, where points are less densely concentrated around the y = x line in this price range.
#
# In the low-price range, the MAE is smaller, even though this range contains the most houses. 
# This suggests that errors are generally less severe, but the model captured proportionality 
# less effectively than in the other ranges.

# It is important to note that, in a business context, it is particularly desirable for the model
# to exhibit small proportional errors for high-priced houses. The same percentage error can
# correspond to drastically different monetary values in the low and high price ranges.
#
# The purpose of this analysis is solely to evaluate how informative the MAE computed over the
# entire dataset actually is. With an overall MAE of approximately 29,000 USD, assuming that this
# error applies uniformly to all cases would lead to misleading conclusions: the model would
# appear extremely poor for low-priced houses (e.g., 50,000 USD) and highly successful for expensive
# properties (>400,000 USD).
#
# Therefore, this analysis aims to interpret the results more cautiously by examining model
# performance across different price ranges.


# =========================================
# Conclusion
# =========================================

# Median neighborhood income and ocean proximity are the most influential features 
# for predicting house prices, according to the model. The model achieved a reasonable MAE of ~29,000 USD, 
# with better accuracy in the medium-price range. While the dataset is old and predictions 
# are not perfect, the model provides meaningful insights into housing price patterns.
# Attempts to refine the model and adopt conservative strategies to make the insights 
# more applicable to real-world business contexts could be explored.