import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from my_ml_lib.model_selection import train_test_split
from my_ml_lib.neighbors import KNNRegressor

# Загружаем California Housing
X, y = fetch_california_housing(return_X_y=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# Твоя реализация KNN
model = KNNRegressor(
    k=5,
    p=2,
    weighted=True,
)

sk_learn_model = KNeighborsRegressor(weights="distance")

model.fit(X_train, y_train)
sk_learn_model.fit(X_train, y_train)

y_pred = model.predict(X_test)
sk_y_pred = sk_learn_model.predict(X_test)

# Метрики
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

sk_mae = mean_absolute_error(y_test, sk_y_pred)
sk_mse = mean_squared_error(y_test, sk_y_pred)
sk_rmse = np.sqrt(mse)
sk_r2 = r2_score(y_test, sk_y_pred)

print("\nMetrics:")
print(f"MAE:  {mae:.4f} | {sk_mae:.4f}")
print(f"MSE:  {mse:.4f} | {sk_mse:.4f}")
print(f"RMSE: {rmse:.4f} | {sk_rmse:.4f}")
print(f"R²:   {r2:.4f} | {sk_r2:.4f}")
