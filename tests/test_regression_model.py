import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from my_ml_lib.linear import LinearRegression
from my_ml_lib.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. Создаём данные
# =========================

rng = np.random.default_rng(42)

X = rng.normal(size=(1000, 3))

# Настоящие параметры модели
true_intercept = 5.0
true_coef = np.array([2.0, -3.0, 0.5])

# Немного шума
noise = rng.normal(0, 0.5, size=1000)

y = true_intercept + X @ true_coef + noise


# =========================
# 2. Обучаем свои модели
# =========================

my_model = LinearRegression()
sk_model = SklearnLinearRegression()

my_model.fit(X, y)
sk_model.fit(X, y)


# =========================
# 3. Получаем предсказания
# =========================

my_pred = my_model.predict(X)
sk_pred = sk_model.predict(X)


# =========================
# 4. Сравниваем коэффициенты
# =========================

print("=== TRUE PARAMETERS ===")
print(f"Intercept: {true_intercept}")
print(f"Coef:      {true_coef}")

print("\n=== MY MODEL ===")
print(f"Intercept: {my_model.intercept_}")
print(f"Coef:      {my_model.coef_}")

print("\n=== SKLEARN ===")
print(f"Intercept: {sk_model.intercept_}")
print(f"Coef:      {sk_model.coef_}")


# =========================
# 5. Ошибка восстановления коэффициентов
# =========================

print("\n=== PARAMETER ERROR ===")

print(
    "My intercept error:",
    abs(my_model.intercept_ - true_intercept),
)

print(
    "My coef error:",
    np.abs(my_model.coef_ - true_coef),
)

print(
    "Sklearn intercept error:",
    abs(sk_model.intercept_ - true_intercept),
)

print(
    "Sklearn coef error:",
    np.abs(sk_model.coef_ - true_coef),
)


# =========================
# 6. Метрики предсказаний
# =========================

print("\n=== MY MODEL METRICS ===")

print("MAE :", mean_absolute_error(y, my_pred))
print("MSE :", mean_squared_error(y, my_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, my_pred)))
print("R2  :", r2_score(y, my_pred))


print("\n=== SKLEARN METRICS ===")

print("MAE :", mean_absolute_error(y, sk_pred))
print("MSE :", mean_squared_error(y, sk_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, sk_pred)))
print("R2  :", r2_score(y, sk_pred))


# =========================
# 7. Сравниваем предсказания
# =========================

print("\n=== PREDICTION DIFFERENCE ===")

print(
    "Max absolute difference:",
    np.max(np.abs(my_pred - sk_pred)),
)

print(
    "Mean absolute difference:",
    np.mean(np.abs(my_pred - sk_pred)),
)


# =========================
# 8. Проверяем, что результаты совпадают
# =========================

assert np.allclose(
    my_model.coef_,
    sk_model.coef_,
)

assert np.isclose(
    my_model.intercept_,
    sk_model.intercept_,
)

assert np.allclose(
    my_pred,
    sk_pred,
)

print("\nAll checks passed!")
