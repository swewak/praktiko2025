import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool

# загрузка данных
data = pd.read_csv('wine8.csv')

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\n")


# признаки и целевая переменная
features = [
    'winery_category', 'wine_category', 'region_category', 'type_category',
    'norm_num_reviews', 'price', 'acidity', 'body', 'year'
]
target = 'rating'
#
#
# разделение данных
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# указание категориальных признаков (если они есть)
cat_features = [
    'winery_category', 'wine_category', 'region_category', 'type_category'
]

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# обучение
cat_model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    random_state=42,
)
cat_model.fit(train_pool, eval_set=test_pool)


# предсказание и оценка
y_pred_cat = cat_model.predict(test_pool)
evaluate_model(y_test, y_pred_cat)

# важность признаков
importances = pd.DataFrame({
    'Feature': features,
    'Importance': cat_model.get_feature_importance()
}).sort_values(by='Importance', ascending=False)
print("Важность признаков в CatBoost:")
print(importances)

model = CatBoostRegressor(iterations=1000, random_state=42, verbose=100)
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

model.save_model("catboost_model.json", format="json")

