# %% [Імпорт необхідних бібліотек]
import warnings  # Бібліотека для роботи з попередженнями
from tempfile import mkdtemp  # Модуль для створення тимчасових директорій
import mlflow  # Бібліотека для відстеження експериментів
import joblib  # Бібліотека для збереження та завантаження моделей
import pandas as pd  # Бібліотека для роботи з даними
import numpy as np  # Бібліотека для числових обчислень
from sklearn.ensemble import GradientBoostingRegressor  # Алгоритм регресії на основі Gradient Boosting
from sklearn.model_selection import cross_val_score, GridSearchCV  # Функції для крос-валідації та підбору параметрів
from sklearn.pipeline import Pipeline  # Модуль для створення конвеєрів

# %% [Налаштування MLflow та встановлення експерименту]
mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')  # Встановлення URI для відстеження MLflow
mlflow.set_experiment('MLflow Tracking')  # Встановлення експерименту з назвою 'MLflow Tracking'

# %% [Завантаження підготовлених даних]
data = pd.read_pickle('../derived/mod_07_topic_13_bigmart_data_upd.pkl.gz')  # Завантаження даних із збереженого pickle файлу

# Розподіл даних на ознаки (X) та цільову змінну (y)
X, y = (data.drop(['Item_Identifier',
                   'Item_Outlet_Sales'],
                  axis=1),
        data['Item_Outlet_Sales'])

# %% [Завантаження попередньо збереженої моделі конвеєра]
with open('../models/mod_07_topic_13_mlpipe.joblib', 'rb') as fl:
    pipe_base = joblib.load(fl)  # Завантаження конвеєра моделі з joblib файлу

# %% [Крос-валідація на основі початкової моделі]
cv_results = cross_val_score(
    estimator=pipe_base,  # Початкова модель
    X=X,
    y=y,
    scoring='neg_root_mean_squared_error',  # Використання метрики RMSE
    cv=5)  # 5-кратна крос-валідація

rmse_cv = np.abs(cv_results).mean()  # Обчислення середнього значення RMSE для крос-валідації

# %% [Навчання початкової моделі на всіх даних]
model_base = pipe_base.fit(X, y)  # Навчання моделі на всіх доступних даних

# %% [Отримання параметрів регресора з початкової моделі]
params_base = pipe_base.named_steps['reg_estimator'].get_params()  # Отримання параметрів алгоритму RandomForestRegressor

# %% [Логування початкової моделі за допомогою MLflow]
# Запуск експерименту MLflow
with mlflow.start_run(run_name='rfr'):
    mlflow.log_params(params_base)  # Логування гіперпараметрів початкової моделі
    mlflow.log_metric('cv_rmse_score', rmse_cv)  # Логування середнього значення RMSE
    mlflow.set_tag('Model', 'RandomForest for BigMart')  # Додавання тега до моделі

    # Обробка попереджень під час створення сигнатури
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Інферування сигнатури моделі для майбутньої верифікації
        signature = mlflow.models.infer_signature(
            X.head(),
            model_base.predict(X.head()))

    # Логування моделі у сховище MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=model_base,
        artifact_path='model_base',
        signature=signature,
        input_example=X.head(),  # Приклад вхідних даних
        registered_model_name='model_base_tracking')  # Зареєстрована назва моделі

# %% [Створення нового конвеєра з оновленою моделлю]
pipe_upd = Pipeline(
    steps=pipe_base.steps[:-1] +  # Використання попередніх кроків конвеєра
    [('reg_model',
      GradientBoostingRegressor(random_state=42))],  # Додавання Gradient Boosting як регресора
    memory=mkdtemp())  # Використання тимчасової пам'яті для конвеєра

# %% [Підбір гіперпараметрів для Gradient Boosting]
parameters = {
    'reg_model__learning_rate': (0.1, 0.3),  # Діапазон для параметра learning_rate
    'reg_model__subsample': (0.75, 0.85),  # Діапазон для subsample
    'reg_model__max_features': ('sqrt', 'log2')}  # Діапазон для max_features

search = (GridSearchCV(
    estimator=pipe_upd,  # Оновлена модель
    param_grid=parameters,  # Гіперпараметри для підбору
    scoring='neg_root_mean_squared_error',  # Використання метрики RMSE
    cv=5,  # 5-кратна крос-валідація
    refit=False)  # Пошук без автоматичного навчання на всіх даних
    .fit(X, y))  # Пошук найкращих параметрів

# %% [Встановлення найкращих параметрів та навчання оновленої моделі]
parameters_best = search.best_params_  # Отримання найкращих параметрів
pipe_upd = pipe_upd.set_params(**parameters_best)  # Встановлення параметрів у модель

model_upd = pipe_upd.fit(X, y)  # Навчання оновленої моделі

# %% [Крос-валідація для оновленої моделі]
cv_results_upd = cross_val_score(
    estimator=pipe_upd,  # Оновлена модель
    X=X,
    y=y,
    scoring='neg_root_mean_squared_error',
    cv=5)  # 5-кратна крос-валідація

rmse_cv_upd = np.abs(cv_results_upd).mean()  # Обчислення середнього RMSE для оновленої моделі

# %% [Логування оновленої моделі в MLflow]
with mlflow.start_run(run_name='gbr'):
    mlflow.log_params(pipe_upd.named_steps['reg_model'].get_params())  # Логування гіперпараметрів Gradient Boosting
    mlflow.log_metric('cv_rmse_score', rmse_cv_upd)  # Логування середнього RMSE
    mlflow.set_tag('Model', 'GradientBoosting model for BigMart')  # Додавання тега до моделі

    # Обробка попереджень під час створення сигнатури
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Інферування сигнатури моделі
        signature = mlflow.models.infer_signature(
            X.head(),
            model_upd.predict(X.head()))

    # Логування моделі у сховище MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=model_upd,
        artifact_path='model_upd',
        signature=signature,
        input_example=X.head(),
        registered_model_name='model_upd_tracking')

# %% [Пошук найкращої моделі]
# best_run = (mlflow
#             .search_runs(
#                 experiment_names=['MLflow Tracking'],
#                 order_by=['metrics.cv_rmse_score'],
#                 max_results=1))

# best_run[['tags.Model', 'metrics.cv_rmse_score']]  # Виведення інформації про найкращий запуск

