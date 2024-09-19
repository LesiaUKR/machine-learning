import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# %%
# Завантажуються дані про вартість житла в Каліфорнії за допомогою 
# fetch_california_housing та зберігаються у змінну data. 
# Виводяться перші кілька рядків даних для ознайомлення

california_housing = fetch_california_housing(as_frame=True)

data = california_housing['frame']
data.head()

# %%
# Виділення цільової змінної
# Видаляється стовпець MedHouseVal, який містить середню вартість житла, 
# і зберігається в окрему змінну target. Цей стовпець є цільовою змінною, 
# яку ми будемо прогнозувати
target = data.pop('MedHouseVal')
target.head()

# %%
# Огляд інформації про дані
# Метод info() надає загальну інформацію про дані, включаючи кількість рядків,
# типи даних та наявність пропущених значень.
data.info()

# %%
# Візуалізація розподілу даних
# Використовується seaborn для візуалізації розподілу всіх змінних за 
# допомогою гістограм. Дані "розплавляються" за допомогою методу melt, 
# що дозволяє легко будувати графіки для всіх змінних на одній площині

sns.set_theme()

melted = pd.concat([data, target], axis=1).melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=3,
                  sharex=False,
                  sharey=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

plt.show()

# %%
# Опис обраних змінних
# Виводиться статистичний опис для кількох важливих змінних, 
# таких як середня кількість кімнат і спалень, середня кількість 
# мешканців і населення

features_of_interest = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
data[features_of_interest].describe()

# %%
# Візуалізація географічного розподілу вартості
# Створюється скаттерплот, що показує вартість житла в залежності від їх 
# географічного розташування за координатами Longitude і Latitude. 
# Значення відображаються через колір та розмір точок

fig, ax = plt.subplots(figsize=(6, 5))

sns.scatterplot(
    data=data,
    x='Longitude',
    y='Latitude',
    size=target,
    hue=target,
    palette='viridis',
    alpha=0.5,
    ax=ax)

plt.legend(
    title='MedHouseVal',
    bbox_to_anchor=(1.05, 0.95),
    loc='upper left')

plt.title('Median house value depending of\n their spatial location')

plt.show()

# %%
# Кореляційна матриця
# Створюється кореляційна матриця між змінними (за винятком координат). 
# Відображається теплова карта з кореляціями, що дозволяє оцінити залежності 
# між різними ознаками.

columns_drop = ['Longitude', 'Latitude']
subset = pd.concat([data, target], axis=1).drop(columns=columns_drop)

corr_mtx = subset.corr()

mask_mtx = np.zeros_like(corr_mtx)
np.fill_diagonal(mask_mtx, 1)

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(subset.corr(),
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            linewidth=0.5,
            square=True,
            mask=mask_mtx,
            ax=ax)

plt.show()

# %%
# Поділ на тренувальний і тестовий набори
# Дані поділяються на тренувальний та тестовий набори в пропорції 80/20

X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42)

# %%
# Масштабування даних
# Масштабуються всі ознаки для тренувального та тестового наборів за допомогою
# стандартного масштабування (нормалізація до середнього 0 та 
# стандартного відхилення 1)
scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Опис масштабованих тренувальних даних
# Виводиться опис масштабованих тренувальних даних для перевірки 
# процесу нормалізації

X_train_scaled.describe()

# %%
# Лінійна регресія і прогнозування
# Створюється модель лінійної регресії, яка тренується на масштабованих даних.
# Робляться прогнози для тестового набору, причому прогнозовані значення 
# обмежуються мінімальними та максимальними значеннями тренувальних даних.

model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

ymin, ymax = y_train.agg(['min', 'max']).values

y_pred = pd.Series(y_pred, index=X_test_scaled.index).clip(ymin, ymax)
y_pred.head()

# %%
# Оцінка якості моделі
# Обчислюються основні метрики для оцінки моделі: коефіцієнт детермінації (R²),
# середня абсолютна помилка (MAE) та середня абсолютна відносна помилка (MAPE).
r_sq = model.score(X_train_scaled, y_train)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')

# %%
# Коефіцієнти моделі
# Виводяться коефіцієнти ознак у лінійній моделі
pd.Series(model.coef_, index=X_train_scaled.columns)

# %%
# Модель з поліноміальними ознаками
# Додаються поліноміальні ознаки другого порядку для поліпшення моделі, 
# після чого модель знову тренується і обчислюються нові метрики
# [a, b] -> [1, a, b, a^2, ab, b^2]
poly = PolynomialFeatures(2).set_output(transform='pandas')

Xtr = poly.fit_transform(X_train_scaled)
Xts = poly.transform(X_test_scaled)

model_upd = LinearRegression().fit(Xtr, y_train)
y_pred_upd = model_upd.predict(Xts)
y_pred_upd = pd.Series(y_pred_upd, index=Xts.index).clip(ymin, ymax)

r_sq_upd = model_upd.score(Xtr, y_train)
mae_upd = mean_absolute_error(y_test, y_pred_upd)
mape_upd = mean_absolute_percentage_error(y_test, y_pred_upd)

print(f'R2: {r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')

# %%
# Аналіз відсоткової похибки
# Створюється скаттерплот для відображення відсоткової похибки між 
# прогнозованими та реальними значеннями, що дозволяє проаналізувати
# точність моделі

pct_error = (y_pred_upd / y_test - 1).clip(-1, 1)

sns.scatterplot(
    x=y_test,
    y=pct_error,
    hue=pct_error.gt(0),
    alpha=0.5,
    s=10,
    legend=False)

plt.show()
