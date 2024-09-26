import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer



# %% Завантаження даних із CSV-файлів
train_data = pd.read_csv('./datasets/mod_04_hw_train_data.csv')
valid_data = pd.read_csv('./datasets/mod_04_hw_valid_data.csv')

# %% Видалення стовпців, які не мають значення для прогнозування 
# (Ім'я та Номер телефону)
train_data.drop(['Name', 'Phone_Number','Date_Of_Birth'], axis=1, inplace=True)
valid_data.drop(['Name', 'Phone_Number','Date_Of_Birth'], axis=1, inplace=True)

# %% Виведення перших кількох рядків для перевірки структури даних
print(train_data.head())

# %% Описова статистика для числових стовпців
print(train_data.describe())

# %% Перевірка на наявність пропущених значень у тренувальному наборі даних
print("Missing values in training data:\n", train_data.isnull().sum())

# Перевірка на наявність пропущених значень у valid_data
print("Missing values in validation data:\n", valid_data.isnull().sum())

# Видаляємо рядки з пропущеними значеннями
train_data_cleaned = train_data.dropna()
valid_data_cleaned = valid_data.dropna()

# Перевіряємо, чи залишилися пропущені значення після очищення
missing_after_cleanup = train_data_cleaned.isnull().sum()
missing_after_cleanup = valid_data_cleaned.isnull().sum()
print("Missing values after cleanup:\n", missing_after_cleanup)
print(missing_after_cleanup)

# %% Розділяємо числові та категоріальні стовпці
num_cols = ['Experience']
cat_cols = ['Qualification', 'University', 'Role', 'Cert']
# %% 
# Імпутація числових значень (середні значення для числових)
num_imputer = SimpleImputer(strategy='mean')
train_data[num_cols] = num_imputer.fit_transform(train_data[num_cols])
valid_data[num_cols] = num_imputer.transform(valid_data[num_cols])

# Імпутація категоріальних значень (найбільш часті для категоріальних)
cat_imputer = SimpleImputer(strategy='most_frequent')
train_data[cat_cols] = cat_imputer.fit_transform(train_data[cat_cols])
valid_data[cat_cols] = cat_imputer.transform(valid_data[cat_cols])

# Дискретизація числових ознак за допомогою KBinsDiscretizer
kbins = KBinsDiscretizer(encode='ordinal', n_bins=5)
num_cols_kbins = valid_data.select_dtypes(include=[np.number]).columns
valid_data[num_cols_kbins] = kbins.fit_transform(valid_data[num_cols_kbins])

# %%
# EDA: Visualize distributions of numeric features (Experience)
sns.histplot(train_data['Experience'], kde=True)
plt.title('Experience Distribution')
plt.show()
# %%
# EDA: Check the target distribution (Salary)
sns.histplot(train_data['Salary'], kde=True)
plt.title('Salary Distribution')
plt.show()

# %%

# One-Hot Encoding для категоріальних змінних перед кореляційною матрицею
encoder = OneHotEncoder(drop='first', sparse_output=False)

# %% Кодування категоріальних змінних для обох наборів даних
train_encoded = pd.DataFrame(encoder.fit_transform(train_data[cat_cols]), 
                             columns=encoder.get_feature_names_out(cat_cols))
valid_encoded = pd.DataFrame(encoder.transform(valid_data[cat_cols]), 
                             columns=encoder.get_feature_names_out(cat_cols))


# %% Додавання закодованих змінних і видалення оригінальних категоріальних змінних
train_data = pd.concat([train_data.drop(cat_cols, axis=1), train_encoded], axis=1)
valid_data = pd.concat([valid_data.drop(cat_cols, axis=1), valid_encoded], axis=1)


# %% Розділення на ознаки (X) та цільову змінну (y)
X_train = train_data.drop(['Salary'], axis=1)
y_train = train_data['Salary']
X_valid = valid_data.drop(['Salary'], axis=1)

# Нормалізуємо числові ознаки для тренувального і валідаційного наборів
scaler = StandardScaler()

# Масштабування тренувальних даних
X_train_scaled = scaler.fit_transform(X_train)

# Масштабування валідаційних даних (без 'Salary')
X_valid_scaled = scaler.transform(X_valid)

# %% Пошук найкращого значення k
param_grid = {'n_neighbors': np.arange(1, 20)}
knn = KNeighborsRegressor()
knn_gscv = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')
knn_gscv.fit(X_train_scaled, y_train)

best_k = knn_gscv.best_params_['n_neighbors']
print(f"Найкраще значення k: {best_k}")

# Тренуємо модель з оптимальним k
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# Прогнозування на валідаційному наборі даних
y_valid_pred = knn.predict(X_valid_scaled)

# Обчислення метрики MAPE
mape = mean_absolute_percentage_error(valid_data['Salary'], y_valid_pred)
print(f'Validation MAPE: {mape:.2%}')


