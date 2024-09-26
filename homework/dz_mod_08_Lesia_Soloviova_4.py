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
train_data.drop(['Name', 'Phone_Number', 'Date_Of_Birth'], axis=1, inplace=True)
valid_data.drop(['Name', 'Phone_Number', 'Date_Of_Birth'], axis=1, inplace=True)

# %% Перевірка на наявність пропущених значень
print("Missing values in training data:\n", train_data.isnull().sum())
print("Missing values in validation data:\n", valid_data.isnull().sum())

# Імпутація числових значень (середні значення для числових)
num_cols = ['Experience']
cat_cols = ['Qualification', 'University', 'Role', 'Cert']

num_imputer = SimpleImputer(strategy='mean')
train_data[num_cols] = num_imputer.fit_transform(train_data[num_cols])
valid_data[num_cols] = num_imputer.transform(valid_data[num_cols])

# Імпутація категоріальних значень (найбільш часті для категоріальних)
cat_imputer = SimpleImputer(strategy='most_frequent')
train_data[cat_cols] = cat_imputer.fit_transform(train_data[cat_cols])
valid_data[cat_cols] = cat_imputer.transform(valid_data[cat_cols])

# %% One-Hot Encoding для категоріальних змінних
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Кодування категоріальних змінних для обох наборів даних
train_encoded = pd.DataFrame(encoder.fit_transform(train_data[cat_cols]), 
                             columns=encoder.get_feature_names_out(cat_cols))
valid_encoded = pd.DataFrame(encoder.transform(valid_data[cat_cols]), 
                             columns=encoder.get_feature_names_out(cat_cols))

# Додавання закодованих змінних і видалення оригінальних категоріальних змінних
train_data = pd.concat([train_data.drop(cat_cols, axis=1), train_encoded], axis=1)
valid_data = pd.concat([valid_data.drop(cat_cols, axis=1), valid_encoded], axis=1)

# %% Будуємо кореляційну матрицю для всіх числових змінних
correlation = train_data.corr()

# Відсортуємо за кореляцією із змінною 'Salary'
correlation_target = correlation['Salary'].sort_values(ascending=False)
print(correlation_target)

# Візуалізація кореляційної матриці
plt.figure(figsize=(10, 8))
plt.title('Correlation of Attributes with Salary')
sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white', cmap='coolwarm')
plt.xticks(rotation=90)
plt.yticks(rotation=30)
plt.show()

# %% Дискретизація числових ознак за допомогою KBinsDiscretizer
kbins = KBinsDiscretizer(encode='ordinal', n_bins=5)
train_data[num_cols] = kbins.fit_transform(train_data[num_cols])
valid_data[num_cols] = kbins.transform(valid_data[num_cols])

# %% Візуалізація розподілу ознаки Experience
sns.histplot(train_data['Experience'], kde=True)
plt.title('Experience Distribution')
plt.show()

# %% Візуалізація розподілу цільової змінної Salary
sns.histplot(train_data['Salary'], kde=True)
plt.title('Salary Distribution')
plt.show()

# %% Розділення на ознаки (X) та цільову змінну (y)
X_train = train_data.drop(['Salary'], axis=1)
y_train = train_data['Salary']
X_valid = valid_data.drop(['Salary'], axis=1)

# %% Нормалізація числових ознак
scaler = StandardScaler()

# Масштабування тренувальних даних
X_train_scaled = scaler.fit_transform(X_train)

# Масштабування валідаційних даних
X_valid_scaled = scaler.transform(X_valid)

# %% Пошук найкращого значення k і ваг для сусідів
param_grid = {
    'n_neighbors': np.arange(1, 30),  # розширений діапазон для пошуку k
    'weights': ['uniform', 'distance']
}
knn = KNeighborsRegressor()
knn_gscv = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')
knn_gscv.fit(X_train_scaled, y_train)

# Найкращі параметри
best_params = knn_gscv.best_params_
best_k = best_params['n_neighbors']
best_weights = best_params['weights']
print(f"Найкращі параметри: k={best_k}, weights={best_weights}")

# Тренування моделі з оптимальними параметрами
knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_weights)
knn.fit(X_train_scaled, y_train)

# Прогнозування на валідаційному наборі
y_valid_pred = knn.predict(X_valid_scaled)

# Обчислення MAPE
mape = mean_absolute_percentage_error(valid_data['Salary'], y_valid_pred)
print(f'Validation MAPE: {mape:.2%}')

