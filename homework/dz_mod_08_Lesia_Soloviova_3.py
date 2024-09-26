import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer

# Завантаження даних із CSV-файлів
train_data = pd.read_csv('./datasets/mod_04_hw_train_data.csv')
valid_data = pd.read_csv('./datasets/mod_04_hw_valid_data.csv')

# Видалення стовпців, які не мають значення для прогнозування 
# (Ім'я та Номер телефону, а також Date_Of_Birth)
train_data.drop(['Name', 'Phone_Number', 'Date_Of_Birth'], axis=1, inplace=True)
valid_data.drop(['Name', 'Phone_Number', 'Date_Of_Birth'], axis=1, inplace=True)

# Перевірка на наявність пропущених значень
print("Missing values in training data:\n", train_data.isnull().sum())
print("Missing values in validation data:\n", valid_data.isnull().sum())

# Імпутація числових значень (середні значення для числових)
num_cols = ['Experience', 'Salary']  # Включаючи числові колонки, такі як 'Salary'
num_imputer = SimpleImputer(strategy='mean')
train_data[num_cols] = num_imputer.fit_transform(train_data[num_cols])
valid_data[num_cols] = num_imputer.transform(valid_data[num_cols])

# Імпутація категоріальних значень (найбільш часті для категоріальних)
cat_cols = ['Qualification', 'Role', 'Cert', 'University']
cat_imputer = SimpleImputer(strategy='most_frequent')
train_data[cat_cols] = cat_imputer.fit_transform(train_data[cat_cols])
valid_data[cat_cols] = cat_imputer.transform(valid_data[cat_cols])

# Перевірка, чи не залишилось пропущених значень після імпутації
print("Missing values after imputation in training data:\n", train_data.isnull().sum())
print("Missing values after imputation in validation data:\n", valid_data.isnull().sum())

# One-Hot Encoding для категоріальних змінних
encoder = OneHotEncoder(drop='first', sparse_output=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train_data[cat_cols]), 
                             columns=encoder.get_feature_names_out(cat_cols))
valid_encoded = pd.DataFrame(encoder.transform(valid_data[cat_cols]), 
                             columns=encoder.get_feature_names_out(cat_cols))

# Додавання закодованих змінних і видалення оригінальних категоріальних змінних
train_data_final = pd.concat([train_data.drop(cat_cols, axis=1), train_encoded], axis=1)
valid_data_final = pd.concat([valid_data.drop(cat_cols, axis=1), valid_encoded], axis=1)

# Побудова кореляційної матриці
corr_matrix = train_data_final.corr()

# Поріг для високої кореляції
threshold = 0.9

# Візуалізація кореляції між ознаками та Salary
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix[['Salary']].sort_values(by='Salary', ascending=False), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Кореляція ознак з цільовою змінною (Salary)')
plt.show()

# Вибір ознак, які мають кореляцію вище порогу
high_corr_features = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold) and column != 'Salary']

# Видалення сильно корельованих ознак
train_data_filtered = train_data_final.drop(high_corr_features, axis=1)
valid_data_filtered = valid_data_final.drop(high_corr_features, axis=1)

print(f"Видалено ознаки з сильною кореляцією: {high_corr_features}")

# Обчислення кореляції з цільовою змінною 'Salary'
corr_with_target = corr_matrix['Salary'].abs()

# Вибір ознак з низькою кореляцією (менше 0.1, наприклад)
low_corr_features = corr_with_target[corr_with_target < 0.1].index

# Видалення ознак з низькою кореляцією
train_data_filtered = train_data_final.drop(low_corr_features, axis=1)
valid_data_filtered = valid_data_final.drop(low_corr_features, axis=1)

print(f"Видалено ознаки з низькою кореляцією з 'Salary': {low_corr_features}")

# Розділення на ознаки (X) та цільову змінну (y)
X_train = train_data_filtered.drop(['Salary'], axis=1)  # Видаляємо 'Salary'
y_train = train_data_filtered['Salary']
X_valid = valid_data_filtered.drop(['Salary'], axis=1)  # Видаляємо 'Salary'

# Нормалізуємо числові ознаки
scaler = StandardScaler()

# Масштабування тренувальних і валідаційних даних
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Пошук найкращого значення k
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
mape = mean_absolute_percentage_error(valid_data_final['Salary'], y_valid_pred)
print(f'Validation MAPE: {mape:.2%}')
