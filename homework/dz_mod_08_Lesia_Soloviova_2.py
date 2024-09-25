import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from datetime import datetime

# Завантаження даних із CSV-файлів
train_data = pd.read_csv('./datasets/mod_04_hw_train_data.csv')
valid_data = pd.read_csv('./datasets/mod_04_hw_valid_data.csv')

# Видалення пропущених значень
train_data.dropna(subset=['Name', 'Phone_Number'], inplace=True)
valid_data.dropna(subset=['Name', 'Phone_Number'], inplace=True)

# Видалення стовпців, які не мають значення для прогнозування
train_data.drop(['Name', 'Phone_Number'], axis=1, inplace=True)
valid_data.drop(['Name', 'Phone_Number'], axis=1, inplace=True)

# Функція для обчислення віку на основі дати народження
def calculate_age(dob):
    return datetime.now().year - pd.to_datetime(dob, format='%d/%m/%Y').dt.year

# Додавання стовпця "Age" для віку
train_data['Age'] = calculate_age(train_data['Date_Of_Birth'])
valid_data['Age'] = calculate_age(valid_data['Date_Of_Birth'])

# Видалення стовпця "Date_Of_Birth"
train_data.drop('Date_Of_Birth', axis=1, inplace=True)
valid_data.drop('Date_Of_Birth', axis=1, inplace=True)

# Розділяємо числові та категоріальні стовпці
num_cols = ['Experience', 'Age']  # числові стовпці
cat_cols = ['Qualification', 'University', 'Role', 'Cert']  # категоріальні стовпці

# Імпутація числових і категоріальних значень
num_imputer = SimpleImputer(strategy='mean')
train_data[num_cols] = num_imputer.fit_transform(train_data[num_cols])
valid_data[num_cols] = num_imputer.transform(valid_data[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
train_data[cat_cols] = cat_imputer.fit_transform(train_data[cat_cols])
valid_data[cat_cols] = cat_imputer.transform(valid_data[cat_cols])

# Видалення аномальних значень
min_age = 18
max_age = 65
min_experience = 0
max_experience = 40

train_data = train_data[
    (train_data['Age'] >= min_age) & 
    (train_data['Age'] <= max_age) & 
    (train_data['Experience'] >= min_experience) & 
    (train_data['Experience'] <= max_experience)
]

valid_data = valid_data[
    (valid_data['Age'] >= min_age) & 
    (valid_data['Age'] <= max_age) & 
    (valid_data['Experience'] >= min_experience) & 
    (valid_data['Experience'] <= max_experience)
]

# Кореляційна матриця та One-Hot Encoding
categorical_cols = ['Qualification', 'University', 'Role', 'Cert']
encoder = OneHotEncoder(drop='first', sparse_output=False)

train_encoded = pd.DataFrame(encoder.fit_transform(train_data[categorical_cols]), 
                             columns=encoder.get_feature_names_out(categorical_cols))
valid_encoded = pd.DataFrame(encoder.transform(valid_data[categorical_cols]), 
                             columns=encoder.get_feature_names_out(categorical_cols))

train_data_encoded = pd.concat([train_data.drop(categorical_cols, axis=1), train_encoded], axis=1)
valid_data_encoded = pd.concat([valid_data.drop(categorical_cols, axis=1), valid_encoded], axis=1)

# Видалення ознак з низькою кореляцією
correlation_threshold = 0.1
correlations = train_data_encoded.corr()['Salary'].abs().sort_values(ascending=False)
low_corr_features = correlations[correlations < correlation_threshold].index.tolist()
train_data_encoded.drop(low_corr_features, axis=1, inplace=True)
valid_data_encoded.drop(low_corr_features, axis=1, inplace=True)

# Перевірка на пропущені значення перед масштабуванням
print("Пропущені значення перед масштабуванням (тренувальні дані):")
print(train_data_encoded.isnull().sum())

print("Пропущені значення перед масштабуванням (валідаційні дані):")
print(valid_data_encoded.isnull().sum())

# Видалення рядків з NaN після обробки
train_data_encoded.dropna(inplace=True)
valid_data_encoded.dropna(inplace=True)

# Розділення на ознаки (X) та цільову змінну (y)
X_train = train_data_encoded.drop('Salary', axis=1)
y_train = train_data_encoded['Salary']
X_valid = valid_data_encoded.drop('Salary', axis=1)
y_valid = valid_data_encoded['Salary']

# Нормалізуємо числові ознаки
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Перевірка наявності пропущених значень після масштабування
print("\nПропущені значення після масштабування у тренувальних даних:")
print(np.isnan(X_train_scaled).sum())

print("\nПропущені значення після масштабування у валідаційних даних:")
print(np.isnan(X_valid_scaled).sum())

# Пошук найкращого значення k за допомогою GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 20)}
knn = KNeighborsRegressor()
knn_gscv = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')
knn_gscv.fit(X_train_scaled, y_train)

best_k = knn_gscv.best_params_['n_neighbors']
print(f"\nНайкраще значення k: {best_k}")

# Створення моделі KNeighborsRegressor з оптимальним k і тренування на нормалізованих даних
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# Прогнозування на валідаційному наборі даних
y_valid_pred = knn.predict(X_valid_scaled)

# Обчислення метрики MAPE
mape = mean_absolute_percentage_error(y_valid, y_valid_pred)
print(f'Validation MAPE: {mape:.2%}')
