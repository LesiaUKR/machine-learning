import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


# Завантаження даних
train_data = pd.read_csv('./datasets/mod_04_hw_train_data.csv')
valid_data = pd.read_csv('./datasets/mod_04_hw_valid_data.csv')

# Видалення колонок Name, Phone_Number
train_data_cleaned = train_data.drop(columns=['Name', 'Phone_Number'])
valid_data_cleaned = valid_data.drop(columns=['Name', 'Phone_Number'])

# Функція для обчислення віку
def calculate_age(birthdate):
    birthdate = datetime.strptime(birthdate, '%d/%m/%Y')
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

# Додавання віку і видалення дати народження
train_data_cleaned['Age'] = train_data_cleaned['Date_Of_Birth'].apply(calculate_age)
valid_data_cleaned['Age'] = valid_data_cleaned['Date_Of_Birth'].apply(calculate_age)
train_data_cleaned = train_data_cleaned.drop(columns=['Date_Of_Birth'])
valid_data_cleaned = valid_data_cleaned.drop(columns=['Date_Of_Birth'])

# 1. Обробка пропущених значень
# Заповнення пропущених значень для числових колонок (Experience)
train_data_cleaned['Experience'].fillna(train_data_cleaned['Experience'].mean(), inplace=True)
valid_data_cleaned['Experience'].fillna(valid_data_cleaned['Experience'].mean(), inplace=True)

# Заповнення пропущених значень для категоріальних колонок (Qualification, Role, Cert)
train_data_cleaned['Qualification'].fillna(train_data_cleaned['Qualification'].mode()[0], inplace=True)
train_data_cleaned['Role'].fillna(train_data_cleaned['Role'].mode()[0], inplace=True)
train_data_cleaned['Cert'].fillna(train_data_cleaned['Cert'].mode()[0], inplace=True)

valid_data_cleaned['Qualification'].fillna(valid_data_cleaned['Qualification'].mode()[0], inplace=True)
valid_data_cleaned['Role'].fillna(valid_data_cleaned['Role'].mode()[0], inplace=True)
valid_data_cleaned['Cert'].fillna(valid_data_cleaned['Cert'].mode()[0], inplace=True)

# Перевірка, що всі пропущені значення оброблені
missing_values_after = train_data_cleaned.isnull().sum()

# 2. Первинний дослідницький аналіз даних (EDA)
# Розподіл заробітної плати
plt.figure(figsize=(10, 6))
sns.histplot(train_data_cleaned['Salary'], kde=True)
plt.title('Розподіл заробітної плати (Salary)')
plt.xlabel('Заробітна плата')
plt.ylabel('Кількість')
plt.show()

# Кореляційна матриця для числових ознак
plt.figure(figsize=(10, 8))
correlation_matrix = train_data_cleaned.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Кореляційна матриця тренувальних даних')
plt.show()

# Залежність заробітної плати від досвіду
plt.figure(figsize=(10, 6))
sns.scatterplot(x=train_data_cleaned['Experience'], y=train_data_cleaned['Salary'])
plt.title('Залежність заробітної плати від досвіду')
plt.xlabel('Досвід (років)')
plt.ylabel('Заробітна плата')
plt.show()

# Розподіл заробітної плати по ролям
plt.figure(figsize=(10, 6))
sns.boxplot(x='Role', y='Salary', data=train_data_cleaned)
plt.title('Розподіл заробітної плати по ролям')
plt.xlabel('Роль')
plt.ylabel('Заробітна плата')
plt.show()

# 3. Обробка числових ознак і кодування категоріальних

# Виділяємо числові та категоріальні ознаки
numeric_features = ['Experience', 'Age']
categorical_features = ['Qualification', 'University', 'Role', 'Cert']

# Створюємо pipeline для обробки числових ознак
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Заповнення пропущених значень
    ('scaler', StandardScaler())  # Масштабування
])

# Створюємо pipeline для обробки категоріальних ознак
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Заповнення пропущених значень
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot кодування
])

# Об'єднуємо трансформації в один ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Масштабування і кодування тренувальних та валідаційних наборів даних
X_train = train_data_cleaned.drop(columns=['Salary'])
X_valid = valid_data_cleaned.drop(columns=['Salary'])

# Застосування обробки до наборів даних
X_train_transformed = preprocessor.fit_transform(X_train)
X_valid_transformed = preprocessor.transform(X_valid)

# Цільова змінна (заробітна плата)
y_train = train_data_cleaned['Salary']
y_valid = valid_data_cleaned['Salary']

# 4. Побудова моделі KNeighborsRegressor

# Ініціалізація KNeighborsRegressor з 5 сусідами
knn_model = KNeighborsRegressor(n_neighbors=5)

# Навчання моделі
knn_model.fit(X_train_transformed, y_train)

# Прогнозування на валідаційному наборі
y_valid_pred = knn_model.predict(X_valid_transformed)

# 5. Оцінка моделі
# Оцінка моделі з виведенням результатів
mae = mean_absolute_error(y_valid, y_valid_pred)
mape = mean_absolute_percentage_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)

# Виведення результатів
print("Оцінки моделі на валідаційному наборі:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
print(f"R-squared (R2): {r2:.2f}")

