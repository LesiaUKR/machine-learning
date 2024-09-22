# 1. Здійснюємо імпорт необхідних пакетів.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack


# %%

# 2. Завантаження даних: читання даних із файлу CSV
# Метод shape показує кількість рядків і стовпців у датасеті.

data = pd.read_csv('../datasets/weatherAUS.csv')
print('Загальна кількість рядків і стовпців у датасеті:', data.shape)

# %%

# 3.1 Видалення колонок з більше ніж 35% пропусків та рядків 
# без значення 'RainTomorrow'
data = data[data.columns[data.isna().mean().lt(0.35)]]
data = data.dropna(subset='RainTomorrow')
print('''Загальна кількість рядків і стовпців у датасеті після
видалення ознак із великою кількістю пропущених значень:''', data.shape)

# %%

# 3.2 Зміна типу колонки Date на тип datetime і створення
# додаткових колонок Year та Month

data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data = data.drop('Date', axis=1)

# %%

# 3.3 Створення підмножини набору даних із числовими та категоріальними ознаками
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = data.select_dtypes(include=[object]).columns.tolist()

# %%

# 3.4 Переміщення колонки Month до підмножини категоріальних ознак та видалення
# колонки Month із numeric_features. Колонка Year вже знаходиться в 
# numeric_features, бо перетворення колонки Date відбувалось безпосередньо 
# в DataFrame data

if 'Month' in numeric_features:
    numeric_features.remove('Month')  # Видаляємо 'Year' із категоріальних ознак
if 'Month' not in categorical_features:
    categorical_features.append('Month')  # Додаємо 'Year' до числових ознак

# %%
# 3.5 Розбиття підмножини на тренувальну і тестову вибірки за такою логікою: 
# до тестової вибірки відносимо всі об'єкти із набору даних із останнім 
# (максимальним) роком спостережень, а для навчання моделі залишаємо всі 
# інші об'єкти.

# Знайдемо максимальний рік
max_year = data['Year'].max()

# Розбиття даних на тренувальну і тестову вибірки
train_data = data[data['Year'] < max_year].copy()
test_data = data[data['Year'] == max_year].copy()

# Перевірка наявності 'RainTomorrow' у вхідних даних
if 'RainTomorrow' in numeric_features:
    numeric_features.remove('RainTomorrow')
if 'RainTomorrow' in categorical_features:
    categorical_features.remove('RainTomorrow')

# Перевірка кількості рядків у тренувальній і тестовій вибірках
print("Тренувальні дані:", train_data.shape)
print("Тестові дані:", test_data.shape)

# %%

# 4. Відновлення пропущених даних за допомогою SimpleImputer

# Перевірка та обробка пропущених значень у цільовій змінній
train_data = train_data.dropna(subset=['RainTomorrow'])
test_data = test_data.dropna(subset=['RainTomorrow'])

imputer = SimpleImputer(strategy='mean')
train_data[numeric_features] = imputer.fit_transform(train_data[numeric_features])
test_data[numeric_features] = imputer.transform(test_data[numeric_features])

# Перевірка на наявність пропущених значень після імпутації у тренувальних даних
print("Пропущені значення у числових ознаках після імпутації (тренувальні дані):")
print(train_data[numeric_features].isna().sum())

# Перевірка на наявність пропущених значень після імпутації у тестових даних
print("\nПропущені значення у числових ознаках після імпутації (тестові дані):")
print(test_data[numeric_features].isna().sum())

# %%

# 5. Нормалізація числових ознак за допомогою StandardScaler

scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
test_data[numeric_features] = scaler.transform(test_data[numeric_features])

# %%

# 6. Кодування категоріальних ознак за допомогою OneHotEncoder

encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
train_categorical = encoder.fit_transform(train_data[categorical_features])
test_categorical = encoder.transform(test_data[categorical_features])

#%%
# 7. Об'єднання підмножин

# Перевірка розмірності перед об'єднанням
X_train_numeric = train_data[numeric_features].values
X_test_numeric = test_data[numeric_features].values

# Об'єднання ознак
X_train = hstack([X_train_numeric, train_categorical])
X_test = hstack([X_test_numeric, test_categorical])

# Цільова змінна
y_train = train_data['RainTomorrow'].values
y_test = test_data['RainTomorrow'].values


#%%

# 8. Розрахунок метрик нової моделі за допомогою classification_report()

model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Прогнозування та оцінка моделі
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#%%
'''ВИСНОВКИ:
    
- Покращення для класу "No":
Нова модель показує покращені метрики для класу "No", особливо в частині Recall 
(з 0.79 до 0.82). Це означає, що модель тепер краще ідентифікує випадки 
відсутності дощу, що позитивно впливає на точність для цього класу.

- Стабільність для класу "Yes":
Precision для класу "Yes" залишається на тому ж рівні, але Recall трохи 
знизився. Це може бути результатом того, що модель зосереджена на покращенні 
передбачення класу "No", що може мати негативний вплив на менш поширений 
клас "Yes".

- Збалансованість моделі:
Нова модель виглядає більш збалансованою щодо передбачення обох класів, але 
покращення для класу "Yes" незначні або відсутні, що свідчить про те, що модель
ще потребує додаткового балансування або інших методів для роботи
з класом меншості.

- Потреба в балансуванні:
Незважаючи на невелике загальне покращення, модель ще може виграти від 
використання технік, таких як oversampling або SMOTE, щоб підвищити точність 
та здатність розпізнавати дощ.

Отже, нова модель продемонструвала покращення для класу "No", але 
продуктивність для класу "Yes" залишилась на тому ж рівні або навіть трохи 
погіршилась.'''






