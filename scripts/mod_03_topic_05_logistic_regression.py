# Цей блок імпортує всі необхідні бібліотеки для роботи з даними, побудови
# моделі та візуалізації. 
# Основні бібліотеки: 
# Pandas для роботи з таблицями
# NumPy для числових операцій
# Sklearn для машинного навчання
# Seaborn і Matplotlib для візуалізації.

import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Завантаження даних: Читання даних із файлу CSV, стисненого у форматі GZIP
# Метод shape показує кількість рядків і стовпців у датасеті.

data = pd.read_csv('../datasets/mod_03_topic_05_weather_data.csv.gz')
data.shape

# %%
# Перегляд перших рядків: Виводяться перші 5 рядків даних, щоб зрозуміти 
# загальну структуру датасету

data.head()

# %%
# Типи даних: Перевірка типів даних у кожному стовпці 
# (числові, категорійні, текстові)
data.dtypes

# %%
# Аналіз пропусків: Обчислення частки пропусків у кожному стовпці та 
# сортування в порядку зменшення

data.isna().mean().sort_values(ascending=False)

# %%
# Візуалізація пропусків за локаціями: Цей блок обчислює частку пропусків для 
# кожної локації, а потім будує теплову карту (heatmap) для їхньої візуалізації
# Це допомагає зрозуміти, які локації мають найбільше пропусків у даних.

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tmp = (data
           .groupby('Location')
           .apply(lambda x:
                  x.drop(['Location', 'Date'], axis=1)
                  .isna()
                  .mean()))

plt.figure(figsize=(9, 13))

ax = sns.heatmap(tmp,
                 cmap='Blues',
                 linewidth=0.5,
                 square=True,
                 cbar_kws=dict(
                     location="bottom",
                     pad=0.01,
                     shrink=0.25))

ax.xaxis.tick_top()
ax.tick_params(axis='x', labelrotation=90)

plt.show()

# %%
# Фільтрація даних: Видаляються стовпці, у яких більше 35% пропусків.
# Додатково видаляються рядки, де немає даних у стовпці 'RainTomorrow'

data = data[data.columns[data.isna().mean().lt(0.35)]]

data = data.dropna(subset='RainTomorrow')

# %%
# Розділення числових та категорійних даних: 
# Створюються окремі таблиці для числових та категорійних стовпців.

data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%
# Візуалізація числових даних: 
# Створюються гістограми для кожної числової змінної, 
# щоб оцінити їхні розподіли

melted = data_num.melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=4,
                  sharex=False,
                  sharey=False,
                  aspect=1.25)

g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

plt.show()

# %%
# Кількість унікальних значень у категорійних стовпцях: 
# Виводиться кількість унікальних значень для кожного категорійного стовпця.
data_cat.nunique()

# %%
# Перегляд унікальних значень: 
# Переглядаються перші 5 унікальних значень у кожному категорійному стовпці

data_cat.apply(lambda x: x.unique()[:5])

# %%
# Перетворення дати: 
# Перетворення стовпця з датою на тип datetime, 
# потім виділяються рік та місяць, 
# які додаються до датасету як окремі стовпці, 
# а оригінальний стовпець 'Date' видаляється

data_cat['Date'] = pd.to_datetime(data['Date'])

data_cat[['Year', 'Month']] = (data_cat['Date']
                               .apply(lambda x:
                                      pd.Series([x.year, x.month])))

data_cat.drop('Date', axis=1, inplace=True)

data_cat[['Year', 'Month']] = data_cat[['Year', 'Month']].astype(str)

data_cat[['Year', 'Month']].head()

# %%
# Поділ даних на тренувальні та тестові набори: Датасет ділиться на тренувальні
# та тестові дані для числових і категорійних змінних. 
# Цільовий стовпець 'RainTomorrow' теж поділяється на тренувальні та 
# тестові набори

X_train_num, X_test_num, X_train_cat,  X_test_cat, y_train, y_test = (
    train_test_split(
        data_num,
        data_cat.drop('RainTomorrow', axis=1),
        data['RainTomorrow'],
        test_size=0.2,
        random_state=42))

# %%
# Заповнення пропусків у числових даних: Пропуски в числових стовпцях 
# заповнюються середніми значеннями, використовуючи SimpleImputer. 
# Потім перевіряється, чи залишилися пропуски.

num_imputer = SimpleImputer().set_output(transform='pandas')
X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)

pd.concat([X_train_num, X_test_num]).isna().sum()

# %%
# Заповнення пропусків у категорійних даних: 
# Пропуски в категорійних стовпцях заповнюються найбільш частими значеннями.

cat_imputer = SimpleImputer(
    strategy='most_frequent').set_output(transform='pandas')
X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

pd.concat([X_train_cat, X_test_cat]).isna().sum()

# %%
# Масштабування числових даних:

# Створюється об'єкт StandardScaler, який масштабуватиме числові ознаки, 
# приводячи їх до стандартного нормального розподілу (середнє значення = 0, 
# стандартне відхилення = 1).
# Метод fit_transform застосовується до тренувальних даних, щоб навчити 
# скейлер і одразу застосувати трансформацію.
# Метод transform застосовується до тестових даних, щоб використовувати ті 
# самі параметри для тестової вибірки.

scaler = StandardScaler().set_output(transform='pandas')

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# %%
# Опція set_output(transform='pandas') забезпечує, що результат повертається 
# у вигляді DataFrame для зручності роботи.
# Кодування категоріальних ознак:

# Створюється об'єкт OneHotEncoder, який перетворює категоріальні змінні 
# в числові значення за допомогою "one-hot" кодування.
# Параметр drop='if_binary' означає, що для бінарних змінних один зі стовпців
# не буде створюватися, щоб уникнути надлишкової інформації.
# fit_transform застосовується до тренувальних даних для навчання кодера, 
# а transform до тестових, щоб зберегти консистентність.
# Параметр sparse_output=False змушує зберігати результат у вигляді щільної 
# матриці.
# Виводиться форма закодованих тренувальних категоріальних даних, 
# щоб перевірити, скільки нових стовпців створено.

encoder = (OneHotEncoder(drop='if_binary',
                         sparse_output=False)
           .set_output(transform='pandas'))

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

X_train_cat.shape

# %%

# Об'єднання числових та категоріальних ознак:

# Числові та категоріальні ознаки об'єднуються у єдиний DataFrame для 
# тренувальної та тестової вибірок за допомогою pd.concat. 
# Об'єднання відбувається по стовпцях (axis=1).
# Виводиться форма об'єднаних даних, щоб переконатися, що всі стовпці 
# додалися правильно.

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

X_train.shape

# %%
# Аналіз розподілу цільової змінної:

# Метод value_counts(normalize=True) показує частки кожного унікального 
# значення у цільовій змінній y_train. Це корисно для аналізу балансування 
# класів у тренувальних даних.

y_train.value_counts(normalize=True)

# %%
# Навчання моделі логістичної регресії:

# Створюється модель логістичної регресії з використанням 
# solver='liblinear' — це алгоритм для малих наборів даних.
# class_weight='balanced' коригує ваги класів для врахування дисбалансу у 
# цільовій змінній.
# random_state=42 забезпечує відтворюваність результатів.
# Модель навчається на тренувальних даних за допомогою fit, а передбачення 
# для тестової вибірки робляться за допомогою predict.

clf = (LogisticRegression(solver='liblinear',
                          class_weight='balanced',
                          random_state=42)
       .fit(X_train, y_train))

pred = clf.predict(X_test)

# %%
# Візуалізація матриці неточностей:

# Використовується ConfusionMatrixDisplay.from_predictions для створення та 
# візуалізації матриці неточностей (confusion matrix) для порівняння фактичних 
# значень y_test з передбаченими pred.
# plt.show() виводить матрицю на екран


ConfusionMatrixDisplay.from_predictions(y_test, pred)

plt.show()

# %%
# Класифікаційний звіт:

# Функція classification_report виводить основні метрики якості моделі 
# для кожного класу: точність (precision), повнота (recall), F1-міра, 
# а також середні значення для всіх класів.

print(classification_report(y_test, pred))

