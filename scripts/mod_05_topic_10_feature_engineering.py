import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

# %% Завантаження даних з файлу
# Відкриття файлу з набором даних, використовуючи модуль pickle, та завантаження всіх наборів даних у змінну `datasets`.
with open('./datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

# %% Вибір набору даних 'autos' та створення нової ознаки
# Завантаження набору даних `autos` та створення нової ознаки `stroke_ratio`, яка є відношенням двох інших ознак — `stroke` і `bore`.
autos = datasets['autos']
autos['stroke_ratio'] = autos['stroke'] / autos['bore']

# Виведення перших 5 рядків для перевірки результату.
autos[['stroke', 'bore', 'stroke_ratio']].head()

# %% Створення логарифмічної трансформації для швидкості вітру
# Завантаження набору даних `accidents` та створення логарифмічної трансформованої ознаки `LogWindSpeed`.
accidents = datasets['accidents']
accidents['LogWindSpeed'] = accidents['WindSpeed'].apply(np.log1p)

# Побудова графіка KDE для оригінальної та трансформованої ознак для порівняння їх розподілу.
sns.set_theme()
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, fill=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, fill=True, ax=axs[1])
plt.show()

# %% Створення сумарної ознаки для дорожніх умов
# Список бінарних ознак, які описують особливості дорожніх умов. Розрахунок нової ознаки `RoadwayFeatures`, що є сумою цих ознак.
roadway_features = [
    'Amenity', 'Bump', 'Crossing', 'GiveWay', 'Junction',
    'NoExit', 'Railway', 'Roundabout', 'Station', 'Stop',
    'TrafficCalming', 'TrafficSignal']
accidents['RoadwayFeatures'] = accidents[roadway_features].sum(axis=1)

# Виведення перших 10 рядків для перевірки нової ознаки.
accidents[roadway_features + ['RoadwayFeatures']].head(10)

# %% Створення ознаки для кількості компонентів у бетоні
# Завантаження набору даних `concrete` та створення нової ознаки `Components`, яка показує кількість компонентів у складі бетону.
concrete = datasets['concrete']
components = [
    'Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water',
    'Superplasticizer', 'CoarseAggregate', 'FineAggregate']
concrete['Components'] = concrete[components].gt(0).sum(axis=1)

# Виведення перших 10 рядків для перевірки нової ознаки.
concrete[components + ['Components']].head(10)

# %% Розбиття текстової ознаки на дві нові категоріальні ознаки
# Розбиття текстової ознаки `Policy` у наборі даних `customer` на дві нові ознаки — `Type` та `Level`.
customer = datasets['customer']
customer[['Type', 'Level']] = customer['Policy'].str.split(' ', expand=True)

# Виведення перших 10 рядків для перевірки.
customer[['Policy', 'Type', 'Level']].head(10)

# %% Об’єднання марок і стилів автомобілів в одну ознаку
# Створення нової ознаки `make_and_style` у наборі даних `autos`, що є комбінацією `make` та `body_style`.
autos['make_and_style'] = autos['make'] + '_' + autos['body_style']

# Виведення перших 5 рядків для перевірки нової ознаки.
autos[['make', 'body_style', 'make_and_style']].head()

# %% Розрахунок середнього доходу для кожного штату
# Додавання до набору даних `customer` нової ознаки `AverageIncome`, яка показує середній дохід у кожному штаті.
customer['AverageIncome'] = customer.groupby('State')['Income'].transform('mean')

# Виведення перших 10 рядків для перевірки нової ознаки.
customer[['State', 'Income', 'AverageIncome']].head(10)

# %% Розрахунок частоти зустрічання кожного штату у наборі даних
# Створення нової ознаки `StateFreq`, що показує частку кожного штату у всьому наборі даних.
customer = customer.assign(StateFreq=lambda x: x.groupby('State')['State'].transform('count') / x['State'].count())

# Виведення перших 10 рядків для перевірки.
customer[['State', 'StateFreq']].head(10)

# %% Розрахунок середньої суми претензій по кожному виду покриття
# Розподіл даних на тренувальний і тестовий набори та створення нової ознаки `AverageClaim` для тренувального набору.
c_train = customer.sample(frac=0.75)
c_test = customer.drop(c_train.index)
c_train['AverageClaim'] = c_train.groupby('Coverage')['ClaimAmount'].transform('mean')

# Додавання середньої суми претензій з тренувального набору до тестового за допомогою злиття.
c_test = c_test.merge(c_train[['Coverage', 'AverageClaim']].drop_duplicates(), on='Coverage', how='left')
c_test[['Coverage', 'AverageClaim']].head(10)

# %% Побудова синусоїдальної кривої та регресійного графіка
# Побудова регресійної моделі для синусоїдальної кривої.
x = np.linspace(0, 2, 50)
y = np.sin(2 * np.pi * 0.25 * x)
sns.regplot(x=x, y=y)
plt.show()

# %% Оцінка взаємної інформації та коефіцієнта кореляції для кривої
# Розрахунок коефіцієнта кореляції та взаємної інформації між `x` та `y`.
mis = mutual_info_regression(x.reshape(-1, 1), y)[0]
cor = np.corrcoef(x, y)[0, 1]
print(f'MI score: {mis:.2f} | Cor index: {cor:.2f}')

# %% Підготовка даних для моделі та розрахунок взаємної інформації
# Кодування категоріальних ознак у наборі `autos` для побудови моделі.
X = autos.copy()
y = X.pop('price')
cat_features = X.select_dtypes('object').columns

# Факторизація категоріальних ознак.
for colname in cat_features:
    X[colname], _ = X[colname].factorize()

# %% Розрахунок взаємної інформації для набору даних
mi_scores = mutual_info_regression(
    X, y, discrete_features=X.columns.isin(cat_features.to_list() + ['num_of_doors', 'num_of_cylinders']), random_state=42)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns).sort_values()

# Випадкова вибірка 5 ознак для перевірки.
mi_scores.sample(5)

# %% Побудова горизонтальної гістограми з результатами MI
plt.figure(figsize=(6, 8))
plt.barh(np.arange(len(mi_scores)), mi_scores)
plt.yticks(np.arange(len(mi_scores)), mi_scores.index)
plt.title('Mutual Information Scores')
plt.show()

# %% Побудова регресійного графіка для ваги автомобіля та ціни
sns.regplot(data=autos, x='curb_weight', y='price', order=2)
plt.show()

# %% Побудова графіка з регресійними лініями для потужності та ціни з групуванням за типом пального
sns.lmplot(data=autos, x='horsepower', y='price', hue='fuel_type', facet_kws={'legend_out': False})
plt.show()

