# pandas для роботи з даними
# seaborn та matplotlib для візуалізації
# sklearn для моделей машинного навчання та імпутації пропусків
# imblearn для обробки незбалансованих даних
# prosphera.projector для візуалізації високорозмірних даних

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from prosphera.projector import Projector

# %%
# Завантаження набору даних із CSV-файлу та перегляд перших кількох рядків. 
# Це дає уявлення про структуру даних та наявні ознаки

data = pd.read_csv('../datasets/mod_03_topic_06_diabets_data.csv')
data.head()

# %%
# Виводиться інформація про типи даних у стовпцях та кількість пропущених 
# значень для кожної ознаки.
data.info()

# %%
# Ознаки (X) відокремлюються від цільової змінної Outcome (y). 
# В деяких стовпцях (наприклад, Glucose, BloodPressure, тощо) значення 0 
# замінюються на NaN, оскільки значення нуля в цих ознаках є нереальними і
# свідчать про відсутні дані.

X, y = (data.drop('Outcome', axis=1), data['Outcome'])

cols = ['Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI']

X[cols] = X[cols].replace(0, np.nan)

# %%
# Побудова діаграми розсіювання для ознак Glucose та BMI, де різні кольори 
# (hue) відповідають класам цільової змінної y. Чорні вертикальні лінії 
# показують значення Glucose 120 і 160, можливо, як референсні точки

ax = sns.scatterplot(x=X['Glucose'], y=X['BMI'], hue=y)
ax.vlines(x=[120, 160],
          ymin=0,
          ymax=X['BMI'].max(),
          color='black',
          linewidth=0.75)

plt.show()

# %%
# Дані поділяються на навчальну (70%) та тестову (30%) вибірки за допомогою 
# функції train_test_split. Використовується параметр random_state для 
# забезпечення відтворюваності розбиття

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42)

# %%
# Імпутація пропущених значень у стовпцях, де раніше було замінено 0 на NaN. 
# SimpleImputer використовується для заміни пропусків середнім значенням у 
# навчальній та тестовій вибірках

imputer = SimpleImputer()

X_train[cols] = imputer.fit_transform(X_train[cols])
X_test[cols] = imputer.fit_transform(X_test[cols])

# %%
# Побудова початкової моделі дерева рішень з використанням навчальних даних. 
# Модель навчається і потім прогнозує результати на тестовій вибірці. 
# Обчислюється збалансована точність (balanced_accuracy_score), 
# яка коригує точність у випадках незбалансованих даних.


clf = (tree.DecisionTreeClassifier(
    random_state=42)
    .fit(X_train, y_train))

y_pred = clf.predict(X_test)

acc = balanced_accuracy_score(y_test, y_pred)

print(f'Acc.: {acc:.1%}')

# %%
# Візуалізація побудованого дерева рішень. Дерево зображається з підписами 
# для ознак та класів, а також з параметрами форматування для кращого
# відображення. Діаграма також зберігається у файл для подальшого аналізу.

plt.figure(figsize=(80, 15), dpi=196)

tree.plot_tree(clf,
               feature_names=X.columns,
               filled=True,
               fontsize=6,
               class_names=list(map(str, y_train.unique())),
               rounded=True)

plt.savefig('../derived/mod_03_topic_06_decision_tree.png')
plt.show()

# %%
# Перевіряється розподіл класів у навчальній вибірці, що може бути 
# важливим для аналізу проблеми незбалансованості.

y_train.value_counts(normalize=True)

# %%
# Застосовується метод балансування SMOTE (Synthetic Minority Over-sampling 
# Technique) для збільшення кількості зразків меншості у навчальній вибірці. 
# Перевіряється новий розподіл класів після балансування.

sm = SMOTE(random_state=42, k_neighbors=15)
X_res, y_res = sm.fit_resample(X_train, y_train)

y_res.value_counts(normalize=True)

# %%
# Навчання нової моделі дерева рішень після балансування вибірки. 
# Задається обмеження на глибину дерева до 5 рівнів для запобігання 
# перенавчанню. Обчислюється нова збалансована точність для оцінки поліпшень.

clf_upd = (tree.DecisionTreeClassifier(
    max_depth=5,
    random_state=42)
    .fit(X_res, y_res))

y_pred_upd = clf_upd.predict(X_test)

acc = balanced_accuracy_score(y_test, y_pred_upd)

print(f'Acc.: {acc:.1%}')

# %%
# Візуалізація оновленого дерева рішень з новими обмеженнями.
# Дерево менш глибоке завдяки обмеженню max_depth=5

plt.figure(figsize=(30, 8))

tree.plot_tree(clf_upd,
               feature_names=X.columns,
               filled=True,
               fontsize=8,
               class_names=list(map(str, y_res.unique())),
               rounded=True)

plt.show()

# %%
# Створюється горизонтальна гістограма, яка показує важливість кожної 
# ознаки у прийнятті рішень деревом. Це допомагає зрозуміти, які ознаки 
# найбільше впливають на модель.

(pd.Series(
    data=clf_upd.feature_importances_,
    index=X.columns)
    .sort_values(ascending=True)
    .plot
    .barh())

plt.show()

# %%

# Цей код використовує бібліотеку Projector для візуалізації даних, 
# зменшуючи розмірність та показуючи класи у навчальній вибірці. 
# Це допомагає побачити, як дані розподіляються у просторовому вигляді 
# після їхньої обробки.

visualizer = Projector()
visualizer.project(data=X_train, labels=y_train)
