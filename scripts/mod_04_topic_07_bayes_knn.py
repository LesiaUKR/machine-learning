# warnings - для обробки та ігнорування попереджень під час виконання коду
# pandas - Бібліотека для роботи з даними у форматі таблиць (DataFrame), 
# зчитування/запису даних, їх обробки та аналізу
# numpy - для виконання математичних операцій, роботи з масивами та числовими даними
# scipy.stats import zscore - функція для обчислення стандартного відхилення
# (z-score) для виявлення аномальних значень (outliers)
# imblearn.over_sampling import SMOTE - техніка для балансування класів у 
# дисбалансованих даних за допомогою створення синтетичних прикладів
# category_encoders - бібліотека для кодування категоріальних змінних, наприклад,
# WOE-кодування для врахування цільової змінної
# sklearn.model_selection import train_test_split - Інструмент для розділення
# даних на тренувальний та тестовий набори
# sklearn.naive_bayes import GaussianNB - Класифікатор наївного Байєса для 
# бінарної або багатокласової класифікації
# sklearn.neighbors import KNeighborsClassifier - Класифікатор K-найближчих 
# сусідів для визначення класу за найближчими сусідами
# sklearn.preprocessing import PowerTransformer - Трансформатор для стабілізації 
# варіації та приведення даних до нормального розподілу
# from sklearn.metrics import balanced_accuracy_score, confusion_matrix - Метрики 
# для оцінки моделей: збалансована точність та матриця невірної класифікації
# seaborn - Бібліотека для візуалізації даних з акцентом на статистичні графіки
# matplotlib.pyplot - Бібліотека для побудови графіків та візуалізації даних
# %%
import warnings
import pandas as pd
import numpy as np
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Завантажуємо набір даних з CSV-файлу та відображаємо перші кілька рядків для
# перевірки структури даних.
data = pd.read_csv('./datasets/mod_04_topic_07_bank_data.csv', sep=';')
data.head()

print('Загальна кількість рядків і стовпців у датасеті:', data.shape)

# %%
# Видаляємо стовпець 'duration', який не є корисним для моделі, оскільки цей 
# стовпець може нести зайву або корельовану інформацію
data.drop('duration', axis=1, inplace=True)

# %%
# Описуємо статистичні показники числових стовпців набору даних (середнє, 
# мінімальне, максимальне значення тощо)
data.describe()

# %%
# Обчислюємо коефіцієнт асиметрії (skewness) для кожної числової ознаки, щоб 
# виявити нерівномірність розподілу даних
data.skew(numeric_only=True)

# %%
# Фільтруємо значення ознаки 'campaign', залишаючи лише ті рядки, де значення
# знаходиться в межах 2 стандартних відхилень (для видалення аномалій)
data = data[zscore(data['campaign']).abs().lt(2)]

# %%
# Створюємо кореляційну матрицю для числових ознак (окрім 'y') та візуалізуємо 
# її у вигляді теплової карти для оцінки взаємозалежностей між ознаками

mtx = data.drop('y', axis=1).corr(numeric_only=True).abs()

fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(mtx,
            cmap='crest',
            annot=True,
            fmt=".2f",
            linewidth=.5,
            mask=np.triu(np.ones_like(mtx, dtype=bool)),
            square=True,
            cbar=False,
            ax=ax)

plt.show()

# %%
# Видаляємо ознаки з високою кореляцією з іншими ознаками, 
# щоб уникнути мультиколінеарності у моделі

data.drop(
    ['emp.var.rate',
     'cons.price.idx',
     'nr.employed'],
    axis=1,
    inplace=True)

# %%
# Перевіряємо кількість унікальних значень для кожної категоріальної ознаки, 
# щоб розуміти структуру даних для категоріального кодування
data.select_dtypes(include='object').nunique()

# %%
# Замінюємо значення цільової змінної 'y' з 'no' та 'yes' на 0 та 1 відповідно 
# для побудови бінарної класифікації. Ігноруємо попередження під час виконання
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    data['y'] = data['y'].replace({'no': 0, 'yes': 1})

# %%
# Розділяємо набір даних на тренувальну та тестову вибірки. 80% даних 
# використовується для тренування, 20% — для тестування. 
# Встановлюємо фіксоване значення random_state для відтворюваності
X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop('y', axis=1),
        data['y'],
        test_size=0.2,
        random_state=42))

# %%
# Визначаємо категоріальні стовпці у тренувальному наборі для подальшого кодування
cat_cols = X_train.select_dtypes(include='object').columns
cat_cols

# %%
# Використовуємо Weight of Evidence (WOE) кодування для категоріальних змінних.
# Це кодування враховує цільову змінну і краще підходить для бінарної класифікації

encoder = ce.WOEEncoder(cols=cat_cols)

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# %%
# Застосовуємо PowerTransformer для приведення даних до нормального розподілу
# (стабілізація варіації та виправлення асиметрії)

power_transform = PowerTransformer().set_output(transform='pandas')

X_train = power_transform.fit_transform(X_train)
X_test = power_transform.transform(X_test)

# %%
# Перевіряємо асиметрію ознак після трансформації, щоб переконатися, 
# що дані стали більш нормалізованими
X_train.skew()

# %%
# Підраховуємо частоту класів цільової змінної у тренувальному наборі,
# щоб оцінити дисбаланс даних
y_train.value_counts(normalize=True)

# %%
# Застосовуємо SMOTE (Synthetic Minority Over-sampling Technique) для 
# збалансування класів шляхом генерації синтетичних прикладів для меншості
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
sm = SMOTE(random_state=42, k_neighbors=50, n_jobs=1)
X_res, y_res = sm.fit_resample(X_train, y_train)

# %%
# Створюємо модель K-найближчих сусідів (KNN) з 7 сусідами. 
# Навчаємо модель на збалансованих даних та прогнозуємо результати для 
# тестового набору. Обчислюємо збалансовану точність моделі

knn_mod = KNeighborsClassifier(n_neighbors=7, n_jobs=-1).fit(X_res, y_res)

knn_preds = knn_mod.predict(X_test)

knn_score = balanced_accuracy_score(y_test, knn_preds)

print(f'KNN model accuracy: {knn_score:.1%}')

# %%
# Навчаємо модель наївного байєсівського класифікатора (GaussianNB) на 
# збалансованих даних. Прогнозуємо результати та обчислюємо точність 
# для тестового набору

gnb_mod = GaussianNB().fit(X_res, y_res)

gnb_preds = gnb_mod.predict(X_test)

gnb_score = balanced_accuracy_score(y_test, gnb_preds)

print(f'GNB model accuracy: {gnb_score:.1%}')

# %%
# Обчислюємо матрицю невірної класифікації для моделі GaussianNB, що 
# дозволяє аналізувати, скільки елементів було правильно та 
# неправильно класифіковано в кожній категорії
confusion_matrix(y_test, gnb_preds)
