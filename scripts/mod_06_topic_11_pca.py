import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# %%

# Завантажуємо набір даних про рак молочної залози з sklearn.
data, target = load_breast_cancer(return_X_y=True, as_frame=True)
# Виводимо перші рядки даних для попереднього огляду.
data.head()

# %%

# Переглядаємо загальну інформацію про набір даних.
data.info()

# %%

# Підраховуємо кількість прикладів у кожному класі цільової змінної.
target.value_counts()

# %%

# Визначаємо викиди (аномальні значення) за допомогою Z-Score для кожної ознаки.
out = (data
       .apply(lambda x: np.abs(zscore(x)).ge(3))  # Застосовуємо Z-Score для кожного стовпця та перевіряємо на наявність аномалій.
       .astype(int)
       .mean(1))  # Розраховуємо середню кількість аномалій для кожного рядка.

# Визначаємо індекси рядків, де кількість аномалій перевищує 20%.
out_ind = np.where(out > 0.2)[0]

# Видаляємо ці рядки з набору даних та цільової змінної.
data.drop(out_ind, inplace=True)
target.drop(out_ind, inplace=True)

# %%

# Перевіряємо розмір даних після видалення викидів.
data.shape

# %%

# Розбиваємо дані на тренувальний (80%) та тестовий (20%) набори.
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


# %%

# Виконуємо стандартизацію ознак, щоб привести їх до одного масштабу (середнє = 0, стандартне відхилення = 1).
scaler = StandardScaler().set_output(transform='pandas')
X_train = scaler.fit_transform(X_train)  # Навчаємо масштабувальник на тренувальних даних.
X_test = scaler.transform(X_test)  # Масштабуємо тестові дані.

# %%

# Виконуємо аналіз головних компонент (PCA) для зменшення кількості вимірів.
pca = PCA().set_output(transform='pandas').fit(X_train)

# %%


# Візуалізуємо накопичену пояснювальну варіацію для кожної компоненти.
sns.set_theme()
explained_variance = np.cumsum(pca.explained_variance_ratio_)
ax = sns.lineplot(explained_variance)
ax.set(xlabel='number of components', ylabel='cumulative explained variance')

# Знаходимо мінімальну кількість компонент, які пояснюють 85% загальної варіації.
n_components = np.searchsorted(explained_variance, 0.85)

# Додаємо візуальні індикатори для обраної кількості компонент та відповідної варіації.
ax.axvline(x=n_components, c='black', linestyle='--', linewidth=0.75)
ax.axhline(y=explained_variance[n_components], c='black', linestyle='--', linewidth=0.75)
plt.show()

# %%

# Трансформуємо дані за допомогою PCA.
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# %%

# Перевіряємо перші рядки зменшених тренувальних даних.
X_train_pca.iloc[:, :n_components].head()

# %%

# Створюємо 3D-візуалізацію перших трьох головних компонент.
plt.figure(figsize=(8, 8))
ax = plt.subplot(projection='3d')
ax.scatter3D(X_train_pca.iloc[:, 0], X_train_pca.iloc[:, 1], X_train_pca.iloc[:, 2], c=y_train, s=20, cmap='autumn', ec='black', lw=0.75)
ax.view_init(elev=30, azim=30)
plt.show()

# %%

# Навчаємо модель Gradient Boosting на повних даних (без PCA).
clf_full = GradientBoostingClassifier()
clf_full.fit(X_train, y_train)
pred_full = clf_full.predict(X_test)
score_full = accuracy_score(y_test, pred_full)
print(f'Model accuracy: {score_full:.1%}')  # Виводимо точність моделі на повних даних.

# %%

# Навчаємо модель Gradient Boosting на даних, перетворених за допомогою PCA.
clf_pca = GradientBoostingClassifier()
clf_pca.fit(X_train_pca.iloc[:, :n_components], y_train)
pred_pca = clf_pca.predict(X_test_pca.iloc[:, :n_components])
score_pca = accuracy_score(y_test, pred_pca)
print(f'Model accuracy (PCA): {score_pca:.1%}')  # Виводимо точність моделі на даних PCA.

# %%

# Візуалізуємо важливість кожної ознаки у повній моделі за допомогою горизонтальних стовпчиків.
plt.figure(figsize=(3, 8))
(pd.Series(data=clf_full.feature_importances_, index=X_train.columns).sort_values(ascending=True).plot.barh())
plt.show()
