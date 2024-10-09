import warnings 
import pickle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Імпортуємо всі необхідні бібліотеки для аналізу даних, кластеризації, стандартизації та візуалізації.
# Використовуємо `warnings` для придушення непотрібних попереджень під час роботи з моделями.

# %%
with open('./datasets/mod_06_topic_12_nci_data.pkl', 'rb') as fl:
    data_dict = pickle.load(fl)

# Завантажуємо дані з файлу `mod_06_topic_12_nci_data.pkl` за допомогою `pickle`.
# Зчитуємо дані в словник `data_dict`.

data = data_dict['data']
target = data_dict['labels']

# Розпаковуємо словник на `data` (вхідні дані) та `target` (мітки для класифікації).

data.shape
# Виводимо розмірність вхідних даних, щоб перевірити кількість ознак та зразків.

# %%
target['label'].value_counts().sort_index()
# Виводимо кількість зразків у кожному класі цільової змінної `target`, щоб побачити розподіл класів.

# %%
X = StandardScaler().fit_transform(data)

# Виконуємо стандартизацію даних, щоб кожна ознака мала середнє значення 0 та стандартне відхилення 1.

pca = PCA(random_state=42).fit(X)
pve = pca.explained_variance_ratio_

# Виконуємо аналіз головних компонент (PCA) для скорочення розмірності даних.
# `pve` - пояснювальна варіація кожної головної компоненти.

# %%
sns.set_theme()
# Застосовуємо стандартну тему візуалізації для `seaborn`.

kneedle = KneeLocator(
    x=range(1, len(pve) + 1),
    y=pve,
    curve='convex',
    direction='decreasing')

# Використовуємо `KneeLocator`, щоб знайти "лікоть" кривої, тобто оптимальну кількість компонент.
# Визначаємо, скільки компонент потрібно, щоб пояснити найбільшу частину варіації даних.

kneedle.plot_knee()
plt.show()
# Візуалізуємо криву залежності варіації від кількості компонент та показуємо точку "лікоть".

# %%
n_components = kneedle.elbow
# Зберігаємо оптимальну кількість компонент, яку знайшли за допомогою методу "лікоть".

ax = sns.lineplot(np.cumsum(pve))
ax.axvline(x=n_components, c='black', linestyle='--', linewidth=0.75)
ax.axhline(y=np.cumsum(pve)[n_components], c='black', linestyle='--', linewidth=0.75)
ax.set(xlabel='number of components', ylabel='cumulative explained variance')
plt.show()
# Візуалізуємо накопичену варіацію та додаємо лінії, що позначають обрану кількість компонент.

# %%
X = pca.transform(X)[:, :n_components]
# Застосовуємо PCA для скорочення даних до обраної кількості компонент.

# %%
model_kmn = KMeans(random_state=42)

# Створюємо модель KMeans для кластеризації даних.

visualizer = KElbowVisualizer(
    model_kmn,
    k=(2, 10),
    timings=False)

# Використовуємо `KElbowVisualizer` для візуалізації оптимальної кількості кластерів у моделі KMeans.

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    visualizer.fit(X)

visualizer.show()
# Відключаємо попередження під час побудови моделі та візуалізуємо кількість кластерів за допомогою графіка "лікоть".

# %%
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for x, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[x] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

# Створюємо функцію `plot_dendrogram`, щоб побудувати дендрограму для агломеративної кластеризації.
# Використовуємо параметри `model` та `kwargs` для налаштування дендрограми.

# %%
model_agg = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model_agg = model_agg.fit(X)

# Створюємо модель агломеративної кластеризації з `distance_threshold=0`, щоб побудувати повне дерево кластерів.
# `n_clusters=None` означає, що ми не обмежуємо кількість кластерів заздалегідь.

# %%
plot_dendrogram(model_agg, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Number of points in node (or index of point if no parenthesis)')
plt.show()
# Будуємо дендрограму для ієрархічної кластеризації та візуалізуємо її.

# %%
k_best = visualizer.elbow_value_

# Знаходимо оптимальну кількість кластерів за результатами KElbowVisualizer.

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model_kmn = KMeans(n_clusters=k_best, random_state=42).fit(X)
    model_agg = AgglomerativeClustering(n_clusters=k_best).fit(X)

# Навчаємо моделі KMeans та агломеративної кластеризації з оптимальною кількістю кластерів `k_best`.
# Використовуємо метод `fit` для навчання моделей.

labels_kmn = pd.Series(model_kmn.labels_, name='k-means')
labels_agg = pd.Series(model_agg.labels_, name='h-clust')

# Зберігаємо результати кластеризації для кожного методу у вигляді серій `labels_kmn` та `labels_agg`.

# %%
pd.crosstab(labels_agg, labels_kmn)
# Перевіряємо, наскільки класифікація обома методами співпадає.

# %%
pd.crosstab(target['label'], labels_kmn)
# Порівнюємо кластеризацію KMeans з фактичними мітками `target['label']`.

# %%
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharey=True)

# Створюємо підплоти для двох графіків.

for i, s in enumerate([target['label'], labels_kmn]):
    ax = axes[i]
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=s, style=s, edgecolor='black',
                    linewidth=0.5, s=60, palette='tab20', legend=False, ax=ax)
    ax.set(title=s.name)

# Створюємо два графіки розсіювання для порівняння фактичних міток та результатів кластеризації.
# Використовуємо колір та стиль маркерів, щоб позначити різні кластери.

plt.show()
# Виводимо порівняльні графіки для візуалізації результатів.

