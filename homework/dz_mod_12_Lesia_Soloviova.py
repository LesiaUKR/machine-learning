import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import pickle

# %% Завантаження набору даних
with open('./datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)
    
# Перевірка доступних даних
print("Available datasets:", datasets.keys())

# %% Вибір набору даних Concrete
data = datasets['concrete']
print(data.head())

# %%

# Створення нової ознаки 'Components' — кількість задіяних складових у рецептурі бетону
data['Components'] = (data.iloc[:, :-1] > 0).sum(axis=1)

# Виведення нової ознаки 'Components' для перших 10 рядків
print("Новий стовпчик 'Components' для перших 10 рядків:")
print(data[['Components']].head(10))

# %%

# Нормалізація даних
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#%%
# Визначення оптимальної кількості кластерів
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(1, 10))
visualizer.fit(data_scaled)
optimal_clusters = visualizer.elbow_value_
print(f"Optimal number of the clusters: {optimal_clusters}")
visualizer.show()

#%%
# Проведення кластеризації методом k-середніх
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

#%%
# Розрахунок описової статистики для кожного кластеру
cluster_stats = data.groupby('Cluster').median()
cluster_stats['Count'] = data['Cluster'].value_counts()

# Додавання кількості компонент у кожному кластері
components_stats = data.groupby('Cluster')['Components'].sum()
cluster_stats['Total Components'] = components_stats

#%%
# Перетворення статистики у формат таблиці
cluster_summary = cluster_stats.reset_index()

# Виведення таблиці за допомогою matplotlib
fig, ax = plt.subplots(figsize=(12, 6))  # Розмір графіка
ax.axis('tight')
ax.axis('off')
# Створення таблиці на основі DataFrame
table = ax.table(cellText=cluster_summary.values,
                 colLabels=cluster_summary.columns,
                 cellLoc='center',
                 loc='center')

# Оформлення таблиці
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Відображення таблиці
plt.title("Cluster Summary Statistics", fontsize=16, pad=20)
plt.show()
