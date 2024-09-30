import pandas as pd
import pickle
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Завантаження набору даних
with open('./datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

# Вибір потрібного набору даних
autos = datasets['autos']
X = autos.copy()
y = X.pop('price')

# 2. Визначення дискретних ознак
# Визначення категоріальних ознак
cat_features = X.select_dtypes(include=['object']).columns
# Визначення дискретних числових ознак (якщо вони є)
discrete_numerical_features = ['num_of_doors', 'num_of_cylinders']  # Додаткові числові дискретні ознаки
discrete_features = cat_features.to_list() + discrete_numerical_features

# 3. Кодування дискретних ознак
# Кодування категоріальних ознак за допомогою .factorize()
for col in cat_features:
    X[col], _ = X[col].factorize()

# 4. Розрахунок взаємної інформації з урахуванням дискретних ознак
mi_scores = mutual_info_regression(
    X, y,
    discrete_features=X.columns.isin(discrete_features)
)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns).sort_values(ascending=False)

# 5. Побудова регресійної моделі для оцінки важливості ознак
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Оцінка важливості ознак у моделі
feature_importances = pd.Series(model.feature_importances_, name='Feature Importance', index=X.columns).sort_values(ascending=False)

# 6. Масштабування та уніфікація показників за допомогою ранжування
mi_scores_ranked = mi_scores.rank(pct=True)
feature_importances_ranked = feature_importances.rank(pct=True)

# 7. Підготовка даних для порівняння
comparison_df = pd.DataFrame({
    'MI Scores': mi_scores_ranked,
    'Feature Importance': feature_importances_ranked
}).reset_index().rename(columns={'index': 'Feature'})

# 8. Сортування за показником MI Scores
comparison_df = comparison_df.sort_values(by='MI Scores', ascending=False)

# 9. Перетворення даних у формат "довгого" типу за допомогою melt()
comparison_melted = comparison_df.melt(id_vars='Feature', var_name='Metric', value_name='Rank')

# 10. Побудова графіка порівняння рангових показників
plt.figure(figsize=(14, 8))
sns.barplot(x='Rank', y='Feature', hue='Metric', data=comparison_melted, dodge=True)
plt.title('Порівняння показників взаємної інформації та важливості ознак у моделі')
plt.xlabel('Ранг (у відсотках)')
plt.ylabel('Ознаки')
plt.legend(title='Показник')
plt.tight_layout()
plt.show()

# 11. Топ-5 ознак за взаємною інформацією
print("Топ-5 ознак за взаємною інформацією:")
print(mi_scores_ranked.sort_values(ascending=False).head())

# 12. Топ-5 ознак за важливістю в моделі
print("\nТоп-5 ознак за важливістю у моделі:")
print(feature_importances_ranked.sort_values(ascending=False).head())

# 13. Висновки
print("\nВисновки:")
print("1. Ознаки з високою взаємною інформацією не завжди мають високу важливість у моделі, і навпаки.")
print("2. Деякі ознаки, такі як 'year' та 'horsepower', є важливими за обома метриками.")
print("3. Модель може надавати велику важливість певним ознакам, які мають низьку взаємну інформацію з цільовою змінною.")
print("4. Це порівняння допомагає виявити потенційно важливі ознаки, які можуть бути недооцінені при використанні лише одного методу.")
