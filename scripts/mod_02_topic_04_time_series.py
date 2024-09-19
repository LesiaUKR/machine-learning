import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error
import statsmodels.api as sm
from scipy.stats import zscore
from prophet import Prophet

# %%
# Читаються дані з файлу CSV і виводяться перші 5 рядків для попереднього 
# перегляду. df.head() допомагає швидко переглянути структуру даних.
df = pd.read_csv('../datasets/mod_02_topic_04_ts_data.csv')
df.head()

# %%
# Стовпець ds перетворюється на тип даних datetime, після чого цей стовпець 
# встановлюється як індекс. Метод squeeze() перетворює DataFrame на серію, 
# якщо можливо.
df['ds'] = pd.to_datetime(df['ds'])
df = df.set_index('ds').squeeze()

# %%
# Метод describe() надає описову статистику для даних: кількість елементів, 
# середнє, стандартне відхилення, мінімальне та максимальне значення тощо.
df.describe()

# %%
# Логарифмується значення ряду, що може допомогти стабілізувати дисперсію 
# часового ряду та зробити його більш лінійним. Переглядаються перші рядки.
df = np.log(df)
df.head()

# %%
# Дані розділяються на історичні (всі рядки, крім останніх 365) та тестові 
# (останні 365 рядків). Це використовується для тренування моделі та 
# перевірки її на нових даних.
df_hist = df.iloc[:-365]
df_test = df.iloc[-365:]

# %%
# Перевіряється наявність пропущених значень в історичних даних
df_hist.isna().sum()

# %%
# Створюється графік з вертикальними лініями для кожної точки історичних даних,
# що допомагає візуалізувати ряд

sns.set_theme()

fig, ax = plt.subplots(figsize=(30, 7))

ax.vlines(
    x=df_hist.index,
    ymin=0,
    ymax=df_hist,
    linewidth=0.5,
    color='grey')

plt.show()

# %%
# Дані перетворюються на щоденну частоту, а відсутні значення інтерполюються,
# що дозволяє уникнути розривів у часовому ряді

df_hist = df_hist.asfreq('D').interpolate()
df_hist.isna().sum()

# %%
# Побудова лінійного тренду для історичних даних за допомогою лінійної регресії
# На графіку відображаються фактичні дані та лінійний тренд.

model = LinearRegression().fit(np.arange(len(df_hist)).reshape(-1, 1), df_hist)
trend = model.predict(np.arange(len(df_hist)).reshape(-1, 1))

ax = plt.subplots(figsize=(10, 3))
sns.scatterplot(df_hist)
sns.lineplot(y=trend, x=df_hist.index, c='black')

plt.show()

# %%
# Тренд видаляється, щоб проаналізувати сезонні коливання. 
# Побудова box-графіка для аналізу сезонності

df_mod = df_hist - trend + trend.mean()

sns.catplot(
    y=df_hist,
    x=df_hist.index.month,
    kind='box',
    showfliers=False)

plt.show()

# %%
# Використовується метод seasonal_decompose для декомпозиції ряду на тренд,
# сезонність і залишки

decomp = sm.tsa.seasonal_decompose(df_hist)
decomp_plot = decomp.plot()

# %%
# Розрахунок Z-оціночного відхилення для залишків декомпозиції,
# що допомагає виявляти аномалії

df_zscore = zscore(decomp.resid, nan_policy='omit')


# %%
# Створюється функція для обчислення Z-оціночного відхилення з ковзним вікном
# для більш точного виявлення аномалій

def zscore_adv(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z


df_zscore_adv = zscore_adv(decomp.resid, window=7)

# %%
# Побудова графіків для порівняння стандартного та ковзного Z-оціночного 
# відхилення. Області між значеннями -3 та 3 підсвічуються

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(10, 7))

for i, d in enumerate([df_zscore, df_zscore_adv]):
    ax = axes[i]
    sns.lineplot(d, ax=ax)
    ax.fill_between(d.index.values, -3, 3, alpha=0.15)

plt.show()

# %%
# Створюється датафрейм з даними про свята та особливі події 
# (playoffs та Super Bowl) для подальшого використання у моделі Prophet

playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2013-01-12',
                        '2014-01-12',
                          '2014-01-19',
                          '2014-02-02',
                          '2015-01-11',
                          '2016-01-17']),
    'lower_window': 0,
    'upper_window': 1})

superbowls = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(['2014-02-02']),
    'lower_window': 0,
    'upper_window': 1})

holidays = pd.concat((playoffs, superbowls)).reset_index(drop=True)

holidays

# %%
# Виявляються аномальні значення (ті, що виходять за межі Z-оціночного
# відхилення -3 та 3) та порівнюються з датами свят, щоб виключити їх як 
# потенційні аномалії

outliers = np.where(~df_zscore_adv.between(-3, 3) * df_zscore_adv.notna())[0]

outliers = list(set(df_hist.index[outliers]).difference(holidays['ds']))

fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(df_hist, ax=ax)
sns.scatterplot(
    x=outliers,
    y=df_hist[outliers],
    color='red',
    ax=ax)

plt.show()

# %%
# Аномальні значення замінюються на NaN, а потім інтерполюються 
# для відновлення даних без аномалій

df_hist.loc[outliers] = np.nan
df_hist = df_hist.interpolate()

# %%
# Скидається індекс для підготовки даних до використання в моделі Prophet

df_hist = df_hist.reset_index()

# %%
# Модель Prophet тренується на історичних даних з урахуванням сезонності 
# та свят

mp = Prophet(holidays=holidays)
mp.add_seasonality(name='yearly', period=365, fourier_order=2)
mp.fit(df_hist)

# %%
# Генерується прогноз на майбутні 365 днів за допомогою моделі Prophet

future = mp.make_future_dataframe(freq='D', periods=365)
forecast = mp.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# %%
# Будуються графіки компонентів прогнозу та самого прогнозу, 
# ігноруючи попередження

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    mp.plot_components(forecast)
    mp.plot(forecast)

# %%
# Порівнюються фактичні дані з тестового набору та прогнозовані значення з 
# моделі Prophet, а також підсвічуються особливі події (свята)


pred = forecast.iloc[-365:][['ds', 'yhat']]

fig, ax = plt.subplots(figsize=(20, 5))

ax.vlines(
    x=df_test.index,
    ymin=5,
    ymax=df_test,
    linewidth=0.75,
    label='fact',
    zorder=1)

ax.vlines(
    x=df_test[df_test.index.isin(holidays['ds'])].index,
    ymin=5,
    ymax=df_test[df_test.index.isin(holidays['ds'])],
    linewidth=0.75,
    color='red',
    label='special events',
    zorder=2)

sns.lineplot(data=pred, y='yhat', x='ds', c='black', label='prophet', ax=ax)

ax.margins(x=0.01)

plt.show()

# %%
# Обчислюється приблизна точність моделі за допомогою метрики 
# median_absolute_error, і результат виводиться у вигляді відсотка точності

approx_mape = median_absolute_error(df_test, pred['yhat'])

print(f'Accuracy: {1 - approx_mape:.1%}')
