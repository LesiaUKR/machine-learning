# Імпортуємо необхідні бібліотеки

import time  # Для вимірювання часу виконання коду
import pandas as pd  # Для роботи з даними у форматі таблиць (DataFrame)
from sklearn.ensemble import (  # Імпорт моделей ансамблю
    StackingClassifier,  # Для побудови стекінгового ансамблю
    VotingClassifier,  # Для побудови ансамблю за методом голосування
    GradientBoostingClassifier,  # Для реалізації градієнтного бустингу
    AdaBoostClassifier,  # Для реалізації AdaBoost
    BaggingClassifier,  # Для реалізації методів бустингу
    RandomForestClassifier)  # Для реалізації випадкового лісу
from sklearn.model_selection import train_test_split  # Для розподілу даних на навчальну та тестову вибірки
from sklearn.naive_bayes import GaussianNB  # Для реалізації наївного баєсівського класифікатора
from sklearn.neighbors import KNeighborsClassifier  # Для реалізації класифікатора на основі найближчих сусідів
from sklearn.linear_model import LogisticRegression  # Для реалізації логістичної регресії
from sklearn.preprocessing import StandardScaler  # Для масштабування ознак
from imblearn.over_sampling import SMOTE  # Для збалансування класів в навчальних даних
import category_encoders as ce  # Для кодування категоріальних ознак
from sklearn.metrics import f1_score  # Для оцінки якості моделей за допомогою метрики F1

# %% Завантаження набору даних
data = pd.read_csv('./datasets/mod_05_topic_09_employee_data.csv')  # Читаємо CSV файл з даними працівників
data.head()  # Виводимо перші кілька рядків набору даних для перевірки

# %% Виведення інформації про набір даних
data.info()  # Показуємо інформацію про набір даних, включаючи типи даних і кількість ненульових значень

# %% Регулювання року приєднання для відображення кількості років з моменту приєднання
data['JoiningYear'] = data['JoiningYear'].max() - data['JoiningYear']  # Обчислюємо кількість років з моменту приєднання

# %% Перетворення типу PaymentTier на строку
data['PaymentTier'] = data['PaymentTier'].astype(str)  # Забезпечуємо, щоб PaymentTier трактувався як категоріальні дані

# %% Розподіл набору даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop('LeaveOrNot', axis=1),  # Ознаки: всі колонки, окрім 'LeaveOrNot'
        data['LeaveOrNot'],  # Цільова змінна: 'LeaveOrNot'
        test_size=0.33,  # Використовуємо 33% даних для тестування
        random_state=42))  # Встановлюємо випадковий стан для відтворюваності

# %% Кодування категоріальних ознак за допомогою цільового кодування
encoder = ce.TargetEncoder()  # Ініціалізуємо кодувальник цільових змінних
X_train = encoder.fit_transform(X_train, y_train)  # Підганяємо і трансформуємо навчальні дані
X_test = encoder.transform(X_test)  # Трансформуємо тестові дані з використанням підганяючого кодувальника

# %% Масштабування ознак для кращої продуктивності моделі
scaler = StandardScaler().set_output(transform='pandas')  # Ініціалізуємо масштабувальник, щоб повертати DataFrame

X_train = scaler.fit_transform(X_train)  # Підганяємо і трансформуємо навчальні ознаки
X_test = scaler.transform(X_test)  # Трансформуємо тестові ознаки

# %% Перевірка розподілу класів у навчальному наборі
y_train.value_counts(normalize=True)  # Виводимо пропорцію кожного класу в цільовій змінній

# %% Обробка дисбалансу класів за допомогою SMOTE
sm = SMOTE(random_state=42)  # Ініціалізуємо SMOTE для надсучасного вибору
X_res, y_res = sm.fit_resample(X_train, y_train)  # Збалансовуємо навчальні дані

# %% Ініціалізуємо словник для зберігання F1 оцінок та часу виконання
f1_scores = {}

# %% Визначення декоратора для вимірювання часу виконання та F1 оцінки прогнозів моделі
def measure_f1_time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Починаємо таймер
        predictions = func(*args, **kwargs)  # Викликаємо функцію прогнозування
        end_time = time.time()  # Зупиняємо таймер
        f1 = f1_score(args[-1], predictions)  # Обчислюємо F1 оцінку, використовуючи справжні та передбачені значення
        model_name = args[0].__class__.__name__  # Отримуємо ім'я моделі
        execution_time = end_time - start_time  # Обчислюємо час виконання
        f1_scores[model_name] = [f1, execution_time]  # Зберігаємо F1 оцінку та час виконання
        print(f'{model_name} F1 Metric: {f1:.4f}')  # Виводимо F1 оцінку
        print(f'{model_name} Inference: {execution_time:.4f} s')  # Виводимо час виконання
        return predictions  # Повертаємо прогнози
    return wrapper

# %% Визначення функції для прогнозування з вимірюванням часу та F1 оцінки
@measure_f1_time_decorator
def predict_with_measure(model, Xt, yt):
    return model.predict(Xt)  # Повертаємо прогнози для заданих ознак

# %% Навчання та оцінка моделі логістичної регресії
mod_log_reg = (LogisticRegression(
    # n_jobs=-1  # Розкоментуйте, щоб використовувати всі ядра для навчання
).fit(X_res, y_res))  # Підганяємо модель на збалансованих даних

prd_log_reg = predict_with_measure(mod_log_reg, X_test, y_test)  # Здійснюємо прогнози та оцінюємо продуктивність

# %% Навчання та оцінка моделі випадкового лісу
mod_rnd_frs = (RandomForestClassifier(
    random_state=42,
    # n_jobs=-1  # Розкоментуйте, щоб використовувати всі ядра для навчання
).fit(X_res, y_res))  # Підганяємо модель на збалансованих даних

prd_rnd_frs = predict_with_measure(mod_rnd_frs, X_test, y_test)  # Здійснюємо прогнози та оцінюємо продуктивність

# %% Навчання та оцінка моделі KNN з бустингом
mod_bag_knn = (BaggingClassifier(
    KNeighborsClassifier(),  # Використовуємо KNeighborsClassifier як базову оцінювальну модель
    max_samples=0.75,  # Максимальна кількість зразків для кожної базової моделі
    max_features=0.75,  # Максимальна кількість ознак для кожної базової моделі
    # n_jobs=-1,  # Розкоментуйте, щоб використовувати всі ядра для навчання
    random_state=42)
    .fit(X_res, y_res))  # Підганяємо модель на збалансованих даних

prd_bag_knn = predict_with_measure(mod_bag_knn, X_test, y_test)  # Здійснюємо прогнози та оцінюємо продуктивність

# %% Навчання та оцінка моделі AdaBoost
mod_ada_bst = (AdaBoostClassifier(
    algorithm='SAMME',  # Використовуємо алгоритм SAMME для бустингу
    random_state=42)
    .fit(X_res, y_res))  # Підганяємо модель на збалансованих даних

prd_ada_bst = predict_with_measure(mod_ada_bst, X_test, y_test)  # Здійснюємо прогнози та оцінюємо продуктивність

# %% Навчання та оцінка моделі градієнтного бустингу
mod_grd_bst = (GradientBoostingClassifier(
    learning_rate=0.3,  # Встановлюємо швидкість навчання для бустингу
    subsample=0.75,  # Частка зразків для підгонки
    max_features='sqrt',  # Використовуємо квадратний корінь ознак для підгонки
    random_state=42)
    .fit(X_res, y_res))  # Підганяємо модель на збалансованих даних

prd_grd_bst = predict_with_measure(mod_grd_bst, X_test, y_test)  # Здійснюємо прогнози та оцінюємо продуктивність

# %% Визначення оцінювачів для Voting Classifier
clf1 = LogisticRegression()  # Модель логістичної регресії
clf2 = KNeighborsClassifier()  # Модель найближчих сусідів
clf3 = GaussianNB()  # Модель наївного баєса

estimators = [('lnr', clf1),  # Список оцінювачів з іменами
              ('knn', clf2),
              ('gnb', clf3)]

# %% Навчання та оцінка Voting Classifier
mod_vot_clf = VotingClassifier(
    estimators=estimators,
    voting='soft').fit(X_res, y_res)  # Підганяємо VotingClassifier на збалансованих даних

prd_vot_clf = predict_with_measure(mod_vot_clf, X_test, y_test)  # Здійснюємо прогнози та оцінюємо продуктивність

# %% Навчання та оцінка Stacking Classifier
final_estimator = GradientBoostingClassifier(
    subsample=0.75,  # Частка зразків для підгонки
    max_features='sqrt',  # Використовуємо квадратний корінь ознак для підгонки
    random_state=42)

mod_stk_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator).fit(X_res, y_res)  # Підганяємо StackingClassifier на збалансованих даних

prd_stk_clf = predict_with_measure(mod_stk_clf, X_test, y_test)  # Здійснюємо прогнози та оцінюємо продуктивність

# %% Збір та відображення F1 оцінок і часу виконання всіх моделей
scores = pd.DataFrame.from_dict(
    f1_scores,
    orient='index',
    columns=['f1', 'time'])  # Створюємо DataFrame з словника оцінок

scores.sort_values('f1', ascending=False)  # Сортуємо DataFrame за F1 оцінкою у спадному порядку
