import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# Настройки отображения
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")

def load_data(file_path):
    """Загрузка данных из CSV файла"""
    df = pd.read_csv(file_path, sep=';', decimal=',')
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df

def explore_data(df):
    """Первичный анализ данных"""
    print('Размер данных:', df.shape, '\n')
    print('Типы данных по колонкам:')
    for col in df.columns:
        unique_types = set(type(x) for x in df[col])
        print(f"{col}: {[t.__name__ for t in unique_types]}")
    print('\nСтатистика:\n', df.describe())
    print('\nПропущенные значения:\n', df.isnull().sum())
 

def check_data_quality(df):
    """Проверка качества данных"""
    issues = []
    price_columns = ['open', 'high', 'low', 'close']
    volume_columns = ['vol']
    
    # 1. Проверка отрицательных значений
    print("1. Проверка отрицательных значений:")
    for col in price_columns + volume_columns:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"   {col}: {negative_count} отрицательных значений")
                issues.append(f"Отрицательные значения в {col}")
            else:
                print(f"   {col}: ✓ Нет отрицательных значений")

    # 2. Проверка нулевых значений
    print("\n2. Проверка нулевых значений:")
    for col in df.columns:
        if col != 'date':
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                print(f"   {col}: {zero_count} нулевых значений")
                if col in volume_columns and zero_count > 0:
                    issues.append(f"Нулевые объемы в {col}")

    # 3. Проверка дубликатов по дате
    print("\n3. Проверка дубликатов:")
    if 'date' in df.columns:
        duplicates = df['date'].duplicated().sum()
        print(f"   Дубликаты по дате: {duplicates}")
        if duplicates > 0:
            issues.append("Дубликаты по дате")

    # 4. Проверка порядка дат
    print("\n4. Проверка порядка дат:")
    if 'date' in df.columns:
        df_sorted = df.sort_values('date')
        if not df['date'].equals(df_sorted['date']):
            print("   Данные не отсортированы по дате")
            issues.append("Данные не отсортированы по дате")
        else:
            print("   ✓ Данные отсортированы по дате")

    # 5. Проверка распределения целевой переменной
    print("\n5. Распределение целевой переменной:")
    if 'target' in df.columns:
        percentages = df['target'].value_counts(normalize=True) * 100
        for value, percent in percentages.items():
            print(f"Значение {value}: {percent:.2f}%")
    
    return issues

def visualize_data(df):
    """Визуализация данных"""
    # Временной ряд
    plt.figure(figsize=(15, 5))
    df.plot.line(x='date', y="close", title='Цена закрытия (close) по датам')
    plt.grid(True)
    plt.show()
    
    # Распределение и выбросы
    numeric_cols = ['open', 'high', 'low', 'close', 'vol']
    plt.figure(figsize=(15, 6))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(1, len(numeric_cols), i)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.suptitle('Распределение значений и выбросы (до обработки)', y=1.02)
    plt.show()

def detect_outliers(df, method='iqr'):
    """Обнаружение выбросов с разными методами"""
    numeric_cols = ['open', 'high', 'low', 'close', 'vol']
    results = {}
    
    if method == 'iqr':
        print("="*50)
        print("Анализ выбросов (IQR метод):")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            results[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'count': len(outliers),
                'iqr': IQR
            }
            print(f"{col}: {len(outliers)} выбросов (границы: {lower_bound:.2f}, {upper_bound:.2f}), IQR: {IQR:.2f}")
    
    elif method == 'quantile':
        print("="*50)
        print("Анализ выбросов (квантильный метод 5%-95%):")
        for col in numeric_cols:
            lower_bound = df[col].quantile(0.05)
            upper_bound = df[col].quantile(0.95)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            results[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'count': len(outliers)
            }
            print(f"{col}: {len(outliers)} выбросов (границы: {lower_bound:.2f}, {upper_bound:.2f})")
    
    return results

def handle_outliers(df, column='vol', method='iqr'):
    """Обработка выбросов"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    else:  # quantile
        lower_bound = df[column].quantile(0.05)
        upper_bound = df[column].quantile(0.95)
    
    # Логарифмическое преобразование
    df[f'{column}_log'] = np.log1p(df[column])
    
    # Индикатор выбросов
    df[f'is_{column}_outlier'] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
    
    print(f"\nОбработка выбросов для {column} ({method} метод):")
    print(f"Всего наблюдений: {len(df)}")
    print(f"Выбросов обнаружено: {df[f'is_{column}_outlier'].sum()} ({df[f'is_{column}_outlier'].mean()*100:.1f}%)")
    print(f"Нижняя граница: {lower_bound:.2f}")
    print(f"Верхняя граница: {upper_bound:.2f}")
    print(f"Применено логарифмическое преобразование: np.log1p({column})")
    
    return df


def feature_engineering(df):
    """Создание новых признаков"""
    # Технические индикаторы
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Лаговые признаки
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    
    # Волатильность
    df['volatility'] = df['high'] - df['low']
    
    # Временные признаки
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Удаление строк с пропусками
    df = df.dropna()
    
    return df

# def plot_correlation(df, predictors):
#     """Визуализация корреляции признаков"""
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(df[predictors].corr(), annot=True, cmap='coolwarm')
#     plt.title('Корреляция признаков')
#     plt.show()
def plot_correlation(df, predictors, annot=True, figsize=(8, 6), cmap='Blues'):
    """
    Визуализация корреляции признаков
    
    Параметры:
    -----------
    df : DataFrame
        Датафрейм с данными
    predictors : list
        Список признаков для анализа
    annot : bool, optional (default=True)
        Показывать ли значения корреляции в ячейках
    figsize : tuple, optional (default=(8, 6))
        Размер графика (ширина, высота)
    cmap : str, optional (default='coolwarm')
        Цветовая схема heatmap
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        df[predictors].corr(), 
        annot=annot, 
        fmt=".2f" if annot else None,  # Формат чисел с 2 знаками после запятой
        cmap=cmap,
        vmin=-1,  # Минимальное значение корреляции
        vmax=1,   # Максимальное значение корреляции
        center=0,  # Центр цветовой шкалы
        linewidths=.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Матрица корреляции признаков')
    plt.tight_layout()
    plt.show()

def train_models(X, y, test_size=0.2, random_state=42):
    """Обучение и сравнение нескольких моделей"""
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Модели для сравнения
    models = {
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state),
        "LightGBM": LGBMClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear')
    }
    
    # Обучение и оценка
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "report": report,
            "y_pred": y_pred
        }
    
    return results, X_test, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """Оценка модели с выводом метрик и матрицы ошибок"""
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Метрики
    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred),
        "Recall": recall_score(y_train, y_train_pred),
        "F1": f1_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1": f1_score(y_test, y_test_pred)
    }
    
    # Матрицы ошибок
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    # Визуализация
    plt.figure(figsize=(15, 6))
    
    # Матрица ошибок для трейна
    plt.subplot(1, 2, 1)
    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Pred 0", "Pred 1"], 
                yticklabels=["True 0", "True 1"])
    plt.title(f"{model_name} | Train Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Матрица ошибок для теста
    plt.subplot(1, 2, 2)
    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Pred 0", "Pred 1"], 
                yticklabels=["True 0", "True 1"])
    plt.title(f"{model_name} | Test Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    
    # Вывод метрик
    print("\n" + "="*50)
    print(f"Отчёт производительности {model_name}")
    print("="*50)
    
    print("\nTrain Metrics:")
    print(pd.DataFrame([train_metrics]).T.rename(columns={0: "Value"}))
    
    print("\nTest Metrics:")
    print(pd.DataFrame([test_metrics]).T.rename(columns={0: "Value"}))
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    
    return train_metrics, test_metrics