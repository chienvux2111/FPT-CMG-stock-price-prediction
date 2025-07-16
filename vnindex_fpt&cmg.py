# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Hàm load dữ liệu
def load_data(file_path):
    return pd.read_csv(file_path)

# Hàm làm sạch dữ liệu
def clean_dataframe(df):
    rename_map = {
        'Ngày': 'Date',
        'Lần cuối': 'Closing Price',
        'Mở': 'Open Price',
        'Cao': 'Highest Price',
        'Thấp': 'Lowest Price',
        'KL': 'Volume',
        '% Thay đổi': 'Price Change %'
    }
    df.rename(columns=rename_map, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.set_index('Date', inplace=True)

    def clean_cell(x):
        if isinstance(x, str):
            x = x.replace(',', '').replace('%', '').replace('−', '-').strip()
            if 'K' in x:
                return float(x.replace('K', '')) * 1_000
            elif 'M' in x:
                return float(x.replace('M', '')) * 1_000_000
            elif 'B' in x:
                return float(x.replace('B', '')) * 1_000_000_000
            else:
                try:
                    return float(x)
                except:
                    return np.nan
        return x

    for col in df.columns:
        if col != 'Date':
            df[col] = df[col].apply(clean_cell)
            if 'Volume' in col:
                df[col] = df[col].fillna(0).astype('int64')
            elif col in ['Closing Price', 'Open Price', 'Highest Price', 'Lowest Price', 'Price Change %']:
                df[col] = df[col].astype('float')

    return df

# Hàm phân tích thống kê cơ bản
def basic_stats(df, name):
    print(name)
    print(df.info())
    print("Số giá trị bị rỗng:")
    print(df.isnull().sum())
    print("Thống kê các giá trị:")
    print(df.describe())
    print("Các giá trị độc nhất của từng trường dữ liệu:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")


# Hàm tạo các feature gián tiếp
def new_feature(market, stock, ticker):
    df = pd.DataFrame(index=market.index)
    df['Average Price (daily)'] = (market['Highest Price'] + market['Lowest Price']) / 2
    df['Intraday price (range)'] = market['Highest Price'] - market['Lowest Price']
    df['Actual price change'] = market['Closing Price_vnindex'] - market['Open Price']
    df['Intraday percentage volatility'] = (market['Highest Price'] - market['Lowest Price']) / 2
    return df
# EDA
def eda(df, ticker):
    """
    Thực hiện phân tích dữ liệu khám phá (EDA) cho một DataFrame.

    Args:
        df (pd.DataFrame): Dữ liệu cần phân tích.
        ticker (str): Tên cổ phiếu hoặc dữ liệu để hiển thị trong báo cáo.

    Returns:
        None
    """
    print(f"📋 EDA Report for {ticker}")
    print("=" * 80)

    # Tổng quan
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Total Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    print("=" * 80)

    # Tóm tắt kiểu dữ liệu
    print("📚 Data Types:")
    print(df.dtypes.value_counts())
    print("=" * 80)

    # Giá trị bị thiếu theo cột
    missing_cols = df.isnull().mean() * 100
    missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)

    if not missing_cols.empty:
        print("🚩 Columns with Missing Values (%):")
        print(missing_cols.round(2))

        # Biểu đồ ma trận giá trị bị thiếu
        import missingno as msno
        plt.figure(figsize=(10, 6))
        msno.matrix(df)
        plt.title('Missing Values Matrix')
        plt.show()

        # Biểu đồ tương quan giá trị bị thiếu
        plt.figure(figsize=(10, 6))
        msno.heatmap(df)
        plt.title('Missing Values Correlation')
        plt.show()
        plt.close()
    else:
        print("✅ No Missing Values")
    print("=" * 80)

    # Cột số và cột phân loại
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print(f"🔢 Numerical Columns ({len(num_cols)}): {num_cols}")
    print(f"🔠 Categorical Columns ({len(cat_cols)}): {cat_cols}")
    print("=" * 80)

    # Tóm tắt các cột số
    if num_cols:
        print("📈 Numerical Features Summary:")
        display(df[num_cols].describe().T)

        # Biểu đồ histogram
        df[num_cols].hist(bins=30, figsize=(16, 12), layout=(int(np.ceil(len(num_cols) / 3)), 3))
        plt.tight_layout()
        plt.suptitle('Histograms of Numerical Features', y=1.02)
        plt.show()
        plt.close()
    else:
        print("⚠️ No Numerical Columns Found")
    print("=" * 80)

    # Tóm tắt các cột phân loại
    if cat_cols:
        print("📊 Categorical Features Summary:")
        for col in cat_cols:
            print(f"\n📝 Column: {col} (Unique: {df[col].nunique()})")
            print(df[col].value_counts(dropna=False).head(5))

            # Biểu đồ countplot
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x=col, order=df[col].value_counts().index)
            plt.xticks(rotation=45)
            plt.title(f'Countplot: {col}')
            plt.show()
            plt.close()
    else:
        print("⚠️ No Categorical Columns Found")
    print("=" * 80)

    # Phân tích tương quan
    if len(num_cols) >= 2:
        print("🔗 Correlation Matrix (Numerical Features):")
        corr = df[num_cols].corr()
        print(corr.round(2))

        # Biểu đồ heatmap tương quan
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title('Correlation Heatmap')
        plt.show()
        plt.close()
    else:
        print("⚠️ Not Enough Numerical Features for Correlation Analysis")
    print("=" * 80) 

# Hàm tạo lag features
def create_lag(df, columns, n_lags=3, prefix=''):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    for col in columns:
        for lag in range(1, n_lags + 1):
            df[f"{prefix}{col}_lag{lag}"] = df[col].shift(lag)

    return df

# 5. Hàm phân tích mô hình
# 5.1. Random Forest Regressor
def rf_regressor_analysis(df, ticker, test_size=0.3, n_estimators=100, random_state=42):
    X = df.drop(columns='Closing Price')
    y = df['Closing Price']
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R²  : {r2_score(y_test, y_pred):.4f}")

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"Feature Importances (Regressor) of {ticker}")
    plt.show()

# 5.2. Random Forest Classifier
def rf_classifier_analysis(df, ticker, test_size=0.3, n_estimators=100, random_state=42):
    df['Target'] = (df['Closing Price'].shift(-1) > df['Closing Price']).astype(int)
    df.dropna(inplace=True)
    X = df.drop(columns=['Closing Price', 'Target'])
    y = df['Target']
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {ticker}")
    plt.show()

# 6. hàm XGBoost
# 6.1. Hàm XGBoost Regressor
def xgb_regressor_analysis(df, ticker, test_size=0.3, random_state=42):
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # 1. Tách features và target
    X = df.drop(columns='Closing Price')
    y = df['Closing Price']

    # 2. Tách tập train/test theo thời gian
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 3. Train XGBoost Regressor
    model = XGBRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # 4. Dự đoán và đánh giá
    y_pred = model.predict(X_test)

    print(f"Đánh giá mô hình XGBoost Regressor - {ticker}:")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R²  : {r2_score(y_test, y_pred):.4f}")

    # 5. Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # 6. Biểu đồ
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"Feature Importances (XGBoost Regressor) - {ticker}")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

# 7. Hàm XGBoost Classifier
def xgb_classifier_analysis(df, ticker, test_size=0.3, random_state=42):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

    # 1. Tạo nhãn tăng/giảm (1 = tăng, 0 = giảm hoặc không đổi)
    df = df.copy()
    df['Target'] = (df['Closing Price'].shift(-1) > df['Closing Price']).astype(int)
    df.dropna(inplace=True)

    # 2. Tách features và target
    X = df.drop(columns=['Closing Price', 'Target'])
    y = df['Target']

    # 3. Tách tập train/test theo thời gian
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 4. Train XGBoost Classifier
    model = XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss')
    model.fit(X_train, y_train)

    # 5. Dự đoán và đánh giá
    y_pred = model.predict(X_test)

    print(f"Đánh giá mô hình XGBoost Classifier - {ticker}:")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {ticker}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # 7. Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"Feature Importances (XGBoost Classifier) - {ticker}")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

# 8. Light GBM
# 8.1. Light GBM regressor
def lgbm_regressor_analysis(df, ticker, test_size=0.3, random_state=42):
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # 1. Tách features và target
    X = df.drop(columns='Closing Price')
    y = df['Closing Price']

    # 2. Tách tập train/test theo thời gian
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 3. Train LightGBM Regressor
    model = LGBMRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # 4. Dự đoán và đánh giá
    y_pred = model.predict(X_test)

    print(f"Đánh giá mô hình LightGBM Regressor - {ticker}:")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R²  : {r2_score(y_test, y_pred):.4f}")

    # 5. Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # 6. Biểu đồ
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"Feature Importances (LightGBM Regressor) - {ticker}")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
# 8.2. Light GBM Classifier
def lgbm_classifier_analysis(df, ticker, test_size=0.3, random_state=42):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

    # 1. Tạo nhãn tăng/giảm (1 = tăng, 0 = giảm hoặc không đổi)
    df = df.copy()
    df['Target'] = (df['Closing Price'].shift(-1) > df['Closing Price']).astype(int)
    df.dropna(inplace=True)

    # 2. Tách features và target
    X = df.drop(columns=['Closing Price', 'Target'])
    y = df['Target']

    # 3. Tách tập train/test theo thời gian
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 4. Train LightGBM Classifier
    model = LGBMClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # 5. Dự đoán và đánh giá
    y_pred = model.predict(X_test)

    print(f"Đánh giá mô hình LightGBM Classifier - {ticker}:")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {ticker}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # 7. Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"Feature Importances (LightGBM Classifier) - {ticker}")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

# 9. Merge DataFrames
def merge_df(target, market, newfeat):
    target = pd.DataFrame(target)
    result = pd.merge(target, market, left_index=True, right_index=True, how='inner')
    result = pd.merge(result, newfeat, left_index=True, right_index=True, how='inner')
    result.dropna(inplace=True)
    return result