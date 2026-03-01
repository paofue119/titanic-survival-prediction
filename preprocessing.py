"""
数据预处理模块：包含数据加载、缺失值填充、特征工程等。
"""
import pandas as pd

def load_data(train_path, test_path):
    """加载训练集和测试集，并打印基本信息。"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print("=" * 50)
    print("训练集样本数:", len(train))
    print("测试集样本数:", len(test))
    print("\n训练集缺失情况:\n", train.isnull().sum())
    print("\n测试集缺失情况:\n", test.isnull().sum())
    print("=" * 50)
    return train, test

def fill_missing(train, test):
    """填充缺失值：Age、Embarked、Fare"""
    age_median = train['Age'].median()
    embarked_mode = train['Embarked'].mode()[0]
    fare_median = train['Fare'].median()

    for df in (train, test):
        df['Age'].fillna(age_median, inplace=True)
        df['Embarked'].fillna(embarked_mode, inplace=True)
    test['Fare'].fillna(fare_median, inplace=True)
    return train, test

def create_features(train, test):
    """创建新特征：FamilySize, IsChild, Title"""
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    for df in (train, test):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsChild'] = (df['Age'] < 12).astype(int)
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    return train, test

def preprocess_pipeline(train, test):
    """完整预处理流程"""
    train, test = fill_missing(train, test)
    train, test = create_features(train, test)
    return train, test

def get_feature_columns():
    """返回用于建模的核心特征列名"""
    return ["Sex", "Pclass", "FamilySize", "Age", "IsChild", "Title"]

def prepare_features(train, test, feature_cols):
    """生成独热编码矩阵，并对齐训练集和测试集"""
    X = pd.get_dummies(train[feature_cols])
    y = train['Survived']
    X_test = pd.get_dummies(test[feature_cols])
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
    feature_names = X.columns.tolist()
    return X, y, X_test, feature_names