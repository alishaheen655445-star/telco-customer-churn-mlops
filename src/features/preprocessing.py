from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(X):

    # الأعمدة الرقمية
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    # الأعمدة النصية
    categorical_features = X.select_dtypes(include=['object']).columns

    # تحويل البيانات
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor