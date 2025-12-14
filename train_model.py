import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = "data/data.adult.csv"
MODEL_PATH = "models/gb_income_model.joblib"

target_col = ">50K,<=50K"

num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]

def main():
    df_dirty = pd.read_csv(DATA_PATH)

    # чистка: выкидываем строки с '?'
    df_clean = df_dirty[~(df_dirty == "?").any(axis=1)].copy()

    y = df_clean[target_col].replace({"<=50K": 0, ">50K": 1})
    X = df_clean[num_cols + cat_cols].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    gb_best = GradientBoostingClassifier(
        n_estimators=29,
        criterion="squared_error",
        max_features=None,
        random_state=101,
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("gb", gb_best),
    ])

    # (опционально) проверка качества как в ДЗ
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"5-Fold ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X, y)

    # сохраним ещё списки колонок для приложения
    bundle = {
        "model": model,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_values": {c: sorted(df_clean[c].unique().tolist()) for c in cat_cols},
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
