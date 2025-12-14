import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/gb_income_model.joblib"

st.set_page_config(
    page_title="Прогноз дохода > $50k",
    page_icon="💼",
    layout="centered",
)

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle["model"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]
cat_values = bundle["cat_values"]

st.title("💼 Прогноз: превысит ли доход $50k?")
st.caption("Введите данные как в анкете — модель оценит вероятность дохода выше $50k.")

with st.container(border=True):
    st.subheader("1) Числовые параметры")

    c1, c2, c3 = st.columns(3)

    age = c1.number_input("Возраст (age)", min_value=17, max_value=100, value=30, step=1)
    fnlwgt = c2.number_input("Вес выборки (fnlwgt)", min_value=1, max_value=2_000_000, value=200_000, step=1000)
    education_num = c3.number_input("Education-num", min_value=1, max_value=16, value=10, step=1)

    c4, c5, c6 = st.columns(3)
    capital_gain = c4.number_input("Capital gain", min_value=0, max_value=100_000, value=0, step=100)
    capital_loss = c5.number_input("Capital loss", min_value=0, max_value=10_000, value=0, step=50)
    hours = c6.number_input("Часов в неделю (hours-per-week)", min_value=1, max_value=99, value=40, step=1)

with st.container(border=True):
    st.subheader("2) Категории")

    c7, c8 = st.columns(2)
    workclass = c7.selectbox("Workclass", cat_values["workclass"])
    education = c8.selectbox("Education", cat_values["education"])

    c9, c10 = st.columns(2)
    marital = c9.selectbox("Marital status", cat_values["marital-status"])
    occupation = c10.selectbox("Occupation", cat_values["occupation"])

    c11, c12 = st.columns(2)
    relationship = c11.selectbox("Relationship", cat_values["relationship"])
    race = c12.selectbox("Race", cat_values["race"])

    sex = st.radio("Sex", cat_values["sex"], horizontal=True)

st.divider()

def predict_proba():
    row = {
        "age": age,
        "fnlwgt": fnlwgt,
        "education-num": education_num,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours,
        "workclass": workclass,
        "education": education,
        "marital-status": marital,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
    }
    X_one = pd.DataFrame([row], columns=num_cols + cat_cols)
    proba = float(model.predict_proba(X_one)[0, 1])
    pred = int(proba >= 0.5)
    return proba, pred

colA, colB = st.columns([1, 1])

with colA:
    go = st.button("🔮 Предсказать", use_container_width=True)

with colB:
    st.button("🧹 Сброс (обновите страницу)", use_container_width=True)

if go:
    proba, pred = predict_proba()

    st.subheader("Результат")
    st.metric("Вероятность дохода > $50k", f"{proba*100:.1f}%")

    st.progress(min(max(proba, 0.0), 1.0))

    if pred == 1:
        st.success("Скорее всего: **доход превысит $50k** ✅")
    else:
        st.info("Скорее всего: **доход не превысит $50k** ℹ️")

    with st.expander("Что это значит?"):
        st.write(
            "Это прогноз модели по данным Adult. Он **не гарантирует** реальный доход конкретного человека, "
            "но показывает, как модель оценивает комбинацию признаков."
        )
