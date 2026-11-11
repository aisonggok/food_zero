import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import Orange
from Orange.data import Domain, Table

st.set_page_config(page_title="푸드제로 AI", page_icon="🌱", layout="centered")
st.title("🌱 푸드제로 AI")
st.subheader("음식물쓰레기 배출량 예측 서비스")
st.write("Orange3 신경망 모델로 연도, 월, 광역시도에 따른 배출량을 예측합니다.")

MODEL_PATH = Path(__file__).parent / "food_waste_model.pkcls"

@st.cache_resource
def load_model(path, modified):
    # 본인이 직접 생성한 신뢰할 수 있는 모델 파일만 불러오세요.
    with open(path, "rb") as f:
        return pickle.load(f)

if not MODEL_PATH.exists():
    st.error("food_waste_model.pkcls 파일을 app.py와 같은 폴더에 넣어 주세요.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as e:
    st.error(f"모델을 불러오지 못했습니다: {e}")
    st.stop()

source_domain = getattr(model, "original_domain", None) or model.domain
feature_vars = list(source_domain.attributes)
feature_names = [v.name for v in feature_vars]
class_vars = list(model.domain.class_vars)

if not class_vars or class_vars[0].name != "배출량_kg":
    st.error(f"Target이 배출량_kg인지 확인하세요. 현재: {[v.name for v in class_vars]}")
    st.stop()
if not class_vars[0].is_continuous:
    st.error("배출량_kg는 연속형 숫자 Target이어야 합니다.")
    st.stop()
if "배출연도" not in feature_names or "배출월" not in feature_names:
    st.error(f"배출연도/배출월 변수를 찾을 수 없습니다: {feature_names}")
    st.stop()

city_var = next((v for v in feature_vars if v.name == "광역시도"), None)
dummy_vars = [v for v in feature_vars if v.name.startswith("광역시도=")]

if city_var is not None and city_var.is_discrete:
    city_options = list(city_var.values)
    city_mode = "범주형"
elif dummy_vars:
    city_options = [v.name.split("=", 1)[1] for v in dummy_vars]
    city_mode = "원핫 인코딩"
else:
    st.error(f"광역시도 변수를 찾을 수 없습니다: {feature_names}")
    st.stop()

st.markdown("### 1. 예측 조건 선택")
c1, c2 = st.columns(2)
with c1:
    year = st.number_input("배출연도", 1900, 2100, 2025, 1)
with c2:
    month = st.selectbox("배출월", list(range(1, 13)), index=7,
                         format_func=lambda x: f"{x}월")
city = st.selectbox("광역시도", city_options)

def make_input(y, m, selected_city):
    values = {"배출연도": float(y), "배출월": float(m)}
    row = []
    for var in feature_vars:
        name = var.name
        if name in values:
            row.append(values[name])
        elif name == "광역시도" and var.is_discrete:
            row.append(selected_city)
        elif name.startswith("광역시도="):
            category = name.split("=", 1)[1]
            row.append(1.0 if category == selected_city else 0.0)
        else:
            raise ValueError(f"처리할 수 없는 모델 입력 변수: {name}")
    return Table.from_list(Domain(feature_vars), [row])

st.markdown("### 2. AI 예측")
if st.button("🤖 음식물쓰레기 배출량 예측하기", type="primary", use_container_width=True):
    try:
        data = make_input(year, month, city)
        prediction = model(data)
        predicted = float(np.asarray(prediction).reshape(-1)[0])
        if not np.isfinite(predicted):
            raise ValueError("유효하지 않은 예측값입니다.")
        st.success("AI 예측이 완료되었습니다.")
        st.metric("예상 음식물쓰레기 배출량", f"{predicted:,.0f} kg")
        result = pd.DataFrame({
            "항목": ["배출연도", "배출월", "광역시도", "AI 예측 배출량"],
            "결과": [f"{int(year)}년", f"{int(month)}월", city, f"{predicted:,.0f} kg"]
        })
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.subheader("예측 결과")
        st.bar_chart(pd.DataFrame(
            {"예측 배출량(kg)": [predicted]},
            index=[f"{city} {int(year)}년 {int(month)}월"]
        ))
        st.caption("※ AI 예측값은 학습 데이터에 기반한 추정치이며 공식 통계가 아닙니다.")
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")

with st.expander("학습 모델 정보"):
    st.write("Orange3 버전:", Orange.__version__)
    st.write("입력 변수:", feature_names)
    st.write("광역시도 처리:", city_mode)
    st.write("Target:", class_vars[0].name)
    st.write("학습된 광역시도:", city_options)

st.divider()
st.caption("푸드제로 AI | Orange3 Neural Network + Streamlit")
