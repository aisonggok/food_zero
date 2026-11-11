
import streamlit as st
import pickle
import pandas as pd

from Orange.data import Domain, Table

# ---------------------------------
# 1. 웹앱 기본 설정
# ---------------------------------

st.set_page_config(
    page_title="푸드제로 AI",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 푸드제로 AI")
st.subheader("음식물쓰레기 배출량 예측 서비스")

st.write(
    "Orange3에서 학습한 신경망 모델을 이용하여 "
    "지역별 음식물쓰레기 배출량을 예측합니다."
)

# ---------------------------------
# 2. Orange3 모델 불러오기
# ---------------------------------

@st.cache_resource
def load_model():
    with open("food_waste_model.pkcls", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"모델을 불러오지 못했습니다: {e}")
    st.stop()

# ---------------------------------
# 3. 학습 모델의 입력 변수 확인
# ---------------------------------

feature_vars = list(model.domain.attributes)

feature_names = [v.name for v in feature_vars]

required_features = ["배출연도", "배출월", "광역시도"]

if set(feature_names) != set(required_features):
    st.error(
        "학습 모델의 Features가 웹앱 설정과 다릅니다. "
        f"현재 모델 변수: {feature_names}"
    )
    st.stop()

city_var = next(
    v for v in feature_vars
    if v.name == "광역시도"
)

city_list = list(city_var.values)

# ---------------------------------
# 4. 사용자 입력 화면
# ---------------------------------

st.markdown("### 📍 예측 조건 선택")

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox(
        "배출연도",
        list(range(2017, 2031))
    )

with col2:
    month = st.selectbox(
        "배출월",
        list(range(1, 13))
    )

city = st.selectbox(
    "광역시도",
    city_list
)

# ---------------------------------
# 5. AI 예측 실행
# ---------------------------------

if st.button("🤖 음식물쓰레기 예측하기",
             use_container_width=True):

    try:
        # Orange3 모델의 원래 변수 순서에 맞춰 입력
        row = []

        for var in feature_vars:
            if var.name == "배출연도":
                row.append(year)

            elif var.name == "배출월":
                row.append(month)

            elif var.name == "광역시도":
                row.append(city)

        # Orange3 데이터 형식으로 변환
        input_domain = Domain(feature_vars)

        input_data = Table.from_list(
            input_domain,
            [row]
        )

        # 학습된 신경망으로 예측
        result = model(input_data)

        predicted_value = float(result[0])

        # ---------------------------------
        # 6. 예측 결과 출력
        # ---------------------------------

        st.success("AI 예측이 완료되었습니다!")

        st.metric(
            label="예상 음식물쓰레기 배출량",
            value=f"{predicted_value:,.0f} kg"
        )

        st.write("### 📋 예측 정보")

        result_df = pd.DataFrame({
            "항목": [
                "배출연도",
                "배출월",
                "광역시도",
                "AI 예측 배출량"
            ],
            "결과": [
                f"{year}년",
                f"{month}월",
                city,
                f"{predicted_value:,.0f} kg"
            ]
        })

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "※ 예측 결과는 Orange3 학습 모델에 기반한 추정값입니다."
        )

    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")

# ---------------------------------
# 7. 하단 안내
# ---------------------------------

st.divider()

st.caption(
    "본 서비스는 Orange3 Neural Network 모델을 활용합니다."
)
st.caption("푸드제로 AI | 음식물쓰레기 감축 프로젝트")
