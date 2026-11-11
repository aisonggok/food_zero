# 푸드제로 AI - Orange3 3.38

1. Orange3에서 저장한 `food_waste_model.pkcls`를 이 폴더에 복사합니다.
2. 모델은 Features 배출연도, 배출월, 광역시도 / Target 배출량_kg 구조여야 합니다.
3. 광역시도가 `광역시도=서울특별시` 형태로 원핫 인코딩된 경우도 자동으로 처리합니다.

Windows 명령 프롬프트:
```bash
python -m pip install -r requirements.txt
python check_model.py
python -m streamlit run app.py
```

저장 환경과 같은 Orange3 버전을 사용하세요. pickle 기반 모델은 본인이 직접 만든 신뢰할 수 있는 파일만 불러오세요.
기초지자체는 현재 학습 변수에 없으므로 앱 입력에 포함하지 않습니다.
