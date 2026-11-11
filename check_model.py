import pickle
from pathlib import Path
import numpy as np
import Orange
from Orange.data import Domain, Table

p = Path(__file__).parent / "food_waste_model.pkcls"
if not p.exists():
    raise FileNotFoundError("food_waste_model.pkcls를 이 폴더에 복사하세요.")
with open(p, "rb") as f:
    model = pickle.load(f)

domain = getattr(model, "original_domain", None) or model.domain
features = list(domain.attributes)
names = [v.name for v in features]
classes = list(model.domain.class_vars)
print("Orange3:", Orange.__version__)
print("Features:", names)
print("Target:", [v.name for v in classes])
assert "배출연도" in names and "배출월" in names
assert classes and classes[0].name == "배출량_kg"

cv = next((v for v in features if v.name == "광역시도"), None)
dummies = [v for v in features if v.name.startswith("광역시도=")]
if cv is not None and cv.is_discrete:
    city = cv.values[0]
elif dummies:
    city = dummies[0].name.split("=", 1)[1]
else:
    raise ValueError("광역시도 입력 변수를 찾지 못했습니다.")

row = []
for v in features:
    if v.name == "배출연도": row.append(2025.0)
    elif v.name == "배출월": row.append(8.0)
    elif v.name == "광역시도" and v.is_discrete: row.append(city)
    elif v.name.startswith("광역시도="):
        row.append(1.0 if v.name.split("=", 1)[1] == city else 0.0)
    else: raise ValueError("처리하지 못하는 변수: " + v.name)

result = model(Table.from_list(Domain(features), [row]))
print("테스트 지역:", city)
print("테스트 예측:", float(np.asarray(result).reshape(-1)[0]), "kg")
print("모델 연동 검사 완료")
