# -*- coding: utf-8 -*-
"""model(gb)ToPickle.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sxm4jE4HNSS3c_biaTkXaWEbgSvhusY9
"""

from google.colab import drive
drive.mount('/content/drive')

from sklearn.ensemble import GradientBoostingClassifier
import joblib
import pickle
import pandas as pd

## 파일 경로 설정
file_path = "/apnea-ecg-develop.pkl"


## 전처리 완료된 train dataset pickle 파일 불러오기
with open(file_path, "rb") as f:
    loaded_data = pickle.load(f)


## pickle 파일 데이터프레임으로 변환
df = pd.DataFrame(columns=[
    "recording", "rmssd", "sdnn", "nn50", "sdsd", "pnn50", "mrri", "mhr",
    "vlf_rri", "lf_rri", "hf_rri", "lf_hf_rri", "lfnu_rri", "hfnu_rri",
    "vlf_edr", "lf_edr", "hf_edr", "lf_hf_edr", "lfnu_edr", "hfnu_edr",
    "sd1", "sd2", "label"
])

for recording, data in loaded_data.items():
    for row_data in data:
        row_data_list = [recording] + row_data.tolist()
        df = df.append(pd.Series(row_data_list, index=df.columns), ignore_index=True)


## 입력 데이터와 타겟 준비
X_train = df.drop(columns = ['recording', 'label'])
y_train = df['label']

## Gradient Boosting Classifier 모델 객체 생성
model_gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)

## 모델 학습
model_gb.fit(X_train, y_train)

# 학습시킨 모델을 pickle 파일로 저장
joblib.dump(model_gb, '/model.pkl')
