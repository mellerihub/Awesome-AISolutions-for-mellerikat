# 📖Solution 

## What is Time Series Forecasting?

Time Series Forecasting(시계열 예측)은 시간 순서대로 배열된 데이터를 분석하고 미래의 값을 예측하는 데 중점을 둔 AI 콘텐츠입니다. 과거 시점의 패턴과 추세를 학습하여, 주어진 데이터로부터 미래의 데이터 포인트의 집합을 예측하는 데 사용합니다. 과거에는 ARIMA, VAR 등 통계 기법으로 미래를 예측했습니다. 현재 딥러닝 기반 방법론이 각광받으면서 LSTM, Transformer 등 다양한 모델들이 시계열 예측에 활용되고 있으며, 본 solution은 비교적 경량화되어 있는 모델이지만 시계열 예측 성능이 좋은 Dlinear 알고리즘 코드입니다.

## When to use Time Series Forecasting?

Time Series Forecasting 적용 가능 분야는 다음과 같습니다:

금융: 금융 데이터를 기반으로 한 전략 수립에 활용됩니다. 시계열 예측을 활용하여 가격 추세 예측, 변동성 분석 등 투자 결정을 최적화할 수 있습니다.
공정 및 장비: 공장에서 사용되는 장비의 센서 데이터를 기반으로, 장비의 마모 상태와 고장 시점을 예측하여 예방 정비를 수행할 수 있습니다. 이를 통해 예기치 않은 고장을 줄이고, 유지보수 비용을 절감할 수 있습니다.
기상: 강수량 및 기온 데이터를 분석하여 농업 생산성을 높이거나, 태풍 경로를 예측하여 재난 대비를 효과적으로 수행할 수 있습니다.

## Key features and benefits 

Time Series Forecasting은 과거 데이터를 기반으로 한 시간 패턴을 효과적으로 분석하여 미래를 대비할 수 있게 합니다. 딥러닝 기반 시계열 예측은 시계열의 복잡한 패턴을 모델의 표현력으로 포착하여 더 정확한 예측을 할 수 있게 합니다.

딥러닝 기반 시계열 예측은 특히 많은 기간을 예측해야 하는 과제나, 훈련과 시험 데이터셋의 특징이 명확하게 차이날 때 효과적입니다.

**실시간 대응 및 최적화**
실시간 데이터를 활용하여 예측 결과를 지속적으로 업데이트할 수 있습니다. 이를 통해 변화하는 환경에 즉각적으로 대응하고, 예측하고자 하는 도메인에서의 빠른 의사결정과 최적화를 도울 수 있습니다.
**장기적 의사결정 지원**
단기적인 변화뿐 아니라 장기적인 패턴을 학습하여, 지속 가능한 성장 전략과 리소스 배분 계획을 수립하는 데 기여합니다.

# 💡Features

## Pipeline

Time series forecasting의 pipeline은 기능 단위인 asset의 조합으로 이루어져 있습니다. 두 가지 pipeline으로 구성되어 있습니다. 

**Train pipeline**
```
Train
```

**Inference pipeline**
```
Inference
```

## Assets

각 단계는 asset으로 구분됩니다.


**Train asset**  

Train asset에서는 DLinear를 기반으로 학습 과정을 실행합니다. 학습 과정에서 get_loader 함수를 사용하여 학습 데이터와 검증 데이터를 생성합니다. 데이터는 여러 개의 윈도우로 구성되며, 각 윈도우에 대해 학습이 이루어집니다. 학습 데이터는 각 윈도우의 입력 순서를 섞은 채 진행되며, 검증 데이터는 검증 데이터셋의 순서대로 검증을 수행합니다. 학습 모델은 DLinear Model을 포함합니다. 학습 단계에서는 정답값과 MSE(Mean Squared Error) 손실 함수를 통해 모델을 학습합니다. 학습 단계의 epoch마다 검증을 수행하여 같은 방식으로 손실 함수를 계산합니다. 최종적인 모델은 검증 단계에서 가장 낮은 손실 함수값을 가진 모델이 저장됩니다. 저장된 파일은 가중치(sample_data_dlinear.pth)로 구성됩니다. Train asset은 최종적으로 학습 결과와 사용된 파라미터를 저장하여 사용자가 최적의 모델을 선택할 수 있도록 지원합니다.

**Inference asset**  
Inference asset은 Train asset에서 학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다. get_loader 함수를 통해 테스트 로더를 기반으로 실행되며, 최적의 모델 가중치(model_best.pth)를 로드합니다. 새로운 데이터(테스트 데이터)에 대해 예측값을 계산하고 predictions.csv 파일로 저장됩니다. 

## Experimental_plan.yaml

내가 갖고 있는 데이터에 AI Contents를 적용하려면 데이터에 대한 정보와 사용할 Contents 기능들을 experimental_plan.yaml 파일에 기입해야 합니다. AI Contents를 solution 폴더에 설치하면 solution 폴더 아래에 contents 마다 기본으로 작성되어있는 experimental_plan.yaml 파일을 확인할 수 있습니다. 이 yaml 파일에 '데이터 정보'를 입력하고 asset마다 제공하는 'user arugments'를 수정/추가하여 ALO를 실행하면, 원하는 세팅으로 데이터 분석 모델을 생성할 수 있습니다.

**experimental_plan.yaml 구조**  
experimental_plan.yaml에는 ALO를 구동하는데 필요한 다양한 setting값이 작성되어 있습니다. 이 setting값 중 '데이터 경로'와 'user arguments'부분을 수정하면 AI Contents를 바로 사용할 수 있습니다.

**데이터 경로 입력(external_path)**  
external_path의 parameter는 불러올 파일의 경로나 저장할 파일의 경로를 지정할 때 사용합니다. save_train_artifacts_path와 save_inference_artifacts_path는 입력하지 않으면 default 경로인 train_artifacts, inference_artifacts 폴더에 모델링 산출물이 저장됩니다.
```
external_path:
    - load_train_data_path: ./solution/sample_data/train/
    - load_inference_data_path: ./solution/sample_data/test/
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```

|파라미터명|DEFAULT|설명 및 옵션|
|---|----|---|
|load_train_data_path|	./solution/sample_data/train/|	학습 데이터가 위치한 폴더 경로를 입력합니다(csv 파일 명 입력 X).|
|load_inference_data_path|	./solution/sample_data/test/|	추론 데이터가 위치한 폴더 경로를 입력합니다(csv 파일 명 입력 X).|

**사용자 파라미터(user_parameters)**  
user_parameters 아래 step은 asset 명을 의미합니다. 아래 step: input은 input asset단계임을 의미합니다.
args는 input asset(step: input)의 user arguments를 의미합니다. user arguments는 각 asset마다 제공하는 데이터 분석 관련 설정 파라미터입니다. 이에 대한 설명은 아래에 User arguments 설명을 확인해주세요.
```
user_parameters:
    - train_pipeline:
      - step: train 
        args:
          - seq_len: 96                 # the length of window
            pred_len: 48                # the length of prediction
            scaler_type : 'standard'    # scaler type, 'standard' means StandardScaler, 'minmax' means MinMaxScaler
            features: 'MS'              # 'MS': Multivariate to Univariate, 'S': Univariate to Univariate, 'M': Multivariate to Multivariate
            individual: True            # Channel-Independent Option, defaults to True
            enc_in: 7                   # the number of features
            batch_size: 32
            train_epoch: 10

    - inference_pipeline:
      - step: inference
        args:
          - seq_len: 96
            pred_len: 48
            features: 'MS'
            individual: True
            enc_in: 7
            batch_size: 32

```
## Algorithms and models

Time Series Forecasting train/inference asset은 DLinear 모델을 포함하고 있습니다. 이 모델은 단순 선형 모델으로 Linear layer에 투입 전 이동 평균법(moving average)으로 추세와 나머지를 분할합니다. 분리된 추세와 나머지는 각각의 선형 모델을 통해 설정된 예측 길이로 투영됩니다. 이 두 개의 결과물을 더하여 최종적인 예측 수치를 산출하게 됩니다.


# 📂Input and Artifacts

## 데이터 준비

**학습 데이터 준비**  
1. 예측 변수 x와 반응 변수 y로 이루어진
2. 모든 csv 파일은 해당 row의 시점인 timestep이 존재해야 합니다.
3. seq_len과 pred_len을 입력하지 않으면 seq_len = 336, pred_len = 96으로 자동 설정됩니다.


**학습 데이터셋 예시**
```
|timestep|x_col_1|x_col_2|y|
|------|---|---|---|
|time 1|value 1_1|value 1_2|y1|
|time 2|value 2_1|value2_2|y2|
|time 3|value 3_1|value3_2|y3|
```
**Input data directory 구조 예시**  
- ALO를 사용하기 위해서는 train과 inference 파일이 분리되어야 합니다. 
- 본 솔루션에서는 동일한 csv 파일 로드 후 데이터셋을 만드는 과정에서 train, valid, test를 구분합니다. 따라서 train과 test에 모두 같은 데이터를 넣어주셔야 합니다.
```
./solution/sample_data/
    ├── train/
    │     ├── data.csv
    ├── test/
    │     ├── data.csv
```

## 데이터 요구사항

**필수 요구사항**  
입력 데이터는 다음 조건을 반드시 만족하여야 합니다.

|index|item|spec.|
|----|----|----|
|1|timetstep 의 개수|1개|
|2|예측하고자 하는 칼럼|1개|

**추가 요구사항**  
최소한의 성능을 보장하기 위한 조건입니다. 하기 조건이 만족되지 않아도 알고리즘은 돌아가지만 성능은 확인되지 않았습니다    

| Index | Item                      | Spec.                                                                                     |
|-------|---------------------------|-------------------------------------------------------------------------------------------|
| 1     | 예측 변수                  | 예측하고자 하는 타겟만 있어도 학습이 진행되지만, 'S' feature 이외 옵션은 다변량 데이터셋을 지원합니다. |
| 2     | seq_len, pred_len         | default 값은 설정되어 있지만, 데이터셋의 수집 주기가 sample과 다른 경우 성능에 영향을 미칠 수 있습니다. |


## 산출물

학습/추론을 실행하면 아래와 같은 산출물이 생성됩니다.  

**Train pipeline**
```
./alo/train_artifacts/
    └ models/train
        └ sample_data_dlinear.pth           # 학습된 모델 가중치
    └ log/
        └ pipeline.log                      # 학습 로그
```

**Inference pipeline**
```
 ./alo/inference_artifacts/
    └ output/
        └ predictions.csv           # 추론 결과
    └ score/
        └ inference_summary.yaml    # 추론 요약
```

각 산출물에 대한 상세 설명은 다음과 같습니다.  

**train_model.pth** 
학습된 모델의 최적 가중치가 저장된 파일입니다.
**pipeline.log**
학습 과정의 손실, 성능 지표가 기록된 로그 파일입니다.

**predictions.csv**
추론 데이터에 대한 예측 클래스가 포함된 CSV 파일입니다.
**inference_summary.yaml**
추론 요약 파일로, 최종 요약 정보를 포함합니다.
