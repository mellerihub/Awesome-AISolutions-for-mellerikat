# 📖Solution 

## What is trace profile?

Trace profile은 사용자별 구매 여정, demographic 정보를 토대로 만든 '사용자 정보'입니다. 해당 solution에서는 사용자의 구매 여정을 파악한뒤, 구매한 물품, 구매한 물품의 순서를 벡터화 하여 구매여정 trace profile을 구성하고, 사용자의 demographic 정보 또한 벡터화 하여 demographic profile을 구성합니다. 이후, 두개의 profile을 합쳐서 고객의 완성된 trace profile을 구상합니다. 해당 solution에서 trace profile을 구성하는 방법 외에도 의사결정자의 needs에 따라 알맞은 trace profile을 구성할수 있으며, 용도에 따라 필요한 정보를 사용하여 trace profile 구성이 가능합니다.

## What is trace clustering? 

Trace clustering은 사전에 구성한 trace profile을 사용하여 사용자를 clustering 하는 방법론입니다. 프로세스 마이닝 분야에서 널리 쓰이는 방법론으로, 복잡한 process중 유사한 trace들 끼리 군집화하여 특성을 파악하는데 사용됩니다. 
Song, M., Günther, C.W., van der Aalst, W.M.P. (2009). Trace Clustering in Process Mining. In: Ardagna, D., Mecella, M., Yang, J. (eds) Business Process Management Workshops. BPM 2008. Lecture Notes in Business Information Processing, vol 17. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-00328-8_11 는 trace clustering을 처음 제시한 논문이며, trace가 무엇인지, trace clustering의 개요를 파악할수 있는 논문입니다.

# 💡Features

## Pipeline

AI Contents의 pipeline은 기능 단위인 asset의 조합으로 이루어져 있습니다. Trace clustering는 단일 asset으로 inference pipeline을 구성하고 있습니다.  

**Inference pipeline**
```
asset_inference.py
```

## Assets


**Inference asset**  
첫번째로, 데이터를 입력받아 결측치 삽입 및 이상치 제거와 같은 전처리를 진행하게 됩니다.

두번째로, 고객의 구매 여정 정보를 통하여 구매여정 profile을 구성합니다. 
구매여정 profile을 구성하기 위하여, 고객이 구매한 상품의 빈도수를 벡터화 하여 저장합니다.
그 이후, 고객이 구매한 순서를 profile에 반영하기 위하여 물품 구매 transition matrix를 생성하여 flatten 후 PCA를 통하여 저차원 벡터화 합니다.
마지막으로 두 벡터를 합쳐서 구매여정 profile을 구성하게 됩니다.

세번째로, 고객의 demographic 정보를 반영하는 demographic profile 구성을 합니다.
데이터 상 존재하는 x(성별, 나이, 지역, 웹 체류시간, LG 가전 구매수,...)를 truncadedsvd 롤 통하여 저차원 벡터화 하여 저장하게 됩니다.

네번째로, 구매여정 profile과 demographic profile를 합쳐서 고객의 완성된 trace profile을 구성하게 됩니다.

다섯번째로, trace profile을 가지고 trace clustering을 진행하게 됩니다. 
해당 solution에서는 k-means clustering 기법을 사용하였으며, k를 정하기 위하여 silhouette score를 사용하여 가장 적절한 k 를 자동으로 도출하여 사용합니다.
trace clustering의 결과로 각 고객들을 대변하는 trace profile이 clustering됩니다.

여섯번째로, 생성된 cluster의 지표를 계산하게 됩니다.
각 cluster별로 평균 고객 구독 비율에 비해 해당 cluster가 얼마나 더 구독을 하였는지를 수치화하는 지표인 구독 지수를 계산하게 됩니다. 예를 들어 전체 고객의 구독률이 15%인데, cluster 1 의 구독률이 20%라면, 해당 cluster의 구독지수는 20%/15%로 1.5가 됩니다.
그 다음으로 각 cluster의 가장 중심에 있는 고객을 대표 고객 persona로 지정하여 추후에 사용할수 있도록 합니다.
추가적으로, 각 cluster과 가장 가깝지만 구독률이 더 높은 cluster를 도출하고, 해당 cluster의 대표 persona를 action persona로 지정하여 추후 구독률을 높이는 action에 활용할수 있도록 합니다.

일곱번째로, 새롭게 들어온 고객을 대상으로 똑같은 방식으로 trace profile을 도출히게 되고, 사전에 구성한 고객 cluster에 새로운 고객을 분류하게 됩니다.
이때, 새로운 고객의 예측 구독지수는 불뉴된 cluster의 구독지수를 사용하게 되고, 대표&액션 페르소나 또한 분류된 cluster의 persona를 사용하게 됩니다.
이를 통하여 새로운 고객이 구독할 확률이 어느정도 되고, 해당 고객과 유사한 고객이 과거에는 어떤 모습을 보였는지, 어떻게 하면 해당 고객의 구독률을 높일 수 있는지를 대표&엑션 페르소나를 통하여 결정할수 있게 됩니다.


## Experimental_plan.yaml

내가 갖고 있는 데이터에 AI Contents를 적용하려면 데이터에 대한 정보와 사용할 Contents 기능들을 experimental_plan.yaml 파일에 기입해야 합니다. AI Contents를 solution 폴더에 설치하면 solution 폴더 아래에 contents 마다 기본으로 작성되어있는 experimental_plan.yaml 파일을 확인할 수 있습니다. 이 yaml 파일에 '데이터 정보'를 입력하고 asset마다 제공하는 'user arugments'를 수정/추가하여 ALO를 실행하면, 원하는 세팅으로 데이터 분석 모델을 생성할 수 있습니다.

**experimental_plan.yaml 구조**  
experimental_plan.yaml에는 ALO를 구동하는데 필요한 다양한 setting값이 작성되어 있습니다. 이 setting값 중 '데이터 경로'와 'user arguments'부분을 수정하면 AI Contents를 바로 사용할 수 있습니다.


## Algorithms and models

sklearn의 kmeans clustering과 truncatedsvd를 사용하였습니다.


# 📂Input and Artifacts

## 데이터 준비

**학습 데이터 준비**  
1. 고객의 대변하는 변수의 값들이 존재하는들이 컬럼으로 이루어진 csv 파일을 준비합니다.
2. 모든 csv 파일은 해당 row를 식별할 수 있는 고객 id와  구독 여부가 존재해야 합니다.


**학습 데이터셋 예시**

|고객id|성별|지역|구매물품1|구매물품2|구매물품3|
|------|---|---|---|----|----|
|1|남|대구|에어컨|에어컨|NA|
|2|여|서울|세탁기|TV|TV|
|3|남|춘천|TV|NA|NA|


## 산출물

학습/추론을 실행하면 아래와 같은 산출물이 생성됩니다.  

**inference pipeline**
```
./alo/inference_artifacts/
    └ models/inference/
        └ svd.pkl
        └ scaler_1.pkl
        └ scaler_2.pkl
        └ pca_for_transition.pkl
        └ kmeans.pkl
    └ output/
        └ output.csv

```

각 산출물에 대한 상세 설명은 다음과 같습니다.  

**output.csv**
- 새로운 고객 데이터의 분류된 cluster와 예측 구독지수, 대표&액션 페르소나가 저장된 파일입니다.
