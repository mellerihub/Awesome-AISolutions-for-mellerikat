# 📖Solution 

## What is Fashion curation?

Fashion curation은 Outfit을 구성하는 아이템들의 이미지 임베딩을 통해 outfit을 학습하고, 주어지는 아이템의 속성에 따라 Outfit의 적합성을 평가하여 새로운 set을 generation을 생성하기 위한 알고리즘입니다.
여기서는 set을 generation하는 부분은 제외하고, 모델을 학습하고 fill in the blank 테스크를 통해 모델을 검증하는 과정까지가 포함되어 있습니다.

Outfit은 fashion curation모델에서 generation하고자 하는 output의 단위입니다. 
하나의 outfit은 ‘가방(액세서리)-아우터-상의-하의-신발’의 item들을 포함하고 있으며, 이 시퀀스대로 bi-lstm기반의 학습이 진행됩니다.

Fill-in-the-blank는 주어진 Outfit에서 하나의 아이템을 제외하고 모델이 정답을 찾는 task입니다.

## Key features and benefits 

**순차적 관계 학습**  
Bi-LSTM은 시퀀스 데이터를 처리하는 데 특화된 모델입니다. 아웃핏은 일반적으로 순차적인 구조(예: 상의 → 하의 → 신발 → 액세서리)를 가지므로, 이를 모델링하는 데 적합합니다.
패션은 아이템 간 조화와 맥락이 중요합니다. 예를 들어, 상의의 스타일은 하의나 신발과의 조화를 고려해야 하며, Bi-LSTM은 이러한 관계를 자연스럽게 학습할 수 있습니다.

**양방향 정보 처리**  
아웃핏의 앞뒤 요소 간 상호작용을 모두 반영하여 더 정교한 추천이 가능합니다.
사용자는 아웃핏을 조합할 때 특정 아이템(예: 신발)을 기준으로 나머지를 선택하거나, 전체적인 스타일을 고려하여 특정 아이템을 결정합니다. Bi-LSTM은 이러한 양방향 접근 방식을 자연스럽게 반영할 수 있습니다.


# 💡Features

## Pipeline

AI Contents의 pipeline은 기능 단위인 asset의 조합으로 이루어져 있습니다. Fashion curation은 총 2가지 asset의 조합으로 pipeline이 구성되어 있습니다.  

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
Bi-LSTM 기반의 딥러닝 모델을 학습하기 위한 워크플로를 구현한 것으로, 데이터 준비, 모델 초기화, 학습, 손실 계산, 가중치 업데이트, 그리고 결과 저장까지의 과정을 포함합니다. 이 과정이 완료되고 나면 생성된 모델 파일이 지정경로에 저장됩니다.

**Inference asset**  
"Fill-in-the-Blank (FITB)" 문제를 해결하기 위한 모델 평가 및 예측 과정입니다. 주어진 질문과 정답 후보를 바탕으로 빈칸에 들어갈 올바른 정답을 예측하며, 결과를 CSV 파일로 저장하고 성능 점수를 계산합니다.

**Configuration**  
Bi-LSTM Polyvore 모델의 설정 및 학습 하이퍼파라미터를 정의하는 코드입니다. 파일은 크게 모델 설정(ModelConfig)과 훈련 설정(TrainingConfig)으로 구성되며, 각 클래스는 모델 학습과 실행에 필요한 매개변수를 관리하는 역할을 합니다.

**Model**  
BiLSTM 모델을 사용하여 패션 이미지 시퀀스를 처리하고 학습하는 PyTorch 모델을 정의합니다. 이미지 특징을 추출하고 임베딩으로 변환한 후, 순방향(f_lstm) 및 역방향(b_lstm) LSTM 네트워크로 시퀀스 데이터를 학습하며, 손실 함수(CrossEntropyLoss)를 통해 학습을 최적화합니다.

**Dataset**  
OutfitDataset 클래스는 PyTorch의 Dataset 클래스를 상속받아 패션 코디 데이터셋을 처리하고 모델 학습에 필요한 데이터를 준비하는 역할을 합니다. 데이터셋의 구성 요소인 이미지와 JSON 파일을 읽고, 전처리 후 모델이 처리할 수 있는 형태로 반환합니다.

**Fitb_dataset**  
"Fill-in-the-Blank (FITB)" 문제를 해결하기 위한 데이터를 처리하는 PyTorch의 Dataset 클래스를 구현한 부분입니다. 주어진 데이터셋에서 질문 및 정답 후보 이미지를 불러와 전처리하고, 모델 학습 및 추론에 사용할 수 있는 형식으로 제공합니다.

## Experimental_plan.yaml

내가 갖고 있는 데이터에 AI Contents를 적용하려면 데이터에 대한 정보와 사용할 Contents 기능들을 experimental_plan.yaml 파일에 기입해야 합니다. AI Contents를 solution 폴더에 설치하면 solution 폴더 아래에 contents 마다 기본으로 작성되어있는 experimental_plan.yaml 파일을 확인할 수 있습니다. 이 yaml 파일에 '데이터 정보'를 입력하고 asset마다 제공하는 'user arugments'를 수정/추가하여 ALO를 실행하면, 원하는 세팅으로 데이터 분석 모델을 생성할 수 있습니다.

**experimental_plan.yaml 구조**  
experimental_plan.yaml에는 ALO를 구동하는데 필요한 다양한 setting값이 작성되어 있습니다. 이 setting값 중 '데이터 경로'와 'user arguments'부분을 수정하면 AI Contents를 바로 사용할 수 있습니다.

**데이터 경로 입력(external_path)**  
external_path의 parameter는 불러올 파일의 경로나 저장할 파일의 경로를 지정할 때 사용합니다. save_train_artifacts_path와 save_inference_artifacts_path는 입력하지 않으면 default 경로인 train_artifacts, inference_artifacts 폴더에 모델링 산출물이 저장됩니다.
```
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path:  ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```

|파라미터명|DEFAULT|설명 및 옵션|
|---|----|---|
|load_train_data_path|	./sample_data/train/|	학습 데이터가 위치한 폴더 경로를 입력합니다.(파일 명 입력 X)|
|load_inference_data_path|	./sample_data/test/|	추론 데이터가 위치한 폴더 경로를 입력합니다.(파일 명 입력 X)|

**사용자 파라미터(user_parameters)**  
user_parameters 아래 step은 asset 명을 의미합니다. 아래 step: input은 input asset단계임을 의미합니다.
args는 input asset(step: input)의 user arguments를 의미합니다. user arguments는 각 asset마다 제공하는 데이터 분석 관련 설정 파라미터입니다. 이에 대한 설명은 아래에 User arguments 설명을 확인해주세요.
```
user_parameters:
    - train_pipeline:
        - step: input
          args:
            - file_type
            ...
          ui_args:
            ...

```
## Algorithms and models

-EfficientNet-B0: 이미지 특징을 추출한다.

EfficientNet-B0는 각 이미지에 대한 고차원 특징 벡터(1280차원)를 생성합니다.
efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)를 통해 ImageNet으로 사전 학습된 가중치를 사용.
마지막 분류 계층(classifier)은 제거(self.img_feature_extracter.classifier[-1] = nn.Identity()), 특징 벡터만 반환.

-BiLSTM (Bidirectional LSTM)
f_lstm: 전방향 LSTM.
b_lstm: 역방향 LSTM.
입력: EfficientNet에서 추출한 이미지 특징 벡터(512차원).
출력: 각 이미지 특징 벡터에 대한 시퀀스 문맥 정보를 포함한 벡터.

# 📂Input and Artifacts

## 데이터 준비

**학습 데이터 준비**  
1. 학습에 사용하고자하는 json형태의 outfit 데이터 파일을 준비합니다. 
2. Outfit 데이터셋에 포함되는 모든 item들은 이미지 파일로 존재해야합니다.
3. Fitb에 사용되는 input 데이터는 이미지가 존재하는 아이템들의 조합으로 질문과 정답 후보를 구성해야합니다.
4. 정답을 모르는 상태에서 Fitb를 수행하려면, answer_index를 임의의 값으로 설정하고 진행합니다.

**학습 데이터셋 예시**

1. outfit_list.json
```
├── Set (Object)
│   ├── set_id (String): 패션 세트의 고유 식별자
│   └── items (Array): 패션 세트에 포함된 아이템 리스트
│       ├── Item (Object)
│       │   ├── item_id (String): 아이템의 고유 식별자
│       │   ├── clo_class (String): 아이템의 의류 분류 (e.g., TOP, BOTTOM, SHOES 등)
│       │   ├── gender (String): 아이템의 성별 대상 (e.g., MEN, WOMEN, 또는 "nan"으로 미지정)
│       │   └── index (Integer): 아이템의 순서 또는 위치를 나타내는 값
│       └── ... (다른 Item 객체 반복)
├── ... (다른 Set 객체 반복)
```
2. fitb.json
```
├── Entry (Object)
│   ├── question (Array)  # 질문 ID 배열
│   │   ├── String (ID)
│   │   ├── String (ID)
│   │   └── ...
│   ├── answer (Array)  # 정답 후보 배열
│   │   ├── String (ID)
│   │   ├── String (ID)
│   │   └── ...
│   ├── blank_index (Integer)  # 빈칸 위치
│   └── answer_index (Integer)  # 정답 위치
├── ... (다른 Entry 객체 반복)
```
**Input data directory 구조 예시**  
- ALO를 사용하기 위해서는 train과 inference 파일이 분리되어야 합니다. 아래와 같이 학습에 사용할 데이터와 추론에 사용할 데이터를 구분해주세요.
- 하나의 폴더 아래 있는 모든 파일을 input asset에서 취합해 하나의 dataframe으로 만든 후 모델링에 사용됩니다. (경로 밑 하위 폴더 안에 있는 파일도 합쳐집니다.)
```
./{train_folder}/
    └ images
        └ image1.jpg
        └ image2.jpg
        ...
    └ json
        └ outfit_list.json
./{inference_folder}/
    └ images
        └ image1.jpg
        └ image2.jpg
        ...
    └ json
        └ fitb.json
```

## 산출물

학습/추론을 실행하면 아래와 같은 산출물이 생성됩니다.  

**Train pipeline**
```
./alo/train_artifacts/
    └ models/train/
        └ bilstm_model.pth
    └ log/
        └ pipeline.log
        └ process.log
    └ extra_output/
    └ output/
    └ report/
    └ score/
```

**Inference pipeline**
```
 ./alo/inference_artifacts/
    └ output/
        └ output.csv
    └ extra_output/
    └ score/
        └ inference_summary.yaml
    └ log/
        └ pipeline.log
        └ process.log

```

각 산출물에 대한 상세 설명은 다음과 같습니다.  

**bilstm_model.pth**  
train단계에서 생성한 bilstm모델파일입니다.

**output.csv**  
inference단계에서 수행한 fitb의 결과를 기록한 파일입니다.
