# 📖 Solution

## What is Zero-Shot Object Detection?

Zero-Shot Object Detection(ZSOD)은 사전 학습된 클래스(label)뿐만 아니라 **새로운 객체를 텍스트 쿼리 기반으로 탐지할 수 있는 AI 기술**입니다. 기존 객체 탐지 모델(YOLO, Faster R-CNN 등)은 미리 학습된 클래스만 탐지할 수 있는 반면, ZSOD는 자연어 기반의 입력을 활용하여 **이전에 학습된 적 없는 객체도 탐지 가능**합니다. 본 솔루션은 **Grounding DINO** 모델을 기반으로 하며, mellerikat MLOps 프레임워크(ALO)에 최적화된 형태로 개발되었습니다.

## When to use Zero-Shot Object Detection?

Zero-Shot Object Detection은 다음과 같은 분야에서 활용할 수 있습니다:

- **산업 자동화 및 공정 관리**: 새로운 부품이나 불량 유형이 발생했을 때, 학습 없이도 즉시 탐지 가능
- **자율 주행 및 스마트 모니터링**: 특정 상황이나 대상(예: "red car" 또는 "pedestrian with blue backpack")을 실시간으로 탐지 가능
- **보안 및 감시 시스템**: 기존 학습된 객체 외에도 "위험한 물체" 또는 "수상한 행동을 하는 사람"과 같은 개념을 탐지 가능
- **의료 영상 분석**: 미리 정의되지 않은 병변이나 특정 조직 패턴을 텍스트 기반으로 탐색 가능

## Key Features and Benefits

Zero-Shot Object Detection은 **사전 학습된 클래스 없이도** 객체를 탐지할 수 있는 강력한 기능을 제공합니다.

- **Open-Vocabulary Detection**: "새로운 클래스(class)"를 추가 학습 없이도 즉시 탐지 가능
- **텍스트 기반 탐지(Text-to-Detect)**: "red car", "person with a hat" 등의 **텍스트 쿼리**를 활용하여 객체를 탐지 가능
- **즉각적인 적용(Zero-Shot Adaptability)**: 새로운 데이터셋이나 환경 변화에도 별도의 학습 없이 탐지 가능
- **실시간 대응 및 최적화**: 실시간 스트리밍 데이터에서도 동작 가능하며, 산업 자동화 및 보안 시스템에서 활용 가능

# 💡 Features

## Pipeline

Zero-Shot Object Detection의 pipeline은 **기능 단위(asset)의 조합**으로 구성됩니다.

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

Train asset에서는 Grounding DINO를 기반으로 학습을 수행합니다. 학습 과정에서는 CLIP 기반 텍스트-이미지 융합 모델을 활용하여 객체 탐지 성능을 최적화합니다. 사전 학습된 모델을 기반으로 하며, 필요 시 추가 학습을 통해 성능을 향상할 수 있습니다. 최적화된 가중치는 `model_best.pth`로 저장됩니다.

**Inference asset**

Inference asset에서는 학습된 모델을 기반으로 **Zero-Shot Object Detection**을 수행합니다. 입력된 이미지와 자연어 텍스트 쿼리를 조합하여 객체를 탐지하고, 탐지 결과를 `predictions.csv` 파일로 저장합니다.

## Experimental_plan.yaml

본 솔루션을 활용하려면 실험 계획 파일(`experimental_plan.yaml`)을 설정해야 합니다. 해당 파일에는 데이터 경로 및 사용자 정의 인자(user arguments) 값이 포함됩니다.

### 데이터 경로 설정 (external_path)
```
external_path:
    - load_train_data_path: ./solution/sample_data/train/
    - load_inference_data_path: ./solution/sample_data/test/
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```

### 사용자 파라미터 설정 (user_parameters)
```
user_parameters:
    - train_pipeline:
      - step: train
        args:
          - batch_size: 16
            train_epoch: 10
            learning_rate: 0.0001
    - inference_pipeline:
      - step: inference
        args:
          - confidence_threshold: 0.5
            max_detections: 100
```

## Algorithms and Models

본 솔루션은 **Grounding DINO** 모델을 기반으로 합니다. Grounding DINO는 **Transformer 기반의 Object Detection 모델**로, 자연어 쿼리를 활용한 Zero-Shot 탐지가 가능합니다. CLIP 기반의 텍스트-이미지 매칭 기법을 사용하여 **사전 정의되지 않은 객체도 탐지**할 수 있습니다.

# 📂 Input and Artifacts

## 데이터 준비

**입력 데이터 요구사항**
- 이미지 데이터 (`.jpg`, `.png` 등)
- 탐지할 객체에 대한 **텍스트 쿼리** (예: "red car", "dog running")

**데이터 디렉토리 구조 예시**
```
./solution/sample_data/
    ├── train/
    │     ├── images/
    │     │     ├── img1.jpg
    │     │     ├── img2.jpg
    ├── test/
    │     ├── images/
    │     │     ├── img1.jpg
    │     │     ├── img2.jpg
```

## 산출물 (Artifacts)

**Train pipeline**
```
./alo/train_artifacts/
    ├── models/
    │     ├── model_best.pth  # 학습된 모델 가중치
    ├── log/
    │     ├── training.log  # 학습 과정 로그
```

**Inference pipeline**
```
./alo/inference_artifacts/
    ├── output/
    │     ├── predictions.csv  # 탐지된 객체 결과
    ├── score/
    │     ├── inference_summary.yaml  # 추론 요약
```

**predictions.csv**
- 탐지된 객체 리스트 및 bounding box 좌표 포함
- 예시:
```
image_id,object,confidence,x_min,y_min,x_max,y_max
img1.jpg,car,0.92,30,50,120,200
img2.jpg,person,0.88,60,80,170,250
```

**inference_summary.yaml**
- Zero-Shot 탐지 결과 요약 정보 제공


<br>
<br>


## :hammer_and_wrench:  Requirements and Install 

Basic Dependencies:

* Python == 3.10
<br>

**Installation:**

<br>
1.Clone the ALO repository from GitHub.

```bash
git clone https://github.com/mellerikat/alo.git zeroshot_od
```

<br>
2. Clone the zeroshot-objectdetection solution repository from GitHub.

```bash
git clone https://github.com/mellerihub/zeroshot-objectdetection.git
```

<br>
3. Download pre=trained tokenizer adn model weights.

```bash
cd assets
cd inference
git clone git@hf.co:google-bert/bert-base-uncased

cd groundingdino
cd checkpoints
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

<br>
4. Install the required dependencies in the solution directory.

```bash
pip install -r requirements.txt
cd assets
cd inference
pip install -e .
```

<br>
5. Local Demo start

```bash
python main.py
```
