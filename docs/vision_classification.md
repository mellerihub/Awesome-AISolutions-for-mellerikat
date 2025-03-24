# Vision Classification (VC) Complete Reference Guide

<div align="right">Updated 2024.05.17</div>

## Table of Contents
- [Introduction](#introduction)
  - [What is Vision Classification?](#what-is-vision-classification)
  - [When to Use Vision Classification](#when-to-use-vision-classification)
  - [Use Cases](#use-cases)
  - [Key Features](#key-features)
- [Quick Start Guide](#quick-start-guide)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Parameter Configuration](#parameter-configuration)
  - [Execution](#execution)
- [Pipeline Architecture](#pipeline-architecture)
  - [Overview](#vc-pipeline)
  - [Pipeline Components](#pipeline-components)
- [Data Preparation](#data-preparation-1)
  - [Requirements](#data-requirements)
  - [Input Preparation](#preparing-training-data)
  - [Directory Structure](#example-of-input-data-directory-structure)
- [Features In Depth](#features-in-depth)
  - [Train Pipeline Features](#train-pipeline-features)
  - [Inference Pipeline Features](#inference-pipeline-features)
- [Output Artifacts](#artifacts)
  - [Train Pipeline Artifacts](#train-pipeline-artifacts)
  - [Inference Pipeline Artifacts](#inference-pipeline-artifacts)
  - [Artifact Descriptions](#artifact-descriptions)
- [Parameter Reference](#parameter-reference)
  - [Structure of experimental_plan.yaml](#structure-of-experimental_planyaml)
  - [User Arguments Summary](#summary-of-user-arguments)
  - [Detailed Parameter Reference](#detailed-explanation-of-user-arguments)
- [Usage Tips](#usage-tips)
  - [Data Preparation Tips](#choosing-ground-truth-data-path)
  - [Model Selection](#selecting-a-model)
  - [Data Augmentation](#choosing-data-augmentation-techniques)
  - [Performance Optimization](#balancing-performance-and-resource-usage)
  - [XAI Visualization](#visual-confirmation-of-inference-areas-xai-explanation)

---

## Introduction

### What is Vision Classification?
Vision Classification is a deep learning-based AI content that automatically classifies images using pre-prepared ground truth. It provides solutions for:

- Detecting visual defects in manufacturing processes
- Automating quality inspections
- Identifying defects and determining their causes
- Collecting data during inspection processes
- Retraining AI models based on user-reviewed images

Additionally, it can be used in retail for product classification and recognition to optimize inventory management and enhance customer service. The included explainable AI (XAI) features help users identify image areas contributing to the inference.

### When to Use Vision Classification
VC can be applied to any image-based classification task where images and ground truth labels are available:

* **Product Inspection:** Automate defect inspection by collecting normal and defective image data
* **Automated Inventory Management:** Verify object/product types using image data
* **Object Classification:** Automatically generate tags for objects in user-uploaded photos

### Use Cases
* **Motor Inspection (TBD):** Image-based quality inspection of various electric vehicle components
* **Box Exterior Inspection (TBD):** Inspecting exterior defects of packaging boxes before shipping

### Key Features

#### Lightweight Models with Fast Speed and Low Memory Usage
VC provides high-accuracy deep learning models that don't require extensive training resources:
- Utilizes lightweight models like mobilenetV1 and mobilenetV3
- Includes a high-resolution model developed by advanced analysts
- Supports fast inference through TensorFlow Lite conversion for embedded devices

#### Visual Confirmation of Inference Basis
The XAI features provide explainability for inference results, highlighting defect locations in images.

#### Flexible Model Selection
Users can easily switch between lightweight and high-resolution models by changing the model name in the experimental plan.

---

## Quick Start Guide

### Installation
```bash
git clone 
```

### Data Preparation
1. Prepare a GroundTruth.csv file with image paths and corresponding labels
2. Ensure all images have consistent size (png or jpeg format)

**Example GroundTruth.csv:**
| label  | image_path                |
| ------ | ------------------------- |
| label1 | ./sample_data/sample1.png |
| label2 | ./sample_data/sample2.png |
| label1 | ./sample_data/sample3.png |

### Parameter Configuration
1. Modify data paths in `vc/experimental_plan.yaml`:
   ```yaml
   external_path:
       - load_train_data_path: ./solution/sample_data/train/
       - load_inference_data_path: ./solution/sample_data/test/
   ```

2. Update image shape parameters to match your data:
   ```yaml
       - step: train
         args:
           - model_type: mobilenetv1
             input_shape: [28, 28, 1]
   ```

3. For advanced settings (XAI, augmentation, training parameters), see the detailed parameter section.

### Execution
* Run via terminal or Jupyter notebook
* Results include trained model files, prediction results, and performance charts
* Note: Currently requires CPU with AVX support

---

## Pipeline Architecture

### VC Pipeline
The pipeline consists of functional units (assets) combined to perform tasks:

#### Train Pipeline
```
Input - Readiness - Train
```
#### Inference Pipeline
```
Input - Readiness - Inference - Output
```

### Pipeline Components

#### Input Asset
- Reads Ground Truth data (image paths and labels)
- During inference, generates file with image information if none exists
- Passes data to the next asset in the pipeline

#### Readiness Asset
- Checks data quality of Ground Truth data
- Raises warnings for missing values
- Proceeds with training by excluding missing data

#### Modeling (Train) Asset
- Reads Ground Truth data from the input
- Loads images into memory using provided paths
- Wraps images in a `tensorflow.Data.dataset` object
- Initializes and trains the model based on experimental plan parameters
- Saves performance metrics and inference results

#### Modeling (Inference) Asset
- Reads images based on paths from the input asset
- Loads the best model saved during training
- Performs inference to predict labels
- Stores results and performance metrics (if ground truth available)
- Saves inference summary for deployment

#### Output Asset
- Customizes inference summary based on solution domain requirements
- For single images: `result` contains predicted label, `score` contains model confidence
- For multiple images: `result` contains count of each type, `score` contains average confidence

---

## Data Preparation

### Data Requirements

#### Mandatory Requirements
| Index | Item                                               | Spec.                  |
| ----- | -------------------------------------------------- | ---------------------- |
| 1     | Single label per image                             | Yes                    |
| 2     | Adherence to class folder format                   | Yes                    |
| 3     | Number of images per class (without duplicates)    | 10-10,000              |
| 4     | Total number of classes                            | 0-20                   |
| 5     | Fixed number of channels: 3 (RGB) or 1 (grayscale) | Yes                    |
| 6     | Image resolution                                   | 32x32-1024x1024 pixels |
| 7     | Inference interval for new images                  | ≥ 10 seconds           |
| 8     | Training interval                                  | ≥ 12 hours             |

#### Additional Requirements for Optimal Performance
| Index | Item                                                                                                         | Spec.                                                |
| ----- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| 1     | Unique image names regardless of class                                                                       | Yes                                                  |
| 2     | Number of images per class (without duplicates)                                                              | 100-800                                              |
| 3     | Total number of classes                                                                                      | 0-20                                                 |
| 4     | Image resolution                                                                                             | 224x224-1280x720 pixels                              |
| 5     | Region of Interest (ROI) size                                                                                | ≥ 10x10 pixels (224x224 basis)                       |
| 6     | Fixed product position/orientation/distance from camera                                                      | Rotation ± 3°, Translation ≤ 1 pixel (224x224 basis) |
| 7     | Sharp image focus                                                                                            | Yes                                                  |
| 8     | No new untrained classes during AI operation                                                                 | Yes                                                  |
| 9     | Consistent image capture environment during training/validation/operation (brightness, lighting, background) | Yes                                                  |

### Preparing Training Data
1. Prepare image data in .png or .jpg format with consistent shape (e.g., 1024x1024 pixels, 3 channels)
2. Create a ground truth file in tabular form with image paths and corresponding labels
3. Ensure each label type has at least 100 images for a stable model
4. Note: Multi-class classification is supported, but multi-label classification is not

**Example of GroundTruth.csv Training Dataset**
| label  | image_path    |
| ------ | ------------- |
| label1 | ./image1.png  |
| label2 | ./image1.jpeg |
| label1 | ./image2.jpeg |

### Example of Input Data Directory Structure
```bash
./{train_folder}/
    └ train_data1.csv
    └ train_data2.csv
    └ train_data3.csv
    └ image1.png
    └ image1.jpeg
    └ image2.jpeg
./{inference_folder}/
    └ data_a.csv
    └ data_b.csv
    └ image_test1.png
    └ image_test1.jpeg
    └ /{folder1}/
        └ image_test2.jpeg
```

**When there is no CSV file for inference data**
```bash
./{inference_folder}/
    └ image_test1.png
    └ image_test1.jpeg
    └ /{inference_folder}/
        └ image_test2.jpeg
```

---

## Features In Depth

### Train Pipeline Features

#### Read Ground Truth File
Reads the Ground Truth file to identify image paths and labels.

#### Check Data Quality
Validates the Ground Truth file for missing values, issues warnings, and proceeds by excluding missing data.

#### Load Image Files and Generate Dataset
- Reads image files based on paths in the Ground Truth file
- Loads images into memory and converts to `dataset` format
- Performs preprocessing (scaling, resizing) based on parameters

#### Split Train/Validation Data
Randomly splits dataset into 80% training and 20% validation data to prevent overfitting.

#### Add Data Augmentation
- Uses RandAugment method ([Paper](https://arxiv.org/abs/1909.13719))
- Allows excluding specific transformations not suitable for the task
- Adjustable parameters: `aug_magnitude` (0-10) and `num_aug` (1-16)
- Enable via `rand_augmentation: True`

Example augmentation table:
|index|image_path|label|1st Augmentation|2nd Augmentation|
|---|---|---|---|---|
|1|OK_image1.jpg|OK|color|autocontrast|
|2|NG_image1.jpg|NG|invert|color|
|3|OK_image2.jpg|OK|shearX|posterize|

#### Initialize Model
- `mobilenetv1`: Stable training and inference with low resource requirements
- `high_resolution`: Enhanced mobilenetV3-Small with CBAM ([Paper](https://arxiv.org/abs/1807.06521)) and ECA ([Paper](https://arxiv.org/abs/1910.03151)) layers

#### Fit Model on Training Data
Trains the model on training and validation datasets for the specified epochs, using checkpoints to save the best model.

#### Evaluate Score and Save Outputs
Calculates performance metrics and saves:
- Summary (`eval.json`)
- Inference results (`prediction.csv`)
- Confusion matrix (`confusion.csv`)
- Inference summary (`inference_summary.yaml`)

### Inference Pipeline Features

#### Read Inference File or Generate It
Reads the inference file if available or generates it using image paths.

#### Load Image Files
- Loads images from provided paths
- Creates dataset object for inference
- If `do_xai` is enabled, saves XAI results with filenames: `original_filename_xai_class_label.png`

#### Predict Label
- Loads appropriate model for inference
- Uses Keras (h5) model if `do_xai` is enabled, otherwise TensorFlow Lite (tflite)
- Inference time: approximately 1 second per image

#### Save Outputs
Saves inference results similar to training pipeline.

---

## Artifacts

### Train Pipeline Artifacts
```bash
./cv/train_artifacts/
    └ models/train/
        └ model.h5
        └ model.tflite
        └ params.json
    └ output/
        └ prediction.csv
    └ extra_output/train/
        └ eval.json
        └ confusion.csv
```

### Inference Pipeline Artifacts
```bash
./cv/inference_artifacts/
    └ output/inference/
        └ prediction.csv
        └ {imagefilename}.png OR {imagefilename}_xai_class{predicted_class}.png
    └ extra_output/inference/
        └ eval.json
        └ confusion.csv
        └ xai_result/
            └ {imagefilename}_xai_class{predicted_class}.png
    └ score/
        └ inference_summary.yaml
```

### Artifact Descriptions

#### model.h5
Trained Keras model file.

#### model.tflite
Trained TensorFlow Lite model suitable for embedded environments.

#### params.json
JSON file containing parameters used during training.

#### prediction.csv
Contains model predictions with columns:
- label: Ground Truth label
- pred_label: Predicted label
- prob_{label}: Probability for each label

#### confusion.csv
Confusion matrix of Ground Truth and predicted labels.

#### eval.json
Classification report representing training performance.

#### {imagefilename}.png OR {imagefilename}_xai_class{predicted_class}.png
Original image or image with highlighted XAI areas (if `do_xai` is enabled).

#### inference_summary.yaml
Summary of inference results including predicted label and confidence level.

---

## Parameter Reference

### Structure of experimental_plan.yaml

The file contains settings needed to run the pipeline. Key sections include:

#### Data Path Configuration (`external_path`)
```yaml
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path: ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```

#### User Parameters (`user_parameters`)
```yaml
user_parameters:
    - train_pipeline:
        - step: input
          args:
            - file_type: csv
            ...
          ui_args:
            ...
```

### Summary of User Arguments

| Asset Name | Argument Type | Argument Name                          | Default       | Description                                                         | Required | ui_args |
|:----------:|:-------------:|:--------------------------------------:|:-------------:|:-------------------------------------------------------------------:|:--------:|:-------:|
| Input      | Custom        | file_type                | csv           | File extension of input data                                      | X        | O       |
| Input      | Custom        | encoding                  | utf-8         | Encoding type of input data                                       | X        | O       |
| Readiness  | Custom        | y_column                  | label         | Column name for classification label                               | X        | O       |
| Readiness  | Custom        | path_column            | image_path    | Column name for image path                                        | X        | O       |
| Readiness  | Custom        | check_runtime        | False         | Check execution time during experiment                             | X        | O       |
| Readiness  | Custom        | check_memory          | False         | Check memory usage during experiment                               | X        | O       |
| Train      | Required      | input_shape            | [28,28,1]     | Size of input image                                                | O        | O       |
| Train      | Required      | resize_shape          | [32,32,3]     | Image size for model training                                      | O        | O       |
| Train      | Custom        | model_type              | mobilenetv1   | Training model selection                                           | X        | O       |
| Train      | Custom        | rand_augmentation| False         | Whether to apply RandAugment                                       | X        | O       |
| Train      | Custom        | exclude_aug_lst    | []            | Exclude inappropriate transformations                              | X        | O       |
| Train      | Custom        | epochs                      | 10            | Number of training epochs                                          | X        | X       |
| Train      | Custom        | batch_size              | 64            | Batch size during training                                         | X        | X       |
| Train      | Custom        | train_ratio            | 0.8           | Ratio of data for training                                         | X        | X       |
| Train      | Custom        | num_aug                    | 2             | Number of random transformations per image                         | X        | X       |
| Train      | Custom        | aug_magnitude        | 10            | Strength of augmentation                                            | X        | X       |
| Inference  | Required      | do_xai                      | False         | Whether to use XAI                                                  | O        | X       |
| Inference  | Custom        | xai_class                | auto          | Class to be analyzed by XAI                                        | X        | X       |
| Inference  | Custom        | mask_threshold      | 0.7           | Sensitivity for XAI results                                         | X        | X       |
| Inference  | Custom        | layer_index            | 2             | Number of layers for analysis during inference                      | X        | X       |

### Detailed Explanation of User Arguments

#### Input Asset Parameters

##### file_type
- **Default**: csv
- **Description**: File extension of the input data
- **Usage**: `file_type: csv`

##### encoding
- **Default**: utf-8
- **Description**: Encoding type of the input data
- **Usage**: `encoding: utf-8`

#### Readiness Asset Parameters

##### y_column
- **Default**: label
- **Description**: Column name for classification label in Ground Truth file
- **Usage**: `y_column: image_label`

##### path_column
- **Default**: image_path
- **Description**: Column name for image path in Ground Truth file
- **Usage**: `path_column: Path`

##### check_runtime
- **Default**: False
- **Description**: Check execution time during modeling
- **Usage**: `check_runtime: True`

##### check_memory
- **Default**: False
- **Description**: Check memory usage during modeling
- **Usage**: `check_memory: True`

#### Train Asset Parameters

##### input_shape
- **Default**: [28,28,1]
- **Description**: Size of the input image (all images must be same size)
- **Usage**: `input_shape: [32,32,1]`

##### resize_shape
- **Default**: [32,32,3]
- **Description**: Image size for model training
- **Usage**: `resize_shape: [32,32,1]`

##### model_type
- **Default**: mobilenetv1
- **Options**: mobilenetv1, high_resolution
- **Description**: Select the training model
- **Usage**: `model_type: high_resolution`

##### rand_augmentation
- **Default**: False
- **Description**: Enable RandAugment data augmentation
- **Usage**: `rand_augmentation: True`

##### exclude_aug_lst
- **Default**: []
- **Options**: solarizeadd, invert, cutout, autocontrast, equalize, rotate, solarize, color, posterize, contrast, brightness, sharpness, shearX, shearY, translateX, translateY
- **Description**: Exclude unsuitable transformations when using rand_augmentation
- **Usage**: `exclude_aug_lst: [invert, color]`

##### epochs
- **Default**: 10
- **Range**: 1-1000
- **Description**: Number of training epochs
- **Usage**: `epochs: 50`

##### batch_size
- **Default**: 64
- **Description**: Number of data points processed at once during training
- **Usage**: `batch_size: 32`

##### train_ratio
- **Default**: 0.8
- **Description**: Ratio of data used for training vs. validation
- **Usage**: `train_ratio: 0.75`

##### num_aug
- **Default**: 2
- **Description**: Number of random transformations per image
- **Usage**: `num_aug: 3`

##### aug_magnitude
- **Default**: 10
- **Range**: 0-10
- **Description**: Strength of augmentation
- **Usage**: `aug_magnitude: 8`

#### Inference Asset Parameters

##### do_xai
- **Default**: False
- **Description**: Enable XAI visualization
- **Usage**: `do_xai: True`

##### xai_class
- **Default**: auto
- **Description**: Class to analyze with XAI
- **Usage**: `xai_class: defect`

##### mask_threshold
- **Default**: 0.7
- **Description**: Sensitivity for XAI results
- **Usage**: `mask_threshold: 0.6`

##### layer_index
- **Default**: 2
- **Description**: Number of layers to analyze during inference
- **Usage**: `layer_index: 3`

---

## Usage Tips

### Choosing Ground Truth Data Path
- Place GT file and images in same folder
- Use `image_path` and `label` columns
- Ensure consistent image sizes
- Preprocess images for uniform size through cropping or resizing

### Selecting a Model
- `mobilenetv1`: Standard model for most datasets (recommended `resize_shape: [224, 224, 3]`)
- `high_resolution`: For higher resolution requirements

### Choosing Data Augmentation Techniques
- Enable `rand_augmentation: True` for datasets with fewer than 1000 images per class
- Exclude transformations that might negatively impact performance
- Configure `aug_magnitude` and `num_aug` based on dataset characteristics

### Balancing Performance and Resource Usage
- Increase epochs for large, complex datasets
- Use checkpoints to save best-performing model
- Adjust batch size based on available resources
- Reduce train_ratio for abundant data requiring more validation

### Visual Confirmation of Inference Areas (XAI Explanation)
- Enable `do_xai` to visualize areas contributing to predictions
- Set `xai_class` to class of interest (e.g., defect class)
- Adjust `mask_threshold` for more extensive coverage
- Increase `layer_index` for deeper layer analysis (requires more resources)

---

**VC Version: 1.5.1**
