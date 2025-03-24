# Visual Anomaly Detection (VAD) Comprehensive Guide

<div align="right">Updated 2024.06.11</div><br/>

## Table of Contents
- [Introduction](#what-is-visual-anomaly-detection)
- [Use Cases](#when-to-use-visual-anomaly-detection)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Feature Overview](#feature-overview)
  - [Pipeline Components](#pipeline-components)
  - [Usage Tips](#tips-for-use)
  - [Feature Details](#feature-details)
- [Data Preparation](#data-preparation)
  - [Requirements](#data-requirements)
  - [Input Structure](#preparation-of-training-data)
- [Artifacts](#artifacts)
  - [Train Pipeline Artifacts](#train-pipeline)
  - [Inference Pipeline Artifacts](#inference-pipeline)
  - [Artifact Descriptions](#detailed-descriptions-for-each-artifact)
- [Parameter Reference](#experimental_planyaml-description)
  - [Structure](#experimental_planyaml-structure)
  - [User Arguments Summary](#user-arguments-summary)
  - [Detailed Parameter Reference](#user-arguments-detailed-description)

---

## What is Visual Anomaly Detection?
Visual Anomaly Detection (VAD) is an AI content that detects abnormal images by learning normal images. By training with normal images or images of the same type, VAD can alert users when there are images different from the trained ones, without the need for users to create a ground truth, thus reducing the cost of creating ground truth.

---

## When to use Visual Anomaly Detection?
VAD can classify images composed of mostly normal or specific types and a few abnormal or other types. VAD can be applied in any field with only normal or specific type images without ground truth labels. The main application areas include:

- **Manufacturing**: In processes such as appearance inspection of products or vision inspection, where most of the products are normal and there are few abnormal products, abnormal products can be automatically classified and verified without a ground truth.
- **Inventory Management**: Images without ground truth labels can be used to verify that other products are not mixed.
- **Agriculture**: Monitoring crop conditions via smartphone images and detecting diseases, pests, abnormal growth, etc., early on. This can increase the efficiency of crop management.

### Example Applications
- **Bolt Inspection**: An image-based anomaly detection solution for inspecting bolts and the proper assembly of various electric vehicle components.
- **Critical Defect Detection**: An image-based anomaly detection solution for detecting defects in parts produced during mass production of electric vehicle components.
- **Box Appearance Inspection**: A solution for detecting defects in the appearance of packaging boxes in warehouses before products are shipped out.

---

## Key Features

### Excellent Performance
VAD can use the latest models such as PatchCore and FastFlow, which have excellent performance among image-based anomaly detection AI models. PatchCore utilizes pre-trained models, so there is no need for training, reducing training costs, and it maintains stable performance. FastFlow is an image anomaly detection model that utilizes generative models that have shown excellence in various fields recently. Both models are lightweight and enable rapid inference.

### Cost Reduction in Initial Labeling
To use an image-based classification model, a process is required to create ground truth labels by confirming each image. However, since VAD can verify whether an image is normal or of a specific type with only normal or specific type images, the cost of creating ground truth labels can be reduced. In the future, by assigning new abnormal type labels only to images classified as abnormal, a variety of image classification AI models can be quickly switched to, thus enabling more diverse types of image classification AI models.

### Convenient Usability and Easy-to-Understand Inference Evidence
Since most of VAD is automated, it can be easily used with only normal image data collected. It also provides images with marked areas showing the evidence area for why an image was classified as abnormal. Therefore, users can quickly verify whether the AI has been trained properly by examining images with marked evidence areas.

---

## Quick Start

### Installation
For Git code access to AI Contents, refer to ([AI Contents Access](https://mellerikat.com/user_guide/data_scientist_guide/ai_contents#access)).
```bash
git clone https://github.com/mellerikat/alo.git {project_name}
cd {project_name}
pip install -r requirements.txt
git clone https://github.com/mellerikat-aicontents/Vision-Anomaly-Detection.git solution
```

### Set Required Parameters
1. Modify the data paths in `solution/experimental_plan.yaml` to user paths.
   ```yaml
   external_path:
       - load_train_data_path: ./solution/sample_data/train/    # Change to user data path
       - load_inference_data_path: ./solution/sample_data/test/ # Change to user data path  
   ```

2. Enter the file_type and encoding in 'args' under 'step: Input' according to the train data.
   ```yaml
       - step: Input
         args:
           - file_type: csv    # Specify the file extension of the input data.
             encoding: utf-8   # Specify the encoding type of the input data.
   ```

3. Configure readiness settings:
   ```yaml
       - step: Readiness
         args:
             train_validate_column:     # Specify the column that distinguishes between train and validation.
             validation_split_rate: 0.1    # Specify the proportion of validation data if train_validate_column doesn't exist.
             threshold_method: F1    # Select method for determining OK and NG during validation.
   ```

4. Configure train settings:
   ```yaml
       - step: Train
         args:
           - model_name: fastflow    # Select model to be used for VAD.
             img_size: 256    # Set image size to be used during training.
             percentile: 0.8    # Select percentile of anomaly score to classify NG if threshold_method is Percentile.
   ```

5. Configure inference settings:
   ```yaml
       - step: Inference
         args:
           - inference_threshold: 0.5    # Specify threshold of anomaly score to classify as abnormal.
   ```

### Run
Navigate to the directory where ALO is installed in the terminal, then execute the command:
```bash
python main.py
```

---

## Feature Overview
### VAD Pipeline
The pipeline of AI Contents consists of a combination of assets at the feature level. The Train pipeline consists of 3 assets, while the Inference pipeline consists of 4 assets.

#### Train Pipeline
```bash
Input - Readiness - Train
```  
#### Inference pipeline
```bash
Input - Readiness - Inference - Output
```

### Pipeline Components

#### Input
VAD operates on tabular data in the form of image paths and correct labels, known as Ground Truth data, during training. Since VAD can be trained with only normal images, if there are only normal images available, you can write the label names indicating normality (e.g., OK, good, normal, etc.). The input asset reads the Ground Truth data and passes it to the next asset. If there is no file containing information about the images during the inference stage, it simply passes on the image paths to create them for inference.

#### Readiness
Checks the data quality of the Ground Truth data. If there are missing values in the image path or label columns, it issues a warning and proceeds with training after excluding the missing values. It also checks and possibly modifies suitable argument settings based on the data structure.

#### Modeling (Train)
The train asset in VAD reads the Ground Truth data passed from the input and loads the training and validation data into PyTorch dataloaders. Depending on the model name specified by the user in the experimental_plan.yaml, VAD creates and trains either the FastFlow or PatchCore model supported by VAD. Once training is complete, if there are abnormal data in the validation data, it outputs the predicted results in the output.csv file, along with confusion matrix, classification report, and performance table and graphs that change when the anomaly_threshold is adjusted.

#### Modeling (Inference)
In the inference asset, it reads the path of the inference data received from the input asset, loads the image, loads the best model saved from the previous training stage, predicts the label, and saves it. If it's a stage for experimentation rather than operation and the correct label is in the file, it saves the performance. It also saves the file for operation, inference_summary.yaml.

#### Output
The required information may vary for each task. Currently, if the inference data infers for a single image, the result contains the inferred label, and the score contains the model's certainty (1-Shannon entropy). If it's inferring for multiple images, the result contains the count for each type, and the score contains the average certainty of the model. Currently, up to 32 characters can be stored in the result due to character limitations.

---

## Tips for Use
### Preparing Ground Truth Files and Image Data
Put the Ground Truth (GT) data and images in the same folder. ALO copies the GT file and images to the execution environment at once. The GT data should consist of image paths (image_path) and labels (label). If you want to differentiate between training and validation datasets, separate the data into train and valid, create new items, and write the corresponding item name in train_validate_column.

It's advisable to have consistent image sizes for all data. If there are images of different sizes, performance may suffer, so it's best to crop or resize them to a consistent size in advance. The data should consist of only normal data or mostly normal data with a few abnormal data.

### Selecting Method for Setting Anomaly Detection Thresholds
While VAD can only be trained with normal data, if there are a few abnormal data, a more rigorous anomaly detection model can be created. If only normal data is available, the threshold_method setting method provided by VAD is Percentile. Similarly, if there are very few abnormal data, it's also good to set it to Percentile. Depending on the number of normal images, if there are slightly more than 5 abnormal data, using the F1 method can find a suitable anomaly_threshold.

### Model Selection
VAD supports two models: FastFlow and PatchCore. Usually, PatchCore has better performance and requires less training time compared to FastFlow, but the inference time is longer, and slightly more RAM memory is needed. Choose according to the operational infrastructure environment.

### Setting Training Conditions
The training conditions that can be set in VAD are mainly image size (`img_size`), batch size (`batch_size`), and maximum epochs(`max_epochs`). To achieve faster and lighter inference, VAD performs resizing based on the entered image size. For FastFlow, the image size is fixed at 256 pixels wide and 256 pixels high, as the model requires a fixed image format. For PatchCore, as it internally cuts patches for inference, it can accept rectangular inputs.

---

## Feature Details
Detailed explanation for utilizing various functions of Visual Anomaly Detection.

### Train pipeline: input asset
#### Read Ground Truth file
Reads the Ground Truth file and identifies image paths and labels in the `input asset`.

### Train pipeline: readiness asset
#### Check data quality
Reads the Ground Truth file, verifies the column names input as arguments, removes missing values, and verifies the existence of images in the image paths. It also checks whether the labels for normal images are correct. VAD divides the training/validation datasets and utilizes the validation dataset to perform training optimally.

### Train pipeline: modeling(train) asset
#### Load image file
Reads images from the data in `image_path` in the Ground Truth.

#### Training parameters
If there are abnormal data in the validation data, the `anomaly_threshold`, which determines abnormality, is trained using the F1-score maximization method. If there are no abnormal data, training proceeds according to the `percentile` argument set in advance, and the anomaly_threshold value is set.

#### Train
Proceeds with training according to the set parameters.

#### Save outputs
Appends predicted results and probabilities for the inference results to the Ground Truth file and saves them in `output.csv`. If the validation data contains abnormal data, confusion matrix and classification report are saved. Both abnormal images and images inferred as abnormal are saved in the `extra_output` folder.

### Inference pipeline: modeling(inference) asset
#### Load image file
Loads the image from the `image_path` in the inference file received and creates a dataloader object for inference.

#### Predict label
Uses the trained model to generate images with marked abnormal areas.

#### Save outputs
The outputs in the inference pipeline are structured the same as in training. Images of abnormal areas (XAI images) are all saved in the `extra_output` folder.

### Inference pipeline: output asset
#### Modify inference summary
Inference summary can be modified according to the Solution of each domain.

---

## Data Preparation
### Preparation of Training Data
1. Prepare normal (or same type) image data in png or .jpg format with the same shape (e.g., 1024x1024, 3 channels).
2. Prepare a tabular-formatted Ground Truth file consisting of image paths and corresponding labels (all normal labels due to the same type).
3. Including some abnormal (or a few different types) image data in the training data leads to a more accurate AI model.
4. The Ground Truth file should be prepared in the following format.
5. There is no need to unify labels for abnormal images with multiple types when preparing the Ground Truth file.
6. If there are validation data desired, input the column name in the `train_validation_column` argument so that images designated for training will be used for training, and other images will be used for validation to assess performance.
7. Prepare Ground Truth files and image files in the same folder.
8. For inference data, prepare either a Ground Truth file like training data or prepare the desired images for inference within a single folder.
9. The types of training data are composed of various types, but classification is performed only into two types: normal and abnormal. 

**Example of Training Dataset**

|  label | image_path  | train_val_split |
|:---:|:---:|:---:|
|  good | ./image0.png  | train |
|  bad | ./image1.png  | valid |
|  good | ./image2.png  | train |
|  good | ./good/image0.png  | train |
|  good | ./good/image1.png  | train |
|  good | ./good/image0.jpg  | train |
|  good | ./good/image1.jpg  | train |

**Example input data directory structure**
```bash
/project_folder/vad/solution/sample_data
└ inference
   └ 10479_0.png
   └ 1080_7.png
   └ 10735_7.png
└ train
   └ train.csv
   └ abnormal
      └ 10019_0.png
      └ 10024_0.png
      └ 10097_0.png
   └ good
      └ 10395_7.png
      └ 10034_7.png
      └ 10852_7.png
```

---

## Data Requirements

### Mandatory Requirements
Input data must meet the following conditions.

|  index | item  | spec. |
|:---:|:---:|:---:|
|  1 | Sufficient number of normal or single-type images  | 100~50000 |
|  2 | Compliance with Ground Truth file naming format (label, image_path)  | Yes |
|  3 | Fixed number of channels: 3 channels or 1 channel (gray)  | Yes |
|  4 | Resolution  | 32x32 ~ 1920x1920 pixels |
|  5 | Inference interval for new images of at least 3 seconds  | Yes |
|  6 | Time interval required for training is at least 6 hours  | Yes |

### Additional Requirements
Conditions necessary to ensure minimal performance. Even if these conditions are not met, the algorithm will run, but performance will not be verified.

|  index | item  | spec. |
|:---:|:---:|:---:|
|  1 | Unique names for images regardless of type  | Yes |
|  2 | Obtain a small number of abnormal images to verify AI performance  | 10 images per abnormal type |
|  3 | Resolution  | 224x224 ~ 1024x1024 pixels |
|  4 | ROI (Region Of Interests) size  | At least 15x15 pixels (based on 224x224) |
|  5 | Fixed product position/direction/distance from camera  | Allowable errors: rotation within +/- 3 degrees, translation within 1 pixel (based on 224x224) |
|  6 | Image focus should be sharp  | Yes |
|  7 | Image measurement environment: Maintain as consistent and stable environment as possible during training/validation/operation (brightness, illumination, background, etc.)  | Yes |  

---

## Artifacts

When training/inference is executed, the following artifacts are generated.

### Train Pipeline
```bash
/project_folder/vad/train_artifacts
└ models
   └ readiness
      └ readiness_config.json
   └ train
      └ train_config.json
      └ model.pickle
      └ patchcore_model.ckpt
      └ best_model_path.json
└ log
   └ experimental_history.json
   └ pipeline.log
   └ process.log
└ extra_output
   └ train
      └ threshold_test_result.png
      └ images
         └ abnormal_OK_WIN_20240521_11_59_25_Pro.jpg
         └ abnormal_OK_WIN_20240521_11_59_01_Pro.jpg
         └ abnormal_OK_WIN_20240521_12_01_16_Pro.jpg
      └ threshold_test_result.csv
└ score
└ output
   └ validation_classification_report.json
   └ output.csv
   └ validation_confusion_matrix.csv
   └ validation_score.csv
└ report
```

### Inference Pipeline
```bash
/project_folder/vad/inference_artifacts
└ log
   └ experimental_history.json
   └ pipeline.log
   └ process.log
└ extra_output
   └ inference
      └ Patchcore
         └ latest
            └ images
               └ NG2.jpg
               └ OK4.jpg
               └ OK3.jpg
└ score
   └ inference_summary.yaml
└ output
   └ output.csv
   └ NG2.jpg
```

### Detailed descriptions for each artifact

#### \{model_name\}_model.ckpt
This is the model file generated after training is completed.

#### output.csv
A tabular file containing the results and probabilities of the inference.

#### validation_confusion_matrix.csv
A file storing the confusion matrix results for the validation data in the training pipeline. It is saved only if there are abnormal data in the validation data.

#### validation_score.csv
A file recording the performance of the validation data in the training pipeline. It includes accuracy, precision, recall, F1-score, and AUROC.

#### \{imagefilename\}.png
An image showing the abnormal areas. In the training pipeline, it saves images that were misclassified and all abnormal images. In the inference pipeline, original images for retraining are saved in the output folder.

#### inference_summary.yaml
A summary file for the inference results. It is used in edge conductor and consists of scores, results, probabilities, etc.

#### threshold_test_result.csv
A table showing the performance changes of the validation data when the threshold criteria for anomaly detection are adjusted.

#### threshold_test_result.png
A graph showing the performance changes when the threshold criteria for anomaly detection are adjusted.

---

## experimental_plan.yaml Description
To apply AI Contents to the data you have, you need to enter information about the data and the Content features to be used in the experimental_plan.yaml file. When you install AI Contents in the solution folder, you can check the experimental_plan.yaml file provided for each content under the solution folder.

### experimental_plan.yaml Structure
experimental_plan.yaml contains various setting values necessary to run ALO. By modifying the 'data path' and 'user arguments' parts, you can use AI Contents immediately.

#### Enter Data Path (`external_path`)
- The parameters of `external_path` are used to specify the path to load or save files. If `save_train_artifacts_path` and `save_inference_artifacts_path` are not entered, the modeling outputs are saved in the default directories `train_artifacts` and `inference_artifacts`.

```yaml
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path:  ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```

|Parameter Name| DEFAULT| Description and Options|
|---|---|---|
| load_train_data_path | ./sample_data/train/ | Enter the folder path where the training data is located (do not enter the csv file name). |
| load_inference_data_path | ./sample_data/test/ | Enter the folder path where the inference data is located (do not enter the csv file name). |

#### User Parameters (`user_parameters`)
- Below `user_parameters`, `step` represents the asset name. `step: input` below indicates the input asset stage.
- `args` represents the user arguments of the input asset (`step: input`). User arguments are parameters related to data analysis provided for each asset.

```yaml
user_parameters:
    - train_pipeline:
        - step: input
          args: 
            - file_type
            ...
          ui_args:
            ...
```

## User Arguments Explanation
### What are User Arguments?
User arguments are parameters for configuring the operation of each asset in the experimental_plan.yaml under the `args` of each asset step. They are used to customize various functionalities of the AI Contents pipeline for your data. User arguments are divided into "Required arguments," which are predefined in the experimental_plan.yaml, and "Custom arguments," which users add based on the provided guide.

## User Arguments Summary

Here's a summary of key user arguments for VAD:

| Asset Name | Argument Name | Default | Description | User Setting Requirement | 
|:----------:|:-------------:|:-------:|:-----------:|:-----------------------:|
| Input | file_type | csv | Enter the file extension of the input data. | O | 
| Input | encoding | utf-8 | Enter the encoding type of the input data. | X | 
| Readiness | ok_class | - | Enter the class representing 'ok' in the y_column. | X | 
| Readiness | train_validate_column | - | Enter the column that distinguishes between train and validation. | X | 
| Readiness | validation_split_rate | 0.1 | If train_validate_column does not exist, this rate generates validation from the input train data. | X | 
| Readiness | threshold_method | F1 | Selects the method for determining OK and NG during validation. | X | 
| Train | model_name | fastflow | Selects the model to be used for VAD. Possible values: fastflow, patchcore | O | 
| Train | img_size | \[256,256\] | Sets the image size for resizing during image training. | X | 
| Train | batch_size | 4 | Sets the batch size during training and validation. | X | 
| Train | max_epochs | 15 | Sets the maximum number of epochs for training. | X | 
| Train | accelerator | cpu | Selects whether to run based on CPU or GPU. | X | 
| Train | percentile | 0.8 | Selects the percentile of the anomaly score to judge as NG when threshold_method is Percentile. | X | 
| Inference | inference_threshold | 0.5 | Threshold of anomaly score for being judged as abnormal. | X | 
| Inference | save_anomaly_maps | False | Whether to save XAI image results. | X | 

For detailed parameter descriptions and additional parameters, please refer to the full VAD documentation.

**VAD Version: 1.0.0**
