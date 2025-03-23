---

# Visual Anomaly Detection (VAD) Documentation

<div align="right">Updated 2024.06.11</div><br/>

**VAD Version: 1.0.0**

---

## VAD Features

### Feature Overview
#### VAD Pipeline
The pipeline of AI Contents consists of a combination of assets at the feature level. The Train pipeline consists of 3 assets, while the Inference pipeline consists of 4 assets.

##### Train Pipeline
```bash
Input - Readiness - Train
```  
##### Inference Pipeline
```bash
Input - Readiness - Inference - Output
```

##### Input
VAD operates on tabular data in the form of image paths and correct labels, known as Ground Truth data, during training. Since VAD can be trained with only normal images, if there are only normal images available, you can write the label names indicating normality (e.g., OK, good, normal, etc.). The input asset reads the Ground Truth data and passes it to the next asset. If there is no file containing information about the images during the inference stage, it simply passes on the image paths to create them for inference. As shown in the diagram above, in the training pipeline, it is passed as train, and in the inference pipeline, it is passed as inference asset.

##### Readiness
Checks the data quality of the Ground Truth data. If there are missing values in the image path or label columns, it issues a warning and proceeds with training after excluding the missing values. It also checks and possibly modifies suitable argument settings based on the data structure.

##### Modeling (Train)
The train asset in VAD reads the Ground Truth data passed from the input and loads the training and validation data into PyTorch dataloaders. Depending on the model name specified by the user in the `experimental_plan.yaml`, VAD creates and trains either the FastFlow or PatchCore model supported by VAD. Once training is complete, if there are abnormal data in the validation data, it outputs the predicted results in the `output.csv` file, along with confusion matrix, classification report, and performance table and graphs that change when the `anomaly_threshold` is adjusted. Depending on the AIOps operational situation, where a model maximizing a specific performance may be needed, you can infer and set an appropriate threshold value based on the table.

##### Modeling (Inference)
In the inference asset, it reads the path of the inference data received from the input asset, loads the image, loads the best model saved from the previous training stage, predicts the label, and saves it. If it's a stage for experimentation rather than operation and the correct label is in the file, it saves the performance. It also saves the file for operation, `inference_summary.yaml`.

##### Output
The required information may vary for each task. Currently, if the inference data infers for a single image, the result contains the inferred label, and the score contains the model's certainty (1-Shannon entropy). If it's inferring for multiple images, the result contains the count for each type, and the score contains the average certainty of the model. Currently, up to 32 characters can be stored in the result due to character limitations.

---

### Tips for Use
#### Preparing Ground Truth Files and Image Data
Put the Ground Truth (GT) data and images in the same folder. ALO copies the GT file and images to the execution environment at once. The GT data should consist of image paths (`image_path`) and labels (`label`). If you want to differentiate between training and validation datasets, separate the data into train and valid, create new items, and write the corresponding item name in `train_validate_column`. Any other items in the GT data are not used for analysis. It's advisable to have consistent image sizes for all data. If there are images of different sizes, performance may suffer, so it's best to crop or resize them to a consistent size in advance. The data should consist of only normal data or mostly normal data with a few abnormal data. Various types of abnormal data can be used. For example, if you want to distinguish between cats and other animals, prepare about 90% cat images and the remaining 10% can be other animals like dogs, mice, or cat dolls. Of course, preparing abnormal data that may occur in real operational environments can prevent performance degradation in operational situations.

#### Selecting Method for Setting Anomaly Detection Thresholds
While VAD can only be trained with normal data, if there are a few abnormal data, a more rigorous anomaly detection model can be created. Strictly speaking, even if abnormal data are not used for training, the `anomaly_threshold`, a numerical value used to determine abnormality, can be adjusted. If significant impact on performance is expected without specific settings, the method of automatic setting may change, but if set according to the situation, an appropriate method suitable for VAD operational situations can be specified. If only normal data is available, the `threshold_method` setting method provided by VAD is `Percentile`. Similarly, if there are very few abnormal data, it's also good to set it to `Percentile`. Depending on the number of normal images, if there are slightly more than 5 abnormal data, using the `F1` method can find a suitable `anomaly_threshold`. By using `percentile` in the training argument, you can determine the level of the `anomaly_score` to be determined as abnormal if only normal images are available.

#### Model Selection
VAD supports two models: FastFlow and PatchCore. Usually, PatchCore has better performance and requires less training time compared to FastFlow, but the inference time is longer, and slightly more RAM memory is needed. Choose according to the operational infrastructure environment.

#### Setting Training Conditions
The training conditions that can be set in VAD are mainly image size (`img_size`), batch size (`batch_size`), and maximum epochs (`max_epochs`). To achieve faster and lighter inference, VAD performs resizing based on the entered image size. For FastFlow, the image size is fixed at 256 pixels wide and 256 pixels high, as the model requires a fixed image format. For PatchCore, as it internally cuts patches for inference, it can accept rectangular inputs. Increasing the batch size, i.e., the number of images trained at once, can lead to faster training completion and improved training stability, but it also consumes more RAM memory of the training infrastructure. The maximum number of epochs is an argument required only for FastFlow, where in deep learning algorithms, training is repeated for the specified number of epochs. If the number of epochs is insufficient, under-fitting may occur, leading to inadequate performance. Increasing the number of epochs will increase the training time, but with the EarlyStopping feature, if the performance is achieved before the specified number of times, the training will be completed.

---

### Feature Details
Detailed explanation for utilizing various functions of Visual Anomaly Detection.

#### Train Pipeline: Input Asset
##### Read Ground Truth File
Reads the Ground Truth file and identifies image paths and labels in the `input asset`.

#### Train Pipeline: Readiness Asset
##### Check Data Quality
Reads the Ground Truth file, verifies the column names input as arguments, removes missing values, and verifies the existence of images in the image paths. It also checks whether the labels for normal images are correct. VAD divides the training/validation datasets and utilizes the validation dataset to perform training optimally. If there is no item to differentiate the training/validation datasets, it creates one and ensures it is separated in the manner required by VAD. VAD can be used with either only normal images or mostly normal images and a few abnormal images. In `readiness`, based on the ratio of normal/abnormal data and the minimum number of abnormal data, it changes to the appropriate user arguments (`threshold_method` and `monitor_metric`) for validation data.

#### Train Pipeline: Modeling (Train) Asset
##### Load Image File
Reads images from the data in `image_path` in the Ground Truth.

##### Training Parameters
If there are abnormal data in the validation data, the `anomaly_threshold`, which determines abnormality, is trained using the F1-score maximization method. If there are no abnormal data, training proceeds according to the `percentile` argument set in advance, and the `anomaly_threshold` value is set. For FastFlow, if the training is insufficient, `max_epochs` can be increased to ensure sufficient training.

##### Train
Proceeds with training according to the set parameters.

##### Save Outputs
Appends predicted results and probabilities for the inference results to the Ground Truth file and saves them in `output.csv`. If the validation data contains abnormal data, confusion matrix and classification report are saved. Both abnormal images and images inferred as abnormal are saved in the `extra_output` folder.

#### Inference Pipeline: Modeling (Inference) Asset
##### Load Image File
Loads the image from the `image_path` in the inference file received and creates a dataloader object for inference.

##### Predict Label
Uses the trained model to generate images with marked abnormal areas.

##### Save Outputs
The outputs in the inference pipeline are structured the same as in training. Images of abnormal areas (XAI images) are all saved in the `extra_output` folder.

#### Inference Pipeline: Output Asset
##### Modify Inference Summary
Inference summary can be modified according to the Solution of each domain.

---

## VAD Input and Artifacts

### Data Preparation
#### Preparation of Training Data
1. Prepare normal (or same type) image data in `.png` or `.jpg` format with the same shape (e.g., 1024x1024, 3 channels).
2. Prepare a tabular-formatted Ground Truth file consisting of image paths and corresponding labels (all normal labels due to the same type).
3. Including some abnormal (or a few different types) image data in the training data leads to a more accurate AI model.
4. The Ground Truth file should be prepared in the following format.
5. There is no need to unify labels for abnormal images with multiple types when preparing the Ground Truth file.
6. If there are validation data desired, input the column name in the `train_validation_column` argument so that images designated for training will be used for training, and other images will be used for validation to assess performance. (Even if only normal images are present, the model will train normally without producing any performance.)
7. Prepare Ground Truth files and image files in the same folder.
8. For inference data, prepare either a Ground Truth file like training data or prepare the desired images for inference within a single folder. (For inference data, if there is no Ground Truth file, the algorithm internally creates files based on the paths of the images.)
9. The types of training data are composed of various types, but classification is performed only into two types: normal and abnormal.

**Example of Training Dataset**

| label | image_path       | train_val_split |
|:-----:|:-----------------|:---------------:|
| good  | ./image0.png     | train           |
| bad   | ./image1.png     | valid           |
| good  | ./image2.png     | train           |
| good  | ./good/image0.png| train           |
| good  | ./good/image1.png| train           |
| good  | ./good/image0.jpg| train           |
| good  | ./good/image1.jpg| train           |

**Example Input Data Directory Structure**
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

### Data Requirements

#### Mandatory Requirements
Input data must meet the following conditions.

| Index | Item                                     | Spec.                 |
|:-----:|:-----------------------------------------|:----------------------|
| 1     | Sufficient number of normal or single-type images | 100~50000             |
| 2     | Compliance with Ground Truth file naming format (label, image_path) | Yes                   |
| 3     | Fixed number of channels: 3 channels or 1 channel (gray) | Yes                   |
| 4     | Resolution                               | 32x32 ~ 1920x1920 pixels |
| 5     | Inference interval for new images of at least 3 seconds | Yes                   |
| 6     | Time interval required for training is at least 6 hours | Yes                   |

#### Additional Requirements
Conditions necessary to ensure minimal performance. Even if these conditions are not met, the algorithm will run, but performance will not be verified.

| Index | Item                                     | Spec.                 |
|:-----:|:-----------------------------------------|:----------------------|
| 1     | Unique names for images regardless of type | Yes                   |
| 2     | Obtain a small number of abnormal images to verify AI performance | 10 images per abnormal type |
| 3     | Resolution                               | 224x224 ~ 1024x1024 pixels |
| 4     | ROI (Region Of Interests) size           | At least 15x15 pixels (based on 224x224) |
| 5     | Fixed product position/direction/distance from camera | Allowable errors: rotation within +/- 3 degrees, translation within 1 pixel (based on 224x224) |
| 6     | Image focus should be sharp              | Yes                   |
| 7     | Image measurement environment            | Maintain as consistent and stable environment as possible during training/validation/operation (brightness, illumination, background, etc.) | Yes |

---

### Artifacts
When training/inference is executed, the following artifacts are generated.

#### Train Pipeline
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

#### Inference Pipeline
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

#### Detailed Descriptions for Each Artifact
- **\{model_name\}_model.ckpt**: Model file generated after training is completed.
- **output.csv**: Tabular file containing the results and probabilities of the inference.
- **validation_confusion_matrix.csv**: Stores the confusion matrix results for the validation data in the training pipeline. Saved only if there are abnormal data in the validation data.
- **validation_score.csv**: Records the performance of the validation data in the training pipeline. Includes accuracy, precision, recall, F1-score, and AUROC.
- **\{imagefilename\}.png**: Image showing the abnormal areas. In the training pipeline, it saves misclassified and all abnormal images. In the inference pipeline, original images for retraining are saved in the output folder.
- **inference_summary.yaml**: Summary file for the inference results. Used in edge conductor and consists of scores, results, probabilities, etc.
- **threshold_test_result.csv**: Table showing the performance changes of the validation data when the threshold criteria for anomaly detection are adjusted.
- **threshold_test_result.png**: Graph showing the performance changes when the threshold criteria for anomaly detection are adjusted.

---

## VAD Parameters

### `experimental_plan.yaml` Description
To apply AI Contents to your data, you need to enter information about the data and the Content features to be used in the `experimental_plan.yaml` file. When you install AI Contents in the solution folder, you can check the `experimental_plan.yaml` file provided for each content under the solution folder. Enter the 'data information' in this YAML file and modify/add the 'user arguments' provided for each asset. When you run ALO with the modified user arguments, you can generate a data analysis model with the desired settings.

#### `experimental_plan.yaml` Structure
`experimental_plan.yaml` contains various setting values necessary to run ALO. By modifying the 'data path' and 'user arguments' parts, you can use AI Contents immediately.

##### Enter Data Path (`external_path`)
- The parameters of `external_path` are used to specify the path to load or save files. If `save_train_artifacts_path` and `save_inference_artifacts_path` are not entered, the modeling outputs are saved in the default directories `train_artifacts` and `inference_artifacts`.

```bash
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path:  ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```

| Parameter Name              | DEFAULT               | Description and Options                                                                 |
|-----------------------------|-----------------------|----------------------------------------------------------------------------------------|
| load_train_data_path        | ./sample_data/train/  | Enter the folder path where the training data is located (do not enter the csv file name). |
| load_inference_data_path    | ./sample_data/test/   | Enter the folder path where the inference data is located (do not enter the csv file name). |

##### User Parameters (`user_parameters`)
- Below `user_parameters`, `step` represents the asset name. `step: input` below indicates the input asset stage.
- `args` represents the user arguments of the input asset (`step: input`). User arguments are parameters related to data analysis provided for each asset.

```bash
user_parameters:
	- train_pipeline:
		- step: input
		  args: 
			- file_type
			...
		  ui_args:
			...
```

---

### User Arguments Explanation
#### What are User Arguments?
User arguments are parameters for configuring the operation of each asset in the `experimental_plan.yaml` under the `args` of each asset step. They are used to customize various functionalities of the AI Contents pipeline for your data. Users can change or add user arguments based on the following guide to tailor the modeling to their data. User arguments are divided into "Required arguments," which are predefined in the `experimental_plan.yaml`, and "Custom arguments," which users add based on the provided guide.

##### Required Arguments
- Required arguments are the basic arguments shown directly in the `experimental_plan.yaml`. Most required arguments have default values built in. For arguments with default values, the user does not need to set a separate value, as it will operate with the default value.
- Among the required arguments in the `experimental_plan.yaml`, data-related arguments must be set by the user (e.g., `path_column`, `y_column`).

##### Custom Arguments
- Custom arguments are not listed in the `experimental_plan.yaml` but are functionalities provided by the asset, which users can add to the `experimental_plan.yaml` for use. They are added to each asset's 'args' section.

The VAD pipeline consists of Input - Readiness - Modeling (train/inference) - Output assets, and user arguments are configured differently for each asset according to its functionality. Start by using the required user arguments listed in the `experimental_plan.yaml` and then add more user arguments to tailor the VAD model to your data!

---

### User Arguments Summary
Here’s a summary of user arguments for VAD.

#### Default
- The 'Default' section indicates the default value of the user argument.
- If there's no default value, it's represented as '-'.
- If there's a logic for the default value, it's indicated as 'Refer to Explanation'.

#### ui_args
- The 'ui_args' indicates whether the argument supports the `ui_args` feature in AI Conductor's UI for changing argument values.
- O: Can be changed in the AI Conductor UI if entered under `ui_args`.
- X: Does not support the `ui_args` feature.

#### User Setting Requirement
- O: Arguments the user needs to input before modeling.
- X: Modeling proceeds with default values if unchanged.

| Asset Name | Argument Type | Argument Name                  | Default                                                                 | Description                                                                                           | User Setting Requirement | ui_args |
|:----------:|:-------------:|:------------------------------|:----------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------|:-----------------------:|:-------:|
| Input      | Custom        | [file_type](#file_type)        | csv                                                                   | Enter the file extension of the input data.                                                          | O                      | X       |
| Input      | Custom        | [encoding](#encoding)          | utf-8                                                                 | Enter the encoding type of the input data.                                                           | X                      | X       |
| Readiness  | Custom        | [ok_class](#ok_class)          | -                                                                     | Enter the class representing 'ok' in the y_column.                                                   | X                      | X       |
| Readiness  | Custom        | [train_validate_column](#train_validate_column) | -                                                                     | Enter the column that distinguishes between train and validation.                                    | X                      | X       |
| Readiness  | Custom        | [validation_split_rate](#validation_split_rate) | 0.1                                                                   | If train_validate_column does not exist, this rate generates validation from the input train data.   | X                      | X       |
| Readiness  | Custom        | [threshold_method](#threshold_method) | F1                                                                    | Selects the method for determining OK and NG during validation.                                      | X                      | X       |
| Readiness  | Custom        | [num_minor_threshold](#num_minor_threshold) | 5                                                                     | If the number of OK/NG images does not exceed this, threshold_method is set to Percentile.           | X                      | X       |
| Readiness  | Custom        | [ratio_minor_threshold](#ratio_minor_threshold) | 0.1                                                                   | If the ratio of OK/NG images does not exceed this, threshold_method is set to Percentile.            | X                      | X       |
| Train      | Required      | [model_name](#model_name)      | fastflow                                                              | Selects the model to be used for VAD.                                                                | O                      | O       |
| Train      | Custom        | [experiment_seed](#experiment_seed) | 42                                                                    | Determines the experiment seed in the PyTorch environment.                                           | X                      | X       |
| Train      | Custom        | [img_size](#img_size)          | [256,256]                                                             | Sets the image size for resizing during image training.                                              | X                      | X       |
| Train      | Custom        | [batch_size](#batch_size)      | 4                                                                     | Sets the batch size during training and validation.                                                  | X                      | X       |
| Train      | Custom        | [max_epochs](#max_epochs)      | 15                                                                    | Sets the maximum number of epochs for training.                                                      | X                      | X       |
| Train      | Custom        | [accelerator](#accelerator)    | cpu                                                                   | Selects whether to run based on CPU or GPU.                                                          | X                      | X       |
| Train      | Custom        | [monitor_metric](#monitor_metric) | image_AUROC                                                           | Selects the criterion for saving the best model.                                                     | X                      | X       |
| Train      | Custom        | [save_validation_heatmap](#save_validation_heatmap) | True                                                                  | Selects whether to save prediction heatmaps for the validation dataset.                              | X                      | X       |
| Train      | Custom        | [percentile](#percentile)      | 0.8                                                                   | Selects the percentile of the anomaly score to judge as NG when threshold_method is Percentile.      | X                      | X       |
| Train      | Custom        | [augmentation_list](#augmentation_list) | []                                                                    | List of transformations applied for random augmentation.                                             | X                      | X       |
| Train      | Custom        | [augmentation_choice](#augmentation_choice) | 3                                                                     | Number of times transformations are applied for random augmentation.                                 | X                      | X       |
| Train      | Custom        | [rotation_angle](#rotation_angle) | 10                                                                    | Maximum angle for rotation in random augmentation.                                                   | X                      | X       |
| Train      | Custom        | [brightness_degree](#brightness_degree) | 0.3                                                                   | Degree of brightness adjustment in random augmentation.                                              | X                      | X       |
| Train      | Custom        | [contrast_degree](#contrast_degree) | 0.3                                                                   | Degree of contrast adjustment in random augmentation.                                                | X                      | X       |
| Train      | Custom        | [saturation_degree](#saturation_degree) | 0.3                                                                   | Degree of saturation adjustment in random augmentation.                                              | X                      | X       |
| Train      | Custom        | [hue_degree](#hue_degree)      | 0.3                                                                   | Degree of hue adjustment in random augmentation.                                                     | X                      | X       |
| Train      | Custom        | [blur_kernel_size](#blur_kernel_size) | 5                                                                     | Maximum kernel size for blur in random augmentation.                                                 | X                      | X       |
| Train      | Custom        | [model_parameters](#model_parameters) | {"fastflow_backborn": 'resnet18', "fastflow_flow_steps": 8, ...}      | Parameters related to model training.                                                                | X                      | X       |
| Inference  | Custom        | [inference_threshold](#inference_threshold) | 0.5                                                                   | Threshold of anomaly score for being judged as abnormal.                                             | X                      | X       |
| Inference  | Custom        | [save_anomaly_maps](#save_anomaly_maps) | False                                                                 | Whether to save XAI image results.                                                                   | X                      | X       |

---

### User Arguments Detailed Description

#### Input Asset
##### file_type 
- **Description**: Enter the file extension of the input data.
- **Argument Type**: Custom
- **Input Type**: string
- **Possible Values**: `csv` (default)
- **Usage**: `file_type: csv`
- **ui_args**: X

##### encoding 
- **Description**: Enter the encoding type of the input data.
- **Argument Type**: Custom
- **Input Type**: string
- **Possible Values**: `utf-8` (default)
- **Usage**: `encoding: utf-8`
- **ui_args**: X

#### Readiness Asset
##### ok_class 
- **Description**: Enter the class representing 'ok' in the y_column. If not entered, `ok_class` is set as the most frequent name in the training dataset.
- **Argument Type**: Custom
- **Input Type**: string
- **Possible Values**: `''` (default)
- **Usage**: `ok_class: good`
- **ui_args**: O

##### train_validate_column 
- **Description**: Enter the column that distinguishes between train and validation. This column should consist of two parameters: `train` and `valid`.
- **Argument Type**: Custom
- **Input Type**: string
- **Possible Values**: `''` (default)
- **Usage**: `train_validate_column: phase`
- **ui_args**: X

##### validation_split_rate 
- **Description**: If `train_validate_column` does not exist, this rate generates validation from the input train data.
- **Argument Type**: Custom
- **Input Type**: float
- **Possible Values**: `0.1` (default)
- **Usage**: `validation_split_rate: 0.1`
- **ui_args**: X

##### threshold_method 
- **Description**: Selects the method for determining OK and NG during validation. Possible values: `F1`, `Percentile` (automatically selected based on validation data size).
- **Argument Type**: Custom
- **Input Type**: string
- **Possible Values**: `F1` (default), `Percentile`
- **Usage**: `threshold_method: Percentile`
- **ui_args**: X

##### num_minor_threshold 
- **Description**: If the number of OK and NG images does not exceed this, `threshold_method` is automatically selected as `Percentile`.
- **Argument Type**: Custom
- **Input Type**: integer
- **Possible Values**: `5` (default)
- **Usage**: `num_minor_threshold: 5`
- **ui_args**: X

##### ratio_minor_threshold 
- **Description**: If the ratio of OK and NG images does not exceed this, `threshold_method` is automatically selected as `Percentile`.
- **Argument Type**: Custom
- **Input Type**: float
- **Possible Values**: `0.1` (default)
- **Usage**: `ratio_minor_threshold: 0.1`
- **ui_args**: X

#### Train Asset
##### model_name 
- **Description**: Selects the model to be used for VAD. Possible values: `fastflow`, `patchcore`.
- **Argument Type**: Required
- **Input Type**: string
- **Possible Values**: `fastflow` (default), `patchcore`
- **Usage**: `model_name: patchcore`
- **ui_args**: O

##### img_size 
- **Description**: Sets the image size for training. If only numbers are entered, the aspect ratio of the original image is maintained during resizing.
- **Argument Type**: Custom
- **Input Type**: integer
- **Possible Values**: `[256,256]` (default), `256`
- **Usage**: `img_size: [256,256]`
- **ui_args**: X

##### batch_size 
- **Description**: Sets the batch size for training and validation.
- **Argument Type**: Custom
- **Input Type**: integer
- **Possible Values**: `4` (default)
- **Usage**: `batch_size: 32`
- **ui_args**: X

##### experiment_seed 
- **Description**: Determines the experiment seed in the PyTorch environment.
- **Argument Type**: Custom
- **Input Type**: integer
- **Possible Values**: `42` (default)
- **Usage**: `experiment_seed: 42`
- **ui_args**: X

##### max_epochs 
- **Description**: Sets the maximum number of epochs for training.
- **Argument Type**: Custom
- **Input Type**: integer
- **Possible Values**: `15` (default)
- **Usage**: `max_epochs: 15`
- **ui_args**: X

##### accelerator 
- **Description**: Selects whether to run based on CPU or GPU.
- **Argument Type**: Custom
- **Input Type**: string
- **Possible Values**: `cpu` (default), `gpu`
- **Usage**: `accelerator: gpu`
- **ui_args**: X

##### monitor_metric 
- **Description**: Selects the criterion for saving the best model. Possible values: `loss`, `image_AUROC`, `image_F1Score`.
- **Argument Type**: Custom
- **Input Type**: string
- **Possible Values**: `image_AUROC` (default), `loss`, `image_F1Score`
- **Usage**: `monitor_metric: loss`
- **ui_args**: X

##### save_validation_heatmap 
- **Description**: Selects whether to save prediction heatmaps for the validation dataset.
- **Argument Type**: Custom
- **Input Type**: bool
- **Possible Values**: `True` (default), `False`
- **Usage**: `save_validation_heatmap: True`
- **ui_args**: X

##### percentile 
- **Description**: Selects the percentile of anomaly score in the validation dataset to determine NG when `threshold_method` is `Percentile`.
- **Argument Type**: Custom
- **Input Type**: float
- **Possible Values**: `0.8` (default)
- **Usage**: `percentile: 0.8`
- **ui_args**: X

##### augmentation_list 
- **Description**: List of transformations applied for random augmentation. Multiple options can be used: `rotation`, `brightness`, `contrast`, `saturation`, `hue`, `blur`.
- **Argument Type**: Custom
- **Input Type**: list
- **Possible Values**: `[]` (default), `rotation`, `brightness`, `contrast`, `saturation`, `hue`, `blur`
- **Usage**: `augmentation_list: [rotation, brightness, contrast, saturation, hue, blur]`
- **ui_args**: X

##### augmentation_choice 
- **Description**: Number of times transformations are applied for random augmentation.
- **Argument Type**: Custom
- **Input Type**: int
- **Possible Values**: `3` (default)
- **Usage**: `augmentation_choice: 2`
- **ui_args**: X

##### rotation_angle 
- **Description**: Maximum angle for rotation in random augmentation. If set to 10, transformation is performed between -10 and 10.
- **Argument Type**: Custom
- **Input Type**: float
- **Possible Values**: `10` (default)
- **Usage**: `rotation_angle: 30`
- **ui_args**: X

##### brightness_degree 
- **Description**: Degree of brightness adjustment in random augmentation. Sets maximum/minimum brightness levels.
- **Argument Type**: Custom
- **Input Type**: float
- **Possible Values**: `0.3` (default)
- **Usage**: `brightness_degree: 0.5`
- **ui_args**: X

##### contrast_degree 
- **Description**: Degree of contrast adjustment in random augmentation.
- **Argument Type**: Custom
- **Input Type**: float
- **Possible Values**: `0.3` (default)
- **Usage**: `contrast_degree: 0.5`
- **ui_args**: X

##### saturation_degree 
- **Description**: Degree of saturation adjustment in random augmentation.
- **Argument Type**: Custom
- **Input Type**: float
- **Possible Values**: `0.3` (default)
- **Usage**: `saturation_degree: 0.5`
- **ui_args**: X

##### hue_degree 
- **Description**: Degree of hue adjustment in random augmentation.
- **Argument Type**: Custom
- **Input Type**: float
- **Possible Values**: `0.3` (default)
- **Usage**: `hue_degree: 0.5`
- **ui_args**: X

##### blur_kernel_size 
- **Description**: Maximum kernel size for blur in random augmentation.
- **Argument Type**: Custom
- **Input Type**: int
- **Possible Values**: `5` (default)
- **Usage**: `blur_kernel_size: 25`
- **ui_args**: X

##### model_parameters 
- **Description**: Parameters related to model training. If not set, the model is trained with default parameters.
- **Argument Type**: Custom
- **Input Type**: dictionary
- **Possible Values**: `{"fastflow_backborn": 'resnet18', "fastflow_flow_steps": 8, "patchcore_backborn": 'wide_resnet50_2', "patchcore_coreset_sampling_ratio": 0.1, "patchcore_layers": ["layer2", "layer3"]}` (default)
- **Usage**: `model_parameters: {"fastflow_backborn": 'resnet18', "fastflow_flow_steps": 8, "patchcore_backborn": 'wide_resnet50_2', "patchcore_coreset_sampling_ratio": 0.1, "patchcore_layers": ["layer2", "layer3"]}`
- **ui_args**: X

#### Inference Asset
##### inference_threshold 
- **Description**: Threshold of anomaly score for being judged as abnormal.
- **Argument Type**: Required
- **Input Type**: float
- **Possible Values**: `0.5` (default)
- **Usage**: `inference_threshold: 0.5`
- **ui_args**: X

##### save_anomaly_maps 
- **Description**: Whether to save XAI image results.
- **Argument Type**: Custom
- **Input Type**: bool
- **Possible Values**: `False` (default), `True`
- **Usage**: `save_anomaly_maps: True`
- **ui_args**: X

---

