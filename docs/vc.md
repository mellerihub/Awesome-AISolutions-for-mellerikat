# Visual Classification (VC) Reference Guide

<div align="right">Updated 2024.05.17</div><br/>

## Table of Contents
- [Overview of Features](#overview-of-features)
  - [VC Pipeline](#vc-pipeline)
  - [Pipeline Components](#pipeline-components)
  - [Usage Tips](#usage-tips)
  - [Detailed Features](#detailed-features)
- [Data Preparation](#data-preparation)
  - [Requirements](#data-requirements)
  - [Input Preparation](#preparing-training-data)
  - [Directory Structure](#example-of-input-data-directory-structure)
- [Artifacts](#artifacts)
  - [Train Pipeline Artifacts](#train-pipeline)
  - [Inference Pipeline Artifacts](#inference-pipeline)
  - [Artifact Descriptions](#the-detailed-descriptions-of-the-artifacts-are-as-follows)
- [Parameter Reference](#explanation-of-experimental_planyaml)
  - [Structure](#structure-of-experimental_planyaml)
  - [User Arguments Summary](#summary-of-user-arguments)
  - [Detailed Parameter Reference](#detailed-explanation-of-user-arguments)

---

## Overview of Features
### VC Pipeline
The AI Contents pipeline is composed of assets, which are functional units combined to perform tasks. The VC pipeline consists of a combination of four assets.

#### Train Pipeline
```bash
Input - Readiness - Train
```
#### Inference Pipeline
```bash
Input - Readiness - Inference - Output
```

### Pipeline Components

#### Input Asset
Computer Vision AI operates with Ground Truth data, which is a tabular form containing image paths and corresponding labels. The input asset reads this Ground Truth data and passes it to the next asset. During inference, if there is no file containing image information, it generates it using the image paths and passes it to the next asset. As shown in the diagram, it passes to the train asset for the training pipeline and the inference asset for the inference pipeline.

#### Readiness Asset
The readiness asset checks the data quality of the Ground Truth data. It raises a warning if there are any missing values in the image path or label fields and proceeds with the training pipeline by excluding the missing values.

#### Modeling (Train) Asset
In the train asset of Computer Vision AI, several tasks are performed. It reads the Ground Truth data from the input, loads images into memory using the paths provided, and wraps them in a `tensorflow.Data.dataset` object for efficient processing. Based on the model name and parameters defined in the experimental plan, it initializes and trains the model. After training, it saves the performance metrics and inference results.

#### Modeling (Inference) Asset
The inference asset reads images based on paths provided by the input asset, loads the best model saved during training, and infers labels. It stores the results and, if running an experiment (not deployment), also saves performance metrics if ground truth labels are available. An inference summary file (`inference_summary.yaml`) is also saved for deployment purposes.

#### Output Asset
The output asset customizes the inference summary according to the specific requirements of the solution domain. If inference data is a single image, the `result` field contains the predicted label and the `score` field contains the model's confidence. For multiple images, the `result` field contains the count of each type, and the `score` field contains the average model confidence. The current limit for characters in the `result` field is 32 characters.

---

## Usage Tips

#### Choosing Ground Truth Data Path
Place the Ground Truth (GT) file and images in the same folder. ALO will copy both the GT file and images to the analysis environment. The GT data should consist of `image_path` and `label` columns. Even if there are other columns in the GT data, they will not be used for analysis. While the `image_path` column name can be changed during analysis (by updating the `path_column` name in the experimental plan), it is recommended to keep it as `image_path` to avoid issues during solution deployment. Ensure all images have a consistent size; performance may degrade if image sizes vary. Preprocess images to have uniform size through cropping or resizing.

#### Selecting a Model
Currently, supported models include `mobilenetv1` and `high_resolution`. `mobilenetv1` is a standard model suitable for most datasets, offering stable training and inference. It is recommended to set `resize_shape` to `[224, 224, 3]`. For higher resolution requirements, select the `high_resolution` model and adjust the `resize_shape` parameter accordingly.

#### Choosing Data Augmentation Techniques
For datasets with fewer than 1000 images per class, enabling `rand_augmentation` to `True` allows training with more diverse and extensive data through augmentation techniques. Ensure the augmentation methods align with the task at hand. Detailed configuration is explained below in the Train Pipeline: Add Data Augmentation section.

#### Balancing Performance and Resource Usage
Increasing the number of epochs can improve performance for large and complex datasets. Use the checkpoint feature to save the best-performing model based on validation data, mitigating overfitting risks. Adjust `batch_size` based on available memory and GPU resources to speed up training and improve model performance. If data is abundant but needs more validation, reduce the `train_ratio` accordingly.

#### Visual Confirmation of Inference Areas (XAI Explanation)
Enable the `do_xai` option to visualize the areas that contributed to the model's predictions. If focusing on identifying defects (e.g., OK/NG classification), set `xai_class` to the class of interest. Adjust `mask_threshold` for more extensive coverage if the XAI sensitivity is too low. Increase `layer_index` to use deeper layers for more accurate results, keeping in mind that this will require more resources.

---

## Detailed Features
A detailed explanation to utilize various features of Computer Vision AI.

### Train Pipeline: Input Asset

#### Read Ground Truth File
Reads the Ground Truth file to identify image paths and labels in the `input asset`.

### Train Pipeline: Readiness Asset

#### Check Data Quality
Validates the Ground Truth file for missing values in image paths and labels. Issues a warning and proceeds by excluding missing values.

### Train Pipeline: Train Asset

#### Load Image Files and Generate Tensorflow.Data.Dataset Object
Reads image files based on paths specified in the Ground Truth file and loads them into memory. Converts images into `dataset` format for efficient processing with TensorFlow APIs. Performs necessary preprocessing like scaling (1/255.) and resizing based on the `resize_shape` parameter defined in the experimental plan.

#### Split Train/Validation Data
Randomly splits the dataset into 80% training and 20% validation data. This prevents overfitting by ensuring the model performs well on unseen data.

#### Add Data Augmentation
Performs data augmentation using the RandAugment method ([Paper](https://arxiv.org/abs/1909.13719)). Exclude specific transformations that may not be suitable for the task. Adjust the augmentation intensity and the number of transformations per image using `aug_magnitude` (0-10) and `num_aug` (1-16). Enable augmentation by setting `rand_augmentation` to `True`.

RandAugment randomly applies a specified number of transformations from a set of 16 to the original image. This method increases the diversity of training data, improving the model's performance on new images. However, inappropriate transformations (e.g., flip or rotation) may degrade performance if they alter the image's characteristics crucial for the task. In such cases, review and exclude unsuitable transformations.

For tasks with abundant data, disable augmentation by setting `rand_augmentation` to `False`.

Example table when `num_aug=2`:

|index|image_path|label|1st Augmentation|2nd Augmentation|
|---|---|---|---|---|
|1|OK_image1.jpg|OK|color|autocontrast|
|2|NG_image1.jpg|NG|invert|color|
|3|OK_image2.jpg|OK|shearX|posterize|
|...|...|...|...|...|

#### Initialize Model
Initializes the model based on the `model_type` parameter. `mobilenetv1` offers stable training and inference with low resource requirements, suitable for various tasks. The `high_resolution` model, developed by LG Electronics' advanced analysts, enhances mobilenetV3-Small with CBAM ([Paper](https://arxiv.org/abs/1807.06521)) and ECA ([Paper](https://arxiv.org/abs/1910.03151)) layers, providing high-resolution capabilities with efficient resource usage.

#### Fit Model on Training Data
Trains the model on the training and validation datasets for the specified number of `epochs`. Utilizes checkpoints to save the model with the lowest validation loss, preventing overfitting.

#### Evaluate Score and Save Outputs
Calculates performance metrics (accuracy, precision, recall, f1-score) on the validation dataset and saves the summary (`eval.json`). Saves inference results (`prediction.csv`) and confusion matrix (`confusion.csv`) for the training/validation datasets. Stores an inference summary (`inference_summary.yaml`) for Edge Application.

Example formats:

**Table 1: prediction.csv (labels: OK, NG)**

|index|image_path|label|pred_label|prob_OK|prob_NG|uncertainty|
|---|---|---|---|---|---|---|
|1|OK_image1.jpg|OK|OK|0.9|0.1|0.5|
|2|NG_image1.jpg|NG|NG|0.15|0.85|1.0|
|3|OK_image2.jpg|OK|OK|0.6|0.4|0.2|
|...|...|...|...|...|...|...|

**Table 2: confusion.csv**

||OK|NG|
|---|---|---|
|OK|10|1|
|NG|0|10|

**Table 3: eval.json**

| |precision|recall|f1-score|support|
|---|---|---|---|---|
|OK|0.50|1.00|0.67|1|
|NG|1.00|0.67|0.80|4|
|accuracy| | |0.60|5|
|macro avg|0.50|0.56|0.49|5|
|weighted avg|0.70|0.60|0.61|5|

**Table 4: inference_summary.yaml**

For a single image:
```
{
    result: 'OK', # predicted label
    score: 0.8, # 1 - uncertainty
}
```

For multiple images:
```
{
    result: 'OK: 12, NG: 3', # value counts of predicted label
    score: 0.9, # 1 - mean of uncertainties of inference data
}
```

### Inference Pipeline: Input Asset

#### Read Inference File or Generate It
Reads the inference file if available. If not, generates it using image paths for inference.

### Inference Pipeline: Inference Asset

#### Load Image Files
Loads images from the paths provided in the inference file into memory and creates a dataset object for inference. If `do_xai` is set to `True`, XAI results are saved with filenames in the format `original_filename_xai_class_label.png`. If `xai_class` is specified, XAI images are saved for that specific class. Otherwise, the original images are resized and saved for inference.

#### Predict Label
Loads the appropriate model for inference. If `do_xai` is `True`, loads the Keras (h5) model. Otherwise, loads the TensorFlow Lite (tflite) model for faster inference. Inference time is approximately 1 second per image in ALO.

#### Save Outputs
Saves inference results similarly to the training pipeline. If no label information is provided, `eval.json` and `confusion.csv` are not saved.

### Inference Pipeline: Output Asset

#### Modify Inference Summary
Customizes the inference summary according to the specific requirements of the solution domain.

---

## Data Preparation

### Preparing Training Data
1. Prepare image data in .png or .jpg format with consistent shape (e.g., 1024x1024 pixels, 3 channels).
2. Create a ground truth file in tabular form containing image paths and corresponding labels.
3. Ensure each label type has at least 100 images to build a stable model.
4. Currently, multi-class classification is supported, but multi-label classification (one image with multiple labels) is not.
5. The ground truth file should be formatted as shown below.
6. Place the ground truth file and image files in the same folder.
7. For inference data, prepare the ground truth file similarly to the training data, or place the images in a single folder.
   (If there is no ground truth file for inference data, the path to the images will be used to generate an internal file.)

**Example of GroundTruth.csv Training Dataset**

| label  | image_path    |
| ------ | ------------- |
| label1 | ./image1.png  |
| label2 | ./image1.jpeg |
| label1 | ./image2.jpeg |
| ...    | ...           |

### Example of Input Data Directory Structure

- Ground truth files can be multiple files but must have consistent column names.

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

**Example of Input Data Directory Structure (When there is no CSV file for inference data)**

```bash
./{inference_folder}/
    └ image_test1.png
    └ image_test1.jpeg
    └ /{inference_folder}/
        └ image_test2.jpeg
```

---

## Data Requirements

### Mandatory Requirements
The input data must meet the following conditions:

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

### Additional Requirements
These conditions ensure minimal performance. If not met, the algorithm will still run, but performance may be affected:

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

---

## Artifacts
Running the training/inference pipeline generates the following artifacts:

### Train Pipeline
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

### Inference Pipeline
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

### The detailed descriptions of the artifacts are as follows:

#### model.h5
The trained Keras model file.

#### model.tflite
The trained TensorFlow Lite model suitable for embedded environments.

#### params.json
A JSON file containing parameters used during training.

#### prediction.csv
Contains the model's predictions for the training data, with the following columns:
- label: Ground Truth label
- pred_label: Predicted label
- prob_\{label\}: Probability for each label

#### confusion.csv
A confusion matrix ([Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)) of the Ground Truth and predicted labels, saved if label information is present in the inference pipeline.

#### eval.json
A classification report ([scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report)) representing the training performance, saved if label information is present in the inference pipeline.

#### \{imagefilename\}.png OR \{imagefilename\}_xai_class\{predicted_class\}.png
If `do_xai` is `True`, saves the image with highlighted XAI areas. If `do_xai` is `False`, saves the original image. Viewable in the edge viewer.

#### inference_summary.yaml
A summary of the inference results displayed in the Mellerikat's edge viewer. Includes fields like date, file_path, note, probability, result, score, and version. In VC, `result` indicates the predicted label and `score` indicates the confidence level (0-1).

---

## Explanation of experimental_plan.yaml

To apply AI Contents to your data, you need to fill in the experimental_plan.yaml file with the data information and the Contents functions you will use. When you install AI Contents in the solution folder, you can find the pre-written experimental_plan.yaml file for each content under the solution folder. By entering 'data information' and modifying/adding 'user arguments' provided by each asset in this yaml file, you can run ALO to generate a data analysis model with the desired settings.

### Structure of experimental_plan.yaml

The experimental_plan.yaml contains various settings needed to run ALO. By modifying the 'data path' and 'user arguments' settings, you can immediately use AI Contents.

#### Entering the Data Path (`external_path`)

- The `external_path` parameter is used to specify the path of files to be loaded or saved. If `save_train_artifacts_path` and `save_inference_artifacts_path` are not entered, modeling artifacts will be saved in the default paths, `train_artifacts` and `inference_artifacts`.

```bash
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path:  ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```
| Parameter Name           | DEFAULT              | Description and Options                                             |
| ------------------------ | -------------------- | ------------------------------------------------------------------- |
| load_train_data_path     | ./sample_data/train/ | Enter the folder path where the training data is located. (Do not enter the CSV file name) |
| load_inference_data_path | ./sample_data/test/  | Enter the folder path where the inference data is located. (Do not enter the CSV file name) |

#### User Parameters (`user_parameters`)

- `user_parameters` under `step` indicates the asset name. Below, `step: input` means the input asset stage.
- `args` means the user arguments for the input asset (`step: input`). User arguments are data analysis-related setting parameters provided by each asset. For more details, refer to the User arguments explanation below.
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

## Explanation of User arguments

### What are User arguments?

User arguments are parameters for setting the operation of each asset, written under `args` for each asset step in the experimental_plan.yaml. Each asset in the AI Contents pipeline provides user arguments so that users can apply various functions to their data. Users can refer to the guide below to change or add user arguments to create a model that fits their data.

User arguments are divided into "Required arguments," which are pre-written in the experimental_plan.yaml, and "Custom arguments," which users add by referring to the guide.

#### Required arguments

- Required arguments are basic arguments that are immediately visible in the experimental_plan.yaml. Most required arguments have default values built-in. If an argument has a default value, users do not need to set the value separately; it will operate with the default value.
- Among the required arguments in the experimental_plan.yaml, users must set the data-related arguments. (e.g., x_columns, y_column)

#### Custom arguments

- Custom arguments are not written in the experimental_plan.yaml by default, but they are functions provided by the asset that users can add to the experimental_plan.yaml. Add these under the 'args' of each asset.

The VC pipeline consists of **Input - Readiness - Modeling (train/inference) - Output** asset stages, with user arguments configured differently according to each asset's function. Try using the required user arguments written in the experimental_plan.yaml first, and then add user arguments to create a VC model that perfectly fits your data!

---

## Summary of User arguments

Below is a summary of the VC user arguments. Click on the 'Argument name' to go to the detailed explanation of that argument.

#### Default

- The 'Default' column shows the default value of the user argument.
- If there is no default value, it is marked as '-'.
- If the default value has logic, it is marked as 'refer to the explanation.' Click on the 'Argument name' to view the detailed explanation.

#### ui_args

- The 'ui_args' column indicates whether the argument supports the `ui_args` function, which allows changing the argument value in the AI Conductor UI.
- O: If you write the argument name under `ui_args` in the experimental_plan.yaml, you can change the argument value in the AI Conductor UI.
- X: Does not support the `ui_args` function.
- For more details on `ui_args`, refer to the following guide: [Write UI Parameter](../../alo/register_ai_solution/write_ui_parameter)
- All potential `ui_args` user arguments are pre-written in the VC experimental_plan.yaml.

#### Required for User Settings

- The 'Required for User Settings' column indicates whether the user must check and change the user argument for AI Contents to operate.
- O: Generally, task or data-related information must be entered by the user before modeling.
- X: If the user does not change the value, the modeling proceeds with the default value.

| Asset Name | Argument Type | Argument Name                          | Default       | Description and Options                                                    | Required for User Settings | ui_args |
|:----------:|:-------------:|:--------------------------------------:|:-------------:|:--------------------------------------------------------------------------:|:---------------------------:|:-------:|
| Input      | Custom        | [file_type](#file_type)                | csv           | Enter the file extension of the input data.                                | X                           | O       |
| Input      | Custom        | [encoding](#encoding)                  | utf-8         | Enter the encoding type of the input data.                                 | X                           | O       |
| Readiness  | Custom        | [y_column](#y_column)                  | label         | Enter the column name where the classification label is written in the Ground Truth file. | X | O |
| Readiness  | Custom        | [path_column](#path_column)            | image_path    | Enter the column name where the image path is written in the Ground Truth file. Do not change during operation. | X | O |
| Readiness  | Custom        | [check_runtime](#check_runtime)        | False         | Set to True if you want to check the execution time of the modeling stage during the experiment. Set `check_memory` to False if you set this to True. | X | O |
| Readiness  | Custom        | [check_memory](#check_memory)          | False         | Set to True if you want to check the memory used during the modeling stage during the experiment. Set `check_runtime` to False if you set this to True. | X | O |
| Train      | Required      | [input_shape](#input_shape)            | [28,28,1]     | Enter the size of the input image.                                          | O                           | O       |
| Train      | Required      | [resize_shape](#resize_shape)          | [32,32,3]     | Enter the image size for model training.                                   | O                           | O       |
| Train      | Custom        | [model_type](#model_type)              | mobilenetv1   | Select the training model.                                                 | X                           | O       |
| Train      | Custom        | [rand_augmentation](#rand_augmentation)| False         | Whether to apply RandAugment.                                               | X                           | O       |
| Train      | Custom        | [exclude_aug_lst](#exclude_aug_lst)    | []            | Exclude inappropriate transformations when using rand_augmentation.        | X                           | O       |
| Train      | Custom        | [epochs](#epochs)                      | 10            | Enter the number of training epochs.                                       | X                           | X       |
| Train      | Custom        | [batch_size](#batch_size)              | 64            | Enter the number of data points to be considered at one time during training. | X                           | X       |
| Train      | Custom        | [train_ratio](#train_ratio)            | 0.8           | Enter the ratio of data to be used for training.                           | X                           | X       |
| Train      | Custom        | [num_aug](#num_aug)                    | 2             | Enter the number of times to randomly transform an image when using RandAugment. | X | X |
| Train      | Custom        | [aug_magnitude](#aug_magnitude)        | 10            | Enter the strength of augmentation when using RandAugment.                 | X                           | X       |
| inference  | Required      | [do_xai](#do_xai)                      | False         | Whether to use XAI.                                                        | O                           | X       |
| inference  | Custom        | [xai_class](#xai_class)                | auto          | Enter the class to be analyzed by XAI.                                     | X                           | X       |
| inference  | Custom        | [mask_threshold](#mask_threshold)      | 0.7           | Enter the sensitivity when outputting XAI results.                         | X                           | X       |
| inference  | Custom        | [layer_index](#layer_index)            | 2             | Enter the number of layers to be analyzed during inference.                | X                           | X       |

---

## Detailed Explanation of User arguments

### Input asset

#### file_type
Enter the file extension of the input data. Currently, AI Solution development only supports CSV files.

- Argument type: Custom
- Input type: string
- Input possible values: **csv (default)**
- Usage: `file_type: csv`  
- ui_args: O

#### encoding
Enter the encoding type of the input data. Currently, AI Solution development only supports UTF-8 encoding.

- Argument type: Custom
- Input type: string
- Input possible values: **utf-8 (default)**
- Usage: `encoding: utf-8`
- ui_args: O

### Readiness asset

#### y_column
Enter the column name where the classification label is written in the Ground Truth file. Users must enter this to match their input data. If the column name is 'label', it does not need to be entered.

- Argument type: Custom
- Input type: string
- Input possible values:
  - **label (default)**
  - Column name
- Usage: `y_column: image_label`
- ui_args: X

#### path_column
This is the column name where the image path is written in the Ground Truth file. During operation, the column name should be designated as image_path. This can be changed during solution development experiments.

- Argument type: Custom
- Input type: string
- Input possible values:
  - **image_path (default)**
  - Column name
- Usage: `path_column: Path`
- ui_args: X

#### check_runtime
If you want to check the time taken during the modeling stage of solution development, set this to True. Set `check_memory` to False to accurately check the execution time.

- Argument type: Custom
- Input type: boolean
- Input possible values:
  - True
  - **False (default)**
- Usage: `check_runtime: True`
- ui_args: X

#### check_memory
If you want to check the memory used during the modeling stage of solution development, set this to True. Set `check_runtime` to False to accurately check the memory usage.

- Argument type: Custom
- Input type: boolean
- Input possible values:
  - True
  - **False (default)**
- Usage: `check_memory: True`
- ui_args: X

### Train asset

#### input_shape
Enter the size of the input image. All input images must be the same size.

- Argument type: Required
- Input type: list(int,int,int)
- Input possible values:
  - **[28,28,1] (default)**
  - [224,224,3]
- Usage: `input_shape: [32,32,1]`
- ui_args: O

#### resize_shape
Enter the image size for model training. If the training image size is large, it will consume a lot of resources, but if the region of interest for classification is small, set it larger. It should be similar to or smaller than input_shape.

- Argument type: Required
- Input type: list(int,int,int)
- Input possible values:
  - **[32,32,3] (default)**
  - [224,224,3]
- Usage: `resize_shape: [32,32,1]`
- ui_args: O

#### model_type
Select the training model. If it is not a high-resolution image, use the default mobilenetv1. If it is a high-resolution image, select high_resolution.

- Argument type: Custom
- Input type: string
- Input possible values:
  - **mobilenetv1 (default)**
  - high_resolution
- Usage: `model_type: high_resolution`
- ui_args: O

#### rand_augmentation
Set whether to apply RandAugment. To improve the model's adaptability to new image data, set this to True. Applying RandAugment will consume additional resources due to data augmentation.

- Argument type: Custom
- Input type: boolean
- Input possible values:
  - True
  - **False (default)**
- Usage: `rand_augmentation: True`
- ui_args: O

#### exclude_aug_lst
When using rand_augmentation, you can exclude transformations that are not suitable. For example, if color is important for image classification, performing transformations like color can degrade the inference performance. Write considering the data and the problem situation to be applied.

- Argument type: Custom
- Input type: list
- Input possible values:
  - **[] (default)**
  - solarizeadd
  - invert
  - cutout
  - autocontrast
  - equalize
  - rotate
  - solarize
  - color
  - posterize
  - contrast
  - brightness
  - sharpness
  - shearX
  - shearY
  - translateX
  - translateY
- Usage: `exclude_aug_lst: [invert, color]`
- ui_args: O

#### epochs
Enter the number of training epochs. Increasing the number will extend the training time. If it is too small, it will result in underfitting, and the model will not be trained. During the training process, the model internally references the validation data, and only the model with the best performance is saved, preventing overfitting.

- Argument type: Custom
- Input type: integer
- Input possible values:
  - **10 (default)**
  - 1-1000
- Usage: `epochs: 50`
- ui_args: X

#### batch_size
The number of data points considered at one time during training. Increasing the number will consume more memory and GPU memory but will improve performance.

- Argument type: Custom
