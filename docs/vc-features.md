# VC Features

<div align="right">Updated 2024.05.17</div><br/>

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

Each step is defined by an asset.

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

<br/>

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

<br/>

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
    result: 'OK:

 12, NG: 3', # value counts of predicted label
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

<br/>

---

**VC Version: 1.5.1**
