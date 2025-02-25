# VAD Input and Artifacts

<div align="right">Updated 2024.06.11</div><br/>

## Data Preparation
#### Preparation of Training Data
1. Prepare normal (or same type) image data in png or .jpg format with the same shape (e.g., 1024x1024, 3 channels).
2. Prepare a tabular-formatted Ground Truth file consisting of image paths and corresponding labels (all normal labels due to the same type).
3. Including some abnormal (or a few different types) image data in the training data leads to a more accurate AI model.
4. The Ground Truth file should be prepared in the following format.
5. There is no need to unify labels for abnormal images with multiple types when preparing the Ground Truth file.
6. If there are validation data desired, input the column name in the `train_validation_column` argument so that images designated for training will be used for training, and other images will be used for validation to assess performance.
(Even if only normal images are present, the model will train normally without producing any performance.)
7. Prepare Ground Truth files and image files in the same folder.
8. For inference data, prepare either a Ground Truth file like training data or prepare the desired images for inference within a single folder.
(For inference data, if there is no Ground Truth file, the algorithm internally creates files based on the paths of the images.)
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

<br/>

## Data Requirements

#### Mandatory Requirements
Input data must meet the following conditions.

|  index | item  | spec. |
|:---:|:---:|:---:|
|  1 | Sufficient number of normal or single-type images  | 100~50000 |
|  2 | Compliance with Ground Truth file naming format (label, image_path)  | Yes |
|  3 | Fixed number of channels: 3 channels or 1 channel (gray)  | Yes |
|  4 | Resolution  | 32x32 ~ 1920x1920 pixels |
|  5 | Inference interval for new images of at least 3 seconds  | Yes |
|  6 | Time interval required for training is at least 6 hours  | Yes |
* Data Examples 참고

#### Additional Requirements
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

<br/>

## Artifacts

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
#### Inference pipeline
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
Detailed descriptions for each artifact are as follows:

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

<br/>

**VAD Version: 1.0.0**
