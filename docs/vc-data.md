# VC Input and Artifacts

<div align="right">Updated 2024.05.17</div>

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

**Example of Input Data Directory Structure**

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

The detailed descriptions of the artifacts are as follows:

### model.h5
The trained Keras model file.

### model.tflite
The trained TensorFlow Lite model suitable for embedded environments.

### params.json
A JSON file containing parameters used during training.

### prediction.csv
Contains the model's predictions for the training data, with the following columns:
- label: Ground Truth label
- pred_label: Predicted label
- prob_\{label\}: Probability for each label

### confusion.csv
A confusion matrix ([Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)) of the Ground Truth and predicted labels, saved if label information is present in the inference pipeline.

### eval.json
A classification report ([scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report)) representing the training performance, saved if label information is present in the inference pipeline.

### \{imagefilename\}.png OR \{imagefilename\}_xai_class\{predicted_class\}.png
If `do_xai` is `True`, saves the image with highlighted XAI areas. If `do_xai` is `False`, saves the original image. Viewable in the edge viewer.

### inference_summary.yaml
A summary of the inference results displayed in the Mellerikat's edge viewer. Includes fields like date, file_path, note, probability, result, score, and version. In VC, `result` indicates the predicted label and `score` indicates the confidence level (0-1).

---

**VC Version: 1.5.1**
