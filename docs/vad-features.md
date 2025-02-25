# VAD Features

<div align="right">Updated 2024.06.11</div><br/>

## Feature Overview
### VAD Pipeline
The pipeline of AI Contents consists of a combination of assets at the feature level. The Train pipeline consists of 3 assets, while the Inference pipeline consists of 4 assets.

#### Train Pipeline
```bash
	Input - Readiness -  Train
```  
#### Inference pipeline
```bash
	Input -  Readiness -  Inference -  Output
```
#### input
VAD operates on tabular data in the form of image paths and correct labels, known as Ground Truth data, during training. Since VAD can be trained with only normal images, if there are only normal images available, you can write the label names indicating normality (e.g., OK, good, normal, etc.). The input asset reads the Ground Truth data and passes it to the next asset. If there is no file containing information about the images during the inference stage, it simply passes on the image paths to create them for inference. As shown in the diagram above, in the training pipeline, it is passed as train, and in the inference pipeline, it is passed as inference asset.

#### readiness
Checks the data quality of the Ground Truth data. If there are missing values in the image path or label columns, it issues a warning and proceeds with training after excluding the missing values. It also checks and possibly modifies suitable argument settings based on the data structure.

#### modeling(train)
The train asset in VAD reads the Ground Truth data passed from the input and loads the training and validation data into PyTorch dataloaders. Depending on the model name specified by the user in the experimental_plan.yaml, VAD creates and trains either the FastFlow or PatchCore model supported by VAD. Once training is complete, if there are abnormal data in the validation data, it outputs the predicted results in the output.csv file, along with confusion matrix, classification report, and performance table and graphs that change when the anomaly_threshold is adjusted. Depending on the AIOps operational situation, where a model maximizing a specific performance may be needed, you can infer and set an appropriate threshold value based on the table.

#### modeling(inference)
In the inference asset, it reads the path of the inference data received from the input asset, loads the image, loads the best model saved from the previous training stage, predicts the label, and saves it. If it's a stage for experimentation rather than operation and the correct label is in the file, it saves the performance. It also saves the file for operation, inference_summary.yaml.

#### output
The required information may vary for each task. Currently, if the inference data infers for a single image, the result contains the inferred label, and the score contains the model's certainty (1-Shannon entropy). If it's inferring for multiple images, the result contains the count for each type, and the score contains the average certainty of the model. Currently, up to 32 characters can be stored in the result due to character limitations.

<br/>

---


## Tips for Use
#### Preparing Ground Truth Files and Image Data
Put the Ground Truth (GT) data and images in the same folder. ALO copies the GT file and images to the execution environment at once. The GT data should consist of image paths (image_path) and labels (label). If you want to differentiate between training and validation datasets, separate the data into train and valid, create new items, and write the corresponding item name in train_validate_column. Any other items in the GT data are not used for analysis. It's advisable to have consistent image sizes for all data. If there are images of different sizes, performance may suffer, so it's best to crop or resize them to a consistent size in advance. The data should consist of only normal data or mostly normal data with a few abnormal data. Various types of abnormal data can be used. For example, if you want to distinguish between cats and other animals, prepare about 90% cat images and the remaining 10% can be other animals like dogs, mice, or cat dolls. Of course, preparing abnormal data that may occur in real operational environments can prevent performance degradation in operational situations.

#### Selecting Method for Setting Anomaly Detection Thresholds
While VAD can only be trained with normal data, if there are a few abnormal data, a more rigorous anomaly detection model can be created. Strictly speaking, even if abnormal data are not used for training, the anomaly_threshold, a numerical value used to determine abnormality, can be adjusted. If significant impact on performance is expected without specific settings, the method of automatic setting may change, but if set according to the situation, an appropriate method suitable for VAD operational situations can be specified. If only normal data is available, the threshold_method setting method provided by VAD is Percentile. Similarly, if there are very few abnormal data, it's also good to set it to Percentile. Depending on the number of normal images, if there are slightly more than 5 abnormal data, using the F1 method can find a suitable anomaly_threshold. By using percentile in the training argument, you can determine the level of the anomaly_score to be determined as abnormal if only normal images are available.

#### Model Selection
VAD supports two models: FastFlow and PatchCore. Usually, PatchCore has better performance and requires less training time compared to FastFlow, but the inference time is longer, and slightly more RAM memory is needed. Choose according to the operational infrastructure environment.

#### Setting Training Conditions
The training conditions that can be set in VAD are mainly image size (`img_size`), batch size (`batch_size`), and maximum epochs(`max_epochs`). To achieve faster and lighter inference, VAD performs resizing based on the entered image size. For FastFlow, the image size is fixed at 256 pixels wide and 256 pixels high, as the model requires a fixed image format. For PatchCore, as it internally cuts patches for inference, it can accept rectangular inputs. Increasing the batch size, i.e., the number of images trained at once, can lead to faster training completion and improved training stability, but it also consumes more RAM memory of the training infrastructure. The maximum number of epochs is an argument required only for FastFlow, where in deep learning algorithms, training is repeated for the specified number of epochs. If the number of epochs is insufficient, under-fitting may occur, leading to inadequate performance. Increasing the number of epochs will increase the training time, but with the EarlyStopping feature, if the performance is achieved before the specified number of times, the training will be completed.


<br/>

--- 
 
## Feature Details
Detailed explanation for utilizing various functions of Visual Anomaly Detection.

### Train pipeline: input asset
#### Read Ground Truth file
Reads the Ground Truth file and identifies image paths and labels in the `input asset`.

### Train pipeline: readiness asset
#### Check data quality
Reads the Ground Truth file, verifies the column names input as arguments, removes missing values, and verifies the existence of images in the image paths. It also checks whether the labels for normal images are correct. VAD divides the training/validation datasets and utilizes the validation dataset to perform training optimally. If there is no item to differentiate the training/validation datasets, it creates one and ensures it is separated in the manner required by VAD. VAD can be used with either only normal images or mostly normal images and a few abnormal images. In `readiness`, based on the ratio of normal/abnormal data and the minimum number of abnormal data, it changes to the appropriate user arguments (`threshold_method` and `monitor_metric`) for validation data.

### Train pipeline: modeling(train) asset
#### Load image file
Reads images from the data in `image_path` in the Ground Truth.

#### Training parameters
If there are abnormal data in the validation data, the `anomaly_threshold`, which determines abnormality, is trained using the F1-score maximization method. If there are no abnormal data, training proceeds according to the `percentile` argument set in advance, and the anomaly_threshold value is set. For FastFlow, if the training is insufficient, `max_epochs` can be increased to ensure sufficient training.

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

<br/>

---

**VAD Version: 1.0.0**
