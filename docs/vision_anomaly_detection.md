# Vision Anomaly Detection (VAD)

<div align="right">Updated 2024.06.11</div><br/>

## What is Visual Anomaly Detection?
Visual Anomaly Detection (VAD) is an AI content that detects abnormal images by learning normal images. By training with normal images or images of the same type, VAD can alert users when there are images different from the trained ones, without the need for users to create a ground truth, thus reducing the cost of creating ground truth.

<br/>

---

## When to use Visual Anomaly Detection?
VAD can classify images composed of mostly normal or specific types and a few abnormal or other types. VAD can be applied in any field with only normal or specific type images without ground truth labels. The main application areas include:

- Manufacturing: In processes such as appearance inspection of products or vision inspection, where most of the products are normal and there are few abnormal products, abnormal products can be automatically classified and verified without a ground truth.
- Inventory Management: Images without ground truth labels can be used to verify that other products are not mixed.
- Agriculture: Monitoring crop conditions via smartphone images and detecting diseases, pests, abnormal growth, etc., early on. This can increase the efficiency of crop management.

Below are three examples of actual cases where VAD was applied:

#### Bolt Inspection (TBD)
An image-based anomaly detection solution for inspecting bolts and the proper assembly of various electric vehicle components.

#### Critical Defect Detection (TBD)
An image-based anomaly detection solution for detecting defects in parts produced during the mass production process of electric vehicle components when workers do not follow assembly instructions or when defects occur in parts produced by automated equipment.

#### Box Appearance Inspection (TBD)
A solution for detecting defects in the appearance of packaging boxes in warehouses before products are shipped out.

<br/>

---

## Key Features

#### Excellent Performance
VAD can use the latest models such as PatchCore and FastFlow, which have excellent performance among image-based anomaly detection AI models. PatchCore utilizes pre-trained models, so there is no need for training, reducing training costs, and it maintains stable performance. FastFlow is an image anomaly detection model that utilizes generative models that have shown excellence in various fields recently. Both models are lightweight and enable rapid inference.

#### Cost Reduction in Initial Labeling
To use an image-based classification model, a process is required to create ground truth labels by confirming each image. However, since VAD can verify whether an image is normal or of a specific type with only normal or specific type images, the cost of creating ground truth labels can be reduced. In the future, by assigning new abnormal type labels only to images classified as abnormal, a variety of image classification AI models can be quickly switched to, thus enabling more diverse types of image classification AI models.

#### Convenient Usability and Easy-to-Understand Inference Evidence
Since most of VAD is automated, it can be easily used with only normal image data collected. It also provides images with marked areas showing the evidence area for why an image was classified as abnormal. Therefore, users can quickly verify whether the AI has been trained properly by examining images with marked evidence areas.

<br/>

---

## Quick Start

#### Installation
```bash
git clone 
```

#### Set Required Parameters
1. Modify the data paths in `solution/experimental_plan.yaml` to user paths.
	```bash
	external_path:
	    - load_train_data_path: ./solution/sample_data/train/    # Change to user data path
        - load_inference_data_path: ./solution/sample_data/test/ # Change to user data path  
	```

2. Enter the file_type and encoding in 'args' under 'step: Input' according to the train data.
    ```bash
        - step: Input
          args:
            - file_type: csv	# Specify the file extension of the input data.
			  encoding: utf-8	# Specify the encoding type of the input data.

	```
3. Enter train_validate_column, validation_split_rate, and threshold_method in 'args' under 'step: Readiness' according to the train data.
    ```bash
        - step: Readiness
          args:
			  train_validate_column: 	# Specify the column that distinguishes between train and validation.
			  validation_split_rate: 0.1	# Specify the proportion of validation data generated from the input train data if train_validate_column does not exist.
			  threshold_method: F1	# Select the method for determining OK and NG during validation.

	```

4. Enter model_name, img_size, and percentile in 'args' under 'step: Train' according to the train data.
    ```bash
        - step: Train
          args:
            - model_name: fastflow	# Select the model to be used for VAD.
			  img_size: 256	# Set the image size to be used during image training.
			  percentile: 0.8	# Select the percentile of the anomaly score of the validation dataset to be used as the criterion for classifying NG if threshold_method is Percentile.
	```

5. Enter inference_threshold in 'args' under 'step: Inference' according to the train data.
    ```bash
        - step: Inference
          args:
            - inference_threshold: 0.5	# Specify the threshold of the anomaly score to be classified as abnormal.

	```

6. Once the above steps are set, you can generate the model! If you want to set more advanced parameters to create a model that fits your data better, please refer to the right page. [Learn more: VAD Parameters](./vad-parameter.md)

#### Run
* Navigate to the directory where ALO is installed in the terminal, then execute the command `python main.py`. [Learn more: Develop AI Solution](../../alo/create_ai_solution/with_contents)

---

## Topics
[Description of VAD Features](./vad-features.md)
[VAD Input Data and Outputs](./vad-data.md)
[VAD Parameters](./vad-parameter.md)


---
**VAD Version: 1.0.0**
