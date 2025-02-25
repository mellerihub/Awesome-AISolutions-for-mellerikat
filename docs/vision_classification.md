# Vision Classification (VC)

<div align="right">Updated 2024.05.17</div><br/>

## What is Vision Classification?
Vision Classification is a deep learning-based AI content that can automatically classify images using pre-prepared ground truth. Whether you want to enhance the efficiency of your manufacturing process or rigorously manage the quality of your final products, our Vision Classification provides the ideal solution. This technology detects visual defects in manufacturing processes and automates quality inspections. It identifies defects and determines their causes to improve production efficiency. Additionally, it automatically collects data during the inspection process and retrains the AI model based on user-reviewed images to maintain high performance consistently.

Moreover, it can be used in retail for product classification and recognition. It automates product classification and recognition to optimize inventory management and enhance customer service. Customers can take a picture of a product to receive information about it, aiding in purchase decisions. Utilizing the explainable AI (XAI) features included in Vision Classification, users can identify the image areas contributing to the inference, quickly locating the object.

From product inspection to product classification and inventory management, Vision Classification is the next-generation technology partner that adds value to your business.

<br/>

---

## When to use Vision Classification?
VC can be used for image-based classification tasks. It can be applied in any field where images and ground truth labels are available. It can be utilized in the following areas:

* **Product Inspection:** Automate the defect inspection process by collecting normal and defective image data of products and training the model.
* **Automated Inventory Management:** Verify the types of objects/products present using image data. Automatically label numerous product images to aid in inventory management.
* **Object Classification:** Automatically generate tags for objects captured in user-uploaded photos.

Use cases for solution application include:

#### Motor Inspection (TBD)
A solution for performing image-based quality inspection of various electric vehicle components, including drive motors, power converters, and integrated systems.

#### Box Exterior Inspection (TBD)
A solution for inspecting the exterior defects of packaging boxes before products are shipped from the warehouse.

<br/>

---

## Key Features
Vision Classification provides high-performance and efficient deep learning models that deliver excellent results without requiring extensive training resources. It also offers explainability features for inference results.

This is useful for customers who want to detect defects using visually distinguishable image data in various situations such as product production processes or final shipping quality inspections. It is also beneficial for customers who want to verify the types of objects or products through image data. Additionally, it assists customers who want to automatically generate tags for objects in user-uploaded photos.

#### Lightweight Models with Fast Speed and Low Memory Usage
VC allows users to easily utilize high-accuracy deep learning models that do not require extensive training resources. By leveraging lightweight models such as mobilenetV1 and mobilenetV3, as well as a high-resolution model developed by advanced analysts, Vision Classification provides reliable and accurate classification models. It supports fast inference by converting models to TensorFlow Lite, suitable for embedded devices.

#### Visual Confirmation of Inference Basis
VC includes XAI features to provide explainability for inference results. If classified as defective, the defect location is highlighted in the image, making it easy to identify the defect location.

#### Functionality to Select Lightweight/High-Resolution Models Based on User Requirements
It offers simple usability by allowing users to switch between lightweight and high-resolution models by changing the model name in the experimental plan.

<br/>

---

## Quick Start
#### Installation
- Install ALO. [Learn more: Start ALO](../../alo/install_alo)
- Install the content using the provided Git address. [Learn more: Use AI Contents (Lv.1)](../../alo/create_ai_solution/with_contents)
- Git URL: [https://github.com/mellerikat-aicontents/Vision-Classification.git](https://github.com/mellerikat-aicontents/Vision-Classification.git)
- Installation command: `git clone https://github.com/mellerikat-aicontents/Vision-Classification.git solution` (Run within the ALO installation folder)

#### Data Preparation
- Prepare a GroundTruth.csv file with image paths and corresponding ground truth labels.
- Ensure all images have the same size and support png or jpeg extensions. [Learn more: VC Data Structure](./data)
**GroundTruth.csv**

    | label  | image_path                |
    | ------ | ------------------------- |
    | label1 | ./sample_data/sample1.png |
    | label2 | ./sample_data/sample2.png |
    | label1 | ./sample_data/sample3.png |
    | ...    | ...                       |

#### Required Parameter Settings
1. Modify the data paths in `vc/experimental_plan.yaml`.
   Update `load_train_data_path` and `load_inference_data_path` if performing inference.
   ```bash
   external_path:
       - load_train_data_path: ./solution/sample_data/train/
       - load_inference_data_path: ./solution/sample_data/test/
   ```

2. Update `input_shape` in the `train_pipeline` section to match your data.
   If performing inference, update the `inference_pipeline` section accordingly.
   ```bash
       - step: train
         args:
           - model_type: mobilenetv1
             input_shape: [28, 28, 1]
   ```

3. For additional advanced parameter settings such as XAI features, augmentation settings, training image size, or training time adjustments, refer to the following page. [Learn more: VC Parameter](./parameter)

#### Execution

* Run the terminal or Jupyter notebook. [Learn more: Develop AI Solution](../../alo/create_ai_solution/with_contents)
* The execution results include trained model files, prediction results, and performance charts.
* Note: Currently, it cannot be executed on CPUs that do not support AVX.

<br/>

---

## Topics
- [VC Features](./vc-features.md)
- [VC Input Data and Artifacts](./data)
- [VC Parameters](./parameter)
- [VC Release Notes](./release)

<br/>

---

**VC Version: 1.5.1, ALO Version: 2.3.4**
