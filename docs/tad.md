# Tabular Anomaly Detection (TAD)
<div align="right">Updated 2024.07.06</div><br/>

## What is Tabular Anomaly Detection?
- TAD (Machine Learning Anomaly Detection) is a system that automatically detects anomalies in data using machine learning.
- This system learns normal data patterns and identifies different patterns as anomalies.
- It is effective in detecting new, unseen anomalies, contributing to risk management and stability across various industries.
<br />

---

## When to use TAD?
---
TAD is useful in the following scenarios:

- When you want to detect new anomalies.
- When the training data consists only of normal data.
- When you want to simplify the pipeline from data preprocessing to model development and deployment for anomaly detection.

TAD can be used for anomaly detection modeling in various domains, including:

- **Manufacturing**: Predicting machine failures, quality control.
- **Finance**: Detecting abnormal transactions, fraud prediction.
- **Healthcare**: Detecting abnormal signs in patients.
- **Public Sector**: Detecting abnormal behavior, crime prediction.

## Key Features
---
- **AutoML Feature**: Automatically finds the optimal model without the need for the user to select and adjust models.
- **Data Preprocessing**: Provides various data preprocessing techniques to improve data quality.
- **Anomaly Detection**: Effectively detects new anomalies based on normal data.
- **User-Friendliness**: Allows users to input a few parameters and execute, creating the desired anomaly detection model for the input data.
- **Code-Free Modeling**: Performs various preprocessing and modeling experiments automatically by inputting parameters in a YAML file.
- **Scalability**: Users can add separate machine learning models to be used along with existing models.

## Quick Start
---
### Installation

- Install ALO. [Learn more: Start ALO](../../alo/install_alo)
- Use the following git address to install the content. [Learn more: Use AI Contents (Lv.1)](../../alo/create_ai_solution/with_contents)
- Git URL: [https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection.git](https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection.git)
- Installation code: `git clone https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection.git solution` (run inside the ALO installation folder)

#### Data Preparation
- Prepare a CSV file containing columns of the data you want to detect anomalies in.
- Each column value should be a float, and if there are empty or NaN values, the corresponding row will be automatically excluded.
  **data.csv**

    | x_col_1 | x_col_2 | time_col(optional) | grouupkey(optional) | y_col(optional) |
    |---------|---------|--------------------|---------------------|-----------------|
    | value 1_1 | value 1_2 | time 1 | group1 | ok |
    | value 2_1 | value2_2 | time 2 | group2 | ok |
    | value 3_1 | value3_2 | time 3 | group1 | ng |
    | ... | ... | ... | ... | ... |

#### Required Parameter Settings
1. Modify the following data paths in `ad/experimental_plan.yaml`.
    If only training is performed, `load_train_inference_data_path` does not need to be modified.
    ```bash
    external_path:
       - load_train_data_path: ./solution/sample_data/train/
       - load_inference_data_path: ./solution/sample_data/test/
    ```

2. Enter `x_columns` and `y_column(optional)` that match the train data in the `args` of `step: readiness`.
   - `x_columns`: Enter the x column names of the user data in a list to use only those columns for model training.
   - `y_column`: Enter the y column name of the user data to use that column as a label. (If blank, it is considered to have no label)

   ```yaml
   - step: readiness
     args:
       - x_columns: [x0, x1, x2, ...]  # Enter the x column names of the user data for training
       - y_column: target              # Enter the y column name of the user data (if blank, it is considered to have no label)
   ```

By setting only steps 1 and 2 and running ALO, you can create a TAD model.

=> For more advanced parameter settings to create a model that better fits your data, refer to the link on the right. [Learn more: TAD Parameter](./parameter)

### Execution

* You can run it in a terminal or a Jupyter notebook. [Learn more: Develop AI Solution](../../alo/create_ai_solution/with_contents)
* The execution results include a trained model file, prediction results, and performance charts.

<br />

---

## Topics
- [TAD Feature Description](./features)
- [TAD Input Data and Outputs](./data)
- [TAD Parameters](./parameter)
- [TAD Release Notes](./release)

<br />

---

**TAD Version: 1.0.1, ALO Version: 2.7.0**
