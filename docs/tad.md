# Tabular Anomaly Detection (TAD)
<div align="right">Updated 2024.07.06</div><br/>

## What is Tabular Anomaly Detection?
- TAD (Machine Learning Anomaly Detection) is a system that automatically detects anomalies in data using machine learning.
- This system learns normal data patterns and identifies different patterns as anomalies.
- It is effective in detecting new, unseen anomalies, contributing to risk management and stability across various industries.

## When to use TAD?
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
- **AutoML Feature**: Automatically finds the optimal model without the need for the user to select and adjust models.
- **Data Preprocessing**: Provides various data preprocessing techniques to improve data quality.
- **Anomaly Detection**: Effectively detects new anomalies based on normal data.
- **User-Friendliness**: Allows users to input a few parameters and execute, creating the desired anomaly detection model for the input data.
- **Code-Free Modeling**: Performs various preprocessing and modeling experiments automatically by inputting parameters in a YAML file.
- **Scalability**: Users can add separate machine learning models to be used along with existing models.

## Quick Start
### Installation

- Install ALO. [Learn more: Start ALO](../../alo/alo-v2/install_alo)
- Use the following git address to install the content. [Learn more: Use AI Contents (Lv.1)](../../alo/alo-v2/create_ai_solution/with_contents)
- Git URL: [https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection.git](https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection.git)
- Installation code: `git clone https://github.com/mellerikat-aicontents/Tabular-Anomaly-Detection.git solution` (run inside the ALO installation folder)

### Data Preparation
- Prepare a CSV file containing columns of the data you want to detect anomalies in.
- Each column value should be a float, and if there are empty or NaN values, the corresponding row will be automatically excluded.
  **data.csv**

    | x_col_1 | x_col_2 | time_col(optional) | grouupkey(optional) | y_col(optional) |
    |---------|---------|--------------------|---------------------|-----------------|
    | value 1_1 | value 1_2 | time 1 | group1 | ok |
    | value 2_1 | value2_2 | time 2 | group2 | ok |
    | value 3_1 | value3_2 | time 3 | group1 | ng |
    | ... | ... | ... | ... | ... |

### Required Parameter Settings
1. Modify the following data paths in `ad/experimental_plan.yaml`.
    If only training is performed, `load_train_inference_data_path` does not need to be modified.
    ```yaml
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

## TAD Features

### TAD's pipeline
The pipeline of AI Contents consists of a combination of assets, which are functional units. The `Train pipeline` and `Inference pipeline` are composed of a combination of 5 assets.

#### Train pipeline
```bash
    Input - Readiness - Preprocess - Train
```
#### Inference pipeline
```bash
    Input - Readiness - Preprocess - Inference
```

Each step is distinguished as an asset.

### Train Pipeline

#### 1. Input Asset
It reads all files in the path specified by the user in `experimental_plan.yaml` and creates a single dataframe. The data path is specified in `load_train_data_path`, and all files in the folder are read and merged.

#### 2. Readiness Asset
It checks if the data is suitable for TAD modeling. Here, it classifies the column types of the data and verifies the minimum required amount of data. It checks whether all columns of the data are properly classified as numeric or categorical, and whether the proportion of missing values is not high.

#### 3. Preprocess Asset
It performs data preprocessing tasks. This includes handling missing values, encoding categorical columns, scaling numeric data, and removing outliers. The default preprocessing method is to fill categorical columns with the most frequent value and numeric columns with the median value if there are missing values.

#### 4. Train Asset
TAD performs Hyperparameter Optimization (HPO) with 5 built-in models (isolation forest, knn, local outlier factor, one-class SVM, dbscan), selects the optimal model, and trains it. HPO uses StratifiedKFold to divide the dataset into multiple folds, trains the model on each fold, and evaluates its performance. Through this process, it finds the optimal parameters.

##### HPO Feature Provision
TAD's HPO (Hyperparameter Optimization) feature can be turned ON/OFF through parameters. When performing HPO, the following settings are used to find the optimal parameters for each model:
1. **KNN**: An anomaly detection technique that identifies outliers based on the information of K nearest neighbors.
    - `n_neighbors`: Number of neighbors, searched as an integer between 3 and 30.
2. **OCSVM**: A technique that determines anomalies in high-dimensional space.
    - `kernel`: Kernel type, chosen from 'rbf', 'poly', 'sigmoid', 'linear'.
    - `degree`: Degree of polynomial kernel, searched as an integer between 2 and 10.
3. **LOF**: A technique that detects outliers using the ratio of distances between nearest neighbors.
    - `n_neighbors`: Number of neighbors, searched as an integer between 2 and 30.
4. **Isolation Forest**: A technique that detects outliers based on the length of isolation paths in the data.
    - `n_estimators`: Number of trees, searched in increments of 50 between 50 and 500.
    - `max_samples`: Sampling ratio for each tree, searched as a float between 0.5 and 1.0.
5. **DBSCAN**: A density-based clustering technique that detects points in low-density regions as outliers.
    - `eps`: Maximum distance between two samples, searched as a float between 0.1 and 10.
    - `min_samples`: Minimum number of samples required to form a cluster, searched as an integer between 2 and 30.

##### Contamination Ratio Search Feature
To improve the performance of the anomaly detection model, it provides a feature that can automatically search for the ratio of outliers in the data. This feature helps maximize model performance even when users don't know the outlier ratio. It takes `contamination` as an argument to explore various ratios and find the optimal ratio.
- **contamination**: Sets the range of outlier ratios. Enter as a list for Search Range or as a single integer. (Ex, [0.001, 0.2] or 0.01) 

##### Visualization Feature Provision
TAD provides a feature to visualize anomaly detection results. This feature helps to visually confirm detected outliers and easily understand the data distribution and location of outliers.
- It visualizes the distribution of actual data and predicted outliers.
- If there's a y_column, it also visualizes the actual outliers.
- **visualization**: Enter as Boolean Type. (True or False)

### Inference Pipeline

#### 1. Input Asset
The Inference pipeline also has a data input step, which loads files from the inference data path. The data path is specified in `load_inference_data_path`.

#### 2. Readiness Asset
In the Inference pipeline, the readiness asset checks the suitability of the inference data. It checks if there are new values in the categorical columns used during training, and if such new values exist, it raises an error so that users can recognize and handle this.

#### 3. Preprocess Asset
It uses the preprocessing model created in the preprocess asset of the Train pipeline to preprocess the inference data. It applies the same preprocessing methods used during training to maintain consistency.

#### 4. Inference Asset
The Inference asset loads the model trained in the train asset, processes the inference data, and returns the results. This allows for anomaly detection on new data.

## TAD Input Data and Output Manual

### Data Preparation
To use TAD (Tabular Anomaly Detection), the following data preparation is necessary. TAD detects anomalies based on unsupervised learning, so a label column (y column) is not mandatory. 
However, preparing about 5% of anomalous data compared to the amount of normal data can help improve performance.

#### Preparing Training Data
1. Prepare a csv file consisting of the data you want to detect anomalies in and the feature columns.
2. The label column is optional.
3. If grouping, there should be a column for grouping.

#### Input Data Directory Structure Example
- To use ALO, train and inference files must be separated. Please separate the data for training and inference as shown below.
- All files under one folder are combined into a single dataframe in the input asset and used for modeling. (Files in subfolders under the path are also combined.)
- The columns of data in one folder must all be the same.
**Training Data Directory Structure Example:**
```bash
./{train_folder}/
    └ train_data1.csv
    └ train_data2.csv
    └ train_data3.csv 
```
**Validation Data Directory Structure Example:**
 
```
./{inference_folder}/
    └ inference_data1.csv
    └ inference_data2.csv
    └ inference_data3.csv 
```

### Data Requirements
#### Essential Requirements
- Input data must satisfy the following conditions:
  - A y column (label) is not necessary.
  - The higher the proportion of normal data, the better. (It's fine to have only normal data.)
#### Additional Requirements
- To ensure minimum performance, we recommend data where the proportion of normal data is greater than that of anomalous data.
- If the data ratios are similar, we recommend proceeding after oversampling the normal data.

### Outputs
When you run TAD's training/inference, the following outputs are generated.

#### Train pipeline
```
./train_artifacts/
   ├── extra_output/
   │   └── train/
   │       └── eval.json
   ├── log/
   │   ├── experimental_history.json
   │   ├── pipeline.log
   │   └── process.log
   ├── models/
   │   ├── input/
   │   │   └── input_config.json
   │   ├── readiness/
   │   │   └── train_config.pickle
   │   │   └── report.csv
   │   ├── preprocess/
   │   │   └── preprocess_config.json
   │   │   └── scaler.pickle
   │   └── train/
   │       ├── isf_default.pkl
   │       ├── knn_default.pkl
   │       ├── lof_default.pkl
   │       ├── ocsvm_default.pkl
   │       ├── dbscan_default.pkl
   │       └── params.json
   ├── output/
   │   └── train_pred.csv
   └── score/
       └── train_summary.yaml
```
#### Inference pipeline
```
./inference_artifacts/
   ├── extra_output/
   │   └── readiness/
   │   │   └── report.csv
   │   └── inference/
   │       └── eval.json
   ├── log/
   │   ├── experimental_history.json
   │   ├── pipeline.log
   │   └── process.log
   ├── output/
   │   └── test_pred.csv
   └── score/
       └── inference_summary.yaml
```

## TAD Parameter

###  experimental_plan.yaml Explanation

- To apply AI Contents to your data, you need to enter information about the data and the Contents features you want to use in the experimental_plan.yaml file. 

- When you install AI Contents in the solution folder, you can find the basic experimental_plan.yaml file written for each content under the solution folder. 

- By entering `data information` in this YAML file and modifying/adding `user arguments` provided by each asset, you can create a data analysis model with the desired settings when running ALO.

### 1. File Overview
`experimental_plan.yaml` is a configuration file that defines the experiment plan for TAD, including the data path and parameters to be used in various pipeline stages. This file allows you to automate the data preprocessing, model training, and deployment process.

### 2. Structure Explanation
`experimental_plan.yaml` consists of the following main sections:

- External data path settings (`external_path`)
- User parameter settings (`user_parameters`)

### 3. External Data Path Settings (external_path)
Specifies the paths for loading data or saving results.

```yaml
external_path:
  - load_train_data_path: ./solution/sample_data/train/
  - load_inference_data_path: ./solution/sample_data/test/
  - save_train_artifacts_path:
  - save_inference_artifacts_path:
  - load_model_path:
```

| Parameter Name             | DEFAULT                     | Description and Options                                                                        |
|----------------------------|-----------------------------|-------------------------------------------------------------------------------------------------|
| load_train_data_path       | ./sample_data/train/        | Specifies the path to load training data. (Do not enter csv file name) All csv files under the entered path will be concatenated. |
| load_inference_data_path   | ./sample_data/test/         | Specifies the path to load inference data. (Do not enter csv file name) All csv files under the entered path will be concatenated. |
| save_train_artifacts_path  | -                           | Specifies the path to save training results.                                                    |
| save_inference_artifacts_path | -                        | Specifies the path to save inference results.                                                   |
| load_model_path            | -                           | Specifies the path to load an existing model.                                                   |

### 4. User Parameter Settings (user_parameters)

#### Pipeline & Asset
`user_parameters` defines the configuration parameters to be used in each pipeline stage. Each pipeline is divided into `train_pipeline` and `inference_pipeline`, and each pipeline consists of several stages (Assets). 
Each Asset performs a specific data processing task and has various parameters that control that task.

- **Pipeline**: The higher concept of data processing flow, consisting of several stages (Assets).
- **Asset**: A unit that performs individual tasks within the pipeline. For example, data preprocessing, model training, etc.
- **args**: Parameters that configure the operation of each Asset.
- **ui_args**: Defines parameters that users can change in the AI Conductor UI.

### User Arguments Summary
Here's a summary of key user arguments you can configure in TAD:

#### Input Asset
- **file_type**: Enter the file extension of the Input data (default: csv)
- **encoding**: Enter the encoding type of the Input data (default: utf-8)

#### Readiness Asset
- **x_columns**: Enter the columns containing the data you want to use for anomaly detection (Required)
- **y_column**: Enter the column containing label information (Optional)
- **groupkey_columns**: Enter the column for grouping data points (Optional)

#### Preprocess Asset
- **handling_missing**: How to handle missing values (Options: drop, most_frequent, mean, median, interpolation, default: drop)
- **handling_scaling**: How to scale the data (Options: standard, minmax, maxabs, robust, normalizer, none, default: none)
- **drop_duplicate_time**: Whether to remove duplicate time rows (default: True)
- **difference_interval**: Whether to perform differencing on data (default: 0)

#### Train Asset
- **hpo_param**: Whether to enable hyperparameter tuning (default: False)
- **contamination**: Sets the range of anomaly ratios (Example: [0.001, 0.1] or 0.01)
- **models**: Which models to use among the 5 built-in models (Options: knn, ocsvm, lof, isf, dbscan)
- **visualization**: Whether to visualize detection results (default: False)

## User Arguments Detailed Explanation

### Input Asset
#### file_type
Enter the file extension of the Input data. Currently, AI Solution development is only possible with csv files.

- Argument type: Required
- Input type: string
- Possible values: **csv (default)**
- Usage: `file_type: csv`  

#### encoding
Enter the encoding type of the Input data. Currently, AI Solution development is only possible with utf-8 encoding.  

- Argument type: Required
- Input type: string
- Possible values: **utf-8 (default)**
- Usage: `encoding: utf-8`

### Readiness Asset
#### x_columns
Enter the columns containing the data you want to use for anomaly detection. Multiple columns are supported.

- Argument type: Required
- Input type: list
- Possible values: Column names
- Usage: `x_columns: [x_col1, x_col2]`

#### y_column
Enter the column containing information about which label each data point belongs to for anomaly detection. Since TAD basically does not require labels, enter this only if you want to get results using labels. The number of unique values should be less than 3.

- Argument type: Custom
- Input type: string
- Possible values: Column name
- Usage: `y_column: y_col`

#### groupkey_columns
Enter the column containing information about which group each data point belongs to if you want to perform anomaly detection by group. If you don't want to proceed by group, leave it blank. Currently supports one group key column.

- Argument type: Required
- Input type: list
- Possible values: Column name
- Usage: `groupkey_columns: [groupkey_col_example]`

###  Preprocess Asset
#### handling_missing
Determines how to handle missing values in the data you want to perform anomaly detection on. If 'drop', it removes the corresponding row. 'most_frequent' fills with the mode, 'mean' with the average, 'median' with the median, and 'interpolation' with the interpolation value of the previous and next values.

- Argument type: Custom
- Input type: string
- Possible values: **drop (default)**, most_frequent, mean, median, interpolation
- Usage: `handling_missing: drop`

#### handling_scaling
Determines how to scale the data you want to perform anomaly detection on. If 'standard', it scales using the mean and std of the train data to have mean 0 and variance 1. If 'minmax', it adjusts the values to be between 0 and 1 using the min and max values of the train data. If 'maxabs', it adjusts the values to be between 0 and 1 using the maximum absolute value of the train data. If 'robust', it scales using the median and quartile values of the train data. If 'normalizer', it scales so that the length of the feature vector of the data becomes 1. If nothing is entered, no separate scaling is performed.

- Argument type: Custom
- Input type: string
- Possible values: **none (default)**, standard, minmax, maxabs, robust, normalizer
- Usage: `handling_scaling: minmax`

#### drop_duplicate_time
Determines how to handle duplicate rows in the time column of the data you want to perform anomaly detection on. If True, it removes all but one of the rows with duplicate time columns.

- Argument type: Custom
- Input type: string
- Possible values: **True (default)**, False
- Usage: `drop_duplicate_time: True`

#### difference_interval
This item determines whether to perform differencing on the data you want to perform anomaly detection on. It is used when you want to perform anomaly detection based on how much the difference between previous values and the current point has changed. When used, the value should be entered as a positive integer greater than 0.

- Argument type: Custom
- Input type: int
- Possible values: **0 (default)**, Positive integer greater than 0
- Usage: `difference_interval: 1`

###  Train Asset
#### hpo_param
hpo_param is an item that determines the hyper parameter tuning of anomaly detection models.

- Argument type: Custom
- Input type: string
- Possible values: **False (default)**, True
- Usage: `hpo_param: True`

#### contamination
contamination is an item that sets the range of anomaly ratios. This feature helps maximize model performance by finding the optimal ratio during the hpo process even when the user doesn't know the anomaly ratio, or allows the user to set the model by entering a known ratio.

- Argument type: Custom
- Input type: float or list
- Possible values: **'' (default)**, [min_value, max_value], specific_value
- Usage: 
  - `contamination: [0.001, 0.1]`
  - `contamination: 0.0001`

#### models
This is an item to select which models to use among the 5 built-in models. If two or more models are selected, the output is provided as an Ensemble result.

- Argument type: Custom
- Input type: string select
- Possible values: **knn, ocsvm, lof, isf (default)**, dbscan
- Usage:
  ```yaml
  models:
    - knn
    - ocsvm
    - lof
    - isf
    - dbscan
  ```

#### visualization
visualization is an item that determines whether to visualize the detection results of anomaly detection models.

- Argument type: Custom
- Input type: string
- Possible values: **False (default)**, True
- Usage: `visualization: True`
