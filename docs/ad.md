# Anomaly Detection (AD)

<div align="right">Updated 2024.05.17</div><br/>

## What is Anomaly Detection?  
Anomaly Detection is AI content that identifies abnormal patterns or outliers within data. It leverages statistical and machine learning techniques to automatically distinguish between normal and abnormal data ranges without the need for separate labeling. Anomaly Detection provides models for detecting point anomalies (PAD), which identify individual data points that deviate from the normal range; contextual anomalies (CAD, TBD), which identify abnormal patterns that deviate from the normal pattern; and multivariate anomalies (MAD, TBD), which comprehensively learn the relationships between multivariate data to detect abnormal points and patterns. This versatility allows Anomaly Detection to be applied to various data characteristics and objectives.

<br />

---

## When to use Anomaly Detection

Anomaly Detection can be applied in the following areas:

* **Manufacturing Process Anomaly Detection:** This feature is for customers who want to monitor the manufacturing process using sensors to detect abnormalities during the manufacturing process. It helps prevent problems by detecting abnormalities in advance.
* **Time Series Anomaly Detection:** This is for customers who want to identify anomalies in time series data, such as stock trends or various trend data, in addition to the manufacturing process. Detecting these anomalies early allows users to take appropriate action.

The solution application use case is as follows:

#### Distribution Partner Return Data Anomaly Detection (TBD)
A solution that detects anomalies when the return volume of distribution partner data suddenly increases, helping to analyze the cause.

<br />

---

## Key Features

Anomaly Detection provides fast and highly efficient statistical models that deliver excellent performance without requiring significant learning resources. It is useful for customers who want to detect anomalies within one-dimensional data obtained during the product production process. It is also beneficial for customers who want to detect sudden anomalies in time series data, such as stock trends.

#### Fast and Low-Memory Statistical Models
Anomaly Detection allows you to easily use efficient statistical and machine learning models that offer excellent performance with minimal learning resources. Based on algorithms such as DynamicThreshold and SpectralResidual, which have fast speed and low memory requirements, AD reliably and quickly detects anomalies.

#### Anomaly Detection for Multiple Columns and Grouped Data
In cases where there are multiple types of data to detect anomalies for each point, or data obtained from different groups or sensors that need anomaly detection by group, AD allows you to easily and quickly perform this by changing the arguments in the experimental plan.

#### Easy Usability with Experimental Plans
Point Anomaly Detection models in Anomaly Detection offer four models: DynamicThreshold, SpectralResidual, stl_DynamicThreshold, and stl_SpectralResidual. These models can be easily used by simply changing the arguments in the experimental plan, providing easy and efficient usability.

<br />

---

## Quick Start

#### Installation
- Install ALO. [Learn more: Start ALO](../../alo/install_alo)
- Use the following git address to install the content. [Learn more: Use AI Contents (Lv.1)](../../alo/create_ai_solution/with_contents)
- Git URL: [https://github.com/mellerikat-aicontents/Anomaly-Detection.git](https://github.com/mellerikat-aicontents/Anomaly-Detection.git)
- Installation code: `git clone https://github.com/mellerikat-aicontents/Anomaly-Detection.git solution` (run in the ALO installation folder)

#### Data Preparation
- Prepare a CSV file where the 1-dimensional data for detecting anomalies exists as columns. It is also possible to have multiple data for each point.
- Each point value should be a real number. If there are empty or NaN values, those rows will be automatically excluded.
  **data.csv**    

    | time column | x column 1 | x column 2 | groupkey |
    | ----------- | ---------- | ---------- | -------- |
    | time 1      | value 1_1  | value 1_2  | group1   |
    | time 2      | value 2_1  | value2_2   | group2   |
    | time 3      | value 3_1  | value3_2   | group1   |
    | ...         | ...        | ...        | ...      |

#### Essential Parameter Settings
1. Modify the data path in the `ad/experimental_plan.yaml`. If only training is to be conducted, you do not need to modify `load_train_inference_data_path`.
    ```bash
    external_path:
       - load_train_data_path: ./solution/sample_data/train/
       - load_inference_data_path: ./solution/sample_data/test/
    ```

2. Enter the columns you want to detect anomalies in the `x_columns` list in the `train_pipeline`. If you want to group the data for anomaly detection, enter the column in the `groupkey` part as a list. And please enter the column containing time values into the `time_column`.
    ```bash
    - step: train
        args:
          - x_columns : [x_column1, x_column2, ...]
            groupkey: [groupkey_column_example]
            time_column: [time_column_example]
            train_models: all
            decision_rule: two
            hpo_repeat: 20
    ```

3. For various other functions and advanced parameter settings, please refer to the page on the right. [Learn more: Anomaly Detection Parameter](./parameter)

#### Execution

* Run in the terminal or Jupyter notebook. [Learn more: Develop AI Solution](../../alo/create_ai_solution/with_contents)
* The execution results will save the trained model file, prediction results, and performance table.

<br />

---

## Topics
- [AD Feature Description](./features)
- [AD Input Data and Outputs](./data)
- [AD Parameters](./parameter)
- [AD Release Notes](./release)

<br />

---

**AD Version: 2.0.1, ALO Version: 2.7.0**
