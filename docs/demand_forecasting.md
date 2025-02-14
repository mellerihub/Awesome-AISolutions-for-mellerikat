# 📖Solution 

## What is Demand Forecasting?

Demand Forecasting is a demand forecasting solution that utilizes the latest AI technology.
It is designed based on Google's TFT (Temporal Fusion Transformer) algorithm,
and appropriately processes observed variables, future planned variables such as promotions and sharing dates, and unchanging variables such as product specifications to make predictions. 
As a result, it demonstrates excellent performance in predicting component and product demand or stock prices that change in accordance with sales and marketing strategies. 
In addition, it provides a wide range of usability to users by increasing the understanding and reliability of the forecast through interval prediction and XAI functions.

## When to use Anomaly Detection?

The fields where the Demand Forecasting solution can be applied are as follows:

- Demand Forecasting: It is easy to utilize not only past data but also the values ​​at that point in time such as temperature, public holiday promotion plans, etc. Therefore, it is suitable for demand forecasting that is greatly affected by the strategy at each point in time.
- Price Forecasting: It is suitable for the price forecasting field where the interpretation of the forecast is important because it identifies the uncertainty of the forecast and provides an explanation of the derived forecast.

## Key features and benefits 

**Easy usability using the experimental plan**

- To use the model and function, you can easily use it by simply changing the argument of the experimental plan, providing easy and efficient usability.

**Simulation function**
- The solution provides a simulation function to identify causal information during the development stage. You can use this function to determine whether the variable was applied as intended.

**Demand forecast suitability**
- The solution is based on Trainsformer and seq2seq LSTM structure.
  Therefore, it is suitable for inputting variables (Known/Unknown/Static) at various points in time.
- Since it performs interval estimation instead of point estimation using Quantile Regression, it can measure the uncertainty of the prediction. 
  You can identify the importance of each point and variable importance through Point Wise Feature Importance Block and Attention Block.

# 💡Features

## Pipeline

The AI ​​Contents pipeline consists of assets for learning and inference and output assets for XAI visualization.

**Train pipeline**
```
Train
```

**Inference pipeline**
```
Inference - Output
```

## Assets

**Train asset**  
This module acquires data according to the experimental plan, updates the Dataset, and trains the model.

**tft_model**
This module loads existing Dataset & Model or creates Dataset and Model according to the experimental plan.

**Inference asset**
It loads the Dataset and Model created from the Train asset and performs interval prediction.
It saves the Point wised Feature Importance and Attention information generated during prediction.

**Inference**
This module performs Inference and XAI.

**Output asset**
The Output asset performs visualization based on the predicted values ​​and XAI values ​​derived from the Inference asset.


## Experimental_plan.yaml

To apply AI Contents to the data I have, I need to enter information about the data and the Contents functions to be used in the experimental_plan.yaml file. If you install AI Contents in the solution folder, you can check the experimental_plan.yaml file that is written by default for each content under the solution folder. If you enter 'data information' in this yaml file and modify/add 'user arguments' provided for each asset and run ALO, you can create a data analysis model with the desired settings.

**experimental_plan.yaml structure**
experimental_plan.yaml contains various setting values ​​required to run ALO. If you modify the 'data path' and 'user arguments' parts of these setting values, you can use AI Contents right away.

**Enter data path (external_path)**
The external_path parameter is used to specify the path of the file to be loaded or the path of the file to be saved. If save_train_artifacts_path and save_inference_artifacts_path are not entered, modeling output will be saved in the default paths, train_artifacts and inference_artifacts folders.
```
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path:  ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```

|Parameter name|DEFAULT|Description and options|
|---|----|---|
|load_train_data_path| ./sample_data/train/| Enter the folder path where the training data is located. (Do not enter a csv file name)|
|load_inference_data_path| ./sample_data/test/| Enter the folder path where the inference data is located. (Do not enter a csv file name)|

**User parameters (user_parameters)**
The step below user_parameters means the asset name. The step below: input means the input asset step.
Args means the user arguments of the input asset (step: input). User arguments are data analysis-related setting parameters provided for each asset. For an explanation of this, please refer to the User arguments description below.
```
user_parameters:
    - train_pipeline:
        - step: input
          args:
            - file_type
            ...
          ui_args:
            ...

```

# 📂Input and Artifacts

## Data Preparation

**Learning Data Preparation**
1. Prepare a csv file containing all the variables you want to use for prediction.
2. The csv file must include the time axis, target, and independent variables.
3. The csv file must not have any missing time axis or values.
4. If you are predicting multiple products, a group separator variable must exist.

**Input data directory structure example**
- In order to use ALO, the train and inference files must be separated. Please separate the data to be used for training and the data to be used for inference as shown below.
- All files under a single folder are collected from the input asset and made into a single dataframe, which is then used for modeling. (Files in subfolders under the path are also merged.)
- All columns of data in a single folder must be the same.
```
./{train_folder}/
    └ train_data1.csv
    └ train_data2.csv
    └ train_data3.csv
./{inference_folder}/
    └ inference_data1.csv
    └ inference_data2.csv
    └ inference_data3.csv
```

## Output

When you run training/inference, the following output is generated:

**Train pipeline**
```
./alo/train_artifacts/
    └ output/train/models.train/
        └ dataset.ckpt
        └ tft_algorithm.ckpt
        └ train_config.yaml
    └ experimental_plan.yaml
```

**Inference pipeline**
```
 ./alo/inference_artifacts/
    └ output/inference/
        └ result_melt.csv
    └ extra_output/inference/
        └ result_weight_og.pickle
        └ weight_avg.csv
        └ weight_pivt.csv
    └ extra_output/output/
        └ decoder.html
        └ encoder.html
        └ static.html
    └ score/
        └ inference_summary.yaml

```

Here are the detailed descriptions of each output:

**result_melt.csv**
A csv file containing the prediction result values.

**result_weight_og.pickle**
A pickle file containing the XAI information derived during the prediction.

**weight_avg.csv**
A value that averages the XAI information by time point.

**weight_pivt.csv**
A value that averages the XAI information per prediction progress.

**decoder.html**
A graph that visualizes the Decoder's XAI value based on weight_pivt.

**encoder.html**
A graph that visualizes the Encoder's XAI value based on weight_pivt.

**static.html**
A graph that visualizes the Static's XAI value based on weight_pivt.
