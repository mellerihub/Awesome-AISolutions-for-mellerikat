# 📖Solution 

## Solution Methodology Description

'Text classification methodology using BERT model' is a binary classification method that can reflect both the column name and the contextual meaning of the text in the values ​​within the column. This methodology involves a process of changing the column name and the values ​​within it into sentence text. For example, for a certain row of table data, let's assume that the value of column 'A' is 'apple', the value of column 'B' is 'banana', and the value of column 'C' is 'carrot'. Then, the sentence data of the row is processed in the form of 'A is apple, B is banana, C is carrot'. If there is a missing value, the missing value is processed by not including the contents of the column in the sentence. For example, if there is nothing written in column 'B' of the row, the sentence becomes 'A is apple, C is carrot'. Using this sentence data, a BERT-based model is used to perform binary classification on the target column.

## Key features and benefits

> Please write from a technical or domain perspective.

**Reflecting the contextual meaning of data**
This methodology can reflect the meaning of the text in the column name and the value in the column, and can learn the relationship and dependency between each column by combining all columns into one sentence.

**Easy preprocessing process**

The column name and its value are processed in the same way regardless of whether they are numeric, character, or mixed, and there is no need for a preprocessing process for each variable type. There is no need to process categorical and numeric variables separately. In addition, missing values ​​can be processed automatically and simply.

# 💡Features

## Pipeline

The pipeline of AI Contents consists of a combination of assets, which are functional units. It is largely divided into Train assets and Inference assets.

## Assets

> In addition to asset_{}.py, please also include descriptions of other modules such as dataset.py.

Each step is divided into assets.

**Train asset**
Train asset uses the 'asset_train.py' file. After loading the original train data (train_df.csv), drop unnecessary columns, leave only valid eigenvalues ​​in the target column, and perform data sentence and tokenization processes. After that, use the TinyBERT model to train, and save the inference results from the training process. The trained model is saved as a joblib file and transferred to the inference asset.

**Inference asset**
The inference asset uses the 'asset_inference.py' file. After loading the original test data (test_df.csv), drop unnecessary columns, leave only valid eigenvalues ​​in the target column, and perform data sentence and tokenization processes, just like the Train asset. Receive the trained model from the Train asset, and perform training with the model. After training is complete, you can see the results of which class the model predicted in the last column 'pred_class' of the generated 'output_csv'. The file for operation, inference_summary.yaml, is also saved, and the 'score' value in this file indicates the f1-score of the inference results of the test data.

## Experimental_plan.yaml

To apply AI Contents to the data I have, I need to enter information about the data and the Contents functions to be used in the experimental_plan.yaml file. If you install AI Contents in the solution folder, you can check the experimental_plan.yaml file written by default for each content under the solution folder. If you enter 'data information' in this yaml file and modify/add 'user arguments' provided for each asset and run ALO, you can create a data analysis model with the desired settings.

**Experimental_plan.yaml structure**
Experimental_plan.yaml contains various setting values ​​required to run ALO. If you modify the 'data path' and 'user arguments' parts of these setting values, you can use AI Contents right away.

**Data path input (external_path)**
The external_path parameter is used to specify the path of the file to be loaded or the path of the file to be saved. If save_train_artifacts_path and save_inference_artifacts_path are not entered, the modeling output will be saved in the default paths, train_artifacts and inference_artifacts folders.
```
external_path:
- load_train_data_path: ./solution/sample_data/train_data/
- load_inference_data_path: ./solution/sample_data/inference_data/
- save_train_artifacts_path:
- save_inference_artifacts_path:
- load_model_path:
```

|Parameter name|DEFAULT|Description and options|
|---|----|---|
|load_train_data_path| ./sample_data/train_data/| Enter the folder path where the training data is located. (Do not enter the csv file name)|
|load_inference_data_path| ./sample_data/inference_data/| Enter the folder path where the inference data is located. (Do not enter the csv file name)|

**User parameters (user_parameters)**
The step under user_parameters means the asset name. The step: train below means the train asset step, and step: inference means the inference asset step.
Args means the user arguments of the train asset (step: train) and test asset (step: inference), respectively. User arguments are data analysis-related setting parameters provided for each asset. Some arguments may need to be changed depending on the format of the training data and inference data, and the desired learning settings. For an explanation of this, please refer to the User arguments description below.
```
user_parameters:
    - train_pipeline:
        - step: train
          args:
            - random_state: 42
              data_dir: './dataset/'
              file_name: 'train_df'
              train_ratio: 0.8
              num_train_epochs: 1
              target: 'lead_status'
              positive_label : 'Converted'
              drop_column : 'not_converted_reason'
              valid_statuses : ['Converted', 'Closed Not Converted']
            
    - inference_pipeline:
        - step: inference
          args:
            - random_state: 42
              data_dir: './dataset/'
              file_name: 'test_df'
              model_file: 'best_model.joblib'
              target: 'lead_status'
              positive_label : 'Converted'
              drop_column : 'not_converted_reason'
              valid_statuses : ['Converted', 'Closed Not Converted']


Arguments that can be changed
1. train_pipeline args
- random_state: Sets the random seed.
- file_name: The name of the original train dataset csv file to be loaded.

- train_ratio: The ratio that the train set will occupy when separating the validation set from the train data during the train asset process. (In the example, train:val = 8:2)
- num_train_epochs: Sets the number of epochs to train the model.

2. inference_pipeline args
- random_state: Sets the random seed. It is recommended to set it to the same as the seed of the train_pipeline.
- file_name: The name of the original test dataset csv file to be loaded.

3. Common args
The following args must be set to the same in both train_pipeline and inference_pipeline.
- target: The target column names of the train data and test data. - positive_label: The target value name to set as a positive class (1 among 0 and 1). You can write one of the two class names in the target column.
- drop_column: Write the columns you want to remove during the learning and inference process.
- valid_statuses: Write only the class names (valid) you want to use in the target column. For example, let's say there are three unique values ​​in the target column: 'Converted', 'Closed Not Converted', and 'Error'. If the classes you want to classify are only 'Converted' and 'Closed Not Converted', and 'Error' is an unwanted class, if you set the 'valid_statuses' arg as in the example above, all rows with the target value of 'Error' will be removed from the data.

# 📂Input and Artifacts

## Data Preparation

**Preparing Learning and Evaluation Data**

1. Prepare a csv file consisting of feature and target columns. (Target columns must exist for both train data and test data.)
2. Since it is a binary classification solution, the target column must have two unique values. If not, you must separately set only the valid unique values ​​in the 'valid_statuses' arg of the user argument described above.

3. The column structures of both train data and test data must be the same.

**Input data directory structure example**

- In order to use ALO, the train and inference files must be separated. Please separate the data to be used for training and the data to be used for inference as follows.

```
./{train_folder}/
    └ train_df.csv
./{inference_folder}/
    └ test_df.csv
```



## Outputs

When running training/inference, the following outputs are generated.

**Train pipeline**
```
./alo/train_artifacts/
    └ models/train/
        └ best_model.joblib
    
```

**Inference pipeline**
```
 ./alo/inference_artifacts/
    └ output/
        └ output.csv
    └ score/
        └ inference_summary.yaml

```

The detailed description of each output is as follows.

**best_model.joblib**

The model that was finally trained and saved in the Train asset. This model is used in the Inference asset.

**output.csv**
The csv file that saves the results after the inference pipeline is finished. In the original test_df.csv, a 'pred_class' column with class prediction results (0 or 1) for each row is added.

**inference_summary.yaml**
The 'score' section records the f1 score for the test set prediction results.
