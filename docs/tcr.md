# Tabular Classification/Regression (TCR)

<div align="right">Updated 2024.05.05</div><br/>

## What is Tabular Classification/Regression?

TCR, short for Tabular Classification/Regression, is an AI content designed to tackle classification and regression problems using tabular data. TCR provides a variety of machine learning models to solve these problems. It leverages years of experience from data analysts who have solved numerous classification and regression issues, selecting the best models and finding the optimal parameters for each. By comparing the performance of different models, TCR helps identify the best one.

One of TCR's strengths is its ease of use. Users can input a few parameters in the experimental_plan.yaml file and run ALO to create the desired classification and regression models based on the input data. Besides modeling, TCR includes various functionalities for tabular data. Following TCR's pipeline, the data is first checked for modeling suitability, automatically preprocessed, and finally used to generate models and predictions. TCR's inspection, preprocessing, and HPO features operate automatically without additional settings, allowing users to easily and conveniently create models with minimal parameter configuration. Additionally, TCR offers template files for adding new machine learning models, making it easy to integrate them into the existing TCR model list and perform HPO with them.

<br/>

---

## When to use Tabular Classification/Regression?

TCR can be used for various classification and regression modeling tasks involving tabular data. It is applicable across various domains as long as the data is in tabular format with multiple variables and label columns. Some areas where TCR can be applied include:

- **Finance:** TCR can be used for tasks such as customer credit rating classification and company bankruptcy prediction. For example, if there is a label column indicating customer credit ratings along with personal information, transaction history, and credit history, TCR can create a model to classify customer credit ratings. Similarly, it can create regression models to predict bankruptcy based on company financial information and market trends.
- **Healthcare:** TCR can classify the presence of specific diseases (e.g., cancer, diabetes) using patient medical records, genetic information, and biosignals as input. This is useful for early disease detection and treatment.
- **Marketing:** TCR can be used for tasks such as customer segmentation classification, customer churn prediction, and advertising effectiveness prediction. For instance, it can create models to classify customer groups based on purchase history, website visit records, and personal information, aiding in customer management and marketing strategy development.
- **Public Sector:** TCR can predict crime, traffic volume, and election outcomes. For example, it can create models to predict the likelihood of crime occurrence in specific areas based on demographic information, past crime records, and economic conditions.

Here are two real-world examples of TCR applications:

#### Bolt Fastening Inspection
Bolt Fastening Inspection analyzes torque and angle data generated during the bolt fastening process to determine the normality of bolt fastening. Developed using TCR, this solution is currently in operation in 18 processes at LG Magna Ramos plant's bolt fastening line, where the risk of bolt mixing is high.

#### Customer Index Development
TCR is used for developing various customer. It has been used to develop indices such as the 'Customer Satisfaction Index' and 'Customer Experience Delivery Index' to identify potential customer complaints, and the 'ThinQ Home Customer Index' to improve the experiences of ThinQ users. TCR is embedded as a default model in the Customer Index Platform, facilitating the active use of TCR for developing customer indices.

<br/>

---

## Key Features

#### Automated Modeling with AutoML
Data analysts no longer need to worry about modeling. TCR automates the processes required for modeling through its AutoML feature, allowing analysts to focus on data discovery and analysis. TCR selects the appropriate hyperparameters for each model and identifies the optimal model for the data. It includes the top 5 machine learning models most frequently used by data analysts, along with their respective hyperparameter sets. With TCR, analysts can easily apply these models to their data and select the best model through performance comparison.

#### Built-in Data Imbalance Handling and Preprocessing Know-How
TCR incorporates data inspection and missing value handling functionalities developed based on the practical know-how of data analysts. Even if users do not specify preprocessing methods required for modeling, TCR automatically applies the appropriate preprocessing methods based on the column types and missing value ratios in the data. Users only need to input the data information into TCR, which then applies the data analyst's know-how to data inspection, preprocessing, and modeling.

#### Rich Modeling Experiments Without Coding
TCR offers a variety of experimental parameters selected by advanced data analysts to facilitate various modeling experiments. Users can refer to the TCR user arguments guide to input various preprocessing and modeling experiment conditions into the experimental_plan.yaml file and run TCR with the desired settings. Simply writing the experimental parameters into the YAML file allows users to conduct various test cases from preprocessing to modeling. No coding is required for modeling experiments. Use TCR's parameter settings to find the optimal model for your data.

<br/>

---

## Quick Start

#### Installation
 For Git code access to AI Contents, refer to ([AI Contents Access](https://mellerikat.com/user_guide/data_scientist_guide/ai_contents#access)).
```bash
git clone https://github.com/mellerikat/alo.git {project_name}
cd {project_name}
pip install -r requirements.txt
git clone https://github.com/mellerikat-aicontents/Tabular-Classification-Regression.git solution
```

#### Essential Parameter Settings
1. **Modify Data Paths:**
    - Edit `solution/experimental_plan.yaml` to change the data paths to your own paths.
    ```bash
    external_path:
        - load_train_data_path: ./solution/sample_data/train/    # Change to your data path
        - load_inference_data_path: ./solution/sample_data/test/ # Change to your data path
    ```

2. **Specify `x_columns` and `y_column`:**
    - Input the appropriate `x_columns` and `y_column` for your train data in the `readiness` step.
    ```bash
        - step: readiness
          args:
            - x_columns: [x0, x1, x2, ...]    # Enter the training column names for your data
              y_column: target                # Enter the y column name for your data
              task_type: classification       # Specify 'classification' or 'regression' based on the task
    ```

3. **Run ALO:** With steps 1 and 2 set, you can run ALO to generate classification or regression models! For advanced parameter settings to create models tailored to your data, refer to [TCR AI Parameter](./parameter).

#### Execution
- Navigate to the directory where ALO is installed and run the following command in the terminal: `python main.py`. [See more: Use AI Contents](../../alo/create_ai_solution/with_contents)

---

## Topics
- [TCR Features](./features)
- [TCR Input Data and Artifacts](./data)
- [TCR Parameters](./parameter)
- [TCR Release Notes](./release)

---
**TCR Version: 2.2.3, ALO Version: 2.7.0**
