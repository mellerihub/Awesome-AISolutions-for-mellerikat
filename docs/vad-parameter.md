# VAD Parameter

<div align="right">Updated 2024.06.11</div><br/>

## experimental_plan.yaml Description
To apply AI Contents to the data you have, you need to enter information about the data and the Content features to be used in the experimental_plan.yaml file. When you install AI Contents in the solution folder, you can check the experimental_plan.yaml file provided for each content under the solution folder. Enter the 'data information' in this yaml file and modify/add the 'user arguments' provided for each asset. When you run ALO with the modified user arguments, you can generate a data analysis model with the desired settings.

### experimental_plan.yaml Structure
experimental_plan.yaml contains various setting values necessary to run ALO. By modifying the 'data path' and 'user arguments' parts, you can use AI Contents immediately.

#### Enter Data Path (`external_path`)
- The parameters of `external_path` are used to specify the path to load or save files. If `save_train_artifacts_path` and `save_inference_artifacts_path` are not entered, the modeling outputs are saved in the default directories `train_artifacts` and `inference_artifacts`.

```bash
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path:  ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```
|Parameter Name| DEFAULT| Description and Options|
|---|---|---|
| load_train_data_path | ./sample_data/train/ | 	Enter the folder path where the training data is located (do not enter the csv file name). |
| load_inference_data_path | ./sample_data/test/ | Enter the folder path where the inference data is located (do not enter the csv file name). |

#### User Parameters (`user_parameters`)
- Below `user_parameters`, `step` represents the asset name. `step: input` below indicates the input asset stage.
- `args` represents the user arguments of the input asset (`step: input`). User arguments are parameters related to data analysis provided for each asset. Please refer to the explanation below for details.
```bash
user_parameters:
	- train_pipeline:
		- step: input
		  args: 
			- file_type
			...
		  ui_args:
			...
```
***

## User Arguments Explanation
### What are User Arguments?
User arguments are parameters for configuring the operation of each asset in the experimental_plan.yaml under the `args` of each asset step. They are used to customize various functionalities of the AI Contents pipeline for your data. Users can change or add user arguments based on the following guide to tailor the modeling to their data.
User arguments are divided into "Required arguments," which are predefined in the experimental_plan.yaml, and "Custom arguments," which users add based on the provided guide.

#### Required Arguments
- Required arguments are the basic arguments shown directly in the experimental_plan.yaml. Most required arguments have default values built in. For arguments with default values, the user does not need to set a separate value, as it will operate with the default value.
- Among the required arguments in the experimental_plan.yaml, data-related arguments must be set by the user. (e.g., path_column, y_column)

#### Custom Arguments
- Custom arguments are not listed in the experimental_plan.yaml but are functionalities provided by the asset, which users can add to the experimental_plan.yaml for use. They are added to each asset's 'args' section.

The VAD pipeline consists of Input - Readiness - Modeling (train/inference) - Output assets, and user arguments are configured differently for each asset according to its functionality. Start by using the required user arguments listed in the experimental_plan.yaml and then add more user arguments to tailor the VAD model to your data!

***
## User Arguments Summary

Here's a summary of user arguments for VAD. Click on the 'Argument Name' to navigate to the detailed explanation of the argument.

#### Default
- The 'Default' section indicates the default value of the user argument.
- If there's no default value, it's represented as '-'.
- If there's a logic for the default value, it's indicated as 'Refer to Explanation'. Click on 'Argument Name' to see the detailed explanation.

#### ui_args
- The 'ui_args' in the table below indicates whether the argument supports the `ui_args` feature in AI Conductor's UI for changing argument values.
- O: If the argument name is entered under the `ui_args` in the experimental_plan.yaml, it can be changed in the AI Conductor UI.
- X: Does not support the `ui_args` feature.
- For more information on `ui_args`, refer to the following guide: [Write UI Parameter](../../alo/register_ai_solution/write_ui_parameter)
- In the FCST experimental_plan.yaml, all user arguments that can be `ui_args` are pre-filled under `ui_args_detail`.

#### User Setting Requirement
- The 'User Setting Requirement' in the table below indicates whether the user needs to confirm and change the user arguments to run AI Contents.
- O: These are typically arguments related to tasks or data that the user needs to input before modeling.
- X: Modeling will proceed with default values if the user does not change the values.

| Asset Name | Argument Type | Argument Name | Default | Description | User Setting Requirement | ui_args |
|:----------:|:-------------:|:-------------:|:-------:|:-----------:|:-----------------------:|:-------:|
| Input | Custom | [file_type](#file_type) | csv | Enter the file extension of the input data. | O | X |
| Input | Custom | [encoding](#encoding) | utf-8 | Enter the encoding type of the input data. | X | X |
| Readiness | Custom | [ok_class](#ok_class) | - | Enter the class representing 'ok' in the y_column. | X | X |
| Readiness | Custom | [train_validate_column](#train_validate_column) | - | Enter the column that distinguishes between train and validation. This column should consist of two parameters: train and valid. | X | X |
| Readiness | Custom | [validation_split_rate](#validation_split_rate) | 0.1 | If train_validate_column does not exist, this rate generates validation from the input train data. | X | X |
| Readiness | Custom | [threshold_method](#threshold_method) | F1 | Selects the method for determining OK and NG during validation. Possible values: F1, Percentile (automatically selected depending on the validation data size). | X | X |
| Readiness | Custom | [num_minor_threshold](#num_minor_threshold) | 5 | If the number of OK and NG images does not exceed this, threshold_method is automatically selected as Percentile. | X | X |
| Readiness | Custom | [ratio_minor_threshold](#ratio_minor_threshold) | 0.1 | If the ratio of OK and NG images does not exceed this, threshold_method is automatically selected as Percentile. | X | X |
| Train | Required | [model_name](#model_name) | fastflow | Selects the model to be used for VAD. Possible values: fastflow, patchcore | O | O |
| Train | Custom | [experiment_seed](#experiment_seed) | 42 | Determines the experiment seed in the PyTorch environment. | X | X |
| Train | Custom | [img_size](#img_size) | \[256,256\] | Sets the image size for resizing during image training. If only numbers are entered, the aspect ratio of the original image is maintained during resize (the shorter side becomes the resize size, maintaining the aspect ratio). | X | X |
| Train | Custom | [batch_size](#batch_size) | 4 | Sets the batch size during training and validation. | X | X |
| Train | Custom | [max_epochs](#max_epochs) | 15 | Sets the maximum number of epochs for training. | X | X |
| Train | Custom | [accelerator](#accelerator) | cpu | Selects whether to run based on CPU or GPU. GPU is recommended in environments where it's available. | X | X |
| Train | Custom | [monitor_metric](#monitor_metric) | image_AUROC | Selects the criterion for saving the best model. If threshold_method is Percentile, loss is automatically selected. Possible values: loss, image_AUROC, image_F1Score | X | X |
| Train | Custom | [save_validation_heatmap](#save_validation_heatmap) | True | Selects whether to save prediction heatmaps for the validation dataset. (Only saves for ok-ng, ng-ok, and ng-ng cases) | X | X |
| Train | Custom | [percentile](#percentile) | 0.8 | Selects the percentile of the anomaly score of the validation dataset to judge as NG when threshold_method is Percentile. | X | X |
| Train | Custom | [augmentation_list](#augmentation_list) | \[\] | List of transformations applied for random augmentation. Can include rotation, brightness, contrast, saturation, hue, and blur. | X | X |
| Train | Custom | [augmentation_choice](#augmentation_choice) | 3 | Number of times transformations are applied for random augmentation. Possible values: non-negative integers | X | X |
| Train | Custom | [rotation_angle](#rotation_angle) | 10 | Maximum angle for rotation in random augmentation. If set to 10, transformation is performed between -10 and 10. Possible values: non-negative integers between 0 and 180 | X | X |
| Train | Custom | [brightness_degree](#brightness_degree) | 0.3 | Degree of brightness adjustment in random augmentation. Sets maximum/minimum brightness levels. Possible values: floating-point numbers between 0 and 1 | X | X |
| Train | Custom | [contrast_degree](#contrast_degree) | 0.3 | Degree of contrast adjustment in random augmentation. Possible values: floating-point numbers between 0 and 1 | X | X |
| Train | Custom | [saturation_degree](#saturation_degree) | 0.3 | Degree of saturation adjustment in random augmentation. Possible values: floating-point numbers between 0 and 1 | X | X |
| Train | Custom | [hue_degree](#hue_degree) | 0.3 | Degree of hue adjustment in random augmentation. Possible values: floating-point numbers between 0 and 1 | X | X |
| Train | Custom | [blur_kernel_size](#blur_kernel_size) | 5 | Maximum kernel size for blur in random augmentation. Possible values: integers between 0 and 255 | X | X |
| Train | Custom | [model_parameters](#model_parameters) | \{"fastflow_backborn": 'resnet18', "fastflow_flow_steps": 8, "patchcore_backborn": 'wide_resnet50_2', "patchcore_coreset_sampling_ratio": 0.1, "patchcore_layers": \["layer2", "layer3"\]\} | Parameters related to model training. If not set, the model is trained with default parameters. For details, refer to the parameter description below. | X | X |
| Inference | Custom | [inference_threshold](#inference_threshold) | 0.5 | Threshold of anomaly score for being judged as abnormal. | X | X |
| Inference | Custom | [save_anomaly_maps](#save_anomaly_maps) | False | Whether to save XAI image results. | X | X |


***

## User Arguments Detailed Description     

### Input asset

#### file_type 
Enter the file extension of the input data.

- Argument type: Custom 
- Input type
    - string
- Possible values
    - **csv (default)**
- Usage
    - file_type: csv
- ui_args: X

***

#### encoding 
Enter the encoding type of the input data.

- Argument type: Custom 
- Input type
    - string
- Possible values
    - **utf-8 (default)**
- Usage
    - encoding: utf-8
- ui_args: X

***

### Readiness asset

#### ok_class 
Enter the class representing 'ok' in the y_column. If not entered, ok_class is entered as the name that occupies the most in the training dataset.

- Argument type: Custom
- Input type
    - string
- Possible values
    - **'' (default)**
- Usage
    - ok_class: good
- ui_args: O

***

#### train_validate_column 
Enter the column that distinguishes between train and validation. This column should consist of two parameters: train and valid.

- Argument type: Custom 
- Input type
    - string
- Possible values
    - **'' (default)**
- Usage
    - train_validate_column: phase
- ui_args: X

***

#### validation_split_rate 
If train_validate_column does not exist, this rate generates validation from the input train data.

- Argument type: Custom
- Input type
    - float
- Possible values
    - **0.1 (default)**
- Usage
    - validation_split_rate: 0.1
- ui_args: X

***

#### threshold_method 
Selects the method for determining OK and NG during validation. Possible values: F1, Percentile (automatically selected based on the number of validation data).

- Argument type: Custom 
- Input type
    - string
- Possible values
    - **F1 (default)**
    - Percentile
- Usage
    - threshold_method: Percentile
- ui_args: X

***

#### num_minor_threshold
If the number of OK and NG images does not exceed this, the threshold_method is automatically selected as Percentile.

- Argument type: Custom 
- Input type
    - integer
- Possible values
    - **5 (default)**
- Usage
    - num_minor_threshold: 5
- ui_args: X

***

#### ratio_minor_threshold
If the ratio of OK and NG images does not exceed this, the threshold_method is automatically selected as Percentile.

- Argument type: Custom 
- Input type
    - float
- Possible values
    - **0.1 (default)**
- Usage
    - ratio_minor_threshold: 0.1
- ui_args: X

***

### Train asset

#### model_name 
Selects the model to be used for VAD. Possible values: fastflow, patchcore

- Argument type: Required
- Input type
    - string
- Possible values
    - **fastflow (default)**
    - patchcore
- Usage
    - model_name: patchcore
- ui_args: O

***

#### img_size  
Sets the image size for training. If only numbers are entered, the aspect ratio of the original image is maintained during resizing.

- Argument type: Custom 
- Input type
    - integer
- Possible values
    - **\[256,256\] (default)**
    - 256
- Usage
    - img_size: \[256,256\]
- ui_args: X

***

#### batch_size 
Sets the batch size for training and validation.

- Argument type: Custom 
- Input type
    - integer
- Possible values
    - **4 (default)**
- Usage
    - batch_size: 32
- ui_args: X

***

#### experiment_seed
Determines the experiment seed in the PyTorch environment.

- Argument type: Custom 
- Input type
    - integer
- Possible values
    - **42 (default)**
- Usage
    - experiment_seed: 42
- ui_args: X

***

#### max_epochs
Sets the maximum number of epochs for training.

- Argument type: Custom 
- Input type
    - integer
- Possible values
    - **15 (default)**
- Usage
    - max_epochs: 15
- ui_args: X

***

#### accelerator
Selects whether to run based on CPU or GPU.

- Argument type: Custom 
- Input type
    - string
- Possible values
    - **cpu (default)**
    - gpu
- Usage
    - accelerator: gpu
- ui_args: X

***

#### monitor_metric 
Selects the criterion for saving the best model. Possible values: loss, image_AUROC, image_F1Score

- Argument type: Custom 
- Input type
    - string
- Possible values
    - **image_AUROC (default)**
    - loss
    - image_F1Score
- Usage
    - monitor_metric: loss
- ui_args: X

***

#### save_validation_heatmap
Selects whether to save prediction heatmaps for the validation dataset.

- Argument type: Custom 
- Input type
    - bool
- Possible values
    - **True (default)**
    - False
- Usage
    - save_validation_heatmap: True
- ui_args: X

***

#### percentile
Selects the percentile of anomaly score in the validation dataset to determine NG when threshold_method is Percentile.

- Argument type: Custom 
- Input type
    - float
- Possible values
    - **0.8 (default)**
- Usage
    - percentile: 0.8
- ui_args: X



***

#### augmentation_list
List of transformations applied for random augmentation. Multiple options can be used: rotation, brightness, contrast, saturation, hue, blur.

- Argument type: Custom 
- Input type
    - list
- Possible values
    - **\[\] (default)**
    - rotation, brightness, contrast, saturation, hue, blur
- Usage
    - augmentation_list: \[rotation, brightness, contrast, saturation, hue, blur\]
- ui_args: X

***

#### augmentation_choice
Number of times transformations are applied for random augmentation.

- Argument type: Custom 
- Input type
    - int
- Possible values
    - **3 (default)**
- Usage
    - augmentation_choice: 2
- ui_args: X

***

#### rotation_angle
Maximum angle for rotation in random augmentation. If set to 10, transformation is performed between -10 and 10.

- Argument type: Custom 
- Input type
    - float
- Possible values
    - **10 (default)**
- Usage
    - rotation_angle: 30
- ui_args: X

***

#### brightness_degree
Degree of brightness adjustment in random augmentation. Sets maximum/minimum brightness levels.

- Argument type: Custom 
- Input type
    - float
- Possible values
    - **0.3 (default)**
- Usage
    - brightness_degree: 0.5
- ui_args: X

***

#### contrast_degree
Degree of contrast adjustment in random augmentation.

- Argument type: Custom 
- Input type
    - float
- Possible values
    - **0.3 (default)**
- Usage
    - contrast_degree: 0.5
- ui_args: X

***

#### saturation_degree
Degree of saturation adjustment in random augmentation.

- Argument type: Custom 
- Input type
    - float
- Possible values
    - **0.3 (default)**
- Usage
    - saturation_degree: 0.5
- ui_args: X

***

#### hue_degree
Degree of hue adjustment in random augmentation.

- Argument type: Custom 
- Input type
    - float
- Possible values
    - **0.3 (default)**
- Usage
    - hue_degree: 0.5
- ui_args: X

***

#### blur_kernel_size
Maximum kernel size for blur in random augmentation.

- Argument type: Custom 
- Input type
    - int
- Possible values
    - **15 (default)**
- Usage
    - blur_kernel_size: 25
- ui_args: X

***

#### model_parameters 
Parameters related to model training. If not set, the model is trained with default parameters. For details, refer to the parameter description below.

- Argument type: Custom 
- Input type
    - dictionary
- Possible values
    - **\{"fastflow_backborn": 'resnet18', "fastflow_flow_steps": 8, "patchcore_backborn": 'wide_resnet50_2', "patchcore_coreset_sampling_ratio": 0.1, "patchcore_layers": \["layer2", "layer3"\]\} (default)**
    - fastflow_backborn: "resnet18", "wide_resnet50_2", "cait_m48_448", "deit_base_distilled_patch16_384"
    - fastflow_flow_steps: int
    - patchcore_backborn: any model available in the timm package, but it should interact with the names of layers in patchcore_layers.
    - patchcore_coreset_sampling_ratio: float between 0 and 1
    - patchcore_layers: list of the names of selected backborn's layers
- Usage
    - model_parameters: \{"fastflow_backborn": 'resnet18', "fastflow_flow_steps": 8, "patchcore_backborn": 'wide_resnet50_2', "patchcore_coreset_sampling_ratio": 0.1, "patchcore_layers": \["layer2", "layer3"\]\}
- ui_args: X

***

### Inference asset 

#### inference_threshold 
Threshold of anomaly score for being judged as abnormal.

- Argument type: Required
- Input type
    - float
- Possible values
    - **0.5 (default)**
- Usage
    - inference_threshold: 0.5
- ui_args: X

***

#### save_anomaly_maps
Whether to save XAI image results.

- Argument type: Custom 
- Input type
    - bool
- Possible values
    - **False (default)**
    - True
- Usage
    - save_anomaly_maps: True
- ui_args: X

***   

**VAD Version: 1.0.0**
