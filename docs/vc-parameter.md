# VC Parameter

<div align="right">Updated 2024.05.17</div><br/>

## Explanation of experimental_plan.yaml

To apply AI Contents to your data, you need to fill in the experimental_plan.yaml file with the data information and the Contents functions you will use. When you install AI Contents in the solution folder, you can find the pre-written experimental_plan.yaml file for each content under the solution folder. By entering 'data information' and modifying/adding 'user arguments' provided by each asset in this yaml file, you can run ALO to generate a data analysis model with the desired settings.

### Structure of experimental_plan.yaml

The experimental_plan.yaml contains various settings needed to run ALO. By modifying the 'data path' and 'user arguments' settings, you can immediately use AI Contents.

#### Entering the Data Path (`external_path`)

- The `external_path` parameter is used to specify the path of files to be loaded or saved. If `save_train_artifacts_path` and `save_inference_artifacts_path` are not entered, modeling artifacts will be saved in the default paths, `train_artifacts` and `inference_artifacts`.

```bash
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path:  ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```
| Parameter Name           | DEFAULT              | Description and Options                                             |
| ------------------------ | -------------------- | ------------------------------------------------------------------- |
| load_train_data_path     | ./sample_data/train/ | Enter the folder path where the training data is located. (Do not enter the CSV file name) |
| load_inference_data_path | ./sample_data/test/  | Enter the folder path where the inference data is located. (Do not enter the CSV file name) |

#### User Parameters (`user_parameters`)

- `user_parameters` under `step` indicates the asset name. Below, `step: input` means the input asset stage.
- `args` means the user arguments for the input asset (`step: input`). User arguments are data analysis-related setting parameters provided by each asset. For more details, refer to the User arguments explanation below.
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

## Explanation of User arguments

### What are User arguments?

User arguments are parameters for setting the operation of each asset, written under `args` for each asset step in the experimental_plan.yaml. Each asset in the AI Contents pipeline provides user arguments so that users can apply various functions to their data. Users can refer to the guide below to change or add user arguments to create a model that fits their data.

User arguments are divided into "Required arguments," which are pre-written in the experimental_plan.yaml, and "Custom arguments," which users add by referring to the guide.

#### Required arguments

- Required arguments are basic arguments that are immediately visible in the experimental_plan.yaml. Most required arguments have default values built-in. If an argument has a default value, users do not need to set the value separately; it will operate with the default value.
- Among the required arguments in the experimental_plan.yaml, users must set the data-related arguments. (e.g., x_columns, y_column)

#### Custom arguments

- Custom arguments are not written in the experimental_plan.yaml by default, but they are functions provided by the asset that users can add to the experimental_plan.yaml. Add these under the 'args' of each asset.

The VC pipeline consists of **Input - Readiness - Modeling (train/inference) - Output** asset stages, with user arguments configured differently according to each asset's function. Try using the required user arguments written in the experimental_plan.yaml first, and then add user arguments to create a VC model that perfectly fits your data!

***

## Summary of User arguments

Below is a summary of the VC user arguments. Click on the 'Argument name' to go to the detailed explanation of that argument.

#### Default

- The 'Default' column shows the default value of the user argument.
- If there is no default value, it is marked as '-'.
- If the default value has logic, it is marked as 'refer to the explanation.' Click on the 'Argument name' to view the detailed explanation.

#### ui_args

- The 'ui_args' column indicates whether the argument supports the `ui_args` function, which allows changing the argument value in the AI Conductor UI.
- O: If you write the argument name under `ui_args` in the experimental_plan.yaml, you can change the argument value in the AI Conductor UI.
- X: Does not support the `ui_args` function.
- For more details on `ui_args`, refer to the following guide: [Write UI Parameter](../../alo/register_ai_solution/write_ui_parameter)
- All potential `ui_args` user arguments are pre-written in the VC experimental_plan.yaml.

#### Required for User Settings

- The 'Required for User Settings' column indicates whether the user must check and change the user argument for AI Contents to operate.
- O: Generally, task or data-related information must be entered by the user before modeling.
- X: If the user does not change the value, the modeling proceeds with the default value.

| Asset Name | Argument Type | Argument Name                          | Default       | Description and Options                                                    | Required for User Settings | ui_args |
|:----------:|:-------------:|:--------------------------------------:|:-------------:|:--------------------------------------------------------------------------:|:---------------------------:|:-------:|
| Input      | Custom        | [file_type](#file_type)                | csv           | Enter the file extension of the input data.                                | X                           | O       |
| Input      | Custom        | [encoding](#encoding)                  | utf-8         | Enter the encoding type of the input data.                                 | X                           | O       |
| Readiness  | Custom        | [y_column](#y_column)                  | label         | Enter the column name where the classification label is written in the Ground Truth file. | X | O |
| Readiness  | Custom        | [path_column](#path_column)            | image_path    | Enter the column name where the image path is written in the Ground Truth file. Do not change during operation. | X | O |
| Readiness  | Custom        | [check_runtime](#check_runtime)        | False         | Set to True if you want to check the execution time of the modeling stage during the experiment. Set `check_memory` to False if you set this to True. | X | O |
| Readiness  | Custom        | [check_memory](#check_memory)          | False         | Set to True if you want to check the memory used during the modeling stage during the experiment. Set `check_runtime` to False if you set this to True. | X | O |
| Train      | Required      | [input_shape](#input_shape)            | [28,28,1]     | Enter the size of the input image.                                          | O                           | O       |
| Train      | Required      | [resize_shape](#resize_shape)          | [32,32,3]     | Enter the image size for model training.                                   | O                           | O       |
| Train      | Custom        | [model_type](#model_type)              | mobilenetv1   | Select the training model.                                                 | X                           | O       |
| Train      | Custom        | [rand_augmentation](#rand_augmentation)| False         | Whether to apply RandAugment.                                               | X                           | O       |
| Train      | Custom        | [exclude_aug_lst](#exclude_aug_lst)    | []            | Exclude inappropriate transformations when using rand_augmentation.        | X                           | O       |
| Train      | Custom        | [epochs](#epochs)                      | 10            | Enter the number of training epochs.                                       | X                           | X       |
| Train      | Custom        | [batch_size](#batch_size)              | 64            | Enter the number of data points to be considered at one time during training. | X                           | X       |
| Train      | Custom        | [train_ratio](#train_ratio)            | 0.8           | Enter the ratio of data to be used for training.                           | X                           | X       |
| Train      | Custom        | [num_aug](#num_aug)                    | 2             | Enter the number of times to randomly transform an image when using RandAugment. | X | X |
| Train      | Custom        | [aug_magnitude](#aug_magnitude)        | 10            | Enter the strength of augmentation when using RandAugment.                 | X                           | X       |
| inference  | Required      | [do_xai](#do_xai)                      | False         | Whether to use XAI.                                                        | O                           | X       |
| inference  | Custom        | [xai_class](#xai_class)                | auto          | Enter the class to be analyzed by XAI.                                     | X                           | X       |
| inference  | Custom        | [mask_threshold](#mask_threshold)      | 0.7           | Enter the sensitivity when outputting XAI results.                         | X                           | X       |
| inference  | Custom        | [layer_index](#layer_index)            | 2             | Enter the number of layers to be analyzed during inference.                | X                           | X       |

***

## Detailed Explanation of User arguments

### Input asset

#### file_type
Enter the file extension of the input data. Currently, AI Solution development only supports CSV files.

- Argument type: Custom


- Input type
    - string
- Input possible values
    - **csv (default)**
- Usage
    - file_type: csv  
- ui_args: O

***

#### encoding
Enter the encoding type of the input data. Currently, AI Solution development only supports UTF-8 encoding.

- Argument type: Custom
- Input type
    - string
- Input possible values
    - **utf-8 (default)**
- Usage
    - encoding: utf-8
- ui_args: O

***

### Readiness asset

#### y_column
Enter the column name where the classification label is written in the Ground Truth file. Users must enter this to match their input data. If the column name is 'label', it does not need to be entered.

- Argument type: Custom
- Input type
    - string
- Input possible values
    - **label (default)**
    - Column name
- Usage
    - y_column: image_label
- ui_args: X

***

#### path_column
This is the column name where the image path is written in the Ground Truth file. During operation, the column name should be designated as image_path. This can be changed during solution development experiments.

- Argument type: Custom
- Input type
    - string
- Input possible values
    - **image_path (default)**
    - Column name
- Usage
    - path_column: Path
- ui_args: X

***

#### check_runtime
If you want to check the time taken during the modeling stage of solution development, set this to True. Set `check_memory` to False to accurately check the execution time.

- Argument type: Custom
- Input type
    - boolean
- Input possible values
    - True
    - **False (default)**
- Usage
    - check_runtime: True
- ui_args: X

***

#### check_memory
If you want to check the memory used during the modeling stage of solution development, set this to True. Set `check_runtime` to False to accurately check the memory usage.

- Argument type: Custom
- Input type
    - boolean
- Input possible values
    - True
    - **False (default)**
- Usage
    - check_memory: True
- ui_args: X

***

### Train asset

#### input_shape
Enter the size of the input image. All input images must be the same size.

- Argument type: Required
- Input type
    - list(int,int,int)
- Input possible values
    - **[28,28,1] (default)**
    - [224,224,3]
- Usage
    - input_shape: [32,32,1]
- ui_args: O

***

#### resize_shape
Enter the image size for model training. If the training image size is large, it will consume a lot of resources, but if the region of interest for classification is small, set it larger. It should be similar to or smaller than input_shape.

- Argument type: Required
- Input type
    - list(int,int,int)
- Input possible values
    - **[32,32,3] (default)**
    - [224,224,3]
- Usage
    - resize_shape: [32,32,1]
- ui_args: O

***

#### model_type
Select the training model. If it is not a high-resolution image, use the default mobilenetv1. If it is a high-resolution image, select high_resolution.

- Argument type: Custom
- Input type
    - string
- Input possible values
    - **mobilenetv1 (default)**
    - high_resolution
- Usage
    - model_type: high_resolution
- ui_args: O

***

#### rand_augmentation
Set whether to apply RandAugment. To improve the model's adaptability to new image data, set this to True. Applying RandAugment will consume additional resources due to data augmentation.

- Argument type: Custom
- Input type
    - boolean
- Input possible values
    - True
    - **False (default)**
- Usage
    - rand_augmentation: True
- ui_args: O

***

#### exclude_aug_lst
When using rand_augmentation, you can exclude transformations that are not suitable. For example, if color is important for image classification, performing transformations like color can degrade the inference performance. Write considering the data and the problem situation to be applied.

- Argument type: Custom
- Input type
    - list
- Input possible values
    - **[] (default)**
    - solarizeadd
    - invert
    - cutout
    - autocontrast
    - equalize
    - rotate
    - solarize
    - color
    - posterize
    - contrast
    - brightness
    - sharpness
    - shearX
    - shearY
    - translateX
    - translateY
- Usage
    - exclude_aug_lst: [invert, color]
- ui_args: O

***

#### epochs
Enter the number of training epochs. Increasing the number will extend the training time. If it is too small, it will result in underfitting, and the model will not be trained. During the training process, the model internally references the validation data, and only the model with the best performance is saved, preventing overfitting.

- Argument type: Custom
- Input type
    - integer
- Input possible values
    - **10 (default)**
    - 1-1000
- Usage
    - epochs: 50
- ui_args: X

***

#### batch_size
The number of data points considered at one time during training. Increasing the number will consume more memory and GPU memory but will improve performance.

- Argument type: Custom
- Input type
    - integer
- Input possible values
    - **64 (default)**
    - 2-256
- Usage
    - batch_size: 32
- ui_args: X

***

#### train_ratio
The ratio of the training data to be actually trained in the model. Random sampling is performed according to the ratio, and unselected data is used as validation data.

- Argument type: Custom
- Input type
    - float
- Input possible values
    - **0.8 (default)**
    - 0-1
- Usage
    - train_ratio: 0.7
- ui_args: X

***

#### num_aug
When using RandAugment, enter the number of times to randomly transform an image. Increasing the number will result in multiple transformations per image, improving the model's adaptability to transformed images. However, increasing it too much may distort the image to an unrealistic level, not helping to improve performance. The paper recommends the default value of 2.

- Argument type: Custom
- Input type
    - integer
- Input possible values
    - **2 (default)**
    - 0-16
- Usage
    - num_aug: 3
- ui_args: X

***

#### aug_magnitude
The strength of augmentation when using RandAugment. Increasing the strength will result in stronger transformations on the image. For example, in rotation, the rotation angle will increase, and in shearX, the X-axis twist will become more pronounced. However, increasing it too much may distort the image to an unrealistic level, not helping to improve performance. The paper recommends the default value of 10.

- Argument type: Custom
- Input type
    - integer
- Input possible values
    - **10 (default)**
    - 0-10
- Usage
    - aug_magnitude: 3
- ui_args: X

***

### Inference asset

#### do_xai
Set whether to use XAI to show which regions of the image were referred to for classification. If inference speed is important and resources are limited, set this to False.

- Argument type: Required
- Input type
    - boolean
- Input possible values
    - True
    - **False (default)**
- Usage
    - do_xai: True
- ui_args: X

***

#### xai_class
Select the class to be analyzed by XAI. If set to auto, the XAI result of the class with the highest probability will be saved. If you want to see the XAI value of the class with the highest probability among specific classes, write the desired class as a list.

- Argument type: Custom
- Input type
    - list
- Input possible values
    - label name
    - **auto (default)**
- Usage
    - xai_class: [OK]
- ui_args: X

***

#### mask_threshold
Sensitivity when outputting XAI results. The lower the value, the wider the area displayed.

- Argument type: Custom
- Input type
    - float
- Input possible values
    - 0-1
    - **0.7 (default)**
- Usage
    - mask_threshold: 0.5
- ui_args: X

***

#### layer_index
The number of layers analyzed during inference. Analyzing more layers will consume more resources.

- Argument type: Custom
- Input type
    - integer
- Input possible values
    - 1-10
    - **2 (default)**
- Usage
    - layer_index: 3
- ui_args: X

***

**VC Version: 1.5.1**
