# Training Process Reward Model for Mathematical Reasoning

## Dataset Preparation
Please prepare a JSON file with the following format:
```
[
    {
        "text": "Janet pays $40/hour for 3 hours per week of clarinet lessons.."    
        "value": [0.8, 0.6, 0.3 ...]
    },
    {
        "text": "Val cuts a single watermelon into 40 slices.."    
        "value": [0.5, 0.7, 0.2 ...]
    }
]
```
Here is a full example of a text
```
Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year? Step 1: Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons. киStep 2: She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week. киStep 3: She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week. киStep 4: Janet spends 120 + 140 = <<120+140=260>>260 on music lessons per week. киStep 5: She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year. The answer is: 13520 ки
```
We use `ки` to separate each step.

Here is a full example of a value
```
[1.0, 0.8, 0.6, 0.7, 1.0]
```
The length of the list must be the same as the number of `ки`, as we want to train the reward model to predict the process reward we specify.

We also provide a sample dataset `sample_dataset.json` for reference

## Training Code

Please run the following bash file
```
bash kl_math_reward_ce.sh
```
You can specify the hyper-parameters within this file (`data_path`, `model_name`, `learning_rate`, `batch_size`...)
