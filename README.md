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

We also provide a sample dataset `sample_dataset.json` for reference.

## Training Code

Please run the following bash file:
```
bash kl_math_reward_ce.sh
```
We apply cross-entropy loss on the ground-truth value and the prediction.

You can specify the hyper-parameters within this file (`data_path`, `model_name`, `learning_rate`, `batch_size`...)


## Evaluation

We adopt a very similar evaluation strategy as Math-Shepherd. 

Here is a sample evaluation code adopted from Math-Shepherd:
```
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

good_token = ' +'
bad_token = ' -'
step_tag = 'ки'

path = "Your-Model-Path"
tokenizer = AutoTokenizer.from_pretrained(path)
plus_tag_id = tokenizer.encode(' +')[-1]
minus_tag_id = tokenizer.encode(' -')[-1]

candidate_tokens = [plus_tag_id,minus_tag_id]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
model = AutoModelForCausalLM.from_pretrained(path).eval()

question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. киStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. киStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. киStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
output2 = """Step 1: Janet's ducks lay 16 eggs per day. киStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. киStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. киStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 17 is wrong

for output in [output1, output2]:
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])

    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,0] 
        step_scores = scores[input_id == step_tag_id]
        print(step_scores)


```
