import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import os
from datasets import load_dataset, Dataset, IterableDataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from pprint import pprint
import re
import wandb
from vllm import SamplingParams

# Check GPU
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected.")

# Environment variables for torch
os.environ["TORCH_LOGS"] = "recompiles"
os.environ['TORCHDYNAMO_CACHE_SIZE_LIMIT'] = '999999999'
import torch._dynamo 
torch._dynamo.config.cache_size_limit = 64

# Login to wandb
wandb.login()

# DATASET
dataset = load_dataset("openai/gsm8k", "main")['train']
print(dataset)

# prompts constant to build dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_hash_answer(text):
    delim = "####"
    if isinstance(text, str) and delim in text:
        return text.split(delim)[1].strip()
    return None

def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def get_gsm8k_questions(split="train") -> object:
    data = load_dataset("openai/gsm8k", "main", split=split)
    if isinstance(data, IterableDataset):
        data = Dataset.from_list(list(data))
    elif isinstance(data, list):
        data = Dataset.from_list(data)
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        },
        remove_columns=["question", "answer"]
    )
    return data

gsm8k = dataset.map(lambda x: { # Note that this map function will keep the original features unless overriden
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": x['question']},
    ],
    'answer': extract_hash_answer(x['answer'])
}) 

pprint(gsm8k[0])

# REWARD FUNCTIONS
def correct_reward_func(prompts, completions, answer, **kwargs):
    reward = kwargs.get('reward', 2.0)
    responses = [completion[0]['content'] for completion in completions]
    extracted_res = [extract_xml_answer(x) for x in responses]
    return [reward if r == a else 0.0 for r,a in zip(extracted_res, answer)]

def integer_reward_func(completions, **kwargs):
    reward = kwargs.get('reward', 0.5)
    responses = [comp[0]['content'] for comp in completions]
    extracted_res = [extract_xml_answer(x) for x in responses]
    return [reward if r.isdigit() else 0.0 for r in extracted_res]

def strict_format_reward_func(completions, **kwargs):
    reward = kwargs.get("reward", 0.5)
    pattern = r"<reasoning>\n.*?\n<reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [reward if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs):
    reward = kwargs.get("reward", 0.5)
    pattern = r"<reasoning>.*?<reasoning>\s<answer>.*?</answer>$"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [reward if match else 0.0 for match in matches]

def count_xml(text):
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")) * 0.001
    if text.count("\n</answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")) * 0.001
    return count

def xml_count_reward_func(completions, **kwargs):
    contents = [completion[0]['content'] for completion in completions]
    return [count_xml(c) for c in contents]



###  MODEL
MODEL_ID = "google/gemma-3-1b-it"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 32
GPU_UTIL=0.6

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True, #enabled vllm
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=GPU_UTIL
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
    ],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing='unsloth', #enable long FT
    random_state = 42
)

### TRAINING 
LR = 5e-6
MAX_PROMPT_LENGTH = 256

# grpo_config = GRPOConfig(
#     learning_rate=LR,
#     adam_beta1 = 0.9,
#     adam_beta2=0.99,
#     weight_decay=0.1,
#     warmup_ratio=0.1,
#     lr_scheduler_type='cosine',
#     optim="paged_adamw_8bit",
#     logging_steps=1,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps= 1,
#     num_generations=8, # generate 8 samples
#     max_prompt_length=MAX_PROMPT_LENGTH,
#     max_completion_length = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
#     max_steps = 250,
#     save_steps = 250,
#     max_grad_norm = 0.1,
#     report_to = 'wandb',
#     output_dir = 'outputs'
# )

# trainer = GRPOTrainer(
#     model = model,
#     reward_funcs = [
#         integer_reward_func, 
#         strict_format_reward_func,
#         soft_format_reward_func,
#         xml_count_reward_func,
#         correct_reward_func, 
#     ],
#     processing_class = tokenizer,
#     args = grpo_config,
#     train_dataset = gsm8k
# )

# trainer.train()


### EVALUATION
from datetime import datetime
VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
print("current_verion = ", VERSION)
model.save_lora(f"outputs/grpo_saved_{VERSION}")

text = tokenizer.apply_chat_template(
    [
        {'role': 'system', "content": SYSTEM_PROMPT},
        {'role': 'user', "content": 'Calculate pi'},
    ],
    tokenize = False,
    add_generation_prompt=True
)

pprint(text)

### REPEAT UNTIL SCORE IMPROVES ON BENCHMARK

