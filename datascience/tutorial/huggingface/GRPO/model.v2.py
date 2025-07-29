# References
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Conversational.ipynb

import torch
from unsloth import FastLanguageModel, FastModel
import transformers
import os
from datasets import load_dataset, Dataset, IterableDataset
from trl import GRPOConfig, GRPOTrainer
from pprint import pprint
import re
import wandb
from vllm import SamplingParams
from transformers import TextStreamer
from datetime import datetime
import sys
import logging
import argparse
import time


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

# Evaluation logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_dir = "outputs"

os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "evaluation.log")
log_format = '%(asctime)s - %(levelname)s - %(message)s'

file_handler = logging.FileHandler(log_file, mode='w')  # override the log file
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
# Remove all existing file handlers to ensure override
for h in logger.handlers[:]:
    if isinstance(h, logging.FileHandler):
        logger.removeHandler(h)
logger.addHandler(file_handler)

# Training logger
train_logger = logging.getLogger("training")
train_logger.setLevel(logging.INFO)
train_log_file = os.path.join(log_dir, "training.log")
train_file_handler = logging.FileHandler(train_log_file, mode='w')
train_file_handler.setLevel(logging.INFO)
train_formatter = logging.Formatter(log_format)
train_file_handler.setFormatter(train_formatter)
# Remove existing handlers
for h in train_logger.handlers[:]:
    if isinstance(h, logging.FileHandler):
        train_logger.removeHandler(h)
train_logger.addHandler(train_file_handler)

logger.info("Hello world")
train_logger.info("Training logger initialized")

# INSERT_YOUR_CODE

# Metrics logger
metrics_logger = logging.getLogger("metrics")
metrics_logger.setLevel(logging.INFO)
metrics_log_file = os.path.join(log_dir, "metrics.log")
metrics_file_handler = logging.FileHandler(metrics_log_file, mode='w')
metrics_file_handler.setLevel(logging.INFO)
metrics_formatter = logging.Formatter('%(message)s')
metrics_file_handler.setFormatter(metrics_formatter)
# Remove existing handlers
for h in metrics_logger.handlers[:]:
    if isinstance(h, logging.FileHandler):
        metrics_logger.removeHandler(h)
metrics_logger.addHandler(metrics_file_handler)


####### PARAMS

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Set this flag to enable training mode')

args = parser.parse_args()
TRAIN = args.train if '--train' in sys.argv else False


####### BUILDING PROMPT
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
    f"""You are given a problem, think about the problem and provide your workout. \n    Place it between {reasoning_start} and {reasoning_end}. Then provide your solution\n    between {solution_start}{solution_end}"""

print(system_prompt)

# DATASET

from datasets import load_dataset

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

dataset = load_dataset('openai/gsm8k', 'main', split='train')
print(dataset)
dataset[0]

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})

# print(dataset)
# pprint(dataset[0])
# assert int(dataset[0]['answer']), "answer not a number format"

# Format match function 
# This regular expression is used to match a specific format in a string, typically for extracting
# the solution part from a text that contains both reasoning and solution sections.
# - It expects the string to start with optional whitespace.
# - Then it looks for the reasoning section, which starts with the value of `reasoning_start`,
#   contains any characters (non-greedy), and ends with `reasoning_end`.
# - After that, it expects the solution section, which starts with `solution_start`,
#   captures everything up to `solution_end` (the solution itself is captured in a group).
# - Finally, it expects optional whitespace at the end of the string.
# - The flags `re.MULTILINE` and `re.DOTALL` allow the regex to match across multiple lines
#   and let the dot (`.`) match newline characters as well.

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

###### REWARDS FUNCTION

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        res = completion[0]['content']
        if match_format.search(res) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approx(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]['content']
        scores.append(
            sum(0.5 if response.count(tag) == 1 else -0.5 
                for tag in [reasoning_start, reasoning_end, solution_start, solution_end])
        )
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]

    extracted_responses = [
        guess.group(1) #Answer within <Solution> 
        if(guess:= match_format.search(res)) is not None else None
        for res in responses
    ]

    scores = []

    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        if guess == true_answer:
            score += 3.0
        elif guess.strip() == true_answer.strip(): # correct answer but there are spaces in between won't get full points
            score += 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    score += 0.5
                elif 0.8 <= ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0 #wrong answer, penalize
            except:
                score -= 0.5 #unknown format 
        scores.append(score)

    return scores

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags = re.MULTILINE | re.DOTALL
)

def extract_response(response):
    guess = match_numbers.search(response)
    return guess.group(1) if guess != None else None

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]['content']
    responses = [completion[0]['content'] for completion in completions]

    extracted_responses = [
        res
        if (res := extract_response(response)) is not None else None
        for response in responses
    ]

    scores = []
    train_logger.info(f"##############################\nQuestion:\n{question}\nAnswer:\n{answer[0]}\nResponse:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}")

    for guess, true_answer in zip(extracted_responses, answer):
        
        if guess is None:
            scores.append(0.0)
            continue
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        
        except:
            scores.append(0.0)
            continue
    
    return scores

# GRPOConfig and GRPOTrainer

if TRAIN:
    max_seq_length = 1024
    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3-1b-it",
        max_seq_length = max_seq_length, # Choose any for long context!
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!
        r = 8,           # Larger = higher accuracy, but might overfit
        lora_alpha = 8,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )

if TRAIN:
    max_prompt_length = 256
    max_seq_length = 1024

    grpo_config = GRPOConfig(
        learning_rate = 5e-6,                # The initial learning rate for the optimizer
        adam_beta1 = 0.9,                    # Beta1 parameter for Adam optimizer (exponential decay rate for first moment estimates)
        adam_beta2 = 0.99,                   # Beta2 parameter for Adam optimizer (exponential decay rate for second moment estimates)
        weight_decay = 0.1,                  # Weight decay (L2 penalty) to prevent overfitting
        warmup_ratio = 0.1,                  # Fraction of total steps used for learning rate warmup
        lr_scheduler_type = "cosine",        # Type of learning rate scheduler ("cosine" annealing)
        optim = "adamw_torch_fused",         # Optimizer type (fused AdamW for efficiency)
        logging_steps = 1,                   # Log training metrics every N steps
        per_device_train_batch_size = 1,     # Batch size per device (GPU/CPU) during training
        gradient_accumulation_steps = 1,     # Number of steps to accumulate gradients before updating weights (increase for larger effective batch size)
        num_generations = 4,                 # Number of generations per prompt (reduce if out of memory)
        max_prompt_length = max_prompt_length,                   # Maximum length of the input prompt
        max_completion_length = max_seq_length - max_prompt_length, # Maximum length of the generated completion
        num_train_epochs = 5,                # Number of training epochs (uncomment and set for full training run)
        # max_steps = 50,                      # Total number of training steps
        save_steps = 50,                     # Save checkpoint every N steps
        max_grad_norm = 0.1,                 # Maximum gradient norm for gradient clipping
        report_to = "none",                  # Reporting backend ("none" disables reporting, can use "wandb" for Weights & Biases)
        output_dir = "outputs"               # Directory to save model checkpoints and outputs
        
    )

    class PrintCallback(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                train_logger.info(logs)
                metrics_logger.info(logs)
        

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approx,
            check_answer,
            check_numbers,
        ],
        args = grpo_config,
        train_dataset = dataset,
        callbacks = [
            # Callback to print rewards after each step
            PrintCallback()
        ],
    )

    train_logger.info("Starting training...")
    train_logger.info(f"Training config: {grpo_config}")
    train_logger.info(f"Dataset size: {len(dataset)}")
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    train_logger.info(f"Training completed successfully, Elapsed = {end_time - start_time:.2f} seconds")
    train_logger.info("Saving model to outputs/gemma-3-tune1")
    model.save_pretrained('outputs/gemma-3-tune1')
    tokenizer.save_pretrained('outputs/gemma-3-tune1')
    train_logger.info("Model saved successfully")
    sys.exit(1)


######## EVALUATE PIPELINE

# load the model weights
# model.save_pretrained('outputs/gemma-3-tune1')
model, tokenizer = FastModel.from_pretrained('outputs/gemma-3-tune1')
print(type(model))
print(type(tokenizer))
print(model.device)

# Sample inference
messages = [
    {'role': 'system', "content": system_prompt},
    {'role': 'user', "content": "What is the square root of 1010?"},
]

token_ids = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt = True,
    return_tensors = "pt",
    tokenize=True
).to('cuda')

# from transformers import TextStreamer

output = model.generate(
    token_ids,
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    # streamer = TextStreamer(tokenizer, skip_prompt = True),
)

sample = tokenizer.decode(output[0])

import re

test_ds = load_dataset('openai/gsm8k', 'main', split="test")

def generate(model, question, **kwargs):
    MAX_NEW_TOKENS = kwargs.get("max_new_tokens", 64)

    # Sample inference
    messages = [
        {'role': 'system', "content": system_prompt},
        {'role': 'user', "content": question},
    ]

    token_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt = True,
        return_tensors = "pt",
        tokenize=True
    ).to('cuda')

    # from transformers import TextStreamer

    output = model.generate(
        token_ids,
        max_new_tokens = MAX_NEW_TOKENS, # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature = 1.0, top_p = 0.95, top_k = 64,
    )

    return output

def get_answer_from_completion(output: torch.Tensor):
    assert type(output) == torch.Tensor, 'Wrong type: output type must be torch.tensor'

    completion = tokenizer.decode(output[0])
    return guess.group(1) if (guess:= match_format.search(completion)) is not None else None, completion


logging.info("##### EVALUATION #####")
from tqdm import tqdm

correct_answer = 0
total = 0 
loop = tqdm(test_ds, desc="Evaluating")
for X in loop:
    total += 1
    try:
        question = X["question"]
        answer = extract_hash_answer(X['answer'])
        y = generate(model, question, max_new_tokens = 128)
        pred_answer, completion = get_answer_from_completion(y)

        # Log for later analysis

        logger.info('question\t' + question)
        logger.info('target_answer\t'+  answer)
        logger.info('pred_answer\t'+ (pred_answer if pred_answer else 'Wrong format!!'))
        logger.info('completion\t'+ completion)

        correct_answer += 1 if (pred_answer != None and pred_answer == answer) else 0

    except Exception as e:
        logger.error("Error in this question ", question)
        logger.errror(e)
    
    accuracy = (correct_answer / total) * 100
    logger.info(f'Accuracy = {accuracy:.2f}')
    logger.info("################################################")
    loop.set_description(f"accuracy = {accuracy}")

# generate(model, test_ds[0]['question']) 