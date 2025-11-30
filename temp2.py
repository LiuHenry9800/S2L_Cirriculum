from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model_path = "EleutherAI/pythia-70m"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("TIGER-Lab/MathInstruct", split="train[:10000]")

# Use prompt-completion format
def format_example(example):
    return {
        "prompt": f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n",
        "completion": example['output']
    }

dataset = dataset.map(format_example)

sft_config = SFTConfig(
    output_dir="./pythia-70m-math",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=1,
    save_steps=1875,
    save_total_limit=4,
    max_seq_length=512,
    # This is the key - trains only on completion by default
    completion_only_loss=True,  # This is actually the default for prompt-completion datasets
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()
