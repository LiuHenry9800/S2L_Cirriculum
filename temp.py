from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

model_path = "EleutherAI/pythia-70m"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def format_example(example):
    prompt = (
        "Below is an instruction that describes a task.\n"
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n"
        "### Response:\n"
        f"{example['output']}"
    )
    return {"text": prompt}

dataset = load_dataset("TIGER-Lab/MathInstruct", split="train[:10000]")
dataset = dataset.map(format)

# Collator that masks instruction tokens
collator = DataCollatorForCompletionOnlyLM(
    response_template="\n### Response:\n",
    tokenizer=tokenizer,
)

training_args = TrainingArguments(
    output_dir="./pythia-70m-math",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=1,
    save_steps=1875,
    save_total_limit=4
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=512,
    args=training_args,
)

trainer.train()
