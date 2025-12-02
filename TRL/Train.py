from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import yaml
import argparse
import os
import wandb


class TrainConfig:
    def __init__(self,config_file):
        self.model_path = "EleutherAI/pythia-70m-deduped"
        self.dataset_name = "TIGER-Lab/MathInstruct"
        self.n_samples = -1
        self.output_dir = "./pythia-70m-math"
        self.num_train_epochs = 3
        self.save_steps = 3750
        self.save_total_limit = 4
        self.logging_steps = 1
        if config_file:
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)
                for key, value in config_dict.items():
                    setattr(self, key, value)

def format_example(example):
    return {
        "prompt": f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n",
        "completion": example['output']
    }

def train_model(config: TrainConfig):
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("Model and Tokenizer Initialized")
    if os.path.exists(config.dataset_name):
        dataset = load_dataset("json", data_files=config.dataset_name)["train"]
        print(f"len dataset {len(dataset)}")
        print(f"first entry {dataset[0]}")
    else:
        split = f"train[:{config.n_samples}]" if config.n_samples > -1 else "train"
        print("Split: ",split)
        dataset = load_dataset(config.dataset_name, split=split)

    dataset = dataset.map(format_example)
    print("Dataset Initialized")

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        optim="adamw_torch",
        report_to="wandb",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=config.num_train_epochs,
        learning_rate=2.0e-5,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        max_length=512,
        bf16=True,
        tf32=True,
        seed=42,
        completion_only_loss=True,
        group_by_length=False,
        full_determinism=True
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print("Trainer Started")
    trainer.train()
    print("Trainer Ended")
    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    wandb.login(key="0944191bcf43ea6231189f995e76d66cc523c13d")
    wandb.init(project="S2L_Cirriculum")
    train_model(TrainConfig(args.config))
    print("Training completed.")
