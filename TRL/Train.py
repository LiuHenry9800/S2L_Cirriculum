from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
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
        self.curriculum_method = False
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

def train_regular(config: TrainConfig, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    print("Model Initialized")
    if os.path.exists(config.dataset_name):
        dataset = load_dataset("json", data_files=config.dataset_name)["train"]
        print(f"len dataset {len(dataset)}")
        print(f"first entry {dataset[0]}")
    else:
        split = f"train[:{config.n_samples}]" if config.n_samples > -1 else "train"
        print("Split: ", split)
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


def train_curriculum(config: TrainConfig, tokenizer):
    dataset = load_dataset("json", data_files=config.dataset_name)["train"]
    print(f"Curriculum dataset loaded: {len(dataset)} samples")
    dataset = dataset.map(format_example)
    print("Dataset ready")
    
    samples_per_epoch = len(dataset) // config.num_train_epochs
    current_model_path = config.model_path
    
    for epoch in range(config.num_train_epochs):
        print(f"Curriculum Epoch {epoch + 1}/{config.num_train_epochs}")
        
        run_name = f"{config.output_dir.split('/')[-1]}_epoch{epoch+1}"
        wandb.init(
            project="S2L_Cirriculum",
            name=run_name,
            reinit=True
        )
        print(f"WandB run: {run_name}")
        
        start_idx = epoch * samples_per_epoch
        end_idx = (epoch + 1) * samples_per_epoch if epoch < config.num_train_epochs - 1 else len(dataset)
        epoch_dataset = dataset.select(range(start_idx, end_idx))
        
        print(f"Training on samples {start_idx}-{end_idx} ({len(epoch_dataset)} samples)")
        print(f"Loading model from: {current_model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(current_model_path)
        
        sft_config = SFTConfig(
            output_dir=config.output_dir,
            optim="adamw_torch",
            report_to="wandb",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=2.0e-5,
            weight_decay=0.0,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_steps=999999,
            save_total_limit=1,
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
            train_dataset=epoch_dataset,
            args=sft_config,
        )
        
        trainer.train()
        
        checkpoint_path = os.path.join(config.output_dir, f"epoch_{epoch + 1}")
        trainer.save_model(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        wandb.finish()
        
        current_model_path = checkpoint_path
    
    trainer.save_model(os.path.join(config.output_dir, "final_model"))


def train_model(config: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer made")
    
    if config.curriculum_method:
        print("Using curriculum train")
        train_curriculum(config, tokenizer)
    else:
        print("Using regular train")
        train_regular(config, tokenizer)
    
    print("Training done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    wandb.login(key="0944191bcf43ea6231189f995e76d66cc523c13d")
    wandb.init(project="S2L_Cirriculum")
    train_model(TrainConfig(args.config))
