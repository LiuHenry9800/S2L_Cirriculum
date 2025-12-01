import argparse
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
LLAMA_IGNORE_INDEX = -100


def format_example(example):
    return {
        "prompt": f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n",
        "output": example['output']
    }


def compute_losses(checkpoint_dir, dataset_name, n_samples=80000):
    losses_dir = os.path.join(checkpoint_dir, "losses")
    os.makedirs(losses_dir,exist_ok=True)
    
    checkpoints = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')])
    
    split = f"train[:{n_samples}]" if n_samples > -1 else "train"
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(format_example)
    
    for ckpt_name in checkpoints:
        ckpt_num = ckpt_name.split('-')[-1]
        loss_file = os.path.join(losses_dir, f"ckpt{ckpt_num}_loss.pt")
        
        if os.path.exists(loss_file):
            print(f"{ckpt_name} losses already exist, skipping")
            continue
        
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        print(f"Processing {ckpt_name}")
        
        model = AutoModelForCausalLM.from_pretrained(ckpt_path).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Loaded model and tokenizer, Begin To Run Examples")
        losses = []
        with torch.no_grad():
            for example in tqdm(dataset):
                prompt = example["prompt"]
                full = example['prompt'] + example['output']

                tok_prompt = tokenizer(prompt,return_tensors="pt")
                tok_full = tokenizer(full, return_tensors="pt",truncation=True,max_length=512)

                labels = tok_full.input_ids.clone()
                labels[0, :tok_prompt.input_ids.shape[1]] = LLAMA_IGNORE_INDEX

                inputs = dict()
                inputs["input_ids"] = tok_full.input_ids.cuda()
                inputs["attention_mask"] = tok_full.attention_mask.cuda()

                loss = model(**inputs, labels=labels.cuda()).loss
                losses.append(loss.cpu())
        
        torch.save(losses, loss_file)
        print(f"Saved to {loss_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True, help='Directory containing checkpoints')
    parser.add_argument('--dataset_name', default='TIGER-Lab/MathInstruct')
    parser.add_argument('--n_samples', type=int, default=80000)
    args = parser.parse_args()
    
    compute_losses(args.checkpoint_dir, args.dataset_name, args.n_samples)
