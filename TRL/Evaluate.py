import re
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(instruction):
    return f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"


def check_match(pred_answer,actual_answer):
    pred_extracts = re.findall(r"[-+]?\d*\.?\d+",pred_answer)
    pred_extract = None
    if(pred_extracts):
        pred_extract = pred_extracts[-1]
    actual_extracts = re.findall(r"[-+]?\d*\.?\d+",actual_answer)
    actual_extract = None
    if(actual_extracts):
        actual_extract = actual_extracts[-1]
    # print("pred_answer: ",pred_answer)
    # print("actual_answer: ",actual_answer)
    # print("extracts: ",actual_extract,pred_extract)
    return actual_extract == pred_extract


def evaluate_model_accuracy(model_path, dataset_path, start_idx, end_idx):
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Evaluator for {model_path} initialized!")
    
    if dataset_path == "EleutherAI/hendrycks_math":
        dataset = load_dataset(dataset_path, split=f"test[{start_idx}:{end_idx}]")
    elif dataset_path == "openai/gsm8k":
        dataset = load_dataset(dataset_path, "main", split=f"test[{start_idx}:{end_idx}]")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_path}")
    
    correct, total = 0, 0
    print("Start Evaluation")
    print("tokenizer pad token: ", tokenizer.pad_token_id)
    
    for example in tqdm(dataset):
        if dataset_path == "EleutherAI/hendrycks_math":
            instruction = example["problem"]
            ans = example["solution"]
        elif dataset_path == "openai/gsm8k":
            instruction = example["question"]
            ans = example["answer"]
        else:
            raise ValueError(f"no such dataset")
        
        prompt = format_prompt(instruction)
        
        model_input = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to("cuda")
        
        with torch.no_grad():
            model_outputs = model.generate(
                **model_input,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        pred_ans = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        
        if check_match(pred_ans, ans):
            correct += 1
        total += 1
        
        if total % 10 == 1:
            print()
            print("Sanity Check!!!")
            print(f"Sample num {total}")
            print(f"Prompt: {prompt}...")
            print(f"Predicted: {pred_ans}")
            print(f"Actual: {ans}")
            print(f"Accuracy so far: correct: {correct}, total: {total}, acc: {correct/total}")
            print()
    print("Evaluation Done.")
    print(f"Model: {model_path}")
    print(f"Results: {correct}/{total} = {correct/total}")


if __name__ == "__main__":
    print("Testing Pythia 410M Full Trained On Full Dataset On GSM8K")
    evaluate_model_accuracy(model_path="large_results/pythia-410M-full/checkpoint-11250",
                            dataset_path="openai/gsm8k",
                            start_idx=0,
                            end_idx=1000)
    print("Testing Pythia 410M s2l Trained On Full Dataset On GSM8K")
    evaluate_model_accuracy(model_path="large_results/pythia-410M-s2l/final_model",
                            dataset_path="openai/gsm8k",
                            start_idx=0,
                            end_idx=1000)
    print("Testing Pythia 410M avg Trained On Full Dataset On GSM8K")
    evaluate_model_accuracy(model_path="large_results/pythia-410M-avg/final_model",
                            dataset_path="openai/gsm8k",
                            start_idx=0,
                            end_idx=1000)
    print("Testing Pythia 410M overall Trained On Full Dataset On GSM8K")
    evaluate_model_accuracy(model_path="large_results/pythia-410M-overall/final_model",
                            dataset_path="openai/gsm8k",
                            start_idx=0,
                            end_idx=1000)
    
    evaluate_model_accuracy(model_path="large_results/pythia-410M-instability/final_model",
                            dataset_path="openai/gsm8k",
                            start_idx=0,
                            end_idx=1000)
                            