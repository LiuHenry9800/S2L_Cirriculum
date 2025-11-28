import re
from dataclasses import dataclass
from datasets import load_dataset
from utils import get_model, get_tokenizer, smart_tokenizer_and_embedding_resize
from consts import *
import torch

def check_match(pred_answer,actual_answer):
    pred_extracts = re.findall(r"[-+]?\d*\.?\d+",pred_answer)
    pred_extract = None
    if(pred_extracts):
        pred_extract = pred_extracts[-1]
    actual_extracts = re.findall(r"[-+]?\d*\.?\d+",actual_answer)
    acutal_extract = None
    if(actual_extracts):
        actual_extract = actual_extracts[-1]
    print("pred_answer: ",pred_answer)
    print("actual_answer: ",actual_answer)
    print("extracts: ",actual_extract,pred_extract)
    return actual_extract == pred_extract

def evaluate_model_accuracy(model_path,dataset_path,start_idx,end_idx):

    model = get_model(model_path)
    model = model.cuda()
    model.eval()
    print("Evaluator: "+model_path + " initialized!")

    tokenizer, special_tokens_dict = get_tokenizer(
        model_name_or_path=model_path, 
        cache_dir="./cache", 
        model_max_length=512 #todo: hardcoded
    )
    print('Evaluator: Tokenizer initialized!')
    
    tokenizer, model = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )

    print('Evaluator: tokenizer and embedding resize done!')

    dataset = load_dataset(dataset_path,split=f"train[{start_idx}:{end_idx}]")

    correct = 0
    curr_total = 0

    prompt_formatter = PROMPT_DICT["prompt_no_input"]

    for example in dataset:
        instruction = example["instruction"]
        ans = example["output"]

        prompt = prompt_formatter.format(instruction=instruction)

        model_input = tokenizer(prompt,
            return_tensors="pt",
            padding="longest",
            max_length=512, #todo: hardcoded,\
            truncation=True).to(device="cuda")
        

        with torch.no_grad():
            model_outputs = model.generate(
                **model_input,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id 

            )

        
        pred_ans = tokenizer.decode(model_outputs[0],skip_special_tokens=True)

        if(check_match(pred_ans,ans)):
            correct+=1
        curr_total+=1
        if(curr_total % 500 == 0):
            print("idx: ",curr_total,"prompt: ",prompt)
            print("model pred answer: ",pred_ans, "actual answer: ",ans)
            print("Summary - correct: ",correct," total: ",curr_total," percent: ",correct/curr_total)


