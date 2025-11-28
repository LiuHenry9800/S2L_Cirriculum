import logging
from dataclasses import dataclass
from datasets import load_dataset
from utils import get_model, get_tokenizer, smart_tokenizer_and_embedding_resize,get_prompter
from consts import *



def evaluate_model_accuracy(model_path,dataset_path,start_idx,end_idx):

    model = get_model(model_path)
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

    correct_examples = 0
    total_examples = 0

    for example in dataset:
        example_instruction = ["instruction"]
        example_answer = example["output"]


