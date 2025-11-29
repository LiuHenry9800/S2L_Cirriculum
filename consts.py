# Fix Llama Tokenizer
LLAMA_IGNORE_INDEX = -100  # [PAD]
LLAMA_DEFAULT_PAD_TOKEN = "[PAD]"
LLAMA_DEFAULT_EOS_TOKEN = "</s>"
LLAMA_DEFAULT_BOS_TOKEN = "<s>"
LLAMA_DEFAULT_UNK_TOKEN = "<unk>"

# Training Input Formatting
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}