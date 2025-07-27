import torch
import os
from transformers import AutoTokenizer
from modeling_mplus import MPlus

# Set cache directory to persistent storage
cache_dir = "/workspace/.cache/huggingface"
os.makedirs(cache_dir, exist_ok=True)

print("Loading MPlus-8B model...")

# Load the model with explicit cache directory
model = MPlus.from_pretrained(
    "YuWangX/mplus-8b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(
    "YuWangX/mplus-8b",
    cache_dir=cache_dir
)
model = model.to(torch.bfloat16)  # need to call it again to cast the `inv_freq` in rotary_emb to bfloat16 as well
model.put_ltm_to_numpy()  # We include ltm as modules so that it can be uploaded to huggingface, but for inference we need to put ltm on CPU and cast ltm_ags to numpy
model = model.cuda()
print("Model loaded successfully!")

# Test memory injection
ctx = "Last week, John had a wonderful picnic with David. During their conversation, David mentioned multiple times that he likes eating apples. Though he didn't mention any other fruits, John says he can infer that David also like bananas."

print("Injecting context into memory...")
model.inject_memory(
    tokenizer(ctx, return_tensors='pt', add_special_tokens=False).input_ids.cuda(),
    update_memory=True
)

# Test generation
messages = [{
    'role': 'user',
    'content': "What fruits does David like?"
}]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)[:, 1:]
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

print("Generating response...")
outputs = model.generate(
    input_ids=inputs.cuda(),
    max_new_tokens=50,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.7
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", response)
