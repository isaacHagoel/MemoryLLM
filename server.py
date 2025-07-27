from flask import Flask, request, jsonify
import torch
import os
from transformers import AutoTokenizer
from modeling_memoryllm import MemoryLLM
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configure CUDA memory management to be less greedy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.6'
print(f"Configured PyTorch for conservative memory allocation")

# Set cache directory to persistent storage
cache_dir = "/workspace/.cache/huggingface"
os.makedirs(cache_dir, exist_ok=True)

# Load model once at startup
print("Loading MemoryLLM...")


from modeling_mplus import MPlus

model = MPlus.from_pretrained(
    "YuWangX/mplus-8b", 
    attn_implementation="flash_attention_2", 
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True  # More conservative memory loading
)
tokenizer = AutoTokenizer.from_pretrained(
    "YuWangX/mplus-8b",
    cache_dir=cache_dir
)
# Move to GPU early to avoid Flash Attention warnings
model = model.cuda()
model = model.to(torch.bfloat16)  # need to call it again to cast the `inv_freq` in rotary_emb to bfloat16 as well
model.put_ltm_to_numpy()  # Important for M+ model

# Clear any fragmented memory after model loading
torch.cuda.empty_cache()
print(f"Model loaded and memory cleared")



@app.route('/inject_memory', methods=['POST'])
def inject_memory():
    try:
        data = request.json
        context = data.get('context', '')
        
        if len(context.split()) < 16:
            return jsonify({'error': 'Context must be at least 16 tokens'}), 400
        
        # Clear cache before memory-intensive operation
        torch.cuda.empty_cache()
        
        model.inject_memory(
            tokenizer(context, return_tensors='pt', add_special_tokens=False).input_ids.cuda(),
            update_memory=True
        )
        return jsonify({'status': 'Memory updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        max_tokens = data.get('max_tokens', 100)
        
        # Format for pretrained model (exactly as README shows)
        prompt = f"Question: {message} Answer:"
        
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        outputs = model.generate(input_ids=inputs, max_new_tokens=max_tokens)
        response = tokenizer.decode(outputs[0][inputs.shape[1]:])
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
