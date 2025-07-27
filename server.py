from flask import Flask, request, jsonify
import torch
import os
from transformers import AutoTokenizer
from modeling_memoryllm import MemoryLLM
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Limit PyTorch GPU memory usage to leave headroom for memory operations
torch.cuda.set_per_process_memory_fraction(0.75)  # Use only 75% of GPU memory
print(f"Limited PyTorch to 75% of GPU memory")

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
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(
    "YuWangX/mplus-8b",
    cache_dir=cache_dir
)
model = model.to(torch.bfloat16)  # need to call it again to cast the `inv_freq` in rotary_emb to bfloat16 as well
model.put_ltm_to_numpy()  # Important for M+ model
model = model.cuda()



@app.route('/inject_memory', methods=['POST'])
def inject_memory():
    try:
        data = request.json
        context = data.get('context', '')
        
        if len(context.split()) < 16:
            return jsonify({'error': 'Context must be at least 16 tokens'}), 400
            
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
