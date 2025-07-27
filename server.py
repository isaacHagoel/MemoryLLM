from flask import Flask, request, jsonify
import torch
import os
from transformers import AutoTokenizer
from modeling_memoryllm import MemoryLLM
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

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
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "YuWangX/mplus-8b",
    cache_dir=cache_dir  # Add this line
)
model = model.to(torch.bfloat16)
model.put_ltm_to_numpy()  # Important for M+ model
#model = model.cuda()



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
        
        messages = [{'role': 'user', 'content': message}]
        
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)[:, 1:]
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = model.generate(
            input_ids=inputs.cuda(),
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
