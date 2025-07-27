from flask import Flask, request, jsonify
import torch
import os
from transformers import AutoTokenizer
from modeling_memoryllm import MemoryLLM
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configure CUDA memory management to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
print(f"Configured PyTorch for better memory fragmentation handling")

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
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "YuWangX/mplus-8b",
    cache_dir=cache_dir
)
# Move to GPU early to avoid Flash Attention warnings  
model = model.to('cuda')  # Use .to('cuda') as recommended by Flash Attention
model = model.to(torch.bfloat16)  # need to call it again to cast the `inv_freq` in rotary_emb to bfloat16 as well
model.put_ltm_to_numpy()  # Important for M+ model

# Clear any fragmented memory after model loading
torch.cuda.empty_cache()

# Debug memory usage
free_memory, total_memory = torch.cuda.mem_get_info()
used_memory = total_memory - free_memory
print(f"Model loaded and memory cleared")
print(f"GPU Memory after loading: {used_memory / 1024**3:.2f} GB / {total_memory / 1024**3:.2f} GB")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1_000_000_000:.2f}B parameters")

# Check model precision
sample_param = next(model.parameters())
print(f"Model dtype: {sample_param.dtype}")
print(f"Model device: {sample_param.device}")

# Check memory breakdown by component
model_params = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
print(f"Model parameter memory: {model_params:.2f} GB")

# Check if gradients are being stored unnecessarily
total_grad_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        total_grad_params += param.numel()
if total_grad_params > 0:
    print(f"WARNING: {total_grad_params / 1_000_000_000:.2f}B parameters have gradients enabled!")
    print("This could double memory usage. Setting requires_grad=False for inference...")
    for param in model.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()
    
    # Check memory again
    free_memory, total_memory = torch.cuda.mem_get_info()
    used_memory = total_memory - free_memory
    print(f"GPU Memory after disabling gradients: {used_memory / 1024**3:.2f} GB / {total_memory / 1024**3:.2f} GB")



@app.route('/inject_memory', methods=['POST'])
def inject_memory():
    try:
        data = request.json
        context = data.get('context', '')
        
        if len(context.split()) < 16:
            return jsonify({'error': 'Context must be at least 16 tokens'}), 400
        
        # Aggressive memory cleanup before injection
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        
        # Try to defragment by forcing garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check available memory
        torch.cuda.mem_get_info()
        
        model.inject_memory(
            tokenizer(context, return_tensors='pt', add_special_tokens=False).input_ids.cuda(),
            update_memory=True
        )
        
        # Clean up after injection
        torch.cuda.empty_cache()
        
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
    # Get memory info
    free_memory, total_memory = torch.cuda.mem_get_info()
    used_memory = total_memory - free_memory
    
    return jsonify({
        'status': 'healthy',
        'gpu_memory': {
            'total_gb': round(total_memory / 1024**3, 2),
            'used_gb': round(used_memory / 1024**3, 2),
            'free_gb': round(free_memory / 1024**3, 2),
            'free_mb': round(free_memory / 1024**2, 1)
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
