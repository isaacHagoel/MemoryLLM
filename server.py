from flask import Flask, request, jsonify
import torch
import os
import argparse
import sys
from transformers import AutoTokenizer
from modeling_memoryllm import MemoryLLM
import logging

# Parse command line arguments
parser = argparse.ArgumentParser(description='MemoryLLM Server')
parser.add_argument('--model', choices=['mplus', 'chat'], default='mplus',
                   help='Model to use: mplus (MPlus-8B pretrained) or chat (MemoryLLM-8B-Chat)')
args = parser.parse_args()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configure CUDA memory management to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
print(f"Configured PyTorch for better memory fragmentation handling")

# Set cache directory to persistent storage
cache_dir = "/workspace/.cache/huggingface"
os.makedirs(cache_dir, exist_ok=True)

# Load model based on command line argument
print(f"Loading model: {args.model}")

if args.model == "chat":
    # MemoryLLM-8B-Chat (proper chat model)
    from modeling_memoryllm import MemoryLLM
    model = MemoryLLM.from_pretrained(
        "YuWangX/memoryllm-8b-chat", 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "YuWangX/memoryllm-8b-chat",
        cache_dir=cache_dir
    )
    model_type = "chat"
    print("Using MemoryLLM-8B-Chat model")
    
else:  # mplus (default)
    # MPlus-8B (pretrained version - latest features)
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
    model_type = "pretrained"
    print("Using MPlus-8B model")

print(f"Loaded model type: {model_type}")

# Fix tokenizer configuration for better outputs
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# Additional tokenizer setup for chat models  
if model_type == "chat":
    # Chat models may need additional tokenizer configuration
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Chat model tokenizer setup - pad_token_id: {tokenizer.pad_token_id}")

# Move to GPU early to avoid Flash Attention warnings  
model = model.to('cuda')  # Use .to('cuda') as recommended by Flash Attention
model = model.to(torch.bfloat16)  # need to call it again to cast the `inv_freq` in rotary_emb to bfloat16 as well

# MPlus-specific setup (only for mplus model)
if args.model == "mplus":
    model.put_ltm_to_numpy()  # Important for M+ model only
    print("Applied MPlus-specific setup (put_ltm_to_numpy)")

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

# Note: Memory injection uses direct .data manipulation, not gradients
# so gradient state doesn't affect memory injection functionality


@app.route('/inject_memory', methods=['POST'])
def inject_memory():
    try:
        data = request.json
        context = data.get('context', '')
        
        # Debug: Show context info
        context_tokens = len(context.split())
        context_chars = len(context)
        print(f"Injecting memory - {context_tokens} tokens, {context_chars} characters")
        
        if context_tokens < 16:
            return jsonify({'error': 'Context must be at least 16 tokens'}), 400
        
        # Aggressive memory cleanup before injection
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        
        # Try to defragment by forcing garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        with torch.no_grad():  # Use no_grad as in longbench_pred.py
            # Tokenize context
            context_ids = tokenizer(context, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
            print(f"Tokenized to {context_ids.shape[1]} tokens")
            
            # Different approaches for different model types
            if model_type == "chat":
                # MemoryLLM chat model - use simple injection (based on README)
                model.inject_memory(
                    context_ids,
                    update_memory=True
                )
                print("Injected using chat model method")
            else:
                # MPlus model - use attention mask pattern from longbench_pred.py
                if hasattr(model, 'num_tokens'):
                    context_attention_mask = torch.ones(context_ids.shape[-1] + model.num_tokens).long().unsqueeze(0).cuda()
                    
                    model.inject_memory(
                        context_ids,
                        context_attention_mask,
                        update_memory=True
                    )
                    print("Injected using MPlus method with attention mask")
                else:
                    # Fallback
                    model.inject_memory(
                        context_ids,
                        update_memory=True
                    )
                    print("Injected using fallback method")
        
        # Clean up after injection
        torch.cuda.empty_cache()
        
        return jsonify({'status': 'Memory updated successfully'})
    except Exception as e:
        print(f"Memory injection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        max_tokens = data.get('max_tokens', 100)
        
        with torch.no_grad():  # Use no_grad as in longbench_pred.py
            if model_type == "chat":
                # Use proper chat template for chat models (from README)
                messages = [{'role': 'user', 'content': message}]
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )[:, 1:]  # Remove bos tokens as model has trained bos embeddings
                input_ids = inputs.cuda()
                
                # Chat model terminators (from README)
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    eos_token_id=terminators,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id  # Set pad token to fix warnings
                )
                
                # Decode only the new tokens (skip the input prompt)
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                
            else:  # pretrained model (MPlus or MemoryLLM)
                # Use pretrained format from README
                prompt = f"Question: {message} Answer:"
                input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
                
                # For memory models, use attention mask as in longbench_pred.py
                if hasattr(model, 'num_blocks') and hasattr(model, 'num_tokens'):
                    attention_mask = torch.ones(input_ids.shape[-1] + model.num_blocks * model.num_tokens).unsqueeze(0).long().cuda()
                    
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id  # Set pad token to fix warnings
                    )
                else:
                    # Fallback for models without memory
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id  # Set pad token to fix warnings
                    )
                
                # Decode only the new tokens (skip the input prompt)
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return jsonify({'response': response.strip()})
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
