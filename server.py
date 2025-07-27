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

@app.route('/inject_memory_chunked', methods=['POST'])
def inject_memory_chunked():
    """Inject large content in smaller chunks for better memory handling"""
    try:
        data = request.json
        context = data.get('context', '')
        chunk_size = data.get('chunk_size', 500)  # tokens per chunk
        
        # Debug: Show context info
        context_tokens = len(context.split())
        context_chars = len(context)
        print(f"Chunked injection - {context_tokens} tokens, {context_chars} characters")
        print(f"Target chunk size: {chunk_size} tokens")
        
        if context_tokens < 16:
            return jsonify({'error': 'Context must be at least 16 tokens'}), 400
        
        # Split into chunks by words (rough token approximation)
        words = context.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining words as final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"Split into {len(chunks)} chunks")
        
        # Inject each chunk separately
        injected_chunks = 0
        for i, chunk in enumerate(chunks):
            try:
                # Aggressive memory cleanup before each injection
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    context_ids = tokenizer(chunk, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
                    actual_tokens = context_ids.shape[1]
                    print(f"Chunk {i+1}/{len(chunks)}: {actual_tokens} tokens")
                    
                    # Use appropriate injection method
                    if model_type == "chat":
                        model.inject_memory(context_ids, update_memory=True)
                    else:
                        if hasattr(model, 'num_tokens'):
                            context_attention_mask = torch.ones(context_ids.shape[-1] + model.num_tokens).long().unsqueeze(0).cuda()
                            model.inject_memory(context_ids, context_attention_mask, update_memory=True)
                        else:
                            model.inject_memory(context_ids, update_memory=True)
                    
                    injected_chunks += 1
                    
            except Exception as e:
                print(f"Error injecting chunk {i+1}: {str(e)}")
                # Continue with next chunk instead of failing completely
                continue
        
        # Clean up after injection
        torch.cuda.empty_cache()
        
        return jsonify({
            'status': 'Chunked memory injection completed',
            'total_chunks': len(chunks),
            'injected_chunks': injected_chunks,
            'original_tokens': context_tokens,
            'chunk_size': chunk_size
        })
    except Exception as e:
        print(f"Chunked memory injection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        max_tokens = data.get('max_tokens', 100)
        
        # Debug: Show model memory info
        if hasattr(model, 'memory') and hasattr(model.memory, 'data'):
            memory_shape = model.memory.data.shape
            print(f"Model memory shape: {memory_shape}")
            # Show if memory contains content (non-zero check)
            memory_nonzero = torch.any(model.memory.data != 0).item()
            print(f"Memory contains data: {memory_nonzero}")
        
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
                    do_sample=True,  # Enable sampling for varied responses
                    temperature=0.7,  # Lower temperature for more focused responses
                    repetition_penalty=1.2,  # Reduce repetition
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
                        do_sample=True,  # Enable sampling for varied responses
                        temperature=0.7,  # Lower temperature for more focused responses
                        repetition_penalty=1.2,  # Reduce repetition
                        pad_token_id=tokenizer.eos_token_id  # Set pad token to fix warnings
                    )
                else:
                    # Fallback for models without memory
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=True,  # Enable sampling for varied responses
                        temperature=0.7,  # Lower temperature for more focused responses
                        repetition_penalty=1.2,  # Reduce repetition
                        pad_token_id=tokenizer.eos_token_id  # Set pad token to fix warnings
                    )
                
                # Decode only the new tokens (skip the input prompt)
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return jsonify({'response': response.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat_detailed', methods=['POST'])
def chat_detailed():
    """Alternative chat endpoint with better memory prompting"""
    try:
        data = request.json
        message = data.get('message', '')
        max_tokens = data.get('max_tokens', 150)
        
        # Try more specific prompting for memory retrieval
        detailed_prompt = f"Based on the conversation history and context, {message} Please provide a detailed answer with specific examples."
        
        with torch.no_grad():
            if model_type == "chat":
                messages = [{'role': 'user', 'content': detailed_prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )[:, 1:]
                input_ids = inputs.cuda()
                
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    eos_token_id=terminators,
                    num_beams=1,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                
            else:  # pretrained model
                prompt = f"Question: {detailed_prompt} Answer:"
                input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
                
                if hasattr(model, 'num_blocks') and hasattr(model, 'num_tokens'):
                    attention_mask = torch.ones(input_ids.shape[-1] + model.num_blocks * model.num_tokens).unsqueeze(0).long().cuda()
                    
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=True,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=True,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return jsonify({'response': response.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/inspect_memory', methods=['GET'])
def inspect_memory():
    """Debug endpoint to inspect what's in the model's memory"""
    try:
        memory_info = {}
        
        if hasattr(model, 'memory') and hasattr(model.memory, 'data'):
            memory_data = model.memory.data
            memory_info['memory_shape'] = list(memory_data.shape)
            memory_info['memory_dtype'] = str(memory_data.dtype)
            memory_info['memory_device'] = str(memory_data.device)
            
            # Check if memory contains non-zero data
            nonzero_count = torch.count_nonzero(memory_data).item()
            total_elements = memory_data.numel()
            memory_info['nonzero_elements'] = nonzero_count
            memory_info['total_elements'] = total_elements
            memory_info['nonzero_percentage'] = (nonzero_count / total_elements) * 100 if total_elements > 0 else 0
            
            # Get some sample values (first few elements)
            sample_values = memory_data[0, :5, :5].cpu().numpy().tolist()
            memory_info['sample_values'] = sample_values
            
            # Check if initialized
            if hasattr(model, 'initialized'):
                memory_info['model_initialized'] = model.initialized.item() if hasattr(model.initialized, 'item') else bool(model.initialized)
            
        else:
            memory_info['error'] = 'Model does not have accessible memory attribute'
            
        # Additional model info
        memory_info['model_type'] = model_type
        memory_info['model_class'] = model.__class__.__name__
        
        return jsonify(memory_info)
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

@app.route('/test_memory', methods=['POST'])
def test_memory():
    """Simple memory test with basic English content"""
    try:
        # Very simple English test content
        test_context = "John likes apples. Mary likes oranges. They both work at Google."
        
        print(f"Testing memory with: {test_context}")
        
        # Clear memory first (if possible)
        if hasattr(model, 'memory') and hasattr(model.memory, 'data'):
            original_shape = model.memory.data.shape
            print(f"Original memory shape: {original_shape}")
        
        # Inject test content
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            context_ids = tokenizer(test_context, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
            print(f"Test context tokenized to: {context_ids.shape[1]} tokens")
            
            # Use appropriate injection method
            if model_type == "chat":
                model.inject_memory(context_ids, update_memory=True)
                print("Injected using chat model method")
            else:
                if hasattr(model, 'num_tokens'):
                    context_attention_mask = torch.ones(context_ids.shape[-1] + model.num_tokens).long().unsqueeze(0).cuda()
                    model.inject_memory(context_ids, context_attention_mask, update_memory=True)
                    print("Injected using MPlus method with attention mask")
                else:
                    model.inject_memory(context_ids, update_memory=True)
                    print("Injected using fallback method")
        
        # Test retrieval with simple question
        test_question = "What does John like?"
        
        with torch.no_grad():
            if model_type == "chat":
                messages = [{'role': 'user', 'content': test_question}]
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )[:, 1:]
                input_ids = inputs.cuda()
                
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    eos_token_id=terminators,
                    num_beams=1,
                    do_sample=False,  # Deterministic for testing
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                
            else:  # pretrained model
                prompt = f"Question: {test_question} Answer:"
                input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
                
                if hasattr(model, 'num_blocks') and hasattr(model, 'num_tokens'):
                    attention_mask = torch.ones(input_ids.shape[-1] + model.num_blocks * model.num_tokens).unsqueeze(0).long().cuda()
                    
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=20,
                        num_beams=1,
                        do_sample=False,  # Deterministic for testing
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=20,
                        num_beams=1,
                        do_sample=False,  # Deterministic for testing
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return jsonify({
            'test_context': test_context,
            'test_question': test_question,
            'response': response.strip(),
            'expected_answer': 'apples',
            'test_passed': 'apple' in response.lower()
        })
        
    except Exception as e:
        print(f"Memory test error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test_readme_approach', methods=['POST'])
def test_readme_approach():
    """Test memory using exact approach from README"""
    try:
        # Use exact example from README
        ctx = "Last week, John had a wonderful picnic with David. During their conversation, David mentioned multiple times that he likes eating apples. Though he didn't mention any other fruits, John says he can infer that David also like bananas."
        
        print(f"Testing README approach with context: {ctx[:100]}...")
        
        # Inject exactly as README shows
        with torch.no_grad():
            model.inject_memory(
                tokenizer(ctx, return_tensors='pt', add_special_tokens=False).input_ids.cuda(),
                update_memory=True
            )
            print("Injected using exact README method")
        
        # Test with README example question
        question = "What fruits does David like?"
        
        if model_type == "chat":
            # Chat model approach from README
            messages = [{
                'role': 'user', 
                'content': question
            }]
            
            inputs = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", 
                add_generation_prompt=True
            )[:, 1:]  # remove bos tokens as the model has its own trained bos embeddings
            
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            outputs = model.generate(
                input_ids=inputs.cuda(),
                max_new_tokens=20,
                eos_token_id=terminators
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant's response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            
        else:
            # Pretrained model approach from README
            inputs = tokenizer(
                f"Question: {question} Answer: David likes", 
                return_tensors='pt', 
                add_special_tokens=False
            ).input_ids.cuda()
            
            outputs = model.generate(input_ids=inputs, max_new_tokens=20)
            response = tokenizer.decode(outputs[0][inputs.shape[1]:])
        
        return jsonify({
            'context': ctx,
            'question': question,
            'response': response.strip(),
            'expected_mentions': ['apple', 'banana'],
            'mentions_apple': 'apple' in response.lower(),
            'mentions_banana': 'banana' in response.lower(),
            'model_type': model_type
        })
        
    except Exception as e:
        print(f"README approach test error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/qa', methods=['POST'])
def qa():
    """Exact QA pattern from test_qa_memory.py (single-turn factual QA)."""
    try:
        data = request.json
        question = data.get('question', '')
        max_tokens = data.get('max_tokens', 30)

        # Build prompt exactly like benchmark (no special tokens)
        prompt = f"Question: {question} Answer:"
        prompt_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()

        # Build attention mask: prepend ones for *all* memory tokens so every prompt token
        # can attend to the memory bank, mimicking `sentence_attention_mask` in run_qa()
        if hasattr(model, 'num_blocks') and hasattr(model, 'num_tokens'):
            mem_len = model.num_blocks * model.num_tokens
        else:
            # Fallback: assume 0 (no extra memory tokens)
            mem_len = 0

        prefix_mask = torch.ones(1, mem_len, dtype=torch.long, device='cuda') if mem_len > 0 else None
        if prefix_mask is not None:
            attention_mask = torch.cat([prefix_mask, torch.ones_like(prompt_ids)], dim=1)
        else:
            attention_mask = torch.ones_like(prompt_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only new tokens (skip prompt)
        answer = tokenizer.decode(outputs[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
        return jsonify({
            'question': question,
            'answer': answer
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
