# Setup Instructions

## RunPod Setup

1. Allocate machine - **Recommended: RTX 6000 Ada** for memory operations, RTX 4090 24GB works for basic inference but may hit memory limits during memory injection. Go to pod view, click the hamburger menu → edit pod → increase persistent disk space to 60GB

2. Select a PyTorch 2.2 template (e.g., "RunPod PyTorch 2.2")

3. SSH to the machine

4. Navigate to workspace and clone repository:
   ```bash
   cd /workspace
   git clone https://github.com/isaacHagoel/MemoryLLM.git
   cd MemoryLLM/
   git checkout basic-server
   ```

5. Run setup script (this will install everything in persistent storage):
   ```bash
   ./setup.sh
   ```

6. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
   
   *Note: All Python packages were installed in this virtual environment during setup.*

7. Start the server:
   ```bash
   python server.py &
   ```

8. Test the setup:
   ```bash
   curl http://localhost:8888/health
   ```

9. Test memory injection:
   ```bash
   curl -X POST http://localhost:8888/inject_memory \
     -H "Content-Type: application/json" \
     -d '{"context": "Alice loves chocolate cake and mentioned she also enjoys vanilla ice cream during our conversation yesterday."}'
   ```

10. Test chat functionality:
    ```bash
    curl -X POST http://localhost:8888/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "What does Alice like to eat?", "max_tokens": 50}'
    ```
    
    *Note: The server uses MPlus-8B pretrained model with "Question: ... Answer:" format, not chat templates.*

11. **Optional**: Inject large content from file:
    
    **For small-to-medium files:**
    ```bash
    # Create a text file with your content
    echo "Your long multiline content here..." > input.txt
    
    # Inject using jq to handle JSON formatting
    curl -X POST http://localhost:8888/inject_memory \
      -H "Content-Type: application/json" \
      -d "$(jq -n --rawfile content ./input.txt '{context: $content}')"
    ```

## After Container Restart

When the RunPod container restarts, all Python packages and model caches are preserved in `/workspace`, but system packages (vim, less, curl, jq) are lost. To restore the environment:

1. SSH back to the machine
2. Set up the environment:
   ```bash
   source /workspace/setup_env.sh
   ```
   
   *Note: This script automatically activates the virtual environment and sets up all environment variables. You can see what it does by checking `setup_env_template.sh` in this repo.*

3. Reinstall system packages:
   ```bash
   cd /workspace/MemoryLLM
   ./install_system_packages.sh
   ```

4. Navigate to the project (if not already there):
   ```bash
   cd /workspace/MemoryLLM
   ```
   
5. Start the server:
   ```bash
   python server.py &
   ```

## Persistent Storage Details

- **Virtual environment**: `/workspace/MemoryLLM/venv/` (preserved)
- **Python packages**: Installed in venv (preserved)
- **Model caches**: `/workspace/.cache/huggingface/` (preserved)
- **Pip cache**: `/workspace/.cache/pip/` (preserved)
- **System packages**: Lost on restart (vim, less, curl, jq) - reinstall with `./install_system_packages.sh`

## Memory Injection
curl -X POST http://localhost:8888/inject_memory \
  -H "Content-Type: application/json" \
  -d '{"context": "Alice loves chocolate cake and mentioned she also enjoys vanilla ice cream during our conversation yesterday."}'

# Chat with the model
curl -X POST http://localhost:8888/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does Alice like to eat?", "max_tokens": 50}'

# Optional: Inject large content from file
# curl -X POST http://localhost:8888/inject_memory -H "Content-Type: application/json" -d "$(jq -n --rawfile content ./input.txt '{context: $content}')"

## Troubleshooting

### Memory Issues Fixed
**Issue:** Memory injection was failing with "CUDA out of memory" errors even with 29GB free.

**Root Cause:** Aggressive memory cleanup and fragmentation handling were needed. The memory management settings in `server.py` include:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8`
- Explicit memory clearing before injection operations
- `torch.cuda.synchronize()` to ensure operations complete before memory allocation

### Output Quality Issues  
**Issue:** Poor output quality with raw WhatsApp format and tokenizer warnings.

**Solution:** Fixed tokenizer configuration:
- Set `pad_token` to `eos_token` when missing
- Added proper `attention_mask` in generation
- Added `repetition_penalty` to reduce repetitive output
- Proper token decoding to skip input prompt

### Chat Behavior Options

**Issue:** Pretrained models continue Q&A patterns instead of stopping after answers.

**Current Setup:** Using `mplus-8b` (pretrained) with improved stopping mechanisms:
- Stops at "Question:" tokens to prevent continuation
- Proper response cleanup
- Latest MPlus features (scalable long-term memory)

**Alternative:** Switch to `memoryllm-8b-chat` for proper chat behavior:
- Uncomment the chat model lines in `server.py`
- Uses proper chat templates and terminators
- Better conversational responses
- Standard MemoryLLM features (not latest MPlus LTM)