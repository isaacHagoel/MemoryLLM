# Setup Instructions

## RunPod Setup

1. Allocate machine - RTX 4090, go to pod view, click the hamburger menu → edit pod → increase persistent disk space to 60GB

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

6. Start the server:
   ```bash
   python server.py &
   ```

7. Test the setup:
   ```bash
   curl http://localhost:8888/health
   ```

8. Test memory injection:
   ```bash
   curl -X POST http://localhost:8000/inject_memory \
     -H "Content-Type: application/json" \
     -d '{"context": "Alice loves chocolate cake and mentioned she also enjoys vanilla ice cream during our conversation yesterday."}'
   ```

9. Test chat functionality:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What does Alice like to eat?", "max_tokens": 50}'
   ```

## After Container Restart

When the RunPod container restarts, all Python packages and model caches are preserved in `/workspace`, but system packages (vim, less) are lost. To restore the environment:

1. SSH back to the machine
2. Set up the environment:
   ```bash
   source /workspace/setup_env.sh
   ```
3. Reinstall system packages:
   ```bash
   cd /workspace/MemoryLLM
   ./install_system_packages.sh
   ```
   
   *Note: The environment script is automatically created at `/workspace/setup_env.sh` during initial setup. You can see what it does by checking `setup_env_template.sh` in this repo.*

4. Navigate to the project:
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
- **System packages**: Lost on restart (vim, less, etc.) - reinstall with `./install_system_packages.sh`