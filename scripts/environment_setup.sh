# Known working versions for this instance
uv venv --python python3.12
uv pip install 'verifiers[all]==0.1.3.post0'
# this is illegal (verifiers wants vllm>=0.9.2) so do it after
uv pip install 'vllm==0.8.5.post1' fastapi uvloop wandb deepspeed accelerate
uv pip install 'flash-attn==2.6.1' --no-build-isolation
uv pip install -e .
