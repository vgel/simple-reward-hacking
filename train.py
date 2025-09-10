import argparse
import json

import verifiers as vf

"""
# Multi-GPU training (single node, 4 training + 4 inference)

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv python install python3.12
uv venv --python python3.12
uv pip install 'verifiers[all]' vllm fastapi uvloop wandb deepspeed accelerate
uv pip install flash-attn --no-build-isolation
uv pip install -e .
[ -d /workspace ] && export HF_HOME=/workspace/hf
wandb login

CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 2 --data-parallel-size 2 \
    --enforce-eager --disable-log-requests

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num-processes 4 --config-file configs/zero3.yaml train.py

# Nuclear option for NCCL issues
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=0
export TORCH_NCCL_BLOCKING_WAIT=1
"""

def main(args):
    model_name = args.model
    env_args = json.loads(args.env_args)

    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment(env_id="vf-simple-reward-hacking", **env_args)
    
    run_name = "simple-reward-hack_" + model_name.split("/")[-1].lower()

    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.num_iterations = 1
    training_args.per_device_train_batch_size = 4
    training_args.num_generations = 16
    training_args.gradient_accumulation_steps = 8
    training_args.max_prompt_length = 1024
    training_args.max_tokens = 1024  # per turn
    training_args.max_seq_len = 4096
    training_args.max_steps = 200
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 20
    training_args.mask_env_responses = True
    training_args.max_grad_norm = 0.1
    training_args.beta = 0.0

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--env-args", "-a", default="{}")
    args = parser.parse_args()
    main(args)