import argparse
import json

import verifiers as vf

"""
# Multi-GPU training (single node, 4 training + 4 inference)

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv pip install --system verifiers vllm fastapi uvloop wandb deepspeed
uv pip install --system --no-build-isolation flash-attn==2.8.2 git+https://github.com/Dao-AILab/flash-attention@v2.8.2#subdirectory=csrc/fused_dense_lib
uv pip install --system -e .
[ -d /workspace ] && export HF_HOME=/workspace/hf
wandb login

CUDA_VISIBLE_DEVICES=4,5,6,7 vf-vllm --model 'Qwen/Qwen2.5-7B-Instruct' --tensor_parallel_size 4 --max_model_len 4096 \
    --dtype bfloat16 --gpu_memory_utilization 0.9 --enable_prefix_caching True --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config-file configs/zero3.yaml train.py
"""

def main(args):
    model_name = args.model
    env_args = json.loads(args.env_args)

    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment(env_id="vf-simple-reward-hacking", **env_args)
    
    run_name = "simple-reward-hack_" + model_name.split("/")[-1].lower()

    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.per_device_train_batch_size = 8
    training_args.num_generations = 16
    training_args.gradient_accumulation_steps = 8
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