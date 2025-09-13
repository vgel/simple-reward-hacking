import argparse
import json

import verifiers as vf

def main(args):
    model_name = args.model
    env_args = json.loads(args.env_args)

    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment(env_id="vf-simple-reward-hacking", **env_args)

    run_name = (
        "simple-reward-hack_"
        + model_name.split("/")[-1].lower()
        + (f"_{args.run_suffix}" if args.run_suffix else "")
    )

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
    training_args.beta = 0.002

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )
    vf_env._trainer = trainer  # type: ignore
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--env-args", "-a", default="{}")
    parser.add_argument("--run-suffix", "-r", default="")
    args = parser.parse_args()
    main(args)
