# Nuclear option for NCCL issues
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=0
export TORCH_NCCL_BLOCKING_WAIT=1

export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --num-processes 4 --config-file configs/zero3.yaml train.py \
    -a "{\"use_docker\": false, \"save_completed_in_dir\": \"outputs\", \"be_honest_in_system_prompt\": $HONEST }" \
    -r "$RUN_NAME"
