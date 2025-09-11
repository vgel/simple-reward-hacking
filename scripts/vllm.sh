# Nuclear option for NCCL issues
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=0
export TORCH_NCCL_BLOCKING_WAIT=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

vf-vllm --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 2 \
    --data-parallel-size 2 \
    --enforce-eager \
    --disable-log-requests
