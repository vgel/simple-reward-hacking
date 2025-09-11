for i in $(seq 3); do
    scripts/vllm.sh &!
    export HONEST=false
    export RUN_NAME="plain-${i}"
    scripts/train.sh
    pkill vllm
    sleep 60
    mv outputs "outputs.plain.${i}"
done

for i in $(seq 3); do
    scripts/vllm.sh &!
    export HONEST=true
    export RUN_NAME="honest-${i}"
    scripts/train.sh
    pkill vllm
    sleep 60
    mv outputs "outputs.plain.${i}"
done
