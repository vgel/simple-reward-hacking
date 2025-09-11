apt update && apt install -y tmux vim
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv python install python3.12
wandb login
