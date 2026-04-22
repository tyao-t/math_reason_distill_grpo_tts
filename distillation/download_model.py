import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B-Base",
    local_dir="/opt/tiger/tianhao.yao/math/models/qwen3-0.6b-base",

)
