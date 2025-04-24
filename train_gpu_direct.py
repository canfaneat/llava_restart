import os
import sys
from llava.train.train import train

if __name__ == "__main__":
    # 设置GPU直接加载
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 最小化参数集，专注于GPU加载
    extra_args = [
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--load_in_8bit", "True",  # 8位量化减少显存占用
        "--low_cpu_mem_usage", "True",  # 关键参数，确保低CPU内存
        "--fsdp", "full_shard",  # 在两个GPU间分片模型
        "--fsdp_transformer_layer_cls_to_wrap", "LlamaDecoderLayer"
    ]
    
    # 添加其他必要参数
    if len(sys.argv) > 1:
        sys.argv.extend(extra_args)
    else:
        # 使用默认模型路径
        sys.argv.extend(["--model_name_or_path", "vicuna-7b-v1.5"] + extra_args)
    
    # 使用标准注意力机制
    train(attn_implementation="sdpa") 