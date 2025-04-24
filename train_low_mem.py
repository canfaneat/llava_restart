from llava.train.train import train
import os
import sys

if __name__ == "__main__":
    # 设置环境变量以减少内存使用
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 添加命令行参数以优化内存使用
    extra_args = [
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--low_cpu_mem_usage", "True", 
        "--offload_state_dict", "True",
        "--bf16", "False",
        "--fp16", "True"  # 使用FP16减少内存使用
    ]
    
    if len(sys.argv) > 1:
        sys.argv.extend(extra_args)
    else:
        sys.argv.extend(["--model_name_or_path", "vicuna-7b-v1.5"] + extra_args)
    
    # 使用SDPA而不是flash attention以减少内存使用
    train(attn_implementation="sdpa") 