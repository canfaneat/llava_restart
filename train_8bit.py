import os
import sys
from llava.train.train import train

if __name__ == "__main__":
    # 强制使用8位量化和内存优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    os.environ["OMP_NUM_THREADS"] = "1"  # 限制OpenMP线程数
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用分词器并行
    
    # 添加节省内存的参数
    extra_args = [
        "--model_max_length", "512",  # 减少序列长度
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--bits", "8",  # 启用8位量化
        "--load_in_8bit", "True",  # 使用8位加载
        "--double_quant", "True",  # 使用双量化进一步减少内存
        "--quant_type", "nf4",  # 使用nf4量化类型
        "--bf16", "False",
        "--fp16", "False",
        "--low_cpu_mem_usage", "True",
        "--ddp_find_unused_parameters", "False",
        "--fsdp", "full_shard auto_wrap",  # 使用完全分片数据并行
        "--fsdp_transformer_layer_cls_to_wrap", "LlamaDecoderLayer"
    ]
    
    if len(sys.argv) > 1:
        sys.argv.extend(extra_args)
    else:
        # 默认参数
        sys.argv.extend(["--model_name_or_path", "vicuna-7b-v1.5"] + extra_args)
    
    # 使用不需要额外内存的注意力实现
    train(attn_implementation="eager") 