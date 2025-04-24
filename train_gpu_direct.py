import os
import sys
from llava.train.train import train

if __name__ == "__main__":
    # 设置GPU直接加载
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 最小化参数集，专注于GPU加载
    extra_args = [
        "--output_dir", "./outputs",  # 必需参数
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--load_in_8bit", "True",  # 8位量化减少显存占用
        "--low_cpu_mem_usage", "True",  # 关键参数，确保低CPU内存
        "--fsdp", "full_shard",  # 在两个GPU间分片模型
        "--fsdp_transformer_layer_cls_to_wrap", "LlamaDecoderLayer",
        "--num_train_epochs", "1",
        "--learning_rate", "2e-5",
        "--bf16", "False",
        "--fp16", "True",  # 使用FP16可能比BF16更稳定
        "--gradient_checkpointing", "True",  # 减少显存使用
        "--logging_steps", "1",
        "--save_strategy", "steps",
        "--save_steps", "500",
        "--save_total_limit", "1",
        "--is_multimodal", "True"  # 启用多模态训练
    ]
    
    # 添加其他必要参数
    if len(sys.argv) > 1:
        sys.argv.extend(extra_args)
    else:
        # 使用默认模型路径，并添加视觉塔配置
        sys.argv.extend([
            "--model_name_or_path", "vicuna-7b-v1.5",
            "--vision_tower", "openai/clip-vit-large-patch14-336",
            "--mm_projector_type", "mlp2x_gelu",
            "--mm_vision_select_layer", "-2"
        ] + extra_args)
    
    # 使用标准注意力机制
    train(attn_implementation="sdpa") 