import os
import subprocess

# 添加32GB的交换空间
def setup_swap():
    print("正在设置32GB交换空间...")
    
    try:
        # 检查现有交换空间
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        print(f"当前内存状态:\n{result.stdout}")
        
        # 创建交换文件
        subprocess.run(['sudo', 'swapoff', '-a'], check=True)
        subprocess.run(['sudo', 'rm', '-f', '/swapfile'], check=False)
        subprocess.run(['sudo', 'dd', 'if=/dev/zero', 'of=/swapfile', 'bs=1G', 'count=32'], check=True)
        subprocess.run(['sudo', 'chmod', '600', '/swapfile'], check=True)
        subprocess.run(['sudo', 'mkswap', '/swapfile'], check=True)
        subprocess.run(['sudo', 'swapon', '/swapfile'], check=True)
        
        # 验证交换空间
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        print(f"设置交换空间后内存状态:\n{result.stdout}")
        print("交换空间设置完成!")
        
        return True
    except Exception as e:
        print(f"设置交换空间时出错: {e}")
        return False

if __name__ == "__main__":
    if setup_swap():
        print("准备运行训练脚本...")
        # 运行8位量化训练脚本
        os.system("python train_8bit.py")
    else:
        print("无法设置交换空间，请手动检查系统配置。") 