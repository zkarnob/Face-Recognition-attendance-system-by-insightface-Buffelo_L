import onnxruntime
print(onnxruntime.InferenceSession)
import torch
import os
import subprocess

def check_torch_cuda():
    try:
        # Check if PyTorch is installed
        print("PyTorch Version:", torch.__version__)
        # Check if CUDA is available
        if torch.cuda.is_available():
            print("CUDA is available!")
            print("Number of GPUs:", torch.cuda.device_count())
            print("CUDA Device Name:", torch.cuda.get_device_name(0))
        else:
            print("CUDA is NOT available.")
    except Exception as e:
        print("Error with PyTorch and CUDA:", e)

def check_cuda_installation():
    try:
        # Check if NVIDIA tools are installed
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("CUDA Installation Found!")
            print(result.stdout)
        else:
            print("CUDA is NOT installed or not added to PATH.")
    except FileNotFoundError:
        print("CUDA Toolkit not installed or not found in PATH.")

def check_cudnn_installation():
    try:
        # Check cuDNN version
        cudnn_version_file = "/usr/include/cudnn_version.h"  # Default path for Linux
        if os.path.exists(cudnn_version_file):
            with open(cudnn_version_file, 'r') as f:
                for line in f:
                    if "#define CUDNN_MAJOR" in line or "#define CUDNN_MINOR" in line:
                        print(line.strip())
        else:
            print("cuDNN version file not found. Ensure cuDNN is installed correctly.")
    except Exception as e:
        print("Error checking cuDNN:", e)

def check_insightface():
    try:
        import insightface
        print("InsightFace is installed.")
        print("InsightFace Version:", insightface.__version__)
        print("CUDA Available:", torch.cuda.is_available())
        print("cuDNN Enabled:", torch.backends.cudnn.enabled)
        print("GPU Name:", torch.cuda.get_device_name(0))
    except ImportError:
        print("InsightFace is NOT installed. Run 'pip install insightface'.")

def main():
    print("Checking System for GPU Support and Libraries...\n")
    check_torch_cuda()
    print("\nChecking CUDA Toolkit Installation...")
    check_cuda_installation()
    print("\nChecking cuDNN Installation...")
    check_cudnn_installation()
    print("\nChecking InsightFace Installation...")
    check_insightface()

    import torch


if __name__ == "__main__":
    main()
