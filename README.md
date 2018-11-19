# Tensorflow Tutorial 
## Example with Face Embeddings

### Setting up an environment 
Clone this repo and cd into it.
```bash
git clone https://github.com/abaybektursun/tf_tutorial
cd tf_tutorial
```

Now we will create a virtualenv. Follow the steps for your operating system
#### Linux
```bash 
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip install --upgrade pip
```
#### Windows
```PowerShell
virtualenv --system-site-packages -p python3 ./venv
.\venv\Scripts\activate
```

### Installing Tensorflow and the other dependencies
If you have an NVIDIA card and want it to be utilized by TF, you can follow the instructions for GPU. Otherwise follow the CPU instructions. 
#### CPU 
```bash
pip3 install -r requirements_cpu.txt
```
#### GPU
```bash
pip3 install -r requirements_gpu.txt
```
In addition, you will have to have CUDA 9.0 and cuDNN installed.
Install CUDA 9.0 from here: https://developer.nvidia.com/cuda-90-download-archive
Install cuDNN by following instructions for your specific platform here: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
