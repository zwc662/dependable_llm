VENV=venv
# Recommended for universal compatibility ${CUDA_VERSION}=11.8.0
ifndef CUDA_VERSION
CUDA_VERSION=11.8.0
endif

# Set library path
LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/:/usr/local/cuda/lib64/:$(shell printenv LD_LIBRARY_PATH)

# Set cuda version for torch
cu0:=$(shell echo $(CUDA_VERSION) | tr -d -c 0-9)
$(eval cu1:=$$$(cu0))
$(eval cu2:=$$$(cu1))
$(eval cu3:=$$$(cu2))
cu4=$(cu0:$(cu3)=)
cu5=$(cu0:$(cu2)=)
 
# Remove 'nightly' if using H400
TORCH_URL:=$(shell echo "https://download.pytorch.org/whl/nightly/cu$(cu4)")

ifndef CUDA_VERSION
DEVICE=''
endif

ifeq ($(DEVICE), H100)
CUDA_VERSION=11.8.0
TORCH_URL=$(shell echo "https://download.pytorch.org/whl/cu$(cu4)")
endif

venv:
	conda create -n ${VENV} python=3.9

cuda:
	conda install -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-nvcc cuda-toolkit cuda-runtime libcusparse-dev libcusolver-dev 

	sudo mkdir -p /usr/local/cuda
	sudo mkdir -p /usr/local/cuda/lib64
	sudo rm -f /usr/local/cuda/lib64/libcudart.so
	sudo rm -f /usr/local/cuda/lib64/libcudart.a
	sudo rm -f /usr/local/cuda/lib64/libcurand.so
	sudo rm -f /usr/local/cuda/lib64/libcurand.a

	sudo ln -s ${CONDA_PREFIX}/lib/libcudart.so* /usr/local/cuda/lib64/
	sudo ln -s ${CONDA_PREFIX}/lib/libcudart_static.a /usr/local/cuda/lib64/libcudart.a
	sudo ln -s ${CONDA_PREFIX}/lib/libcurand.so* /usr/local/cuda/lib64/
	sudo ln -s ${CONDA_PREFIX}/lib/libcurand_static.a /usr/local/cuda/lib64/libcurand.a

lib:
	conda install -c anaconda git
	python -m pip install torch==2.0.1 --pre --extra-index-url ${TORCH_URL}
	python -m pip install torchvision==0.15.2 --pre --extra-index-url ${TORCH_URL} 
	python -m pip install -r requirements.txt
	python -m pip install mosaicml tensorboard jupyter ipykernel
	python -m ipykernel install --user --name=venv
	

mpt: CUDA_VERSION=11.8.0 
mpt: cuda lib
	sudo rm -rf llm-foundry
	conda install -c anaconda git
	git clone https://github.com/mosaicml/llm-foundry/
	cd llm-foundry && python -m pip install -e .[gpu]
	python -m ipykernel install --user --name=venv

# Do not recommend cuda 12.0.0 and H100
# If having to use H100, please use cuda 11.8.0 and H100 together
# If having to use H100 with cuda >= 12.0.0, make sure the variable in Makefile in bitsandbytes is revised as below
CC_cublasLt111 := " -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

qlora: cuda lib
	sudo rm -rf bitsandbytes
	git clone https://github.com/TimDettmers/bitsandbytes
	cd bitsandbytes && CUDA_VERSION=${cu4} make CC_cublasLt111=${CC_cublasLt111} cuda${cu5}x && LD_LIBRARY_PATH=${LD_LIBRARY_PATH} python setup.py install && LD_LIBRARY_PATH=${LD_LIBRARY_PATH} python -m bitsandbytes
	python -m ipykernel install --user --name=venv

#python -m pip install -q -U git+https://github.com/huggingface/transformers.git
#python -m pip install -q -U git+https://github.com/huggingface/transformers.git
#python -m pip install -q -U git+https://github.com/huggingface/peft.git
#python -m pip install -q -U git+https://github.com/huggingface/accelerate.git
#python -m pip install -q -U einops
#python -m pip install -q -U safetensors
#python -m pip install -q -U xformers 
		


