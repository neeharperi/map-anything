rm -rf sam2

conda install -y -c conda-forge cmake
conda install -y -c conda-forge cxx-compiler==1.5.2

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

cd checkpoints && \
./download_ckpts.sh && \
cd ../..

pip install xformers==0.0.28.post3
pip install diffusers[torch] transformers
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

pip install git+https://github.com/microsoft/MoGe.git

pip install -e .
