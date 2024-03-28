conda create -n iopaint python=3.9 --yes
conda activate iopaint

pip install http://192.168.177.150/softwares/python/torch-2.1.1%2Bcu118-cp39-cp39-linux_x86_64.whl
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .


iopaint start --enable-interactive-seg --interactive-seg-device=cuda --port=8081
iopaint start --model=lama --device=cpu --port=8081

