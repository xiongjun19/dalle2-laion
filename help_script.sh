1. 环境准备
git clone https://github.com/lucidrains/DALLE2-pytorch.git
git checkout 1.1.0




docker run --gpus all -p 9982:9982 --name cnt_ll2 --ipc=host -it -v /u/jxiong/workspace/dalle2-laion:/workspace/dalle2 -v /u/jxiong/workspace/DALLE2-pytorch:/workspace/dal2_torch  dl2

CUDA_VISIBLE_DEVICES=3 python example_inference.py dream
