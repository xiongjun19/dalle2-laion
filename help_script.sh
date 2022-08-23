docker run --gpus all -p 9982:9982 --name cnt_ll2 --ipc=host -it -v /u/jxiong/workspace/dalle2-laion:/workspace/dalle2 -v /u/jxiong/workspace/DALLE2-pytorch:/workspace/dal2_torch  dl2
