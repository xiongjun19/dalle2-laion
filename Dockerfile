FROM nvcr.io/nvidia/pytorch:21.05-py3


COPY dock_req.txt .
RUN pip3 install -r dock_req.txt
