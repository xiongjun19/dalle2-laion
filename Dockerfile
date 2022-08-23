FROM nvcr.io/nvidia/pytorch:21.07-py3


COPY dock_req.txt .
RUN pip3 install -r dock_req.txt
ENV PYTHONPATH="${PYTHONPATH}:/workspace/dal2_torch"
