FROM python:3.10-slim
WORKDIR /app

COPY . /app

VOLUME /app/data
VOLUME /app/vsevolo_de_bert
VOLUME /app/llama_w

RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama_cpp_python --verbose
RUN pip3 install -r requirements.txt

RUN chmod +x /app/make_prediction.py

CMD ["python3","/app/make_prediction.py"]
