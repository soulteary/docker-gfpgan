FROM nvcr.io/nvidia/pytorch:23.04-py3
LABEL org.opencontainers.image.authors="soulteary@gmail.com"

RUN pip install gfpgan==1.3.8 realesrgan==0.3.0 facexlib==0.3.0 gradio==3.39.0

WORKDIR /app
COPY src/app.py ./

CMD ["python", "app.py"]