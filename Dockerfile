FROM soulteary/docker-pytorch-playground:2022.05.19
LABEL maintainer=soulteary@gmail.com

# Install gfpgan related dependencies
RUN pip install gfpgan realesrgan

# Pre-download dependent models
# https://github.com/xinntao/facexlib/releases/tag/v0.1.0
RUN mkdir -p /opt/conda/lib/python3.9/site-packages/facexlib/weights/
RUN apt install -y curl && \
    curl -L "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" -o "/opt/conda/lib/python3.9/site-packages/facexlib/weights/detection_Resnet50_Final.pth" && \
    apt remove -y curl
# OR copy from local
# COPY detection_Resnet50_Final.pth /opt/conda/lib/python3.9/site-packages/facexlib/weights/detection_Resnet50_Final.pth

# Install towhee related dependencies
# Save IPython directly displayed results as a file
RUN pip install IPython pandas
RUN sed -i -e "s/display(HTML(table))/with open('result.html', 'w') as file:\n            file.write(HTML(table).data)/" /opt/conda/lib/python3.9/site-packages/towhee/functional/mixins/display.py

# Set user workdir
WORKDIR /data
# Copy entrypoint
COPY app.py /entrypoint.py
# Set entrypoint
CMD ["python", "/entrypoint.py"]