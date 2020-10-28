FROM python:3.8

ENV LANG=C.UTF-8
RUN mkdir /app
WORKDIR /app
ENV MODEL /app/models/final_model_VGG16.h5
COPY requirements.txt /app
RUN pip install --no-cache-dir -r ./requirements.txt
COPY . /app
CMD ["python", "run_keras_server.py"]