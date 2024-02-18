FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3-pip
RUN pip install --upgrade pip setuptools
RUN apt-get update && apt-get install -y build-essential patchelf ninja-build
COPY . /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
CMD python application.py

