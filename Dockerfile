FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3-pip
RUN pip install --upgrade pip setuptools
RUN apk add --no-cache build-base patchelf ninja
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt --global-option="build_requires --no-binary :python:3.8"
CMD python application.py

