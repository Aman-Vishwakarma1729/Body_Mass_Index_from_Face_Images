FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev libgl1-mesa-glx cmake\
    libglib2.0-0 libv4l-dev libx264-dev build-essential patchelf ninja-build

RUN pip install --upgrade pip setuptools

WORKDIR /app
COPY . /app
COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

CMD ["python3", "application.py"]
