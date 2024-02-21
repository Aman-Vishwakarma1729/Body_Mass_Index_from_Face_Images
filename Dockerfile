FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3 python3-pip libgl1-mesa-glx
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN pip install --upgrade pip setuptools
RUN apt-get update && apt-get install -y build-essential patchelf ninja-build
WORKDIR /app
COPY . /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
CMD ["python3", "application.py"]

