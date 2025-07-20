docker build --build-arg IMAGE_NAME="blk-hacking-mx-tomas-estrada" -t blk-hacking-mx-tomas-estrada .
ARG IMAGE_NAME=blk-hacking-mx-tomas-estrada
FROM python:3.9.13

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /Project
COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . /Project/

EXPOSE 80

# Set environment variables
ENV CONTAINER_PORT=80
ENV HOST_PORT=5477
ENV IMAGE_NAME=$IMAGE_NAME
LABEL image_name=$IMAGE_NAME

# Define the command to run your application
CMD ["python", "BASECODE_3.py"]