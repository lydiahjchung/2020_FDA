FROM ubuntu:18.04
FROM python:3

COPY . /app
WORKDIR /app

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt install vim -y
RUN pip install -r requirements.txt

CMD ["python", "./flask/app.py"]