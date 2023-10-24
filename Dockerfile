FROM python:3.9-slim
# FROM python:3.9.18

WORKDIR /app

COPY requirements.txt .
RUN apt-get update -y
RUN apt-get install -y gcc build-essential libtool automake
RUN /usr/local/bin/python -m pip install --upgrade pip


RUN pip3 install -r requirements.txt \
        && rm -rf /root/.cache

# copy dependencies for the app
COPY . .

ENV LOGLEVEL "debug"

EXPOSE 80

CMD [ "python", "./index.py"]