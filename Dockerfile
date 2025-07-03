# cpu base image
FROM python:3.11.10-slim

# install linux packages
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y ffmpeg git git-lfs

# copy source code
COPY . /opt/spkrid/
RUN pip install -r /opt/spkrid/requirements.txt && pip install /opt/spkrid/.
WORKDIR /opt
