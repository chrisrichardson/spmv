FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update \
 && apt-get install -y \
       g++ cmake git \
       libeigen3-dev libmpich-dev \
       libmkl-dev libmkl-computational-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*