FROM continuumio/miniconda

RUN mkdir /opt/amazon-classifier
ADD . /opt/amazon-classifier
WORKDIR /opt/amazon-classifier

RUN conda config --add channels conda-forge \
    && conda env create -f environment.yml \
    && /bin/bash -c "source activate amazon-classifier"
