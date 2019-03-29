from ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive

ENV LANG=C.UTF-8

# Run apt to install OS packages
RUN apt update
RUN apt install -y tree vim curl python3 python3-pip git
RUN apt-get install poppler-utils -y
RUN apt install python3-tk -y
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y openjdk-8-jre
RUN update-alternatives --config java
RUN update-alternatives --config javac
RUN apt install -y maven

# Python 3 package install example
RUN pip3 install --upgrade setuptools
RUN apt-get install aspell aspell-en dictionaries-common emacsen-common enchant hunspell-en-us libaspell15 libenchant1c2a libhunspell-1.6-0 libtext-iconv-perl
RUN pip3 install ipython Cython matplotlib==2.2.3 numpy==1.15.4 pandas==0.23.4 scikit-learn scipy six==1.11.0 spacy==2.0.16 thinc==6.12.1 langdetect==1.0.7 pyenchant==2.0.0
RUN pip3 install sklearn_crfsuite rasa-nlu==0.13.8 pytest==4.0.1 gensim==3.7.0
RUN python3 -m spacy download en
RUN python3 -m spacy download en_vectors_web_lg

# create directory for "work".
RUN mkdir /work

# clone the rich context repo into /rich-context-competition
RUN git clone https://github.com/Coleridge-Initiative/rich-context-competition.git /rich-context-competition


LABEL maintainer="jonathan.morgan@nyu.edu"