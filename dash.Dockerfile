FROM python:3.10.12

WORKDIR /root/app
# WORKDIR /home/knl/DSAI/NLP/w1/a1/app

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install pandas
RUN pip3 install torch==2.2
RUN pip3 install torchtext==0.17
RUN pip3 install datasets
RUN pip3 install dash_chat
RUN pip3 install spacy
RUN python3 -m spacy download en_core_web_sm
# RUN pip3 install pickle


# Testing module
RUN pip3 install dash[testing]

COPY ./app /root/app/
# COPY . ./
# CMD tail -f /dev/null

CMD python3 main.py