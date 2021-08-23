FROM node:16.7.0-slim as node
FROM python:3.9.0-slim as python
FROM rust:1.54.0-slim as rust
FROM httpd:2.4.48

RUN apt-get update && apt-get install -y \
  libc-dev \
  libffi-dev \
  gcc \
  g++ \
  make \
  cmake \
  curl \
  ca-certificates \
  perl

# node
COPY --from=node /usr/local/bin /usr/local/bin
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm

# python
COPY --from=python /usr/local/bin /usr/local/bin
COPY --from=python /usr/local/lib /usr/local/lib
COPY --from=python /usr/local/include /usr/local/include
RUN ln /usr/local/lib/libpython3.9.so.1.0 /lib64/libpython3.9.so.1.0 && ldconfig

# rust
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV PATH $PATH:/usr/local/cargo/bin/
ENV RUSTUP_HOME /usr/local/rustup

# copy repo
COPY . /masked-lm-wasm
WORKDIR /masked-lm-wasm

# download and convert bert model
RUN pip install torch transformers && pip show transformers
RUN mkdir tmp \
  && python /usr/local/lib/python3.9/site-packages/transformers/convert_graph_to_onnx.py --pipeline fill-mask --model bert-base-cased --framework pt tmp/bert-masked.onnx

# download vocab.txt
RUN cd tmp && curl -o vocab.txt https://huggingface.co/bert-base-cased/raw/main/vocab.txt
RUN mv tmp/* maskedlm/ && rm -r tmp

# rust project setup
RUN cd maskedlm && cargo install wasm-pack

# node project setup
RUN npm install
RUN npx webpack build
RUN cp ./dist/* /usr/local/apache2/htdocs/ && cp ./config/httpd.conf /usr/local/apache2/conf/httpd.conf

EXPOSE 80
