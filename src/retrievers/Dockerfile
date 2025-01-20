ARG BASE_IMAGE_TAG=3.10-slim

FROM python:${BASE_IMAGE_TAG} as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install required ubuntu packages
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 libmagic1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install any example specific dependency if available
ARG EXAMPLE_PATH
COPY ${EXAMPLE_PATH} /opt/${EXAMPLE_PATH}
RUN if [ -f "/opt/${EXAMPLE_PATH}/requirements.txt" ] ; then \
    pip3 install --no-cache-dir -r /opt/${EXAMPLE_PATH}/requirements.txt ; else \
    echo "Skipping example dependency installation, since requirements.txt was not found" ; \
    fi

RUN if [ "${EXAMPLE_PATH}" = "src/retrievers/unstructured_data" ]; then \
    mkdir -p /tmp-data/nltk_data/ && \
    chmod -R 777 /tmp-data && \
    chown -R 1000:1000 /tmp-data && \
    export NLTK_DATA=/tmp-data/nltk_data/ && \
    export HF_HOME=/tmp-data && \
    python3 -m nltk.downloader averaged_perceptron_tagger && \
    python3 -m nltk.downloader averaged_perceptron_tagger_eng && \
    python3 -m nltk.downloader stopwords && \
    python3 -m nltk.downloader punkt && \
    python3 -m nltk.downloader punkt_tab \
; fi

# export inside the above block is not working
ENV NLTK_DATA=/tmp-data/nltk_data/
ENV HF_HOME=/tmp-data

# Copy required common modules
COPY src/common /opt/src/common
COPY src/retrievers/server.py /opt/src/retrievers/
COPY src/retrievers/base.py /opt/src/retrievers/

WORKDIR /opt
ENTRYPOINT ["uvicorn", "src.retrievers.server:app"]
