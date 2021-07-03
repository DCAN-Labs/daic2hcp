FROM dcanlabs/internal-tools:v1.0.2

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG="en_US.UTF-8" \
    LC_ALL="C.UTF-8" \
    ND_ENTRYPOINT="/neurodocker/startup.sh"

RUN apt-get update && apt-get install -yq --no-install-recommends \
        bc \
        dc \
        graphviz \
        libgomp1 \
        libxmu6 \
        libxt6 \
        libfontconfig1 \
        libfreetype6 \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libgraphviz-dev \
        libice6 \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        python3 \
        python3-dev \
        python3-pip \
        tcsh \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install setuptools wheel && \
    pip3 install nipype graphviz pygraphviz pydot && \
    pip install Pillow

ENV FREESURFER_HOME=/opt/freesurfer

RUN mkdir -p /mriproc /output /app
WORKDIR /opt/dcan-tools

COPY ["app", "/app"]
COPY ["./SetupEnv.sh", "/SetupEnv.sh"]
COPY ["./entrypoint.sh", "/entrypoint.sh"]
COPY ["LICENSE", "/LICENSE"]
ENTRYPOINT ["/entrypoint.sh"]
WORKDIR /output
CMD ["--help"]
