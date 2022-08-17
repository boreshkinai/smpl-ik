# ===========
# FIRST STAGE
# ===========
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime as pytorch

RUN date
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

RUN python -m pip install pip -U

# Install Jupyter
RUN conda install -y jupyter
# Install tini, which will keep the container up as a PID 1
RUN apt-get install -y curl grep sed dpkg && \
    curl -L "https://github.com/krallin/tini/releases/download/v0.19.0/tini_0.19.0.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html


# ============
# SECOND STAGE
# ============
FROM pytorch

# Export port for TensorBoard
EXPOSE 6006
# Export port 8888 for jupyter
EXPOSE 8888

RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = 'sha1:3f84353ad3f5:d1b6eeb440acbc49330646714898ae27c8dd56c2'" >> /root/.jupyter/jupyter_notebook_config.py

ENTRYPOINT [ "/usr/bin/tini", "--" ]

CMD ["jupyter", "notebook", "--allow-root"]

WORKDIR /workspace/protores

