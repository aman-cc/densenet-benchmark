FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

LABEL description="SatSure Assignment"
LABEL maintainer="Aman Kumar <amankumar528491@gmail.com>"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8 
ENV PYTHON_VERSION=3.11.8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    wget \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    gcc

# ========================== INSTALL DEPENDENCIES ==========================
# Install uv
ENV UV_RELEASE_NAME=uv-x86_64-unknown-linux-gnu
RUN curl -L -o /usr/local/bin/uv.tar.gz https://github.com/astral-sh/uv/releases/download/0.6.9/$UV_RELEASE_NAME.tar.gz
RUN tar xzf /usr/local/bin/uv.tar.gz -C /usr/local/bin
RUN mv /usr/local/bin/$UV_RELEASE_NAME/uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

COPY requirements/requirements.txt /tmp/requirements.txt
WORKDIR /srv/
RUN uv venv --python $PYTHON_VERSION
RUN uv pip install -r /tmp/requirements.txt \
    --index-strategy unsafe-best-match \
    --trusted-host pypi.ngc.nvidia.com
COPY . /srv/

CMD ["uv", "run", "app/benchmark.py"]
