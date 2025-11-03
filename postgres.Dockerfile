FROM postgres:16-bookworm

# 使用官方源安装 ca-certificates
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# 清华源
RUN sed -i 's|URIs: http://deb.debian.org|URIs: https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|URIs: http://security.debian.org|URIs: https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

# 安装其他构建依赖
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    postgresql-server-dev-16 \
    flex \
    bison \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 pgvector
RUN git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && make install && \
    cd .. && rm -rf pgvector

# 安装 Apache AGE
RUN git clone https://github.com/apache/age.git && \
    cd age && \
    make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config && \
    make install && \
    cd .. && rm -rf age