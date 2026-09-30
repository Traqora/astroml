# Production Apache Airflow image for AstroML pipeline orchestration
FROM apache/airflow:2.8.1-python3.10

USER root

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Set working directory
WORKDIR /opt/airflow

# Copy requirements and install AstroML dependencies
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        apache-airflow-providers-postgres \
        apache-airflow-providers-redis \
        apache-airflow-providers-http \
        apache-airflow-providers-amazon

# Copy AstroML package and DAGs
COPY astroml /opt/airflow/astroml
COPY configs/pipeline/airflow_config.yaml /opt/airflow/config/airflow_config.yaml

# Set Python path
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"

# Default command
CMD ["airflow", "standalone"]
