FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3
COPY astroml/serving /app/astroml/serving
WORKDIR /app
CMD ["python3", "-m", "astroml.serving.grpc_server"]
