FROM vllm/vllm-openai:latest

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY monitor-client/ /app/monitor-client/

ENTRYPOINT ["python3", "/app/monitor-client/entrypoint.py"]
