FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md ./
COPY driftguard ./driftguard
COPY examples ./examples

RUN pip install --no-cache-dir .

CMD ["python", "-m", "driftguard", "--help"]

