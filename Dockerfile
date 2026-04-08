FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv pip install --system .

COPY . .

EXPOSE 7860

CMD ["python", "-m", "server.app"]