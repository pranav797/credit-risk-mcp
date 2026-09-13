# Container image for the remote (HTTP) credit-risk MCP server.
# Build:  docker build -t credit-risk-mcp .
# Run:    docker run -e CREDIT_RISK_API_KEY=secret -p 8000:8000 credit-risk-mcp
FROM python:3.12-slim

# uv for fast, reproducible installs (copied from the official uv image).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependency manifests first so Docker can cache the install layer.
COPY pyproject.toml README.md ./
COPY uv.lock* ./

# Source + the model artifacts the server loads at startup.
COPY src ./src
COPY artifacts ./artifacts

# Install into a project venv (no dev deps in the image).
RUN uv sync --no-dev

ENV HOST=0.0.0.0
# Most platforms inject $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# CREDIT_RISK_API_KEY must be provided at run time (as a platform secret).
CMD ["uv", "run", "--no-dev", "credit-risk-http"]
