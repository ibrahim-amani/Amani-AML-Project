# Use a stable, slim Python base image
FROM python:3.12.8-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# -----------------------------------------------------------
# 1) System dependencies
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# 2) Dependency layer (cached)
#    Copy only dependency manifests + wheels first
# -----------------------------------------------------------
COPY pyproject.toml uv.lock README.md ./
COPY dists/ ./dists/

# Install dependencies ONLY (do not install the project yet)
RUN uv sync --frozen --no-install-project --no-dev

# -----------------------------------------------------------
# 3) Playwright browsers
# -----------------------------------------------------------
RUN uv run playwright install chromium --with-deps

# -----------------------------------------------------------
# 4) Application layer
# -----------------------------------------------------------
COPY src/ ./src/

# Now install the project (entrypoints included)
RUN uv sync --frozen --no-dev

# -----------------------------------------------------------
# 5) Run API
# -----------------------------------------------------------
CMD ["uvicorn", "sanction_parser.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
