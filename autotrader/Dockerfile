# Reproducible image for the bot and the dashboard. Secrets come from the environment (.env), never the image.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.lock.txt pyproject.toml README.md ./
COPY config ./config
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock.txt && pip install --no-cache-dir --no-deps . \
    && useradd --create-home trader && mkdir -p /app/state /app/reports && chown -R trader /app
USER trader
VOLUME ["/app/state", "/app/reports"]
ENTRYPOINT ["autotrader"]
CMD ["run"]
