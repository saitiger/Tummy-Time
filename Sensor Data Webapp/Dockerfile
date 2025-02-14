FROM python:3.9-slim-buster AS base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV STREAMLIT_SERVER_PORT 8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE $STREAMLIT_SERVER_PORT

LABEL maintainer="sairaina@usc.edu"
LABEL version="1.0"
LABEL description="Streamlit application for data analysis"

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port=$STREAMLIT_SERVER_PORT", "--server.address=0.0.0.0"]
