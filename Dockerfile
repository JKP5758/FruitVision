FROM python:3.11-slim

# deps untuk opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# pastikan folder ada
RUN mkdir -p data_buah/apel data_buah/pisang data_buah/jeruk data_buah/durian output app

EXPOSE 5000
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

CMD ["python", "web.py"]
