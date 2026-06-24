FROM python:3.13-slim

WORKDIR /main

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn","backend.main:app","--host","0.0.0.0","--port","8000"]