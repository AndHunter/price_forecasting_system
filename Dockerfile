FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и данные
COPY . .

# Указываем порт
EXPOSE 5000

# Команда для запуска
CMD ["python", "main.py"]