FROM python:3.9

WORKDIR /app

# Копируем все файлы в контейнер
COPY . /app

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт 5000
EXPOSE 5000

# Запускаем Flask-приложение
CMD ["python", "app.py"]