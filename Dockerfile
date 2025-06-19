# Dockerfile
FROM python:3.9-slim

# 1. Establece el directorio de trabajo
WORKDIR /app

# 2. Copia e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copia el resto del código
COPY . .

# 4. Expone el puerto de Flask
EXPOSE 5000

# 5. Modo desarrollo para recarga automática
ENV FLASK_ENV=development

# 6. Comando por defecto
CMD ["python", "app.py"]
