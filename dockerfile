# Usa uma imagem Python oficial compatível
FROM python:3.12-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta do Flask
EXPOSE 3001

# Comando para rodar o servidor Flask
CMD ["python", "/api/main.py"] 
