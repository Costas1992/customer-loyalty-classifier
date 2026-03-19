
# BASE IMAGE
# Official lightweight Python image
FROM python:3.11-slim

# SET WORKING DIRECTORY 
# All commands run from this folder inside container
WORKDIR /app

# COPY & INSTALL DEPENDENCIES 
# Copy requirements first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY PROJECT FILES 
COPY app/    ./app/
COPY model/  ./model/
COPY data/   ./data/

# EXPOSE PORT 
# Tell Docker our app runs on port 8000
EXPOSE 8000

# START COMMAND 
# This runs when the container starts

CMD ["sh", "-c", "python model/train.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]