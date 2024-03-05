# Use the base PyTorch image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy just the requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the PyTorch model file and Firebase credentials file into the Docker image
COPY CNN3_LSTM1_DO_2_5_7.pth /app/CNN3_LSTM1_DO_2_5_7.pth
COPY iotproject-a0272-firebase-adminsdk-dcegr-8b5b94b32d.json /app/

# Copy the FastAPI application code into the Docker image
COPY . /app/

# Expose port 8000 (assuming your FastAPI app runs on this port)
EXPOSE 8080

# Set the command to run your FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]