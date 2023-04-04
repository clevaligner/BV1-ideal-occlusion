FROM clevaligner/clevaligner-base:v0.0.1

# Copy simple requirements file
COPY requirements.txt .

# Installing packages
RUN pip install -r requirements.txt

# Set working directory
WORKDIR /app

USER root

# Install runtime dependencies
RUN mkdir -p /var/lib/apt/lists/partial \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libx11-dev libgl1-mesa-glx libgomp1 libxrender1

# Copy application source code
COPY . /app

# Create our output folder
RUN mkdir /app/outputs/ \
    && chown -R user:user /app/outputs

# Switch to our user
USER user

# Set the entrypoint
ENTRYPOINT ["python", "testing_yehiel_with_transfo_02_04_23.py"]
