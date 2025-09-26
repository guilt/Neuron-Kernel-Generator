FROM --platform=linux/amd64 ubuntu:22.04

# Get version
RUN . /etc/os-release

# Install dependencies
RUN apt-get update -y && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Neuron SDK repository key
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# Add Neuron SDK repository
RUN echo "deb https://apt.repos.neuron.amazonaws.com jammy main" | tee /etc/apt/sources.list.d/neuron.list || \
    echo "deb [signed-by=/usr/share/keyrings/neuron-keyring.gpg] https://apt.repos.neuron.amazonaws.com jammy main" | tee /etc/apt/sources.list.d/neuron.list

# Update and install Neuron SDK
RUN apt-get update && \
    (apt-get install -y aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools || \
     echo "Neuron SDK installation failed, continuing with mock setup") && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace

# Copy source code
COPY . .

# Build with Neuron SDK
RUN ./generate_all_kernels.sh

# Default command
CMD ["bash"]
