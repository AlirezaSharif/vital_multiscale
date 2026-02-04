FROM ubuntu:20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- 1. SYSTEM DEPENDENCIES ---
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    vim \
    curl \
    bzip2 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxt-dev \
    libx11-dev \
    libxml2-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- 2. INSTALL MICROMAMBA ---
# Download and install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
    && mv bin/micromamba /usr/local/bin/ \
    && rm -rf bin

# Setup Mamba Environment Variables
ENV MAMBA_ROOT_PREFIX=/opt/conda
# FIX: Removed "-p $MAMBA_ROOT_PREFIX". Micromamba detects the ENV var automatically.
RUN mkdir -p $MAMBA_ROOT_PREFIX && \
    micromamba shell init -s bash

# --- 3. CREATE CONDA ENVIRONMENT ---
COPY environment.yml /app/environment.yml
RUN micromamba create -f /app/environment.yml -y && \
    micromamba clean --all --yes

# Add Conda to PATH
ENV PATH="/opt/conda/envs/femSolver/bin:$PATH"

# --- 4. BUILD VTK 8.1.2 (System Version) ---
RUN wget -q https://www.vtk.org/files/release/8.1/VTK-8.1.2.tar.gz && \
    tar -xzf VTK-8.1.2.tar.gz && \
    mv VTK-8.1.2 VTK-source && \
    rm VTK-8.1.2.tar.gz

RUN mkdir -p /app/vtk_build && cd /app/vtk_build && \
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DVTK_Group_Rendering=ON \
      -DVTK_Group_StandAlone=ON \
      -DCMAKE_CXX_VISIBILITY_PRESET=default \
      -DCMAKE_VISIBILITY_INLINES_HIDDEN=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ../VTK-source && \
    make -j$(nproc) && \
    make install && \
    rm -rf /app/vtk_build /app/VTK-source

# --- 5. BUILD VItA ---
# Ensure the folder "VItA" exists next to your Dockerfile!
COPY ./VItA /app/VItA-source

RUN mkdir -p /app/vita_build && cd /app/vita_build && \
    cmake \
      -DVTK_DIR=/usr/local/lib/cmake/vtk-8.1 \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ../VItA-source && \
    make -j$(nproc) && \
    make install && \
    rm -rf /app/vita_build /app/VItA-source

# --- 6. RUNTIME CONFIG ---
# FIX: Removed ":${LD_LIBRARY_PATH}" to suppress the "Undefined Variable" warning.
ENV LD_LIBRARY_PATH="/usr/local/lib"

WORKDIR /code
CMD ["/bin/bash"]
