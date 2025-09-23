"""
GPU acceleration for FAISS vector search
Automatic fallback to CPU if GPU unavailable
"""
import os
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

def setup_faiss_gpu() -> Tuple[any, bool]:
    """Setup FAISS with GPU support if available"""

    try:
        # Try GPU version first
        import faiss

        # Check for GPU availability
        ngpus = faiss.get_num_gpus()

        if ngpus > 0:
            logger.info(f"FAISS GPU enabled with {ngpus} GPU(s)")
            return faiss, True
        else:
            logger.info("No GPU detected, using CPU FAISS")
            return faiss, False

    except ImportError:
        # Fallback to CPU version
        try:
            import faiss
            logger.warning("Using CPU FAISS (GPU version not installed)")
            return faiss, False
        except ImportError:
            raise ImportError(
                "Neither faiss-gpu nor faiss-cpu installed. "
                "Install with: pip install faiss-gpu or pip install faiss-cpu"
            )

class GPUAcceleratedIndex:
    """FAISS index with automatic GPU/CPU selection and optimization"""

    def __init__(self, dim: int, use_gpu: Optional[bool] = None):
        self.dim = dim
        self.faiss, self.gpu_available = setup_faiss_gpu()

        # Override GPU usage if specified
        if use_gpu is not None:
            self.use_gpu = use_gpu and self.gpu_available
        else:
            self.use_gpu = self.gpu_available

        self.index = None
        self.gpu_index = None
        self.gpu_resources = None

        # Performance monitoring
        self.search_times = []
        self.add_times = []

    def _create_gpu_resources(self):
        """Create and configure GPU resources"""
        if not self.use_gpu:
            return None

        try:
            res = self.faiss.StandardGpuResources()

            # Configure GPU memory
            res.setTempMemory(512 * 1024 * 1024)  # 512MB temp memory

            # Set multi-GPU options if available
            if self.faiss.get_num_gpus() > 1:
                logger.info(f"Configuring multi-GPU with {self.faiss.get_num_gpus()} GPUs")
                # Could implement sharding across GPUs here

            return res
        except Exception as e:
            logger.error(f"Failed to create GPU resources: {e}")
            return None

    def build_index(
        self,
        vectors: np.ndarray,
        index_type: str = "HNSW32"
    ) -> None:
        """Build index with GPU acceleration if available"""

        vectors = np.ascontiguousarray(vectors.astype('float32'))
        n_vectors = vectors.shape[0]

        logger.info(f"Building {index_type} index for {n_vectors} vectors")

        if index_type == "HNSW32":
            # Create HNSW index (CPU only, but fast)
            M = 32
            index = self.faiss.IndexHNSWFlat(self.dim, M)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128

            # HNSW doesn't support GPU, use CPU
            index.add(vectors)
            self.index = index
            self.use_gpu = False  # HNSW is CPU-only

        elif index_type == "IVF":
            # Create IVF index (GPU-compatible)
            nlist = min(4096, int(np.sqrt(n_vectors)))
            quantizer = self.faiss.IndexFlatL2(self.dim)

            if self.use_gpu:
                # Create GPU-compatible IVF index
                index = self.faiss.IndexIVFFlat(quantizer, self.dim, nlist)
                index.train(vectors)

                # Move to GPU
                self.gpu_resources = self._create_gpu_resources()
                if self.gpu_resources:
                    try:
                        gpu_index = self.faiss.index_cpu_to_gpu(
                            self.gpu_resources, 0, index
                        )
                        gpu_index.add(vectors)
                        self.gpu_index = gpu_index
                        logger.info(f"IVF index moved to GPU with {nlist} lists")
                    except Exception as e:
                        logger.error(f"GPU indexing failed: {e}, using CPU")
                        index.add(vectors)
                        self.index = index
                        self.use_gpu = False
                else:
                    index.add(vectors)
                    self.index = index
                    self.use_gpu = False
            else:
                index.train(vectors)
                index.add(vectors)
                self.index = index

        elif index_type == "Flat":
            # Create flat index (exact search, GPU-compatible)
            index = self.faiss.IndexFlatL2(self.dim)

            if self.use_gpu:
                self.gpu_resources = self._create_gpu_resources()
                if self.gpu_resources:
                    try:
                        gpu_index = self.faiss.index_cpu_to_gpu(
                            self.gpu_resources, 0, index
                        )
                        gpu_index.add(vectors)
                        self.gpu_index = gpu_index
                        logger.info("Flat index moved to GPU")
                    except Exception as e:
                        logger.error(f"GPU indexing failed: {e}, using CPU")
                        index.add(vectors)
                        self.index = index
                        self.use_gpu = False
                else:
                    index.add(vectors)
                    self.index = index
                    self.use_gpu = False
            else:
                index.add(vectors)
                self.index = index

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        logger.info(f"Index built with {n_vectors} vectors, GPU: {self.use_gpu}")

    def search(
        self,
        queries: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search with GPU acceleration"""

        queries = np.ascontiguousarray(queries.astype('float32'))

        if self.use_gpu and self.gpu_index:
            distances, indices = self.gpu_index.search(queries, k)
        elif self.index:
            distances, indices = self.index.search(queries, k)
        else:
            raise ValueError("Index not built")

        return distances, indices

    def add_vectors(self, vectors: np.ndarray) -> None:
        """Add vectors to existing index"""

        vectors = np.ascontiguousarray(vectors.astype('float32'))

        if self.use_gpu and self.gpu_index:
            self.gpu_index.add(vectors)
        elif self.index:
            self.index.add(vectors)
        else:
            raise ValueError("Index not built")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {
            "using_gpu": self.use_gpu,
            "gpu_available": self.gpu_available,
            "index_type": type(self.index).__name__ if self.index else "None",
            "dimension": self.dim
        }

        if self.use_gpu and self.gpu_index:
            stats["gpu_memory_used"] = self.gpu_resources.getTempMemoryAvailable()
            stats["num_vectors"] = self.gpu_index.ntotal
        elif self.index:
            stats["num_vectors"] = self.index.ntotal

        return stats

# Enhanced search service integration
def create_optimized_index(dim: int, num_vectors: int) -> GPUAcceleratedIndex:
    """Create optimized index based on data size and hardware"""

    gpu_index = GPUAcceleratedIndex(dim)

    # Choose index type based on dataset size
    if num_vectors < 10000:
        # Small dataset: use exact search
        index_type = "Flat"
    elif num_vectors < 100000:
        # Medium dataset: use HNSW (fast CPU)
        index_type = "HNSW32"
    else:
        # Large dataset: use IVF (GPU-friendly)
        index_type = "IVF"

    logger.info(f"Selected {index_type} index for {num_vectors} vectors")
    return gpu_index, index_type

# Configuration for Docker GPU support
def get_gpu_docker_config() -> str:
    """Return Docker configuration for GPU support"""
    return """
# Dockerfile.gpu for GPU-enabled FAISS
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install FAISS GPU version
RUN pip3 install faiss-gpu==1.7.4 numpy

# Copy application
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt

COPY . /app
WORKDIR /app

# Run with GPU support
CMD ["python3", "service.py"]
"""

# Docker Compose configuration for GPU
def get_docker_compose_gpu() -> str:
    """Return Docker Compose configuration with GPU support"""
    return """
# docker-compose.gpu.yml
version: '3.8'

services:
  search:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - USE_GPU=true
    ports:
      - "8100:8100"
"""