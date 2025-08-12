"""
DALRN Vector Search Service
Implements FAISS HNSW index with gRPC and HTTP interfaces
"""

import os
import json
import time
import hashlib
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent import futures
from dataclasses import dataclass, asdict

import numpy as np
import faiss
import grpc
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import generated gRPC code
try:
    from . import search_pb2
    from . import search_pb2_grpc
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import search_pb2
    import search_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
VECTOR_DIM = 768
HNSW_M = 32
HNSW_EF_SEARCH = 128
HNSW_EF_CONSTRUCTION = 200
DEFAULT_K = 10
MAX_REWEIGHT_ITERS = 10
P95_LATENCY_TARGET_MS = 600

# Set random seed for reproducibility
np.random.seed(42)


@dataclass
class SearchMetrics:
    """Metrics for search operations"""
    query_id: str
    k: int
    recall_at_10: float
    latency_ms: float
    reweight_enabled: bool
    total_vectors: int


@dataclass
class PoDPReceipt:
    """Proof of Data Processing receipt"""
    receipt_type: str = "SEARCH_QUERY"
    query_id: str = ""
    timestamp: float = 0.0
    metrics: Optional[SearchMetrics] = None


class VectorIndex:
    """FAISS HNSW Index wrapper with thread-safe operations"""
    
    def __init__(self, dimension: int = VECTOR_DIM, m: int = HNSW_M):
        self.dimension = dimension
        self.m = m
        self.index = None
        self.vectors = None  # Store original vectors for recall calculation
        self.lock = threading.RLock()
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize HNSW index with specified parameters"""
        with self.lock:
            self.index = faiss.IndexHNSWFlat(self.dimension, self.m)
            self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            self.index.hnsw.efSearch = HNSW_EF_SEARCH
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
            logger.info(f"Initialized HNSW index: dim={self.dimension}, M={self.m}")
    
    def build(self, embeddings: np.ndarray, append: bool = False) -> int:
        """Build or append to index with L2 normalization"""
        with self.lock:
            if not append or self.index.ntotal == 0:
                self._initialize_index()
            
            # Ensure correct shape and type
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            embeddings = embeddings.astype(np.float32)
            
            # L2 normalize vectors (unit norm)
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Store for recall calculation
            if append:
                self.vectors = np.vstack([self.vectors, embeddings])
            else:
                self.vectors = embeddings.copy()
            
            return self.index.ntotal
    
    def search(self, query_vector: np.ndarray, k: int = DEFAULT_K,
               reweight_iters: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Search k nearest neighbors with optional reweighting"""
        with self.lock:
            if self.index.ntotal == 0:
                return np.array([]), np.array([])
            
            # Prepare query
            query_vector = query_vector.astype(np.float32)
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # L2 normalize
            faiss.normalize_L2(query_vector)
            
            # Adjust k if needed
            k = min(k, self.index.ntotal)
            
            # Search
            distances, indices = self.index.search(query_vector, k)
            
            # Apply reweighting if enabled
            if reweight_iters > 0:
                distances, indices = self._apply_reweighting(
                    query_vector, distances[0], indices[0], reweight_iters
                )
                return indices.reshape(1, -1), distances.reshape(1, -1)
            
            return indices, distances
    
    def _apply_reweighting(self, query: np.ndarray, distances: np.ndarray,
                          indices: np.ndarray, iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply quantum-inspired iterative reweighting"""
        weights = 1.0 - distances  # Convert distances to similarities
        
        for _ in range(min(iterations, MAX_REWEIGHT_ITERS)):
            # Quantum-inspired reweighting: amplify high similarities
            # Ensure weights are non-negative to avoid warnings
            weights = np.maximum(weights, 0)
            weights = np.power(weights, 1.5)
            weights = weights / (np.sum(weights) + 1e-10)  # Normalize with small epsilon
            
            # Adjust scores based on weights
            adjusted_scores = weights * (1.0 - distances)
            
            # Re-sort based on adjusted scores
            sorted_idx = np.argsort(-adjusted_scores)
            indices = indices[sorted_idx]
            distances = distances[sorted_idx]
            weights = weights[sorted_idx]
        
        return distances, indices
    
    def calculate_recall(self, query: np.ndarray, retrieved: np.ndarray,
                        k: int = 10) -> float:
        """Calculate recall@k against brute-force ground truth"""
        if self.vectors is None or len(self.vectors) == 0:
            return 0.0
        
        with self.lock:
            # Normalize query
            query = query.astype(np.float32)
            if len(query.shape) == 1:
                query = query.reshape(1, -1)
            faiss.normalize_L2(query)
            
            # Brute-force search for ground truth
            similarities = np.dot(self.vectors, query.T).flatten()
            ground_truth = np.argsort(-similarities)[:k]
            
            # Calculate recall
            retrieved_set = set(retrieved[:k])
            ground_truth_set = set(ground_truth)
            
            if len(ground_truth_set) == 0:
                return 0.0
            
            recall = len(retrieved_set & ground_truth_set) / len(ground_truth_set)
            return recall
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        with self.lock:
            return {
                "total_vectors": int(self.index.ntotal) if self.index else 0,
                "dimension": self.dimension,
                "index_type": "HNSW",
                "m_parameter": self.m,
                "ef_search": HNSW_EF_SEARCH,
                "ef_construction": HNSW_EF_CONSTRUCTION
            }
    
    def save(self, path: str):
        """Save index to disk"""
        with self.lock:
            faiss.write_index(self.index, path)
            np.save(path + ".vectors", self.vectors)
            logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load index from disk"""
        with self.lock:
            self.index = faiss.read_index(path)
            self.vectors = np.load(path + ".vectors.npy")
            logger.info(f"Loaded index from {path}")


# Global index instance
vector_index = VectorIndex()


# Pydantic models for HTTP API
class BuildRequest(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Vector embeddings")
    append: bool = Field(False, description="Append to existing index")


class QueryRequest(BaseModel):
    query: List[float] = Field(..., description="Query vector")
    k: int = Field(DEFAULT_K, description="Number of neighbors")
    reweight_iters: int = Field(0, description="Reweighting iterations (0=OFF)")
    query_id: Optional[str] = Field(None, description="Query ID for tracking")


class BuildResponse(BaseModel):
    total_vectors: int
    success: bool
    message: str


class QueryResponse(BaseModel):
    indices: List[int]
    scores: List[float]
    recall_at_10: float
    latency_ms: float
    query_id: str


class StatsResponse(BaseModel):
    total_vectors: int
    dimension: int
    index_type: str
    m_parameter: int
    ef_search: int


# gRPC Service Implementation
class SearchServicer(search_pb2_grpc.SearchServiceServicer):
    """gRPC service implementation"""
    
    def Build(self, request, context):
        """Build index from embeddings"""
        try:
            # Convert proto vectors to numpy array
            embeddings = []
            for vector in request.embeddings:
                embeddings.append(list(vector.values))
            
            if not embeddings:
                return search_pb2.BuildResponse(
                    success=False,
                    message="No embeddings provided"
                )
            
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Build index
            total = vector_index.build(embeddings, append=request.append)
            
            return search_pb2.BuildResponse(
                total_vectors=total,
                success=True,
                message=f"Successfully indexed {total} vectors"
            )
        
        except Exception as e:
            logger.error(f"Build error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return search_pb2.BuildResponse(
                success=False,
                message=str(e)
            )
    
    def Query(self, request, context):
        """Query k nearest neighbors"""
        try:
            start_time = time.time()
            
            # Generate query ID if not provided
            query_id = request.query_id or hashlib.md5(
                str(time.time()).encode()
            ).hexdigest()[:8]
            
            # Convert query vector
            query_vector = np.array(list(request.query_vector.values), dtype=np.float32)
            
            # Search
            indices, distances = vector_index.search(
                query_vector,
                k=request.k,
                reweight_iters=request.reweight_iters
            )
            
            if len(indices) == 0:
                return search_pb2.QueryResponse(
                    query_id=query_id,
                    recall_at_10=0.0,
                    latency_ms=0.0
                )
            
            # Calculate metrics
            recall = vector_index.calculate_recall(query_vector, indices[0], k=10)
            latency_ms = (time.time() - start_time) * 1000
            
            # Convert distances to similarity scores
            scores = (1.0 - distances[0]).tolist()
            
            # Emit PoDP receipt
            self._emit_podp_receipt(query_id, request.k, recall, latency_ms,
                                   request.reweight_iters > 0)
            
            return search_pb2.QueryResponse(
                indices=indices[0].tolist(),
                scores=scores,
                recall_at_10=recall,
                latency_ms=latency_ms,
                query_id=query_id
            )
        
        except Exception as e:
            logger.error(f"Query error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return search_pb2.QueryResponse()
    
    def GetStats(self, request, context):
        """Get index statistics"""
        try:
            stats = vector_index.get_stats()
            return search_pb2.StatsResponse(**stats)
        except Exception as e:
            logger.error(f"Stats error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return search_pb2.StatsResponse()
    
    def HealthCheck(self, request, context):
        """Health check"""
        try:
            healthy = vector_index.index is not None
            return search_pb2.HealthResponse(
                healthy=healthy,
                status="OK" if healthy else "Index not initialized"
            )
        except Exception as e:
            return search_pb2.HealthResponse(
                healthy=False,
                status=str(e)
            )
    
    def _emit_podp_receipt(self, query_id: str, k: int, recall: float,
                          latency_ms: float, reweight_enabled: bool):
        """Emit PoDP receipt for query"""
        metrics = SearchMetrics(
            query_id=query_id,
            k=k,
            recall_at_10=recall,
            latency_ms=latency_ms,
            reweight_enabled=reweight_enabled,
            total_vectors=vector_index.index.ntotal if vector_index.index else 0
        )
        
        receipt = PoDPReceipt(
            query_id=query_id,
            timestamp=time.time(),
            metrics=metrics
        )
        
        # Log receipt (in production, send to PoDP system)
        logger.info(f"PoDP Receipt: {json.dumps(asdict(receipt), default=str)}")


# FastAPI HTTP Application
app = FastAPI(title="DALRN Vector Search Service")


@app.post("/build", response_model=BuildResponse)
async def build_index(request: BuildRequest):
    """Build index from embeddings"""
    try:
        embeddings = np.array(request.embeddings, dtype=np.float32)
        total = vector_index.build(embeddings, append=request.append)
        
        return BuildResponse(
            total_vectors=total,
            success=True,
            message=f"Successfully indexed {total} vectors"
        )
    except Exception as e:
        logger.error(f"Build error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """Query k nearest neighbors"""
    try:
        start_time = time.time()
        
        # Generate query ID if not provided
        query_id = request.query_id or hashlib.md5(
            str(time.time()).encode()
        ).hexdigest()[:8]
        
        # Search
        query_vector = np.array(request.query, dtype=np.float32)
        indices, distances = vector_index.search(
            query_vector,
            k=request.k,
            reweight_iters=request.reweight_iters
        )
        
        if len(indices) == 0:
            return QueryResponse(
                indices=[],
                scores=[],
                recall_at_10=0.0,
                latency_ms=0.0,
                query_id=query_id
            )
        
        # Calculate metrics
        recall = vector_index.calculate_recall(query_vector, indices[0], k=10)
        latency_ms = (time.time() - start_time) * 1000
        
        # Convert distances to similarity scores
        scores = (1.0 - distances[0]).tolist()
        
        return QueryResponse(
            indices=indices[0].tolist(),
            scores=scores,
            recall_at_10=recall,
            latency_ms=latency_ms,
            query_id=query_id
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get index statistics"""
    try:
        stats = vector_index.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    try:
        healthy = vector_index.index is not None
        return JSONResponse(
            content={
                "healthy": healthy,
                "status": "OK" if healthy else "Index not initialized"
            },
            status_code=200 if healthy else 503
        )
    except Exception as e:
        return JSONResponse(
            content={"healthy": False, "status": str(e)},
            status_code=503
        )


def run_grpc_server(port: int = 50051):
    """Run gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    search_pb2_grpc.add_SearchServiceServicer_to_server(
        SearchServicer(), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"gRPC server started on port {port}")
    server.wait_for_termination()


def run_http_server(host: str = "0.0.0.0", port: int = 8000):
    """Run HTTP server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "grpc":
        run_grpc_server()
    else:
        # Default to HTTP server
        run_http_server()