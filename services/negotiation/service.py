"""
Enhanced DALRN Negotiation Service with production features.
Provides Nash equilibrium computation with PoDP receipts, explanations, and CIDs.
"""

import nashpy as nash
import numpy as np
import time
import json
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import our enhanced modules
from services.negotiation.validation import (
    NegotiationRequest, GameValidator, 
    validate_equilibrium, sanitize_strategy,
    detect_degenerate_game
)
from services.negotiation.explanation import (
    ExplanationGenerator, NegotiationContext,
    generate_concise_explanation
)
from services.negotiation.cid_generator import (
    NegotiationCIDGenerator, create_enhanced_cid
)

# Import common utilities
from services.common.podp import Receipt, ReceiptChain
from services.common.ipfs import put_json, get_json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for service
negotiation_cache = {}
receipt_chains = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    logger.info("Starting DALRN Negotiation Service (Enhanced)")
    yield
    logger.info("Shutting down DALRN Negotiation Service")

app = FastAPI(
    title="DALRN Negotiation Service",
    description="Enhanced negotiation service with Nash equilibrium computation, PoDP receipts, and explanations",
    version="2.0.0",
    lifespan=lifespan
)

class NegReq(BaseModel):
    """Legacy request model for backward compatibility."""
    A: list[list[float]]
    B: list[list[float]]
    rule: str = "nsw"
    batna: tuple[float, float] = (0.0, 0.0)

class EnhancedNegotiationRequest(NegotiationRequest):
    """Enhanced request with all production features."""
    generate_explanation: bool = Field(True, description="Generate explanation memo")
    generate_cid: bool = Field(True, description="Generate Causal Influence Diagram")
    generate_receipt: bool = Field(True, description="Generate PoDP receipt")
    store_in_ipfs: bool = Field(False, description="Store results in IPFS")
    include_sensitivity: bool = Field(False, description="Include sensitivity analysis")
    
class NegotiationResponse(BaseModel):
    """Comprehensive negotiation response."""
    dispute_id: str
    equilibrium: Dict[str, List[float]]
    utilities: Dict[str, float]
    selection_rule: str
    computation_time: float
    num_equilibria: int
    explanation_summary: Optional[str] = None
    explanation_ipfs: Optional[str] = None
    cid_ipfs: Optional[str] = None
    receipt_id: Optional[str] = None
    receipt_chain_ipfs: Optional[str] = None
    fairness_metrics: Optional[Dict[str, float]] = None
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}

class NegotiationService:
    """Enhanced negotiation service with production features."""
    
    def __init__(self):
        self.validator = GameValidator()
        self.explanation_generator = ExplanationGenerator()
        self.cid_generator = NegotiationCIDGenerator()
        self.computation_timeout = 30.0
        
    async def negotiate_with_receipts(
        self,
        request: EnhancedNegotiationRequest
    ) -> NegotiationResponse:
        """
        Perform negotiation with full production features.
        
        Args:
            request: Enhanced negotiation request
            
        Returns:
            Comprehensive negotiation response
        """
        start_time = time.time()
        dispute_id = f"NEG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{np.random.randint(1000, 9999)}"
        warnings = []
        
        # Initialize receipt chain
        receipt_chain = ReceiptChain(dispute_id=dispute_id)
        
        try:
            # Step 1: Validation with receipt
            validation_receipt = Receipt(
                dispute_id=dispute_id,
                step="VALIDATION",
                inputs={
                    "matrix_A_shape": f"{len(request.A)}x{len(request.A[0])}",
                    "matrix_B_shape": f"{len(request.B)}x{len(request.B[0])}",
                    "rule": request.rule,
                    "batna": list(request.batna)
                },
                params={"tolerance": request.tolerance},
                ts=datetime.utcnow().isoformat() + "Z"
            )
            
            A = np.array(request.A)
            B = np.array(request.B)
            
            validation_result = self.validator.validate_game(A, B)
            if not validation_result['valid']:
                raise ValueError(f"Game validation failed: {validation_result['errors']}")
            
            if validation_result['warnings']:
                warnings.extend(validation_result['warnings'])
            
            validation_receipt.artifacts["validation_result"] = validation_result
            validation_receipt.finalize()
            receipt_chain.add_receipt(validation_receipt)
            
            # Step 2: Equilibrium computation with receipt
            computation_receipt = Receipt(
                dispute_id=dispute_id,
                step="EQUILIBRIUM_COMPUTATION",
                inputs={
                    "payoff_A_hash": hash(A.tobytes()),
                    "payoff_B_hash": hash(B.tobytes())
                },
                params={
                    "algorithm": "lemke_howson_enumeration",
                    "max_iterations": request.max_iterations
                },
                ts=datetime.utcnow().isoformat() + "Z"
            )
            
            # Compute equilibria with timeout protection
            equilibria = await self._compute_equilibria_async(
                A, B, request.timeout_seconds
            )
            
            if not equilibria:
                # Try support enumeration as fallback
                equilibria = await self._compute_support_enumeration_async(
                    A, B, request.timeout_seconds
                )
            
            if not equilibria:
                # Generate bargaining solution as last resort
                equilibrium, utilities = self._compute_bargaining_solution(
                    A, B, request.batna
                )
                equilibria = [(equilibrium[0], equilibrium[1])]
                warnings.append("No Nash equilibrium found; using bargaining solution")
            
            computation_receipt.artifacts["num_equilibria"] = len(equilibria)
            computation_receipt.finalize()
            receipt_chain.add_receipt(computation_receipt)
            
            # Step 3: Equilibrium selection with receipt
            selection_receipt = Receipt(
                dispute_id=dispute_id,
                step="EQUILIBRIUM_SELECTION",
                inputs={"num_candidates": len(equilibria)},
                params={
                    "rule": request.rule,
                    "batna": list(request.batna)
                },
                ts=datetime.utcnow().isoformat() + "Z"
            )
            
            selected_eq, selected_utilities, selected_index = self._select_equilibrium(
                equilibria, A, B, request.rule, request.batna
            )
            
            # Validate and sanitize selected equilibrium
            is_valid, error_msg = validate_equilibrium(
                selected_eq, A.shape, request.tolerance
            )
            if not is_valid:
                logger.warning(f"Equilibrium validation warning: {error_msg}")
                selected_eq = (
                    sanitize_strategy(selected_eq[0]),
                    sanitize_strategy(selected_eq[1])
                )
                warnings.append(f"Equilibrium sanitized: {error_msg}")
            
            selection_receipt.artifacts["selected_index"] = selected_index
            selection_receipt.artifacts["utilities"] = list(selected_utilities)
            selection_receipt.finalize()
            receipt_chain.add_receipt(selection_receipt)
            
            # Step 4: Generate explanation if requested
            explanation_summary = None
            explanation_ipfs = None
            
            if request.generate_explanation:
                explanation_receipt = Receipt(
                    dispute_id=dispute_id,
                    step="EXPLANATION_GENERATION",
                    inputs={"generate_full": True},
                    ts=datetime.utcnow().isoformat() + "Z"
                )
                
                context = NegotiationContext(
                    payoff_A=A,
                    payoff_B=B,
                    equilibrium=selected_eq,
                    utilities=selected_utilities,
                    selection_rule=request.rule,
                    batna=request.batna,
                    all_equilibria=equilibria,
                    computation_time=time.time() - start_time,
                    metadata={
                        'iterations': len(equilibria),
                        'selected_index': selected_index
                    }
                )
                
                full_memo = self.explanation_generator.generate_memo(
                    context, dispute_id
                )
                
                explanation_summary = generate_concise_explanation(
                    A, B, selected_eq, selected_utilities, request.rule
                )
                
                if request.store_in_ipfs:
                    try:
                        explanation_ipfs = put_json({
                            'dispute_id': dispute_id,
                            'memo': full_memo,
                            'summary': explanation_summary,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        explanation_receipt.artifacts["ipfs_cid"] = explanation_ipfs
                    except Exception as e:
                        logger.error(f"Failed to store explanation in IPFS: {e}")
                        warnings.append("IPFS storage failed for explanation")
                
                explanation_receipt.finalize()
                receipt_chain.add_receipt(explanation_receipt)
            
            # Step 5: Generate CID if requested
            cid_ipfs = None
            
            if request.generate_cid:
                cid_receipt = Receipt(
                    dispute_id=dispute_id,
                    step="CID_GENERATION",
                    inputs={"include_fairness": True},
                    ts=datetime.utcnow().isoformat() + "Z"
                )
                
                # Calculate fairness metrics
                fairness_metrics = self._calculate_fairness_metrics(
                    selected_utilities, request.batna
                )
                
                cid_data = create_enhanced_cid(
                    A, B, selected_eq, request.batna,
                    fairness_metrics,
                    metadata={
                        'rule': request.rule,
                        'selected_index': selected_index,
                        'uncertainty': {'implementation_risk': 0.05}
                    }
                )
                
                if request.store_in_ipfs:
                    try:
                        cid_ipfs = put_json(cid_data)
                        cid_receipt.artifacts["ipfs_cid"] = cid_ipfs
                    except Exception as e:
                        logger.error(f"Failed to store CID in IPFS: {e}")
                        warnings.append("IPFS storage failed for CID")
                
                cid_receipt.finalize()
                receipt_chain.add_receipt(cid_receipt)
            else:
                fairness_metrics = self._calculate_fairness_metrics(
                    selected_utilities, request.batna
                )
            
            # Step 6: Finalize receipt chain
            receipt_chain.finalize()
            receipt_chain_ipfs = None
            
            if request.generate_receipt and request.store_in_ipfs:
                try:
                    receipt_chain_ipfs = put_json(receipt_chain.dict())
                except Exception as e:
                    logger.error(f"Failed to store receipt chain: {e}")
                    warnings.append("IPFS storage failed for receipt chain")
            
            # Cache results
            negotiation_cache[dispute_id] = {
                'request': request.dict(),
                'response': {
                    'equilibrium': selected_eq,
                    'utilities': selected_utilities,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            if request.generate_receipt:
                receipt_chains[dispute_id] = receipt_chain
            
            # Build response
            response = NegotiationResponse(
                dispute_id=dispute_id,
                equilibrium={
                    'row_strategy': selected_eq[0].tolist(),
                    'col_strategy': selected_eq[1].tolist()
                },
                utilities={
                    'player_1': float(selected_utilities[0]),
                    'player_2': float(selected_utilities[1])
                },
                selection_rule=request.rule,
                computation_time=time.time() - start_time,
                num_equilibria=len(equilibria),
                explanation_summary=explanation_summary,
                explanation_ipfs=explanation_ipfs,
                cid_ipfs=cid_ipfs,
                receipt_id=dispute_id if request.generate_receipt else None,
                receipt_chain_ipfs=receipt_chain_ipfs,
                fairness_metrics=fairness_metrics,
                warnings=warnings,
                metadata={
                    'game_properties': validation_result.get('properties', {}),
                    'selected_equilibrium_index': selected_index,
                    'convergence_iterations': len(equilibria)
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Negotiation failed for {dispute_id}: {str(e)}")
            
            # Create error receipt
            error_receipt = Receipt(
                dispute_id=dispute_id,
                step="ERROR",
                inputs={"error_type": type(e).__name__},
                artifacts={"error_message": str(e)},
                ts=datetime.utcnow().isoformat() + "Z"
            ).finalize()
            
            receipt_chain.add_receipt(error_receipt)
            
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _compute_equilibria_async(
        self,
        A: np.ndarray,
        B: np.ndarray,
        timeout: float
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute equilibria with async timeout protection."""
        try:
            game = nash.Game(A, B)
            
            # Run computation in executor with timeout
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None,
                lambda: list(game.lemke_howson_enumeration())
            )
            
            equilibria = await asyncio.wait_for(future, timeout=timeout)
            return equilibria
            
        except asyncio.TimeoutError:
            logger.warning("Lemke-Howson enumeration timed out")
            return []
        except Exception as e:
            logger.error(f"Error in equilibrium computation: {e}")
            return []
    
    async def _compute_support_enumeration_async(
        self,
        A: np.ndarray,
        B: np.ndarray,
        timeout: float
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute equilibria using support enumeration."""
        try:
            game = nash.Game(A, B)
            
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None,
                lambda: list(game.support_enumeration())
            )
            
            equilibria = await asyncio.wait_for(future, timeout=timeout/2)
            return equilibria
            
        except asyncio.TimeoutError:
            logger.warning("Support enumeration timed out")
            return []
        except Exception as e:
            logger.error(f"Error in support enumeration: {e}")
            return []
    
    def _compute_bargaining_solution(
        self,
        A: np.ndarray,
        B: np.ndarray,
        batna: Tuple[float, float]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        """Compute Nash bargaining solution as fallback."""
        # Simplified bargaining solution
        n_rows, n_cols = A.shape
        
        # Find cooperative solution (max sum of utilities)
        best_sum = -np.inf
        best_outcome = None
        
        for i in range(n_rows):
            for j in range(n_cols):
                u1 = A[i, j]
                u2 = B[i, j]
                if u1 >= batna[0] and u2 >= batna[1]:
                    total = u1 + u2
                    if total > best_sum:
                        best_sum = total
                        best_outcome = (i, j, u1, u2)
        
        if best_outcome:
            i, j, u1, u2 = best_outcome
            row_strat = np.zeros(n_rows)
            row_strat[i] = 1.0
            col_strat = np.zeros(n_cols)
            col_strat[j] = 1.0
            return (row_strat, col_strat), (u1, u2)
        
        # Default to uniform mixed strategy
        row_strat = np.ones(n_rows) / n_rows
        col_strat = np.ones(n_cols) / n_cols
        u1 = float(row_strat @ A @ col_strat)
        u2 = float(row_strat @ B @ col_strat)
        return (row_strat, col_strat), (u1, u2)
    
    def _select_equilibrium(
        self,
        equilibria: List[Tuple[np.ndarray, np.ndarray]],
        A: np.ndarray,
        B: np.ndarray,
        rule: str,
        batna: Tuple[float, float]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], int]:
        """Select equilibrium based on rule with deterministic tie-breaking."""
        if not equilibria:
            raise ValueError("No equilibria to select from")
        
        if len(equilibria) == 1:
            eq = equilibria[0]
            u1 = float(eq[0] @ A @ eq[1])
            u2 = float(eq[0] @ B @ eq[1])
            return eq, (u1, u2), 0
        
        # Calculate scores for each equilibrium
        scores = []
        utilities = []
        
        for eq in equilibria:
            row_strat, col_strat = eq
            u1 = float(row_strat @ A @ col_strat)
            u2 = float(row_strat @ B @ col_strat)
            utilities.append((u1, u2))
            
            if rule == "nsw":
                score = max(u1 - batna[0], 0) * max(u2 - batna[1], 0)
            elif rule == "egal":
                score = min(u1, u2)
            elif rule == "utilitarian":
                score = u1 + u2
            elif rule == "kalai_smorodinsky":
                # Simplified K-S approximation
                if u1 > batna[0] and u2 > batna[1]:
                    score = min((u1 - batna[0]) / (u2 - batna[1]), 
                               (u2 - batna[1]) / (u1 - batna[0]))
                else:
                    score = 0
            else:  # nash_bargaining
                score = (u1 - batna[0]) * (u2 - batna[1])
            
            scores.append(score)
        
        # Find best score with deterministic tie-breaking
        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if abs(s - best_score) < 1e-9]
        
        if len(best_indices) > 1:
            # Tie-breaking: prefer equilibrium with more balanced utilities
            balances = [1 / (1 + abs(utilities[i][0] - utilities[i][1])) 
                       for i in best_indices]
            best_balance_idx = best_indices[np.argmax(balances)]
            selected_index = best_balance_idx
        else:
            selected_index = best_indices[0]
        
        return equilibria[selected_index], utilities[selected_index], selected_index
    
    def _calculate_fairness_metrics(
        self,
        utilities: Tuple[float, float],
        batna: Tuple[float, float]
    ) -> Dict[str, float]:
        """Calculate comprehensive fairness metrics."""
        u1, u2 = utilities
        b1, b2 = batna
        
        metrics = {
            'nash_social_welfare': float(max(u1 - b1, 0) * max(u2 - b2, 0)),
            'egalitarian': float(min(u1, u2)),
            'utilitarian': float(u1 + u2),
            'gini_coefficient': float(abs(u1 - u2) / (u1 + u2)) if (u1 + u2) > 0 else 0,
            'min_max_ratio': float(min(u1, u2) / max(u1, u2)) if max(u1, u2) > 0 else 1,
            'surplus_player1': float(u1 - b1),
            'surplus_player2': float(u2 - b2),
            'total_surplus': float((u1 - b1) + (u2 - b2))
        }
        
        return metrics

# Create service instance
service = NegotiationService()

@app.post("/negotiate")
async def negotiate(req: NegReq):
    """Legacy endpoint for backward compatibility."""
    enhanced_req = EnhancedNegotiationRequest(
        A=req.A,
        B=req.B,
        rule=req.rule,
        batna=req.batna,
        generate_explanation=False,
        generate_cid=False,
        generate_receipt=False
    )
    
    response = await service.negotiate_with_receipts(enhanced_req)
    
    # Return legacy format
    return {
        "row": response.equilibrium['row_strategy'],
        "col": response.equilibrium['col_strategy'],
        "u1": response.utilities['player_1'],
        "u2": response.utilities['player_2']
    }

@app.post("/negotiate/enhanced", response_model=NegotiationResponse)
async def negotiate_enhanced(
    request: EnhancedNegotiationRequest,
    background_tasks: BackgroundTasks
):
    """Enhanced negotiation endpoint with all production features."""
    response = await service.negotiate_with_receipts(request)
    
    # Schedule background cleanup
    if response.dispute_id:
        background_tasks.add_task(
            cleanup_old_cache,
            response.dispute_id
        )
    
    return response

@app.get("/negotiate/{dispute_id}")
async def get_negotiation(dispute_id: str):
    """Retrieve cached negotiation results."""
    if dispute_id not in negotiation_cache:
        raise HTTPException(status_code=404, detail="Negotiation not found")
    
    return negotiation_cache[dispute_id]

@app.get("/receipt/{dispute_id}")
async def get_receipt_chain(dispute_id: str):
    """Retrieve receipt chain for a negotiation."""
    if dispute_id not in receipt_chains:
        raise HTTPException(status_code=404, detail="Receipt chain not found")
    
    return receipt_chains[dispute_id].dict()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "negotiation",
        "version": "2.0.0",
        "cache_size": len(negotiation_cache),
        "receipt_chains": len(receipt_chains)
    }

async def cleanup_old_cache(dispute_id: str, delay: int = 3600):
    """Clean up old cache entries after delay."""
    await asyncio.sleep(delay)
    
    if dispute_id in negotiation_cache:
        del negotiation_cache[dispute_id]
        logger.info(f"Cleaned up cache for {dispute_id}")
    
    if dispute_id in receipt_chains:
        del receipt_chains[dispute_id]
        logger.info(f"Cleaned up receipt chain for {dispute_id}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("NEGOTIATION_PORT", 8003))
    logger.info(f"Starting Negotiation Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
