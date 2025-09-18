"""
Input validation and error handling for DALRN Negotiation Service.
Provides comprehensive validation for game matrices, parameters, and results.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, code: str = "VALIDATION_ERROR", details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class NegotiationRequest(BaseModel):
    """Enhanced negotiation request with comprehensive validation."""
    A: List[List[float]] = Field(..., description="Payoff matrix for player 1")
    B: List[List[float]] = Field(..., description="Payoff matrix for player 2")
    rule: str = Field("nsw", description="Selection rule: nsw (Nash Social Welfare) or egal (Egalitarian)")
    batna: Tuple[float, float] = Field((0.0, 0.0), description="Best Alternative to Negotiated Agreement")
    max_iterations: int = Field(1000, ge=1, le=10000, description="Maximum iterations for equilibrium computation")
    tolerance: float = Field(1e-9, gt=0, le=1e-3, description="Numerical tolerance for convergence")
    timeout_seconds: float = Field(30.0, gt=0, le=300, description="Maximum computation time in seconds")
    n_players: Optional[int] = Field(2, ge=2, le=10, description="Number of players (for future n-party support)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for tracking")
    
    @field_validator('A', 'B', mode='before')
    @classmethod
    def validate_matrix(cls, v, info):
        """Validate payoff matrices."""
        field_name = info.field_name
        if not v or not all(v):
            raise ValueError(f"{field_name}: Matrix cannot be empty")
        
        rows = len(v)
        if rows == 0:
            raise ValueError(f"{field_name}: Matrix must have at least one row")
        
        cols = len(v[0])
        if cols == 0:
            raise ValueError(f"{field_name}: Matrix must have at least one column")
        
        # Check rectangular shape
        for i, row in enumerate(v):
            if len(row) != cols:
                raise ValueError(f"{field_name}: Inconsistent row lengths. Row {i} has {len(row)} columns, expected {cols}")
        
        # Check for valid numeric values
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(f"{field_name}: Non-numeric value at position [{i}][{j}]")
                if np.isnan(val) or np.isinf(val):
                    raise ValueError(f"{field_name}: Invalid value (NaN or Inf) at position [{i}][{j}]")
        
        return v
    
    @model_validator(mode='after')
    def validate_matrix_dimensions(self):
        """Ensure both matrices have compatible dimensions."""
        if self.A and self.B:
            a_shape = (len(self.A), len(self.A[0]))
            b_shape = (len(self.B), len(self.B[0]))
            if a_shape != b_shape:
                raise ValueError(f"Matrix dimension mismatch: A is {a_shape}, B is {b_shape}")
        return self
    
    @field_validator('rule')
    @classmethod
    def validate_rule(cls, v):
        """Validate selection rule."""
        valid_rules = ['nsw', 'egal', 'utilitarian', 'kalai_smorodinsky', 'nash_bargaining']
        if v not in valid_rules:
            raise ValueError(f"Invalid rule '{v}'. Must be one of: {', '.join(valid_rules)}")
        return v
    
    @field_validator('batna')
    @classmethod
    def validate_batna(cls, v):
        """Validate BATNA values."""
        if len(v) != 2:
            raise ValueError(f"BATNA must have exactly 2 values, got {len(v)}")
        for i, val in enumerate(v):
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Invalid BATNA value at position {i}")
        return v

def validate_equilibrium(
    equilibrium: Tuple[np.ndarray, np.ndarray],
    game_shape: Tuple[int, int],
    tolerance: float = 1e-9
) -> Tuple[bool, Optional[str]]:
    """
    Validate computed equilibrium strategies.
    
    Args:
        equilibrium: Tuple of strategy vectors (row_strategy, col_strategy)
        game_shape: Shape of the game matrices (n_rows, n_cols)
        tolerance: Numerical tolerance for validation
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        row_strat, col_strat = equilibrium
        n_rows, n_cols = game_shape
        
        # Check dimensions
        if len(row_strat) != n_rows:
            return False, f"Row strategy dimension mismatch: expected {n_rows}, got {len(row_strat)}"
        if len(col_strat) != n_cols:
            return False, f"Column strategy dimension mismatch: expected {n_cols}, got {len(col_strat)}"
        
        # Check probability constraints
        if not np.allclose(row_strat.sum(), 1.0, atol=tolerance):
            return False, f"Row strategy doesn't sum to 1: {row_strat.sum()}"
        if not np.allclose(col_strat.sum(), 1.0, atol=tolerance):
            return False, f"Column strategy doesn't sum to 1: {col_strat.sum()}"
        
        # Check non-negativity
        if np.any(row_strat < -tolerance):
            return False, f"Negative values in row strategy: {row_strat}"
        if np.any(col_strat < -tolerance):
            return False, f"Negative values in column strategy: {col_strat}"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_payoff_range(
    A: np.ndarray,
    B: np.ndarray,
    max_abs_value: float = 1e6
) -> Tuple[bool, Optional[str]]:
    """
    Validate that payoff values are within reasonable ranges.
    
    Args:
        A, B: Payoff matrices
        max_abs_value: Maximum absolute value allowed
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    a_max = np.abs(A).max()
    b_max = np.abs(B).max()
    
    if a_max > max_abs_value:
        return False, f"Matrix A has values too large: max abs value = {a_max}"
    if b_max > max_abs_value:
        return False, f"Matrix B has values too large: max abs value = {b_max}"
    
    # Check for numerical stability issues
    a_range = A.max() - A.min()
    b_range = B.max() - B.min()
    
    if a_range > 0 and a_range < 1e-10:
        logger.warning(f"Matrix A has very small range {a_range}, may cause numerical issues")
    if b_range > 0 and b_range < 1e-10:
        logger.warning(f"Matrix B has very small range {b_range}, may cause numerical issues")
    
    return True, None

def sanitize_strategy(
    strategy: np.ndarray,
    tolerance: float = 1e-9
) -> np.ndarray:
    """
    Sanitize a strategy vector to ensure it's a valid probability distribution.
    
    Args:
        strategy: Strategy vector to sanitize
        tolerance: Values below this are set to 0
    
    Returns:
        Sanitized strategy vector
    """
    # Set very small values to 0
    strategy[strategy < tolerance] = 0
    
    # Normalize to sum to 1
    total = strategy.sum()
    if total > 0:
        strategy = strategy / total
    else:
        # Uniform distribution if all zeros
        strategy = np.ones_like(strategy) / len(strategy)
    
    return strategy

def detect_degenerate_game(
    A: np.ndarray,
    B: np.ndarray,
    tolerance: float = 1e-9
) -> Tuple[bool, str]:
    """
    Detect if a game is degenerate or has special structure.
    
    Args:
        A, B: Payoff matrices
        tolerance: Numerical tolerance
    
    Returns:
        Tuple of (is_degenerate, description)
    """
    # Check for constant games
    if np.allclose(A, A[0, 0], atol=tolerance):
        return True, "Matrix A is constant"
    if np.allclose(B, B[0, 0], atol=tolerance):
        return True, "Matrix B is constant"
    
    # Check for zero-sum
    if np.allclose(A + B, 0, atol=tolerance):
        return False, "Game is zero-sum"
    
    # Check for dominance
    n_rows, n_cols = A.shape
    
    # Row dominance in A
    for i in range(n_rows):
        for j in range(n_rows):
            if i != j and np.all(A[i, :] >= A[j, :] + tolerance):
                if np.any(A[i, :] > A[j, :] + tolerance):
                    return True, f"Row {i} dominates row {j} in matrix A"
    
    # Column dominance in B
    for i in range(n_cols):
        for j in range(n_cols):
            if i != j and np.all(B[:, i] >= B[:, j] + tolerance):
                if np.any(B[:, i] > B[:, j] + tolerance):
                    return True, f"Column {i} dominates column {j} in matrix B"
    
    return False, "No degeneracy detected"

class GameValidator:
    """Comprehensive game validation with caching and performance optimization."""
    
    def __init__(self, tolerance: float = 1e-9, max_abs_value: float = 1e6):
        self.tolerance = tolerance
        self.max_abs_value = max_abs_value
        self._validation_cache = {}
    
    def validate_game(
        self,
        A: np.ndarray,
        B: np.ndarray,
        check_degeneracy: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive game validation.
        
        Args:
            A, B: Payoff matrices
            check_degeneracy: Whether to check for game degeneracy
        
        Returns:
            Dictionary with validation results
        """
        # Create cache key
        cache_key = (A.tobytes(), B.tobytes(), check_degeneracy)
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'properties': {}
        }
        
        # Basic shape validation
        if A.shape != B.shape:
            results['valid'] = False
            results['errors'].append(f"Shape mismatch: A={A.shape}, B={B.shape}")
            return results
        
        # Range validation
        range_valid, range_error = validate_payoff_range(A, B, self.max_abs_value)
        if not range_valid:
            results['valid'] = False
            results['errors'].append(range_error)
        
        # Degeneracy check
        if check_degeneracy:
            is_degenerate, desc = detect_degenerate_game(A, B, self.tolerance)
            results['properties']['degenerate'] = is_degenerate
            results['properties']['degeneracy_desc'] = desc
            if is_degenerate:
                results['warnings'].append(f"Game degeneracy detected: {desc}")
        
        # Game properties
        results['properties']['shape'] = A.shape
        results['properties']['zero_sum'] = np.allclose(A + B, 0, atol=self.tolerance)
        results['properties']['symmetric'] = (
            A.shape[0] == A.shape[1] and 
            np.allclose(A, B.T, atol=self.tolerance)
        )
        
        # Cache results
        self._validation_cache[cache_key] = results
        
        return results