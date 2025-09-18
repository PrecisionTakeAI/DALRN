"""
M/M/1 Queue Model for SOAN Node Simulation.

This module implements an M/M/1 queueing system with stability detection,
full PoDP compliance, and ε-ledger budget tracking.
"""

import json
import hashlib
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque
import heapq

# Import PoDP utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.podp import Receipt, ReceiptChain, keccak


@dataclass
class QueueMetrics:
    """Metrics for queue performance analysis."""
    average_queue_length: float
    average_wait_time: float
    average_system_time: float
    utilization: float
    throughput: float
    loss_probability: float
    is_stable: bool
    epsilon_used: float


@dataclass
class CustomerEvent:
    """Event in the queueing system."""
    event_type: str  # 'arrival' or 'departure'
    time: float
    customer_id: int


class MM1Queue:
    """
    M/M/1 Queue implementation with Poisson arrivals and exponential service times.

    Parameters:
    - arrival_rate (λ): Average arrival rate (customers per time unit)
    - service_rate (μ): Average service rate (customers per time unit)
    - max_queue_length: Maximum queue capacity (None for infinite)
    """

    # ε-ledger budget allocations
    EPSILON_QUEUE_INIT = 0.0002
    EPSILON_SIMULATION_STEP = 0.0001
    EPSILON_METRICS_CALC = 0.0003

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        max_queue_length: Optional[int] = None
    ):
        """Initialize M/M/1 queue with specified parameters."""
        if arrival_rate <= 0:
            raise ValueError(f"Arrival rate must be positive, got {arrival_rate}")
        if service_rate <= 0:
            raise ValueError(f"Service rate must be positive, got {service_rate}")

        # Ensure service rate is in valid range [1.0, 2.0]
        if not 1.0 <= service_rate <= 2.0:
            service_rate = np.clip(service_rate, 1.0, 2.0)

        self.arrival_rate = arrival_rate  # λ
        self.service_rate = service_rate  # μ
        self.max_queue_length = max_queue_length

        # Queue state
        self.queue = deque()
        self.current_time = 0.0
        self.customers_served = 0
        self.customers_lost = 0
        self.total_wait_time = 0.0
        self.total_system_time = 0.0

        # Event management
        self.event_heap = []
        self.next_customer_id = 0

        # Tracking
        self.queue_length_samples = []
        self.wait_times = []
        self.system_times = []

        # PoDP tracking
        self.receipt_chain = ReceiptChain(
            dispute_id=f"mm1_queue_{int(time.time())}"
        )
        self.epsilon_spent = 0.0

        # Stability check
        self.rho = arrival_rate / service_rate  # Traffic intensity
        self.is_stable = self.rho < 1.0

        # Create initialization receipt
        self._create_init_receipt()

    def _create_init_receipt(self):
        """Create PoDP receipt for queue initialization."""
        receipt = Receipt(
            receipt_id=f"queue_init_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="queue_initialization",
            inputs={
                "arrival_rate": self.arrival_rate,
                "service_rate": self.service_rate,
                "max_queue_length": self.max_queue_length
            },
            params={
                "queue_type": "M/M/1",
                "traffic_intensity": self.rho,
                "stability": self.is_stable
            },
            artifacts={
                "theoretical_avg_queue_length": self._theoretical_queue_length() if self.is_stable else "infinite",
                "theoretical_avg_wait_time": self._theoretical_wait_time() if self.is_stable else "infinite",
                "epsilon_used": self.EPSILON_QUEUE_INIT
            },
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        self.receipt_chain.add_receipt(receipt)
        self.epsilon_spent += self.EPSILON_QUEUE_INIT

    def _theoretical_queue_length(self) -> float:
        """Calculate theoretical average queue length (L) for stable M/M/1."""
        if not self.is_stable:
            return float('inf')
        return self.rho / (1 - self.rho)

    def _theoretical_wait_time(self) -> float:
        """Calculate theoretical average wait time (W) for stable M/M/1."""
        if not self.is_stable:
            return float('inf')
        return self.rho / (self.service_rate * (1 - self.rho))

    def _theoretical_system_time(self) -> float:
        """Calculate theoretical average system time (W_s) for stable M/M/1."""
        if not self.is_stable:
            return float('inf')
        return 1 / (self.service_rate - self.arrival_rate)

    def generate_arrival_time(self) -> float:
        """Generate next arrival time using exponential distribution."""
        return np.random.exponential(1.0 / self.arrival_rate)

    def generate_service_time(self) -> float:
        """Generate service time using exponential distribution."""
        return np.random.exponential(1.0 / self.service_rate)

    def simulate(
        self,
        duration: float,
        warm_up: float = 0.1
    ) -> Tuple[QueueMetrics, Receipt]:
        """
        Run discrete event simulation of the queue with PoDP receipt.

        Args:
            duration: Simulation duration in time units
            warm_up: Warm-up period fraction (default 0.1 = 10%)

        Returns:
            Tuple of (QueueMetrics, PoDP receipt)
        """
        start_time = time.time()

        # Create entry receipt
        entry_receipt = Receipt(
            receipt_id=f"sim_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="simulation_entry",
            inputs={
                "duration": duration,
                "warm_up": warm_up
            },
            params={
                "arrival_rate": self.arrival_rate,
                "service_rate": self.service_rate
            },
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Reset simulation state
        self._reset()

        # Schedule first arrival
        first_arrival_time = self.generate_arrival_time()
        heapq.heappush(
            self.event_heap,
            (first_arrival_time, CustomerEvent('arrival', first_arrival_time, self.next_customer_id))
        )
        self.next_customer_id += 1

        warm_up_time = duration * warm_up
        events_processed = 0

        # Main simulation loop
        while self.current_time < duration:
            if not self.event_heap:
                # Schedule next arrival if no events
                next_arrival = self.current_time + self.generate_arrival_time()
                heapq.heappush(
                    self.event_heap,
                    (next_arrival, CustomerEvent('arrival', next_arrival, self.next_customer_id))
                )
                self.next_customer_id += 1

            # Process next event
            event_time, event = heapq.heappop(self.event_heap)
            self.current_time = event_time

            if self.current_time >= duration:
                break

            if event.event_type == 'arrival':
                self._process_arrival(event)
            else:  # departure
                self._process_departure(event)

            # Collect statistics after warm-up
            if self.current_time > warm_up_time:
                self.queue_length_samples.append(len(self.queue))

            events_processed += 1
            self.epsilon_spent += self.EPSILON_SIMULATION_STEP

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Create exit receipt
        simulation_time = (time.time() - start_time) * 1000
        exit_receipt = Receipt(
            receipt_id=f"sim_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="simulation_exit",
            inputs={
                "entry_receipt_id": entry_receipt.receipt_id
            },
            params={
                "events_processed": events_processed,
                "simulation_time_ms": simulation_time
            },
            artifacts={
                "metrics": asdict(metrics),
                "customers_served": self.customers_served,
                "customers_lost": self.customers_lost,
                "final_queue_length": len(self.queue),
                "epsilon_used": self.EPSILON_SIMULATION_STEP * events_processed,
                "epsilon_total": self.epsilon_spent
            },
            hashes={
                "metrics_hash": keccak(json.dumps(asdict(metrics), sort_keys=True))
            },
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Add to receipt chain
        self.receipt_chain.add_receipt(entry_receipt)
        self.receipt_chain.add_receipt(exit_receipt)

        return metrics, exit_receipt

    def _process_arrival(self, event: CustomerEvent):
        """Process customer arrival event."""
        # Check if queue is full
        if self.max_queue_length and len(self.queue) >= self.max_queue_length:
            self.customers_lost += 1
        else:
            # Add customer to queue with arrival time
            customer_data = {
                'id': event.customer_id,
                'arrival_time': event.time
            }

            # If server is idle, start service immediately
            if len(self.queue) == 0:
                # Schedule departure
                service_time = self.generate_service_time()
                departure_time = event.time + service_time
                heapq.heappush(
                    self.event_heap,
                    (departure_time, CustomerEvent('departure', departure_time, event.customer_id))
                )
                customer_data['service_start_time'] = event.time
                self.wait_times.append(0)  # No wait time
            else:
                customer_data['service_start_time'] = None

            self.queue.append(customer_data)

        # Schedule next arrival
        next_arrival_time = event.time + self.generate_arrival_time()
        heapq.heappush(
            self.event_heap,
            (next_arrival_time, CustomerEvent('arrival', next_arrival_time, self.next_customer_id))
        )
        self.next_customer_id += 1

    def _process_departure(self, event: CustomerEvent):
        """Process customer departure event."""
        if not self.queue:
            return  # No customer to depart

        # Remove departing customer
        customer = self.queue.popleft()
        self.customers_served += 1

        # Calculate system time
        system_time = event.time - customer['arrival_time']
        self.system_times.append(system_time)
        self.total_system_time += system_time

        # Start service for next customer if any
        if self.queue:
            next_customer = self.queue[0]
            if next_customer['service_start_time'] is None:
                next_customer['service_start_time'] = event.time
                wait_time = event.time - next_customer['arrival_time']
                self.wait_times.append(wait_time)
                self.total_wait_time += wait_time

                # Schedule their departure
                service_time = self.generate_service_time()
                departure_time = event.time + service_time
                heapq.heappush(
                    self.event_heap,
                    (departure_time, CustomerEvent('departure', departure_time, next_customer['id']))
                )

    def _calculate_metrics(self) -> QueueMetrics:
        """Calculate queue performance metrics."""
        # Average queue length
        avg_queue_length = np.mean(self.queue_length_samples) if self.queue_length_samples else 0

        # Average wait time
        avg_wait_time = np.mean(self.wait_times) if self.wait_times else 0

        # Average system time
        avg_system_time = np.mean(self.system_times) if self.system_times else 0

        # Utilization (rho)
        utilization = min(self.rho, 1.0)

        # Throughput
        throughput = self.customers_served / self.current_time if self.current_time > 0 else 0

        # Loss probability
        total_arrivals = self.customers_served + self.customers_lost
        loss_probability = self.customers_lost / total_arrivals if total_arrivals > 0 else 0

        metrics = QueueMetrics(
            average_queue_length=avg_queue_length,
            average_wait_time=avg_wait_time,
            average_system_time=avg_system_time,
            utilization=utilization,
            throughput=throughput,
            loss_probability=loss_probability,
            is_stable=self.is_stable,
            epsilon_used=self.EPSILON_METRICS_CALC
        )

        self.epsilon_spent += self.EPSILON_METRICS_CALC

        return metrics

    def _reset(self):
        """Reset simulation state."""
        self.queue.clear()
        self.current_time = 0.0
        self.customers_served = 0
        self.customers_lost = 0
        self.total_wait_time = 0.0
        self.total_system_time = 0.0
        self.event_heap.clear()
        self.queue_length_samples.clear()
        self.wait_times.clear()
        self.system_times.clear()

    def check_stability(self) -> Tuple[bool, Dict[str, Any], Receipt]:
        """
        Check queue stability with theoretical analysis and PoDP receipt.

        Returns:
            Tuple of (is_stable, stability_metrics, PoDP receipt)
        """
        # Create stability check receipt
        receipt = Receipt(
            receipt_id=f"stability_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            dispute_id=self.receipt_chain.dispute_id,
            step="stability_check",
            inputs={
                "arrival_rate": self.arrival_rate,
                "service_rate": self.service_rate
            },
            params={
                "method": "traffic_intensity"
            },
            artifacts={},
            hashes={},
            signatures=[],
            ts=datetime.utcnow().isoformat()
        )

        # Calculate stability metrics
        stability_metrics = {
            "is_stable": self.is_stable,
            "traffic_intensity": self.rho,
            "stability_condition": "λ < μ",
            "condition_met": self.arrival_rate < self.service_rate,
            "margin": self.service_rate - self.arrival_rate if self.is_stable else 0
        }

        if self.is_stable:
            stability_metrics.update({
                "theoretical_avg_queue_length": self._theoretical_queue_length(),
                "theoretical_avg_wait_time": self._theoretical_wait_time(),
                "theoretical_avg_system_time": self._theoretical_system_time()
            })
        else:
            stability_metrics.update({
                "warning": "Queue is unstable - will grow unbounded",
                "recommendation": f"Increase service rate to > {self.arrival_rate} or decrease arrival rate"
            })

        receipt.artifacts = stability_metrics
        receipt.hashes["metrics_hash"] = keccak(json.dumps(stability_metrics, sort_keys=True))

        self.receipt_chain.add_receipt(receipt)

        return self.is_stable, stability_metrics, receipt

    def get_current_state(self) -> Dict[str, Any]:
        """Get current queue state."""
        return {
            "queue_length": len(self.queue),
            "current_time": self.current_time,
            "customers_served": self.customers_served,
            "customers_lost": self.customers_lost,
            "is_stable": self.is_stable,
            "traffic_intensity": self.rho,
            "epsilon_spent": self.epsilon_spent
        }

    def get_receipt_chain(self) -> ReceiptChain:
        """Get the complete receipt chain for all operations."""
        return self.receipt_chain

    def get_epsilon_budget_status(self) -> Dict[str, float]:
        """Get current epsilon budget status."""
        return {
            "spent": self.epsilon_spent,
            "remaining": 4.0 - self.epsilon_spent,  # Total budget of 4.0
            "breakdown": {
                "initialization": self.EPSILON_QUEUE_INIT,
                "simulation": self.EPSILON_SIMULATION_STEP * len([r for r in self.receipt_chain.receipts if "sim" in r.step]),
                "metrics": self.EPSILON_METRICS_CALC * len([r for r in self.receipt_chain.receipts if "metrics" in r.step])
            }
        }