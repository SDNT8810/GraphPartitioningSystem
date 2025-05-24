"""
Autonomous Node Agent Implementation
Self-Partitioning Graphs for Industrial Data Management

This implements the lightweight decision-making agents embedded within each node
as described in the research framework.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
import logging

class DecisionState(Enum):
    """Node decision states for autonomous behavior"""
    STABLE = "stable"
    EXPLORING = "exploring"
    MIGRATING = "migrating"
    RECOVERING = "recovering"
    COOPERATING = "cooperating"

@dataclass
class NodeIntelligence:
    """Intelligence metrics for autonomous decision making"""
    load_capacity: float
    communication_cost: float
    failure_probability: float
    cooperation_score: float
    adaptation_rate: float
    experience_level: int

class AutonomousNodeAgent(nn.Module):
    """
    Lightweight decision-making agent embedded within each graph node.
    
    Implements:
    - Reinforcement learning for adaptive behavior
    - Game theory for cooperation
    - Real-time decision making
    - Resource-constrained operation
    """
    
    def __init__(self, node_id: int, initial_capacity: float = 1.0):
        super().__init__()
        self.node_id = node_id
        self.state = DecisionState.STABLE
        self.intelligence = NodeIntelligence(
            load_capacity=initial_capacity,
            communication_cost=0.1,
            failure_probability=0.01,
            cooperation_score=1.0,
            adaptation_rate=0.1,
            experience_level=0
        )
        
        # Lightweight neural network for decision making
        self.decision_network = nn.Sequential(
            nn.Linear(12, 32),  # State features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)    # Decision outputs
        )
        
        # Experience buffer for learning
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
        # Cooperation memory
        self.cooperation_history = {}
        self.trust_scores = {}
        
        # Performance tracking
        self.performance_history = []
        self.decision_count = 0
        
        self.logger = logging.getLogger(f"NodeAgent-{node_id}")
        
    def get_state_vector(self, graph_state: Dict, neighbors: List[int]) -> torch.Tensor:
        """Extract current state features for decision making"""
        features = []
        
        # Node-specific features
        features.extend([
            self.intelligence.load_capacity,
            self.intelligence.communication_cost,
            self.intelligence.failure_probability,
            self.intelligence.cooperation_score,
            self.intelligence.adaptation_rate,
            float(self.intelligence.experience_level) / 1000.0  # Normalized
        ])
        
        # Graph context features
        features.extend([
            len(neighbors),  # Degree
            graph_state.get('global_load', 0.5),
            graph_state.get('network_congestion', 0.1),
            graph_state.get('failure_rate', 0.01),
            float(self.state.value == 'stable'),  # Current stability
            time.time() % 86400 / 86400.0  # Time of day normalized
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def make_autonomous_decision(self, graph_state: Dict, neighbors: List[int]) -> Dict:
        """
        Make autonomous decision based on current state and intelligence.
        
        Returns decision with action type and parameters.
        """
        state_vector = self.get_state_vector(graph_state, neighbors)
        
        with torch.no_grad():
            decision_logits = self.decision_network(state_vector)
            decision_probs = torch.softmax(decision_logits, dim=0)
        
        # Decision mapping
        decisions = ['stay', 'migrate', 'cooperate', 'replicate', 'optimize']
        action = decisions[torch.argmax(decision_probs).item()]
        confidence = torch.max(decision_probs).item()
        
        # Generate detailed decision with reasoning
        decision = {
            'action': action,
            'confidence': confidence,
            'reasoning': self._generate_reasoning(action, state_vector),
            'parameters': self._get_action_parameters(action, graph_state, neighbors),
            'timestamp': time.time(),
            'node_id': self.node_id,
            'state': self.state.value
        }
        
        # Update experience and learning
        self._update_experience(state_vector, decision)
        self.decision_count += 1
        
        self.logger.info(f"Node {self.node_id} decided: {action} (confidence: {confidence:.3f})")
        
        return decision
    
    def _generate_reasoning(self, action: str, state_vector: torch.Tensor) -> str:
        """Generate human-readable reasoning for the decision"""
        load = state_vector[0].item()
        comm_cost = state_vector[1].item()
        failure_prob = state_vector[2].item()
        
        if action == 'migrate':
            return f"High load ({load:.2f}) and communication cost ({comm_cost:.2f}) suggest migration"
        elif action == 'cooperate':
            return f"Cooperation beneficial with current trust scores and network state"
        elif action == 'replicate':
            return f"Failure probability ({failure_prob:.3f}) indicates need for replication"
        elif action == 'optimize':
            return f"Stable conditions allow for optimization improvements"
        else:
            return f"Current state optimal, maintaining position"
    
    def _get_action_parameters(self, action: str, graph_state: Dict, neighbors: List[int]) -> Dict:
        """Get specific parameters for the chosen action"""
        if action == 'migrate':
            # Find best migration target based on cooperation scores
            best_target = max(neighbors, key=lambda n: self.trust_scores.get(n, 0.5))
            return {
                'target_node': best_target,
                'migration_weight': min(0.3, self.intelligence.load_capacity * 0.5),
                'expected_benefit': self._calculate_migration_benefit(best_target)
            }
        
        elif action == 'cooperate':
            # Select cooperation partners
            partners = [n for n in neighbors if self.trust_scores.get(n, 0.5) > 0.7]
            return {
                'partners': partners[:3],  # Max 3 partners
                'cooperation_type': 'load_sharing',
                'resource_commitment': 0.2
            }
        
        elif action == 'replicate':
            return {
                'replication_factor': 2,
                'backup_nodes': neighbors[:2],
                'sync_frequency': 'high'
            }
        
        elif action == 'optimize':
            return {
                'optimization_type': 'local_structure',
                'intensity': 'moderate',
                'duration': 10.0
            }
        
        return {}
    
    def update_cooperation_score(self, neighbor_id: int, interaction_result: float):
        """Update trust and cooperation scores based on interactions"""
        if neighbor_id not in self.trust_scores:
            self.trust_scores[neighbor_id] = 0.5
        
        # Exponential moving average for trust update
        alpha = 0.1
        self.trust_scores[neighbor_id] = (
            alpha * interaction_result + 
            (1 - alpha) * self.trust_scores[neighbor_id]
        )
        
        # Update cooperation history
        if neighbor_id not in self.cooperation_history:
            self.cooperation_history[neighbor_id] = []
        
        self.cooperation_history[neighbor_id].append({
            'timestamp': time.time(),
            'result': interaction_result,
            'trust_level': self.trust_scores[neighbor_id]
        })
        
        # Keep only recent history
        if len(self.cooperation_history[neighbor_id]) > 100:
            self.cooperation_history[neighbor_id] = self.cooperation_history[neighbor_id][-100:]
    
    def adapt_intelligence(self, performance_feedback: Dict):
        """Adapt intelligence parameters based on performance feedback"""
        # Update capacity based on actual performance
        actual_load = performance_feedback.get('actual_load', 0.5)
        target_load = performance_feedback.get('target_load', 0.7)
        
        if actual_load > target_load * 1.1:
            self.intelligence.load_capacity *= 0.95  # Reduce capacity estimate
        elif actual_load < target_load * 0.9:
            self.intelligence.load_capacity *= 1.05  # Increase capacity estimate
        
        # Update communication cost based on measured latency
        measured_latency = performance_feedback.get('avg_latency', 0.1)
        self.intelligence.communication_cost = 0.9 * self.intelligence.communication_cost + 0.1 * measured_latency
        
        # Update failure probability based on recent failures
        failure_rate = performance_feedback.get('failure_rate', 0.01)
        self.intelligence.failure_probability = 0.8 * self.intelligence.failure_probability + 0.2 * failure_rate
        
        # Increase experience level
        self.intelligence.experience_level += 1
        
        # Update adaptation rate based on environment volatility
        volatility = performance_feedback.get('volatility', 0.1)
        self.intelligence.adaptation_rate = np.clip(0.05 + volatility * 0.5, 0.01, 0.3)
        
        self.logger.debug(f"Node {self.node_id} intelligence adapted: "
                         f"capacity={self.intelligence.load_capacity:.3f}, "
                         f"comm_cost={self.intelligence.communication_cost:.3f}")
    
    def _calculate_migration_benefit(self, target_node: int) -> float:
        """Calculate expected benefit of migrating to target node"""
        trust = self.trust_scores.get(target_node, 0.5)
        cooperation_history_len = len(self.cooperation_history.get(target_node, []))
        
        # Benefit based on trust and cooperation history
        benefit = trust * 0.7 + min(cooperation_history_len / 50.0, 0.3)
        
        return benefit
    
    def _update_experience(self, state_vector: torch.Tensor, decision: Dict):
        """Update experience buffer for learning"""
        experience = {
            'state': state_vector.clone(),
            'decision': decision['action'],
            'confidence': decision['confidence'],
            'timestamp': decision['timestamp']
        }
        
        self.experience_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary for this agent"""
        return {
            'node_id': self.node_id,
            'decisions_made': self.decision_count,
            'current_state': self.state.value,
            'intelligence': {
                'load_capacity': self.intelligence.load_capacity,
                'communication_cost': self.intelligence.communication_cost,
                'failure_probability': self.intelligence.failure_probability,
                'cooperation_score': self.intelligence.cooperation_score,
                'experience_level': self.intelligence.experience_level
            },
            'trust_network_size': len(self.trust_scores),
            'avg_trust_score': np.mean(list(self.trust_scores.values())) if self.trust_scores else 0.0,
            'cooperation_partners': len([t for t in self.trust_scores.values() if t > 0.7]),
            'experience_buffer_size': len(self.experience_buffer)
        }
    
    def enter_recovery_mode(self, failure_type: str):
        """Enter recovery mode when failure is detected"""
        self.state = DecisionState.RECOVERING
        self.logger.warning(f"Node {self.node_id} entering recovery mode: {failure_type}")
        
        # Adjust intelligence based on failure
        self.intelligence.failure_probability = min(0.5, self.intelligence.failure_probability * 1.5)
        self.intelligence.cooperation_score *= 0.9  # Reduce cooperation temporarily
    
    def exit_recovery_mode(self):
        """Exit recovery mode and return to normal operation"""
        self.state = DecisionState.STABLE
        self.intelligence.cooperation_score = min(1.0, self.intelligence.cooperation_score * 1.1)
        self.logger.info(f"Node {self.node_id} recovered, returning to stable operation")
