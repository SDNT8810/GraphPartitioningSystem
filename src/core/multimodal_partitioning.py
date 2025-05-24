"""
Multi-Modal Partitioning Framework
Self-Partitioning Graphs for Industrial Data Management

Implements the hybrid partitioning system that dynamically combines multiple 
strategies based on real-time conditions as described in Proposed_Method.
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging
from abc import ABC, abstractmethod

class PartitioningStrategy(Enum):
    """Available partitioning strategies"""
    GRAPH_STRUCTURAL = "graph_structural"
    WORKLOAD_AWARE = "workload_aware"
    DATA_LOCALITY = "data_locality"
    TEMPORAL_PATTERN = "temporal_pattern"
    HYBRID_ADAPTIVE = "hybrid_adaptive"

@dataclass
class SystemConditions:
    """Real-time system state for strategy selection"""
    network_latency: float
    cpu_utilization: float
    memory_usage: float
    io_load: float
    failure_rate: float
    data_stream_velocity: float
    temporal_variability: float
    communication_cost: float

@dataclass
class PartitioningObjectives:
    """Multi-objective optimization targets"""
    communication_cost_weight: float = 0.3
    load_balance_weight: float = 0.3
    response_time_weight: float = 0.2
    fault_tolerance_weight: float = 0.2

class PartitioningStrategyInterface(ABC):
    """Interface for individual partitioning strategies"""
    
    @abstractmethod
    def evaluate_suitability(self, conditions: SystemConditions, graph: nx.Graph) -> float:
        """Evaluate how suitable this strategy is for current conditions"""
        pass
    
    @abstractmethod
    def execute_partitioning(self, graph: nx.Graph, num_partitions: int, **kwargs) -> Dict:
        """Execute the partitioning strategy"""
        pass
    
    @abstractmethod
    def get_transition_cost(self, current_partition: Dict) -> float:
        """Calculate cost of transitioning to this strategy"""
        pass

class GraphStructuralStrategy(PartitioningStrategyInterface):
    """Graph-based structural optimization strategy"""
    
    def __init__(self):
        self.name = "GraphStructural"
        self.logger = logging.getLogger(f"Strategy-{self.name}")
    
    def evaluate_suitability(self, conditions: SystemConditions, graph: nx.Graph) -> float:
        """Evaluate suitability based on graph structure characteristics"""
        # Better for stable, well-connected graphs
        density = nx.density(graph)
        clustering = nx.average_clustering(graph)
        
        # Prefer when network is stable and graph structure is important
        stability_factor = 1.0 - conditions.temporal_variability
        structure_factor = (density + clustering) / 2.0
        
        suitability = 0.4 * stability_factor + 0.6 * structure_factor
        
        # Penalize if high communication costs (structure-based needs communication)
        if conditions.communication_cost > 0.5:
            suitability *= 0.7
        
        return np.clip(suitability, 0.0, 1.0)
    
    def execute_partitioning(self, graph: nx.Graph, num_partitions: int, **kwargs) -> Dict:
        """Execute graph structural partitioning using spectral clustering"""
        try:
            # Use spectral clustering for structural partitioning
            laplacian = nx.normalized_laplacian_matrix(graph).toarray()
            eigenvals, eigenvecs = np.linalg.eigh(laplacian)
            
            # Use Fiedler vector and subsequent eigenvectors for partitioning
            partition_vectors = eigenvecs[:, 1:num_partitions]
            
            # K-means clustering on eigenvectors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_partitions, random_state=42)
            partition_labels = kmeans.fit_predict(partition_vectors)
            
            # Create partition mapping
            partition_map = {}
            for node, label in enumerate(partition_labels):
                partition_map[node] = int(label)
            
            # Calculate metrics
            cut_size = self._calculate_cut_size(graph, partition_map)
            balance = self._calculate_balance(partition_map, num_partitions)
            
            result = {
                'partition_map': partition_map,
                'strategy': self.name,
                'cut_size': cut_size,
                'balance': balance,
                'execution_time': time.time(),
                'quality_score': (1.0 - cut_size / graph.number_of_edges()) * balance
            }
            
            self.logger.info(f"Structural partitioning: cut={cut_size}, balance={balance:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Structural partitioning failed: {e}")
            return self._fallback_partition(graph, num_partitions)
    
    def get_transition_cost(self, current_partition: Dict) -> float:
        """Low transition cost for structural changes"""
        return 0.2
    
    def _calculate_cut_size(self, graph: nx.Graph, partition_map: Dict) -> int:
        """Calculate cut size for partition"""
        cut_size = 0
        for u, v in graph.edges():
            if partition_map[u] != partition_map[v]:
                cut_size += 1
        return cut_size
    
    def _calculate_balance(self, partition_map: Dict, num_partitions: int) -> float:
        """Calculate partition balance"""
        partition_sizes = [0] * num_partitions
        for partition in partition_map.values():
            partition_sizes[partition] += 1
        
        max_size = max(partition_sizes)
        min_size = min(partition_sizes)
        
        if max_size == 0:
            return 0.0
        
        return min_size / max_size
    
    def _fallback_partition(self, graph: nx.Graph, num_partitions: int) -> Dict:
        """Simple fallback partitioning"""
        nodes = list(graph.nodes())
        partition_size = len(nodes) // num_partitions
        
        partition_map = {}
        for i, node in enumerate(nodes):
            partition_map[node] = min(i // partition_size, num_partitions - 1)
        
        return {
            'partition_map': partition_map,
            'strategy': f"{self.name}-Fallback",
            'cut_size': self._calculate_cut_size(graph, partition_map),
            'balance': self._calculate_balance(partition_map, num_partitions),
            'execution_time': time.time(),
            'quality_score': 0.5
        }

class WorkloadAwareStrategy(PartitioningStrategyInterface):
    """Workload-aware partitioning strategy"""
    
    def __init__(self):
        self.name = "WorkloadAware"
        self.logger = logging.getLogger(f"Strategy-{self.name}")
    
    def evaluate_suitability(self, conditions: SystemConditions, graph: nx.Graph) -> float:
        """Evaluate suitability based on workload characteristics"""
        # Better for high CPU/memory utilization scenarios
        workload_factor = (conditions.cpu_utilization + conditions.memory_usage) / 2.0
        
        # Consider I/O load and data stream velocity
        io_factor = min(conditions.io_load * 2.0, 1.0)
        velocity_factor = min(conditions.data_stream_velocity * 1.5, 1.0)
        
        suitability = 0.4 * workload_factor + 0.3 * io_factor + 0.3 * velocity_factor
        
        # Bonus if temporal variability is high (workload changes frequently)
        if conditions.temporal_variability > 0.6:
            suitability *= 1.2
        
        return np.clip(suitability, 0.0, 1.0)
    
    def execute_partitioning(self, graph: nx.Graph, num_partitions: int, **kwargs) -> Dict:
        """Execute workload-aware partitioning"""
        try:
            # Get node workload data (simulated if not provided)
            node_workloads = kwargs.get('node_workloads', self._estimate_node_workloads(graph))
            
            # Sort nodes by workload
            sorted_nodes = sorted(node_workloads.items(), key=lambda x: x[1], reverse=True)
            
            # Distribute nodes using round-robin for balance
            partition_map = {}
            partition_loads = [0.0] * num_partitions
            
            for node, workload in sorted_nodes:
                # Assign to partition with lowest current load
                target_partition = np.argmin(partition_loads)
                partition_map[node] = target_partition
                partition_loads[target_partition] += workload
            
            # Calculate metrics
            cut_size = self._calculate_cut_size(graph, partition_map)
            load_balance = self._calculate_load_balance(partition_loads)
            
            result = {
                'partition_map': partition_map,
                'strategy': self.name,
                'cut_size': cut_size,
                'load_balance': load_balance,
                'partition_loads': partition_loads,
                'execution_time': time.time(),
                'quality_score': load_balance * (1.0 - cut_size / graph.number_of_edges())
            }
            
            self.logger.info(f"Workload-aware partitioning: cut={cut_size}, load_balance={load_balance:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Workload-aware partitioning failed: {e}")
            return self._fallback_partition(graph, num_partitions)
    
    def get_transition_cost(self, current_partition: Dict) -> float:
        """Medium transition cost for workload redistribution"""
        return 0.4
    
    def _estimate_node_workloads(self, graph: nx.Graph) -> Dict[int, float]:
        """Estimate node workloads based on graph properties"""
        workloads = {}
        for node in graph.nodes():
            # Estimate based on degree and centrality
            degree = graph.degree(node)
            try:
                centrality = nx.betweenness_centrality(graph)[node]
            except:
                centrality = 0.1
            
            # Combine degree and centrality with some randomness
            workload = 0.6 * (degree / graph.number_of_nodes()) + 0.4 * centrality
            workload += np.random.normal(0, 0.1)  # Add variability
            workloads[node] = max(0.1, workload)
        
        return workloads
    
    def _calculate_cut_size(self, graph: nx.Graph, partition_map: Dict) -> int:
        """Calculate cut size for partition"""
        cut_size = 0
        for u, v in graph.edges():
            if partition_map[u] != partition_map[v]:
                cut_size += 1
        return cut_size
    
    def _calculate_load_balance(self, partition_loads: List[float]) -> float:
        """Calculate load balance across partitions"""
        if max(partition_loads) == 0:
            return 1.0
        
        return min(partition_loads) / max(partition_loads)
    
    def _fallback_partition(self, graph: nx.Graph, num_partitions: int) -> Dict:
        """Simple fallback partitioning"""
        nodes = list(graph.nodes())
        partition_size = len(nodes) // num_partitions
        
        partition_map = {}
        for i, node in enumerate(nodes):
            partition_map[node] = min(i // partition_size, num_partitions - 1)
        
        return {
            'partition_map': partition_map,
            'strategy': f"{self.name}-Fallback",
            'cut_size': self._calculate_cut_size(graph, partition_map),
            'load_balance': 0.5,
            'execution_time': time.time(),
            'quality_score': 0.5
        }

class MultiModalPartitioningFramework:
    """
    Multi-modal partitioning framework that dynamically combines multiple 
    partitioning approaches for optimal performance under varying conditions.
    """
    
    def __init__(self, objectives: PartitioningObjectives = None):
        self.objectives = objectives or PartitioningObjectives()
        self.strategies = {
            PartitioningStrategy.GRAPH_STRUCTURAL: GraphStructuralStrategy(),
            PartitioningStrategy.WORKLOAD_AWARE: WorkloadAwareStrategy(),
        }
        
        self.current_strategy = None
        self.current_partition = None
        self.strategy_history = []
        self.performance_history = []
        
        self.logger = logging.getLogger("MultiModalFramework")
        self.logger.info("Multi-modal partitioning framework initialized")
    
    def select_optimal_strategy(self, conditions: SystemConditions, graph: nx.Graph) -> PartitioningStrategy:
        """
        Dynamically select optimal partitioning strategy based on real-time conditions.
        """
        strategy_scores = {}
        
        for strategy_type, strategy in self.strategies.items():
            # Base suitability score
            suitability = strategy.evaluate_suitability(conditions, graph)
            
            # Consider transition cost if we have a current strategy
            transition_cost = 0.0
            if self.current_strategy and self.current_partition:
                transition_cost = strategy.get_transition_cost(self.current_partition)
            
            # Stability bonus for keeping current strategy (avoid thrashing)
            stability_bonus = 0.0
            if strategy_type == self.current_strategy:
                stability_bonus = 0.1
            
            # Final score considering all factors
            final_score = suitability - (0.2 * transition_cost) + stability_bonus
            strategy_scores[strategy_type] = final_score
            
            self.logger.debug(f"Strategy {strategy_type.value}: "
                            f"suitability={suitability:.3f}, "
                            f"transition_cost={transition_cost:.3f}, "
                            f"final_score={final_score:.3f}")
        
        # Select best strategy
        optimal_strategy = max(strategy_scores, key=strategy_scores.get)
        
        self.logger.info(f"Selected strategy: {optimal_strategy.value} "
                        f"(score: {strategy_scores[optimal_strategy]:.3f})")
        
        return optimal_strategy
    
    def execute_intelligent_partitioning(self, 
                                       graph: nx.Graph, 
                                       num_partitions: int,
                                       conditions: SystemConditions,
                                       **kwargs) -> Dict:
        """
        Execute intelligent partitioning with automatic strategy selection.
        """
        start_time = time.time()
        
        # Select optimal strategy
        selected_strategy = self.select_optimal_strategy(conditions, graph)
        strategy_obj = self.strategies[selected_strategy]
        
        # Check if strategy switch is needed
        strategy_switched = (selected_strategy != self.current_strategy)
        if strategy_switched and self.current_strategy:
            self.logger.info(f"Strategy switch: {self.current_strategy.value} -> {selected_strategy.value}")
        
        # Execute partitioning
        partition_result = strategy_obj.execute_partitioning(graph, num_partitions, **kwargs)
        
        # Calculate multi-objective score
        multi_objective_score = self._calculate_multi_objective_score(
            partition_result, graph, conditions
        )
        
        # Enhanced result with framework metadata
        enhanced_result = {
            **partition_result,
            'framework_strategy': selected_strategy.value,
            'strategy_switched': strategy_switched,
            'multi_objective_score': multi_objective_score,
            'system_conditions': conditions,
            'framework_execution_time': time.time() - start_time,
            'objectives': self.objectives
        }
        
        # Update framework state
        self.current_strategy = selected_strategy
        self.current_partition = enhanced_result
        
        # Record history
        self.strategy_history.append({
            'timestamp': time.time(),
            'strategy': selected_strategy.value,
            'conditions': conditions,
            'performance': multi_objective_score
        })
        
        self.performance_history.append(enhanced_result)
        
        # Keep history bounded
        if len(self.strategy_history) > 1000:
            self.strategy_history = self.strategy_history[-1000:]
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        self.logger.info(f"Intelligent partitioning completed: "
                        f"strategy={selected_strategy.value}, "
                        f"score={multi_objective_score:.3f}, "
                        f"time={time.time() - start_time:.2f}s")
        
        return enhanced_result
    
    def _calculate_multi_objective_score(self, 
                                       partition_result: Dict, 
                                       graph: nx.Graph,
                                       conditions: SystemConditions) -> float:
        """Calculate multi-objective optimization score"""
        
        # Communication cost component
        cut_size = partition_result.get('cut_size', 0)
        total_edges = graph.number_of_edges()
        comm_cost_score = 1.0 - (cut_size / max(total_edges, 1))
        
        # Load balance component
        balance_score = partition_result.get('balance', partition_result.get('load_balance', 0.5))
        
        # Response time component (inverse of execution time)
        exec_time = partition_result.get('execution_time', time.time())
        response_score = 1.0 / (1.0 + exec_time)
        
        # Fault tolerance component (based on partition distribution)
        fault_tolerance_score = self._calculate_fault_tolerance_score(partition_result)
        
        # Weighted combination
        multi_objective_score = (
            self.objectives.communication_cost_weight * comm_cost_score +
            self.objectives.load_balance_weight * balance_score +
            self.objectives.response_time_weight * response_score +
            self.objectives.fault_tolerance_weight * fault_tolerance_score
        )
        
        return multi_objective_score
    
    def _calculate_fault_tolerance_score(self, partition_result: Dict) -> float:
        """Calculate fault tolerance score based on partition distribution"""
        partition_map = partition_result.get('partition_map', {})
        if not partition_map:
            return 0.5
        
        # Count nodes per partition
        partition_counts = {}
        for partition in partition_map.values():
            partition_counts[partition] = partition_counts.get(partition, 0) + 1
        
        # Better fault tolerance when partitions are more evenly distributed
        if len(partition_counts) <= 1:
            return 0.0
        
        counts = list(partition_counts.values())
        variance = np.var(counts)
        mean_count = np.mean(counts)
        
        # Lower variance relative to mean indicates better fault tolerance
        if mean_count == 0:
            return 0.0
        
        coefficient_of_variation = np.sqrt(variance) / mean_count
        fault_tolerance_score = 1.0 / (1.0 + coefficient_of_variation)
        
        return fault_tolerance_score
    
    def get_framework_analytics(self) -> Dict:
        """Get comprehensive analytics about framework performance"""
        if not self.strategy_history:
            return {"status": "no_data"}
        
        # Strategy usage statistics
        strategy_counts = {}
        for entry in self.strategy_history:
            strategy = entry['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Performance trends
        recent_scores = [entry['performance'] for entry in self.strategy_history[-50:]]
        performance_trend = np.mean(recent_scores) if recent_scores else 0.0
        
        # Strategy switches
        switches = sum(1 for i in range(1, len(self.strategy_history)) 
                      if self.strategy_history[i]['strategy'] != self.strategy_history[i-1]['strategy'])
        
        analytics = {
            'total_partitioning_operations': len(self.strategy_history),
            'strategy_usage': strategy_counts,
            'current_strategy': self.current_strategy.value if self.current_strategy else None,
            'strategy_switches': switches,
            'recent_performance_trend': performance_trend,
            'framework_stability': 1.0 - (switches / max(len(self.strategy_history), 1)),
            'objectives': {
                'communication_cost_weight': self.objectives.communication_cost_weight,
                'load_balance_weight': self.objectives.load_balance_weight,
                'response_time_weight': self.objectives.response_time_weight,
                'fault_tolerance_weight': self.objectives.fault_tolerance_weight
            }
        }
        
        return analytics
    
    def adapt_objectives(self, performance_feedback: Dict):
        """Adapt optimization objectives based on performance feedback"""
        # Increase weight for objectives that are performing poorly
        if performance_feedback.get('communication_cost_high', False):
            self.objectives.communication_cost_weight *= 1.1
        
        if performance_feedback.get('load_imbalance_detected', False):
            self.objectives.load_balance_weight *= 1.1
        
        if performance_feedback.get('response_time_slow', False):
            self.objectives.response_time_weight *= 1.1
        
        if performance_feedback.get('failures_detected', False):
            self.objectives.fault_tolerance_weight *= 1.1
        
        # Normalize weights
        total_weight = (self.objectives.communication_cost_weight + 
                       self.objectives.load_balance_weight +
                       self.objectives.response_time_weight + 
                       self.objectives.fault_tolerance_weight)
        
        self.objectives.communication_cost_weight /= total_weight
        self.objectives.load_balance_weight /= total_weight
        self.objectives.response_time_weight /= total_weight
        self.objectives.fault_tolerance_weight /= total_weight
        
        self.logger.info(f"Objectives adapted: comm={self.objectives.communication_cost_weight:.3f}, "
                        f"balance={self.objectives.load_balance_weight:.3f}, "
                        f"response={self.objectives.response_time_weight:.3f}, "
                        f"fault_tolerance={self.objectives.fault_tolerance_weight:.3f}")
