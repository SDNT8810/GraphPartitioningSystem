"""
Industrial IoT Integration Module
Self-Partitioning Graphs for Industrial Data Management

Implements real-time stream processing and dynamic load balancing 
for industrial IoT environments as described in Proposed_Method.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import json
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

class StreamType(Enum):
    """Types of industrial data streams"""
    SENSOR_DATA = "sensor_data"
    CONTROL_SIGNALS = "control_signals"
    DIAGNOSTIC_INFO = "diagnostic_info"
    PERFORMANCE_METRICS = "performance_metrics"
    ALARM_EVENTS = "alarm_events"

class ProcessingPriority(Enum):
    """Processing priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class IndustrialDataPoint:
    """Individual data point from industrial system"""
    timestamp: float
    source_id: str
    stream_type: StreamType
    priority: ProcessingPriority
    data: Dict
    processing_deadline: Optional[float] = None
    requires_real_time: bool = False

@dataclass
class StreamMetrics:
    """Metrics for data stream monitoring"""
    throughput: float = 0.0
    latency_avg: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    backlog_size: int = 0
    processing_time: float = 0.0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_count: int = 0
    total_count: int = 0

@dataclass
class IndustrialNode:
    """Industrial processing node with capabilities"""
    node_id: str
    processing_capacity: float
    memory_capacity: float
    network_bandwidth: float
    specializations: List[StreamType] = field(default_factory=list)
    current_load: float = 0.0
    status: str = "active"
    last_heartbeat: float = field(default_factory=time.time)
    processed_count: int = 0

class RealTimeStreamProcessor:
    """
    Real-time stream processor for industrial IoT data.
    
    Features:
    - Multi-stream processing with prioritization
    - Dynamic load balancing
    - Fault tolerance and recovery
    - Real-time performance monitoring
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.processing_queues = {
            ProcessingPriority.CRITICAL: asyncio.Queue(maxsize=100),
            ProcessingPriority.HIGH: asyncio.Queue(maxsize=500),
            ProcessingPriority.NORMAL: asyncio.Queue(maxsize=1000),
            ProcessingPriority.LOW: asyncio.Queue(maxsize=2000)
        }
        
        self.stream_metrics = {}
        self.industrial_nodes = {}
        self.load_balancer = DynamicLoadBalancer()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=1000)
        self.throughput_counter = 0
        self.throughput_start_time = time.time()
        
        # Real-time constraints
        self.real_time_threshold = 0.100  # 100ms for real-time processing
        self.deadline_violations = 0
        
        self.logger = logging.getLogger("RealTimeStreamProcessor")
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize with default nodes
        self._create_default_nodes()
        
    async def start_processing(self):
        """Start the real-time processing system"""
        self.is_running = True
        self.logger.info("Starting real-time stream processing system")
        
        # Start priority-based processing tasks
        tasks = []
        for priority in ProcessingPriority:
            task = asyncio.create_task(self._process_priority_queue(priority))
            tasks.append(task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_performance())
        tasks.append(monitor_task)
        
        # Start load balancing task
        load_balance_task = asyncio.create_task(self._dynamic_load_balancing())
        tasks.append(load_balance_task)
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Processing system error: {e}")
        finally:
            self.is_running = False
    
    async def ingest_data_point(self, data_point: IndustrialDataPoint) -> bool:
        """Ingest a single data point for processing"""
        try:
            # Check if real-time deadline can be met
            if data_point.requires_real_time:
                current_time = time.time()
                if (data_point.processing_deadline and 
                    current_time > data_point.processing_deadline):
                    self.deadline_violations += 1
                    self.logger.warning(f"Deadline violation for {data_point.source_id}")
                    return False
            
            # Route to appropriate priority queue
            queue = self.processing_queues[data_point.priority]
            
            # Non-blocking put with immediate feedback
            try:
                queue.put_nowait(data_point)
                self.throughput_counter += 1
                return True
            except asyncio.QueueFull:
                self.logger.warning(f"Queue full for priority {data_point.priority}")
                
                # Try to process in lower priority queue if possible
                if data_point.priority != ProcessingPriority.LOW:
                    return await self._route_to_lower_priority(data_point)
                
                return False
                
        except Exception as e:
            self.logger.error(f"Data ingestion error: {e}")
            return False
    
    async def _process_priority_queue(self, priority: ProcessingPriority):
        """Process data points from specific priority queue"""
        queue = self.processing_queues[priority]
        
        while self.is_running:
            try:
                # Timeout based on priority level
                timeout = {
                    ProcessingPriority.CRITICAL: 0.001,  # 1ms
                    ProcessingPriority.HIGH: 0.010,      # 10ms
                    ProcessingPriority.NORMAL: 0.100,    # 100ms
                    ProcessingPriority.LOW: 1.0          # 1s
                }[priority]
                
                data_point = await asyncio.wait_for(queue.get(), timeout=timeout)
                
                # Process the data point
                start_time = time.time()
                success = await self._process_data_point(data_point)
                processing_time = time.time() - start_time
                
                # Update metrics
                if hasattr(self, 'processing_times'):
                    self.processing_times.append(processing_time)
                self._update_stream_metrics(data_point, processing_time, success)
                
                # Check real-time constraints
                if (data_point.requires_real_time and 
                    processing_time > self.real_time_threshold):
                    self.logger.warning(f"Real-time constraint violated: {processing_time:.3f}s")
                
            except asyncio.TimeoutError:
                # No data available, continue monitoring
                continue
            except Exception as e:
                self.logger.error(f"Priority queue processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_data_point(self, data_point: IndustrialDataPoint) -> bool:
        """Process individual data point"""
        try:
            # Select optimal processing node
            selected_node = self.load_balancer.select_optimal_node(
                self.industrial_nodes, data_point
            )
            
            if not selected_node:
                self.logger.error("No available nodes for processing")
                return False
            
            # Execute processing based on stream type
            processing_result = await self._execute_stream_processing(
                data_point, selected_node
            )
            
            # Update node load and heartbeat
            if selected_node.node_id in self.industrial_nodes:
                self.industrial_nodes[selected_node.node_id].current_load += 0.1
                # Update heartbeat to show node is active
                self.update_node_heartbeat(selected_node.node_id)
                # Increment processed count
                if processing_result:
                    self.industrial_nodes[selected_node.node_id].processed_count += 1
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Data point processing failed: {e}")
            return False
    
    async def _execute_stream_processing(self, 
                                       data_point: IndustrialDataPoint, 
                                       node: IndustrialNode) -> bool:
        """Execute actual stream processing logic"""
        
        # Simulate processing based on stream type
        processing_functions = {
            StreamType.SENSOR_DATA: self._process_sensor_data,
            StreamType.CONTROL_SIGNALS: self._process_control_signals,
            StreamType.DIAGNOSTIC_INFO: self._process_diagnostic_info,
            StreamType.PERFORMANCE_METRICS: self._process_performance_metrics,
            StreamType.ALARM_EVENTS: self._process_alarm_events
        }
        
        processor = processing_functions.get(data_point.stream_type, self._process_generic)
        
        # Execute in thread pool for CPU-intensive work
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, processor, data_point, node
        )
        
        return result
    
    def _process_sensor_data(self, data_point: IndustrialDataPoint, node: IndustrialNode) -> bool:
        """Process sensor data stream"""
        try:
            # Extract sensor values
            sensor_values = data_point.data.get('values', [])
            
            # Apply filtering and validation
            filtered_values = [v for v in sensor_values if -1000 <= v <= 1000]
            
            # Detect anomalies
            anomalies = []
            if filtered_values:
                mean_val = np.mean(filtered_values)
                std_val = np.std(filtered_values)
                anomalies = [v for v in filtered_values if abs(v - mean_val) > 3 * std_val]
                
                if anomalies:
                    self.logger.warning(f"Anomalies detected in sensor {data_point.source_id}: {anomalies}")
            
            # Store processed data
            processed_data = {
                'original_count': len(sensor_values),
                'filtered_count': len(filtered_values),
                'mean': np.mean(filtered_values) if filtered_values else 0,
                'anomaly_count': len(anomalies),
                'processing_node': node.node_id,
                'timestamp': time.time()
            }
            
            # Simulate storage operation
            time.sleep(0.001)  # 1ms processing time
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sensor data processing error: {e}")
            return False
    
    def _process_control_signals(self, data_point: IndustrialDataPoint, node: IndustrialNode) -> bool:
        """Process control signals - high priority, real-time"""
        try:
            control_commands = data_point.data.get('commands', [])
            
            # Validate control commands
            valid_commands = []
            for cmd in control_commands:
                if self._validate_control_command(cmd):
                    valid_commands.append(cmd)
                else:
                    self.logger.warning(f"Invalid control command: {cmd}")
            
            # Execute control logic
            for cmd in valid_commands:
                self._execute_control_command(cmd, node)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Control signal processing error: {e}")
            return False
    
    def _process_diagnostic_info(self, data_point: IndustrialDataPoint, node: IndustrialNode) -> bool:
        """Process diagnostic information"""
        try:
            diagnostic_data = data_point.data.get('diagnostics', {})
            
            # Analyze diagnostic patterns
            health_score = self._calculate_health_score(diagnostic_data)
            
            # Generate alerts if needed
            if health_score < 0.7:
                self.logger.warning(f"Low health score for {data_point.source_id}: {health_score:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Diagnostic processing error: {e}")
            return False
    
    def _process_performance_metrics(self, data_point: IndustrialDataPoint, node: IndustrialNode) -> bool:
        """Process performance metrics"""
        try:
            metrics = data_point.data.get('metrics', {})
            
            # Update performance tracking
            for metric_name, value in metrics.items():
                self._update_performance_tracking(metric_name, value, data_point.source_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Performance metrics processing error: {e}")
            return False
    
    def _process_alarm_events(self, data_point: IndustrialDataPoint, node: IndustrialNode) -> bool:
        """Process alarm events - critical priority"""
        try:
            alarm_data = data_point.data.get('alarm', {})
            severity = alarm_data.get('severity', 'low')
            
            # Immediate escalation for critical alarms
            if severity == 'critical':
                self.logger.critical(f"CRITICAL ALARM from {data_point.source_id}: {alarm_data}")
                # Trigger immediate response protocols
                self._trigger_emergency_response(alarm_data, data_point.source_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Alarm processing error: {e}")
            return False
    
    def _process_generic(self, data_point: IndustrialDataPoint, node: IndustrialNode) -> bool:
        """Generic processing for unknown stream types"""
        try:
            # Basic validation and storage
            data_size = len(str(data_point.data))
            
            # Simulate processing time based on data size
            processing_time = min(data_size / 10000, 0.1)  # Max 100ms
            time.sleep(processing_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Generic processing error: {e}")
            return False
    
    async def _monitor_performance(self):
        """Monitor system performance in real-time"""
        while self.is_running:
            try:
                # Calculate current metrics
                current_time = time.time()
                time_elapsed = current_time - self.throughput_start_time
                
                if time_elapsed > 0:
                    current_throughput = self.throughput_counter / time_elapsed
                else:
                    current_throughput = 0
                
                # Processing time statistics
                if self.processing_times:
                    avg_processing_time = np.mean(self.processing_times)
                    p99_processing_time = np.percentile(self.processing_times, 99)
                else:
                    avg_processing_time = 0
                    p99_processing_time = 0
                
                # Queue backlogs
                total_backlog = sum(q.qsize() for q in self.processing_queues.values())
                
                # Log performance metrics
                self.logger.info(f"Performance: throughput={current_throughput:.1f}/s, "
                               f"avg_time={avg_processing_time:.3f}s, "
                               f"p99_time={p99_processing_time:.3f}s, "
                               f"backlog={total_backlog}, "
                               f"deadline_violations={self.deadline_violations}")
                
                # Reset counters periodically
                if time_elapsed > 60:  # Reset every minute
                    self.throughput_counter = 0
                    self.throughput_start_time = current_time
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _dynamic_load_balancing(self):
        """Dynamic load balancing across industrial nodes"""
        while self.is_running:
            try:
                # Ensure we have at least one node
                if not self.industrial_nodes:
                    self._create_default_nodes()
                
                # Update node loads and health
                for node_id, node in self.industrial_nodes.items():
                    # Simulate load decay
                    node.current_load *= 0.95
                    
                    # Update heartbeat for active processing (simulate node activity)
                    if node.current_load > 0.01:  # Node is processing something
                        node.last_heartbeat = time.time()
                    
                    # Check node health with more lenient timeout
                    if time.time() - node.last_heartbeat > 60:  # 60 seconds timeout
                        if node.status == "active":
                            node.status = "unhealthy"
                            self.logger.warning(f"Node {node_id} appears unhealthy")
                    else:
                        if node.status == "unhealthy":
                            node.status = "active"
                            self.logger.info(f"Node {node_id} recovered")
                
                # Rebalance if needed
                await self.load_balancer.rebalance_if_needed(self.industrial_nodes)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(10)
    
    def add_industrial_node(self, node: IndustrialNode):
        """Add new industrial processing node"""
        self.industrial_nodes[node.node_id] = node
        self.logger.info(f"Added industrial node: {node.node_id}")
    
    def _create_default_nodes(self):
        """Create default industrial nodes if none exist"""
        if not self.industrial_nodes:
            # Create a few default nodes with different specializations
            default_nodes = [
                IndustrialNode(
                    node_id="industrial_0",
                    processing_capacity=10.0,
                    memory_capacity=8.0,
                    network_bandwidth=100.0,
                    specializations=[StreamType.SENSOR_DATA, StreamType.PERFORMANCE_METRICS]
                ),
                IndustrialNode(
                    node_id="industrial_1", 
                    processing_capacity=12.0,
                    memory_capacity=16.0,
                    network_bandwidth=100.0,
                    specializations=[StreamType.CONTROL_SIGNALS, StreamType.ALARM_EVENTS]
                ),
                IndustrialNode(
                    node_id="industrial_2",
                    processing_capacity=8.0,
                    memory_capacity=4.0,
                    network_bandwidth=50.0,
                    specializations=[StreamType.DIAGNOSTIC_INFO]
                )
            ]
            
            for node in default_nodes:
                self.add_industrial_node(node)
            
            self.logger.info("Created default industrial nodes")
    
    def update_node_heartbeat(self, node_id: str):
        """Update heartbeat for a specific node"""
        if node_id in self.industrial_nodes:
            self.industrial_nodes[node_id].last_heartbeat = time.time()
            if self.industrial_nodes[node_id].status == "unhealthy":
                self.industrial_nodes[node_id].status = "active"
                self.logger.info(f"Node {node_id} status restored to active")
    
    def remove_industrial_node(self, node_id: str):
        """Remove industrial processing node"""
        if node_id in self.industrial_nodes:
            del self.industrial_nodes[node_id]
            self.logger.info(f"Removed industrial node: {node_id}")
    
    def get_real_time_metrics(self) -> Dict:
        """Get real-time system metrics"""
        current_time = time.time()
        time_elapsed = current_time - self.throughput_start_time
        
        return {
            'throughput': self.throughput_counter / max(time_elapsed, 1),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'p99_processing_time': np.percentile(self.processing_times, 99) if self.processing_times else 0,
            'total_backlog': sum(q.qsize() for q in self.processing_queues.values()),
            'deadline_violations': self.deadline_violations,
            'active_nodes': len([n for n in self.industrial_nodes.values() if n.status == "active"]),
            'queue_sizes': {p.name: q.qsize() for p, q in self.processing_queues.items()},
            'node_loads': {nid: n.current_load for nid, n in self.industrial_nodes.items()}
        }
    
    # Helper methods for processing functions
    def _validate_control_command(self, command: Dict) -> bool:
        """Validate control command"""
        required_fields = ['target', 'action', 'value']
        return all(field in command for field in required_fields)
    
    def _execute_control_command(self, command: Dict, node: IndustrialNode):
        """Execute validated control command"""
        # Simulate control execution
        time.sleep(0.001)  # 1ms execution time
    
    def _calculate_health_score(self, diagnostic_data: Dict) -> float:
        """Calculate health score from diagnostic data"""
        # Simple health scoring based on available metrics
        scores = []
        
        if 'temperature' in diagnostic_data:
            temp = diagnostic_data['temperature']
            temp_score = 1.0 if 20 <= temp <= 80 else 0.5
            scores.append(temp_score)
        
        if 'vibration' in diagnostic_data:
            vibration = diagnostic_data['vibration']
            vib_score = 1.0 if vibration < 10 else 0.3
            scores.append(vib_score)
        
        if 'pressure' in diagnostic_data:
            pressure = diagnostic_data['pressure']
            press_score = 1.0 if 0.8 <= pressure <= 1.2 else 0.4
            scores.append(press_score)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _update_performance_tracking(self, metric_name: str, value: float, source_id: str):
        """Update performance tracking for metrics"""
        # Store metrics for analysis
        if source_id not in self.stream_metrics:
            self.stream_metrics[source_id] = StreamMetrics()
        
        metrics = self.stream_metrics[source_id]
        
        # Update the specific metric if it exists
        if hasattr(metrics, metric_name):
            if isinstance(getattr(metrics, metric_name), deque):
                getattr(metrics, metric_name).append(float(value))
            else:
                setattr(metrics, metric_name, float(value))
        else:
            # Create a new deque for this metric
            setattr(metrics, metric_name, deque([float(value)], maxlen=100))
    
    def _trigger_emergency_response(self, alarm_data: Dict, source_id: str):
        """Trigger emergency response protocols"""
        # Implement emergency response logic
        self.logger.critical(f"EMERGENCY RESPONSE TRIGGERED for {source_id}")
        
        # Notify all relevant systems
        # Increase processing priority for this source
        # Activate backup systems if needed
    
    async def _route_to_lower_priority(self, data_point: IndustrialDataPoint) -> bool:
        """Route to lower priority queue when higher priority is full"""
        priority_order = [ProcessingPriority.CRITICAL, ProcessingPriority.HIGH, 
                         ProcessingPriority.NORMAL, ProcessingPriority.LOW]
        
        current_index = priority_order.index(data_point.priority)
        
        for i in range(current_index + 1, len(priority_order)):
            lower_priority = priority_order[i]
            queue = self.processing_queues[lower_priority]
            
            try:
                queue.put_nowait(data_point)
                self.logger.info(f"Routed {data_point.source_id} from {data_point.priority} to {lower_priority}")
                return True
            except asyncio.QueueFull:
                continue
        
        return False
    
    def _update_stream_metrics(self, data_point: IndustrialDataPoint, processing_time: float, success: bool):
        """Update metrics for specific stream"""
        source_id = data_point.source_id
        
        if source_id not in self.stream_metrics:
            self.stream_metrics[source_id] = StreamMetrics()
        
        metrics = self.stream_metrics[source_id]
        
        # Update processing time
        try:
            if hasattr(metrics, 'processing_times') and metrics.processing_times is not None:
                metrics.processing_times.append(processing_time)
            else:
                metrics.processing_times = deque([processing_time], maxlen=100)
        except AttributeError:
            # Fallback: recreate the processing_times deque
            metrics.processing_times = deque([processing_time], maxlen=100)
        
        # Update success rate
        if hasattr(metrics, 'success_count') and hasattr(metrics, 'total_count'):
            metrics.success_count += 1 if success else 0
            metrics.total_count += 1
        else:
            metrics.success_count = 1 if success else 0
            metrics.total_count = 1
        
        # Update throughput and latency
        metrics.processing_time = processing_time
        if metrics.total_count > 0:
            metrics.error_rate = 1.0 - (metrics.success_count / metrics.total_count)

class DynamicLoadBalancer:
    """Dynamic load balancer for industrial processing nodes"""
    
    def __init__(self):
        self.logger = logging.getLogger("DynamicLoadBalancer")
    
    def select_optimal_node(self, 
                          nodes: Dict[str, IndustrialNode], 
                          data_point: IndustrialDataPoint) -> Optional[IndustrialNode]:
        """Select optimal node for processing data point"""
        
        if not nodes:
            self.logger.warning("No nodes available")
            return None
        
        # Filter healthy nodes
        healthy_nodes = [node for node in nodes.values() if node.status == "active"]
        
        if not healthy_nodes:
            self.logger.warning("No healthy nodes available")
            # As a fallback, try to use any available node
            all_nodes = list(nodes.values())
            if all_nodes:
                fallback_node = all_nodes[0]
                fallback_node.status = "active"  # Force reactivate
                fallback_node.last_heartbeat = time.time()
                self.logger.info(f"Reactivated fallback node: {fallback_node.node_id}")
                return fallback_node
            return None
        
        # Score nodes based on multiple factors
        node_scores = {}
        
        for node in healthy_nodes:
            # Base score from current load (lower is better)
            load_score = 1.0 - min(node.current_load, 1.0)
            
            # Specialization bonus
            specialization_score = 1.0
            if data_point.stream_type in node.specializations:
                specialization_score = 1.5
            
            # Capacity score
            capacity_score = node.processing_capacity / 10.0  # Normalize
            
            # Combined score
            total_score = load_score * 0.5 + specialization_score * 0.3 + capacity_score * 0.2
            node_scores[node.node_id] = total_score
        
        # Select node with highest score
        if not node_scores:
            return None
            
        best_node_id = max(node_scores.keys(), key=lambda k: node_scores[k])
        selected_node = nodes[best_node_id]
        
        self.logger.debug(f"Selected node {best_node_id} for {data_point.source_id} "
                         f"(score: {node_scores[best_node_id]:.3f})")
        
        return selected_node
    
    async def rebalance_if_needed(self, nodes: Dict[str, IndustrialNode]):
        """Rebalance load if needed"""
        if not nodes:
            return
        
        loads = [node.current_load for node in nodes.values() if node.status == "active"]
        
        if not loads:
            return
        
        # Check if rebalancing is needed
        max_load = max(loads)
        min_load = min(loads)
        
        if max_load - min_load > 0.5:  # Significant imbalance
            self.logger.info(f"Load imbalance detected: max={max_load:.3f}, min={min_load:.3f}")
            # Implement rebalancing logic here
            await self._execute_rebalancing(nodes)
    
    async def _execute_rebalancing(self, nodes: Dict[str, IndustrialNode]):
        """Execute load rebalancing"""
        # Simple rebalancing: reduce load on heavily loaded nodes
        for node in nodes.values():
            if node.current_load > 0.8:
                reduction = min(0.2, node.current_load - 0.6)
                node.current_load -= reduction
                self.logger.info(f"Reduced load on node {node.node_id} by {reduction:.3f}")
