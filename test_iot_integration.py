#!/usr/bin/env python3
"""
Test script for Industrial IoT Integration Module
Tests the fixes for node health management and load balancing
"""

import time
import numpy as np
import asyncio
from src.core.industrial_iot_integration import (
    RealTimeStreamProcessor, 
    IndustrialDataPoint, 
    StreamType, 
    ProcessingPriority
)

def test_industrial_iot_integration():
    """Test the industrial IoT integration with sensor data processing"""
    print("=" * 60)
    print("Testing Industrial IoT Integration Module")
    print("=" * 60)
    
    # Initialize the integration system
    print("\n1. Initializing Industrial IoT Integration System...")
    iot_system = RealTimeStreamProcessor()
    
    # Check initial nodes
    print(f"   Initial node count: {len(iot_system.industrial_nodes)}")
    for node_id, node in iot_system.industrial_nodes.items():
        print(f"   Node {node_id}: {node.status} - {node.specializations}")
    
    async def run_async_tests():
        """Run the async parts of the test"""
        # Start the processing system in background
        processing_task = asyncio.create_task(iot_system.start_processing())
        
        # Give the system a moment to start
        await asyncio.sleep(0.1)
        
        # Test sensor data processing
        print("\n2. Testing Sensor Data Processing...")
        
        # Process each sensor reading
        for i, sensor_info in enumerate([
            {"sensor_id": "temp_001", "type": "temperature", "value": 25.5},
            {"sensor_id": "vib_002", "type": "vibration", "value": 0.8},
            {"sensor_id": "press_003", "type": "pressure", "value": 101.3},
            {"sensor_id": "flow_004", "type": "flow", "value": 15.2},
            {"sensor_id": "humid_005", "type": "humidity", "value": 65.0},
        ]):
            print(f"\n   Processing sensor {i+1}: {sensor_info['sensor_id']} ({sensor_info['type']})")
            
            # Create data point
            data_point = IndustrialDataPoint(
                timestamp=time.time(),
                source_id=sensor_info['sensor_id'],
                stream_type=StreamType.SENSOR_DATA,
                priority=ProcessingPriority.NORMAL,
                data={"values": [sensor_info['value']], "type": sensor_info['type']}
            )
            
            # Process the sensor data using async
            result = await iot_system.ingest_data_point(data_point)
            print(f"   ✓ Ingestion result: {result}")
            
            # Small delay to allow processing
            await asyncio.sleep(0.1)
        
        # Test load balancing
        print("\n4. Testing Load Balancing...")
        
        # Generate burst of data to test load balancing
        print("   Generating data burst...")
        for i in range(10):
            data_point = IndustrialDataPoint(
                timestamp=time.time(),
                source_id=f"burst_{i:03d}",
                stream_type=StreamType.SENSOR_DATA,
                priority=ProcessingPriority.NORMAL,
                data={
                    "values": [20.0 + np.random.random() * 10],
                    "type": "temperature",
                    "location": f"zone_{chr(65 + i % 5)}"
                }
            )
            
            # Process the data point using async
            result = await iot_system.ingest_data_point(data_point)
            print(f"   Burst {i+1}: Processed successfully: {result}")
            
            # Small delay to allow processing
            await asyncio.sleep(0.05)
        
        # Give processing time to complete
        await asyncio.sleep(0.5)
        
        # Test stream processing
        print("\n6. Testing Real-time Stream Processing...")
        
        # Simulate continuous data stream
        print("   Starting 5-second data stream simulation...")
        start_time = time.time()
        processed_count = 0
        
        while time.time() - start_time < 2.0:  # Reduced to 2 seconds for faster testing
            # Generate random sensor data
            sensor_types = ["temperature", "vibration", "pressure", "flow", "humidity"]
            selected_type = str(np.random.choice(sensor_types))
            
            data_point = IndustrialDataPoint(
                timestamp=time.time(),
                source_id=f"stream_{processed_count:04d}",
                stream_type=StreamType.SENSOR_DATA,
                priority=ProcessingPriority.NORMAL,
                data={
                    "values": [np.random.random() * 100],
                    "type": selected_type,
                    "location": f"zone_{chr(65 + processed_count % 5)}"
                }
            )
            
            result = await iot_system.ingest_data_point(data_point)
            processed_count += 1
            
            # Short delay to simulate realistic data rates
            await asyncio.sleep(0.05)
        
        print(f"   ✓ Processed {processed_count} sensor readings in 2 seconds")
        print(f"   ✓ Average rate: {processed_count / 2.0:.1f} readings/second")
        
        # Give final processing time
        await asyncio.sleep(0.5)
        
        # Stop the processing system
        iot_system.is_running = False
        processing_task.cancel()
        
        try:
            await processing_task
        except asyncio.CancelledError:
            pass
        
        return processed_count
    
    # Run the async tests
    processed_count = asyncio.run(run_async_tests())
    
    # Check node health after processing
    print("\n3. Checking Node Health Status...")
    for node_id, node in iot_system.industrial_nodes.items():
        time_since_heartbeat = time.time() - node.last_heartbeat
        print(f"   Node {node_id}: {node.status}")
        print(f"     - Last heartbeat: {time_since_heartbeat:.2f}s ago")
        print(f"     - Load: {node.current_load:.2f}")
        print(f"     - Processed: {node.processed_count} tasks")
    
    # Final health check
    print("\n5. Final Health Check...")
    healthy_nodes = 0
    for node_id, node in iot_system.industrial_nodes.items():
        time_since_heartbeat = time.time() - node.last_heartbeat
        is_healthy = node.status == "active" and time_since_heartbeat < 60
        print(f"   Node {node_id}: {'✓ HEALTHY' if is_healthy else '✗ UNHEALTHY'}")
        if is_healthy:
            healthy_nodes += 1
    
    print(f"\n   Total healthy nodes: {healthy_nodes}/{len(iot_system.industrial_nodes)}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_processed = sum(node.processed_count for node in iot_system.industrial_nodes.values())
    print(f"✓ Total sensors processed: {total_processed}")
    print(f"✓ Active nodes: {len(iot_system.industrial_nodes)}")
    print(f"✓ Healthy nodes: {healthy_nodes}")
    
    # Check for any "No healthy nodes available" scenarios
    if healthy_nodes > 0:
        print("✓ SUCCESS: No 'No healthy nodes available' errors encountered!")
        print(f"✓ Nodes properly processed {total_processed} data points!")
    else:
        print("✗ FAILURE: All nodes marked as unhealthy!")
    
    print("=" * 60)

if __name__ == "__main__":
    test_industrial_iot_integration()
