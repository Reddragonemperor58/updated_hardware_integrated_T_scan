#!/usr/bin/env python3
"""
Test script for hardware integration components
"""

import sys
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import our hardware components
from hardware_data_acquisition import HardwareDataAcquisition, HardwareDataProcessor
from hardware_connection_dialog import HardwareConnectionDialog
from points_array import PointsArray

def test_hardware_acquisition():
    """Test the hardware data acquisition class"""
    print("Testing HardwareDataAcquisition...")
    
    # Create instance
    hw_acq = HardwareDataAcquisition()
    
    # Test available ports
    ports = hw_acq.get_available_ports()
    print(f"Available ports: {ports}")
    
    # Test connection status
    print(f"Connection status: {hw_acq.get_connection_status()}")
    
    # Clean up
    hw_acq.cleanup()
    print("HardwareDataAcquisition test completed\n")

def test_hardware_processor():
    """Test the hardware data processor class"""
    print("Testing HardwareDataProcessor...")
    
    # Create instance
    points_array = PointsArray()
    hw_proc = HardwareDataProcessor(points_array)
    
    # Test with dummy data
    dummy_data = [i * 10 for i in range(2288)]  # 44 * 52 = 2288
    
    # Test force matrix conversion
    force_matrix, valid_positions = hw_proc.convert_to_force_matrix(dummy_data)
    print(f"Force matrix shape: {force_matrix.shape}")
    print(f"Valid positions count: {len(valid_positions)}")
    
    # Test force at position
    force = hw_proc.get_force_at_position(dummy_data, 25, 25)
    print(f"Force at position (25, 25): {force}")
    
    # Test max force
    max_force = hw_proc.get_max_force(dummy_data)
    print(f"Max force: {max_force}")
    
    print("HardwareDataProcessor test completed\n")

def test_qt_integration():
    """Test Qt integration with hardware components"""
    print("Testing Qt integration...")
    
    app = QApplication(sys.argv)
    
    # Create hardware acquisition
    hw_acq = HardwareDataAcquisition()
    
    # Create connection dialog
    dialog = HardwareConnectionDialog(hw_acq)
    
    # Show dialog for a few seconds
    dialog.show()
    
    # Set up timer to close after 3 seconds
    timer = QTimer()
    timer.singleShot(3000, dialog.close)
    timer.singleShot(3500, app.quit)
    
    # Run the application
    app.exec_()
    
    # Clean up
    hw_acq.cleanup()
    print("Qt integration test completed\n")

def main():
    """Run all tests"""
    print("Starting hardware integration tests...\n")
    
    try:
        test_hardware_acquisition()
        test_hardware_processor()
        test_qt_integration()
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 