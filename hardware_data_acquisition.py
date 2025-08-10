import serial
import serial.tools.list_ports
import threading
import time
import logging
import numpy as np
from collections import deque
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

class HardwareDataAcquisition(QObject):
    """
    Hardware data acquisition class that reads serial data from your dental sensor hardware
    and provides it in a format compatible with your existing Qt application.
    """
    
    # Signals for Qt integration
    data_received = pyqtSignal(list)  # Emits the raw sensor data array
    connection_status_changed = pyqtSignal(bool, str)  # Emits (connected, status_message)
    error_occurred = pyqtSignal(str)  # Emits error messages
    
    def __init__(self, baudrate=460800, frame_size=6343):
        super().__init__()
        self.serial_port = None
        self.baudrate = baudrate
        self.frame_size = frame_size
        self.running = False
        self.connected = False
        
        # Data storage
        self.rows = 44
        self.cols = 52
        self.data = [0] * 2288  # 44 * 52 = 2288 total cells
        self.data_buffer = deque(maxlen=100)  # Keep last 100 frames
        
        # Serial reading thread
        self.serial_thread = None
        
        # Timer for periodic data emission
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.emit_latest_data)
        self.data_timer.start(50)  # Emit data every 50ms (20 FPS)
        
        logging.info("HardwareDataAcquisition initialized")
    
    def get_available_ports(self):
        """Get list of available serial ports"""
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            return ports
        except Exception as e:
            logging.error(f"Error getting ports: {e}")
            return []
    
    def connect_to_port(self, port):
        """Connect to a specific serial port"""
        if self.connected:
            self.disconnect()
        
        try:
            self.serial_port = serial.Serial(port, self.baudrate, timeout=0.1)
            self.connected = True
            self.running = True
            
            # Start serial reading thread
            self.serial_thread = threading.Thread(target=self._read_serial_loop, daemon=True)
            self.serial_thread.start()
            
            status_msg = f"Connected to {port}"
            self.connection_status_changed.emit(True, status_msg)
            logging.info(status_msg)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to {port}: {e}"
            self.connection_status_changed.emit(False, error_msg)
            self.error_occurred.emit(error_msg)
            logging.error(error_msg)
            return False
    
    def disconnect(self):
        """Disconnect from serial port"""
        self.running = False
        self.connected = False
        
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=1.0)
        
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except Exception as e:
                logging.error(f"Error closing serial port: {e}")
        
        self.connection_status_changed.emit(False, "Disconnected")
        logging.info("Disconnected from hardware")
    
    def _read_serial_loop(self):
        """Main serial reading loop running in separate thread"""
        buffer = bytearray()
        
        while self.running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    buffer.extend(data)
                    
                    # Process complete frames
                    while len(buffer) >= self.frame_size:
                        start_idx = buffer.find(b'\xff')
                        if start_idx == -1:
                            # No start marker found, clear buffer
                            buffer.clear()
                            break
                        
                        if start_idx + self.frame_size <= len(buffer):
                            # Complete frame available
                            frame = buffer[start_idx+1:start_idx+self.frame_size-1]
                            buffer = buffer[start_idx+self.frame_size:]
                            self._process_frame(frame)
                        else:
                            # Incomplete frame, wait for more data
                            break
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                error_msg = f"Serial read error: {e}"
                self.error_occurred.emit(error_msg)
                logging.error(error_msg)
                break
        
        # Clean up if loop exits
        if self.running:
            self.connection_status_changed.emit(False, "Serial connection lost")
    
    def _process_frame(self, frame):
        """Process a complete data frame from the hardware"""
        try:
            # Decode frame and parse data
            frame_string = frame.decode('latin1').rstrip('\xfe\xff')
            data_array_string = frame_string.split(',')
            
            # Update data array
            for i in range(min(len(data_array_string), len(self.data))):
                try:
                    self.data[i] = int(data_array_string[i])
                except ValueError:
                    self.data[i] = 0
            
            # Store in buffer for history
            self.data_buffer.append(self.data.copy())
            
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
    
    def emit_latest_data(self):
        """Emit the latest data to connected Qt components"""
        if self.connected and self.data:
            self.data_received.emit(self.data.copy())
    
    def get_latest_data(self):
        """Get the latest sensor data array"""
        return self.data.copy()
    
    def get_data_history(self):
        """Get recent data history"""
        return list(self.data_buffer)
    
    def is_connected(self):
        """Check if currently connected to hardware"""
        return self.connected
    
    def get_connection_status(self):
        """Get current connection status string"""
        if self.connected:
            return f"Connected to {self.serial_port.port if self.serial_port else 'Unknown'}"
        return "Disconnected"
    
    def cleanup(self):
        """Clean up resources"""
        self.disconnect()
        if self.data_timer:
            self.data_timer.stop()


class HardwareDataProcessor:
    """
    Processor class that converts raw hardware data into the format expected by your visualization components
    """
    
    def __init__(self, points_array):
        self.points_array = points_array
        self.rows = 44
        self.cols = 52
        
    def convert_to_force_matrix(self, raw_data):
        """
        Convert raw hardware data array to force matrix format
        Returns: (force_matrix, valid_positions)
        """
        if not raw_data or len(raw_data) != 2288:
            return np.array([]), []
        
        # Create force matrix based on valid sensor positions
        valid_positions = []
        force_values = []
        
        for col in range(self.cols):
            for row in range(self.rows):
                if self.points_array.is_valid(col, row):
                    idx = row * self.cols + col
                    if idx < len(raw_data):
                        force_values.append(raw_data[idx])
                        valid_positions.append((col, row))
        
        if force_values:
            force_matrix = np.array(force_values).reshape(1, -1)
            return force_matrix, valid_positions
        
        return np.array([]), []
    
    def get_force_at_position(self, raw_data, col, row):
        """Get force value at specific grid position"""
        if not raw_data or len(raw_data) != 2288:
            return 0
        
        if self.points_array.is_valid(col, row):
            idx = row * self.cols + col
            if idx < len(raw_data):
                return raw_data[idx]
        
        return 0
    
    def get_max_force(self, raw_data):
        """Get maximum force value from current data"""
        if not raw_data:
            return 0
        return max(raw_data) if raw_data else 0
    
    def normalize_force(self, force_value, max_force=1000):
        """Normalize force value to 0-1 range"""
        if max_force <= 0:
            return 0
        return min(1.0, max(0.0, force_value / max_force)) 