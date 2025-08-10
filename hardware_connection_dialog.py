from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QPushButton, QLabel, QGroupBox, QGridLayout, QSpinBox,
                             QCheckBox, QProgressBar, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import logging
import time

class HardwareConnectionDialog(QDialog):
    """
    Dialog for connecting to hardware and controlling data acquisition
    """
    
    # Signals
    connection_established = pyqtSignal()  # Emitted when hardware connection is established
    connection_lost = pyqtSignal()        # Emitted when hardware connection is lost
    
    def __init__(self, hardware_acquisition, parent=None):
        super().__init__(parent)
        self.hardware_acquisition = hardware_acquisition
        self.setup_ui()
        self.setup_connections()
        self.update_ui_state()
        
        # Timer for updating UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui_state)
        self.update_timer.start(1000)  # Update every second
        
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Hardware Connection")
        self.setModal(False)
        self.resize(500, 400)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Connection group
        connection_group = QGroupBox("Hardware Connection")
        connection_layout = QGridLayout()
        
        # Port selection
        connection_layout.addWidget(QLabel("Serial Port:"), 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(150)
        connection_layout.addWidget(self.port_combo, 0, 1)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setMaximumWidth(80)
        connection_layout.addWidget(self.refresh_button, 0, 2)
        
        # Baudrate selection
        connection_layout.addWidget(QLabel("Baudrate:"), 1, 0)
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(['115200', '230400', '460800', '921600'])
        self.baudrate_combo.setCurrentText('460800')
        connection_layout.addWidget(self.baudrate_combo, 1, 1)
        
        # Connection buttons
        self.connect_button = QPushButton("Connect")
        self.connect_button.setMaximumWidth(100)
        connection_layout.addWidget(self.connect_button, 2, 1)
        
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setMaximumWidth(100)
        self.disconnect_button.setEnabled(False)
        connection_layout.addWidget(self.disconnect_button, 2, 2)
        
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
        
        # Status group
        status_group = QGroupBox("Connection Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.connection_progress = QProgressBar()
        self.connection_progress.setVisible(False)
        status_layout.addWidget(self.connection_progress)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Data group
        data_group = QGroupBox("Data Acquisition")
        data_layout = QGridLayout()
        
        # Frame size
        data_layout.addWidget(QLabel("Frame Size:"), 0, 0)
        self.frame_size_spin = QSpinBox()
        self.frame_size_spin.setRange(1000, 10000)
        self.frame_size_spin.setValue(6343)
        data_layout.addWidget(self.frame_size_spin, 0, 1)
        
        # Data rate
        data_layout.addWidget(QLabel("Data Rate (Hz):"), 1, 0)
        self.data_rate_label = QLabel("0")
        data_layout.addWidget(self.data_rate_label, 1, 1)
        
        # Auto-connect checkbox
        self.auto_connect_check = QCheckBox("Auto-connect on startup")
        data_layout.addWidget(self.auto_connect_check, 2, 0, 1, 2)
        
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)
        
        # Log group
        log_group = QGroupBox("Connection Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Clear log button
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.setMaximumWidth(100)
        log_layout.addWidget(self.clear_log_button)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.setMaximumWidth(100)
        close_layout.addWidget(self.close_button)
        main_layout.addLayout(close_layout)
        
        self.setLayout(main_layout)
        
    def setup_connections(self):
        """Setup signal connections"""
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.connect_to_hardware)
        self.disconnect_button.clicked.connect(self.disconnect_from_hardware)
        self.clear_log_button.clicked.connect(self.clear_log)
        self.close_button.clicked.connect(self.close)
        
        # Hardware acquisition signals
        self.hardware_acquisition.connection_status_changed.connect(self.on_connection_status_changed)
        self.hardware_acquisition.error_occurred.connect(self.on_error_occurred)
        self.hardware_acquisition.data_received.connect(self.on_data_received)
        
        # Initial port refresh
        self.refresh_ports()
        
    def refresh_ports(self):
        """Refresh the list of available serial ports"""
        ports = self.hardware_acquisition.get_available_ports()
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        
        if ports:
            self.log_message(f"Found {len(ports)} available ports: {', '.join(ports)}")
        else:
            self.log_message("No serial ports found")
            
    def connect_to_hardware(self):
        """Connect to the selected hardware port"""
        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, "No Port Selected", "Please select a serial port first.")
            return
            
        baudrate = int(self.baudrate_combo.currentText())
        frame_size = self.frame_size_spin.value()
        
        # Update hardware acquisition parameters
        self.hardware_acquisition.baudrate = baudrate
        self.hardware_acquisition.frame_size = frame_size
        
        self.log_message(f"Attempting to connect to {port} at {baudrate} baud...")
        self.connection_progress.setVisible(True)
        self.connection_progress.setRange(0, 0)  # Indeterminate progress
        
        # Attempt connection
        if self.hardware_acquisition.connect_to_port(port):
            self.log_message(f"Successfully connected to {port}")
        else:
            self.log_message(f"Failed to connect to {port}")
            
    def disconnect_from_hardware(self):
        """Disconnect from hardware"""
        self.hardware_acquisition.disconnect()
        self.log_message("Disconnected from hardware")
        
    def on_connection_status_changed(self, connected, status_message):
        """Handle connection status changes"""
        self.update_ui_state()
        self.log_message(f"Status: {status_message}")
        
        if connected:
            self.connection_established.emit()
        else:
            self.connection_lost.emit()
            
    def on_error_occurred(self, error_message):
        """Handle error messages from hardware acquisition"""
        self.log_message(f"ERROR: {error_message}")
        
    def on_data_received(self, data):
        """Handle received data (for logging purposes)"""
        # Update data rate display
        if hasattr(self, '_last_data_time'):
            current_time = time.time()
            time_diff = current_time - self._last_data_time
            if time_diff > 0:
                rate = 1.0 / time_diff
                self.data_rate_label.setText(f"{rate:.1f}")
            self._last_data_time = current_time
        else:
            self._last_data_time = time.time()
            
    def update_ui_state(self):
        """Update UI state based on connection status"""
        connected = self.hardware_acquisition.is_connected()
        
        # Update button states
        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)
        self.port_combo.setEnabled(not connected)
        self.baudrate_combo.setEnabled(not connected)
        self.frame_size_spin.setEnabled(not connected)
        
        # Update status display
        if connected:
            status = self.hardware_acquisition.get_connection_status()
            self.status_label.setText(status)
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.connection_progress.setVisible(False)
        else:
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.connection_progress.setVisible(False)
            
    def log_message(self, message):
        """Add a message to the log"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear_log(self):
        """Clear the log display"""
        self.log_text.clear()
        
    def closeEvent(self, event):
        """Handle dialog close event"""
        # Don't disconnect hardware when closing dialog
        # Just hide the dialog
        event.ignore()
        self.hide()
        
    def showEvent(self, event):
        """Handle dialog show event"""
        super().showEvent(event)
        self.refresh_ports()
        self.update_ui_state() 