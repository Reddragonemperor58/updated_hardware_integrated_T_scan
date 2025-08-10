# Hardware Integration for Dental Force Visualization

This document explains how to use the hardware integration features that have been added to your Qt application to replace the simulation data with real hardware data from your dental sensor system.

## Overview

The hardware integration consists of three main components:

1. **HardwareDataAcquisition** - Handles serial communication with your hardware
2. **HardwareDataProcessor** - Converts raw hardware data into visualization format
3. **HardwareConnectionDialog** - User interface for connecting to hardware

## Features

- **Real-time data acquisition** from your dental sensor hardware
- **Automatic data processing** and conversion to visualization format
- **User-friendly connection interface** with port selection and status monitoring
- **Seamless integration** with existing visualization components
- **Fallback to simulation** when hardware is not available

## Hardware Requirements

Your hardware should provide:
- Serial communication (USB-to-Serial or direct serial)
- Data format: Comma-separated values with frame markers
- Frame size: 6343 bytes (configurable)
- Baudrate: 460800 (configurable)
- Data structure: 2288 sensor values (44 rows × 52 columns)

## Installation

1. Ensure you have the required Python packages:
   ```bash
   pip install pyserial PyQt5 numpy
   ```

2. The following files should be in your project directory:
   - `hardware_data_acquisition.py`
   - `hardware_connection_dialog.py`
   - `points_array.py` (already exists)

## Usage

### Starting the Application

1. Run your main application as usual:
   ```bash
   python main_qt_app.py
   ```

2. Click the **"Hardware Connection"** button (green button) in the control panel

3. The hardware connection dialog will open

### Connecting to Hardware

1. **Select Serial Port**: Choose your hardware's serial port from the dropdown
2. **Set Baudrate**: Use 460800 (default) or adjust as needed
3. **Set Frame Size**: Use 6343 (default) or adjust based on your hardware
4. **Click Connect**: The application will attempt to connect to your hardware

### Connection Status

- **Green status**: Successfully connected
- **Red status**: Disconnected or error
- **Connection log**: Shows detailed connection information and errors

### Data Visualization

Once connected:
- Your visualizations will automatically update with real hardware data
- The animation system will use hardware data instead of simulation
- Data rate is displayed in the connection dialog
- You can disconnect/reconnect without restarting the application

## Configuration

### Serial Parameters

- **Baudrate**: 460800 (default), supports 115200, 230400, 460800, 921600
- **Frame Size**: 6343 bytes (default), configurable from 1000-10000
- **Timeout**: 0.1 seconds (hardcoded)

### Data Processing

- **Grid dimensions**: 44 rows × 52 columns (hardcoded)
- **Sensor validation**: Uses your existing `PointsArray` class
- **Data format**: Raw integer values from hardware

## Troubleshooting

### Common Issues

1. **No ports found**
   - Check USB connections
   - Install appropriate USB-to-Serial drivers
   - Verify hardware is powered on

2. **Connection fails**
   - Verify correct port selection
   - Check baudrate settings
   - Ensure hardware is not connected to another application

3. **Data not updating**
   - Check connection status
   - Verify hardware is sending data
   - Check frame size configuration

4. **Application crashes**
   - Check serial port permissions
   - Verify hardware compatibility
   - Check error logs

### Debug Information

Enable debug logging by modifying the logging level in your main application:
```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Testing

Run the test script to verify hardware integration:
```bash
python test_hardware_integration.py
```

This will test:
- Hardware acquisition class
- Data processing
- Qt integration
- Connection dialog

## Integration Details

### Data Flow

1. **Hardware** → Serial Port → `HardwareDataAcquisition`
2. **Raw Data** → `HardwareDataProcessor` → Visualization Format
3. **Processed Data** → Qt Application → Visualizations

### Signal Connections

- `data_received`: Emits when new hardware data arrives
- `connection_status_changed`: Emits connection status updates
- `error_occurred`: Emits error messages

### Fallback Behavior

When hardware is not available:
- Application falls back to simulation data
- All existing functionality remains intact
- Hardware connection button remains available

## Customization

### Adding New Hardware Support

1. Extend `HardwareDataAcquisition` class
2. Implement custom data parsing in `_process_frame`
3. Update frame size and baudrate defaults

### Modifying Data Processing

1. Extend `HardwareDataProcessor` class
2. Add new data conversion methods
3. Update visualization data format

### UI Customization

1. Modify `HardwareConnectionDialog` class
2. Add new connection parameters
3. Customize status display

## Performance Considerations

- **Data rate**: Hardware data is processed at up to 20 FPS
- **Memory usage**: Last 100 frames are buffered
- **CPU usage**: Serial processing runs in separate thread
- **Visualization**: Updates are throttled to prevent UI lag

## Security Notes

- Serial port access may require elevated permissions on some systems
- Hardware data is processed locally (no network transmission)
- No sensitive data is logged or stored

## Support

For issues or questions:
1. Check the connection log in the hardware dialog
2. Review application logs for error details
3. Test with the provided test script
4. Verify hardware compatibility and settings

## Future Enhancements

Potential improvements:
- Data recording and playback
- Multiple hardware support
- Advanced error handling
- Performance optimization
- Data export functionality 