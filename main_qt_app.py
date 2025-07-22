# --- START OF FILE main_qt_app.py ---
import sys
import logging
import time
import numpy as np
import cv2
import atexit
import os
import collections # For deque if we were to implement more complex buffering

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QLineEdit
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
import vedo
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Assuming these are in the same directory or Python path
from data_acquisition import SensorDataReader # Your data simulation/reading
from data_processing import DataProcessor   # Your data processing
from graph_visualization_qt import GraphVisualizerQt # Your Matplotlib graph
from points_array import PointsArray         # Your PointsArray definition
from hardware_grid_visualizer_qt import HardwareGridVisualizerQt # Your 2D grid
from hardware_3d_bar_visualizer_qt import Hardware3DBarVisualizerQt # Your "working git version"

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from vedo import Plotter, settings # Import base Plotter
import vtk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# You can set specific loggers to DEBUG if needed, e.g.:
# logging.getLogger('hardware_3d_bar_visualizer_qt').setLevel(logging.DEBUG)

_main_app_window_instance_for_atexit = None
def cleanup_on_exit():
    global _main_app_window_instance_for_atexit
    if _main_app_window_instance_for_atexit and hasattr(_main_app_window_instance_for_atexit, 'video_writer'):
        if _main_app_window_instance_for_atexit.video_writer is not None and _main_app_window_instance_for_atexit.video_writer.isOpened():
            logging.info("ATEIXT: Releasing OpenCV video writer...");
            _main_app_window_instance_for_atexit.video_writer.release()
            _main_app_window_instance_for_atexit.video_writer = None
            logging.info("ATEIXT: OpenCV video writer released.")
atexit.register(cleanup_on_exit)

class VedoQtCanvas(QVTKRenderWindowInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
        # It's good practice to initialize the interactor if the render window exists.
        # Plotter usually handles this, but being explicit doesn't hurt.
        if self.GetRenderWindow() and self.GetRenderWindow().GetInteractor():
            self.GetRenderWindow().GetInteractor().Initialize()
        else:
            logging.warning("VedoQtCanvas: RenderWindow or Interactor not immediately available after super().__init__")

    def GetPlotter(self, **kwargs_for_plotter):
        # Plotter needs to be created with this QVTK widget as its qt_widget
        plt = Plotter(qt_widget=self, **kwargs_for_plotter)
        return plt

    def closeEvent(self, event):
        # Finalize the interactor when the widget is closed
        self.Finalize()
        super().closeEvent(event)

class EmbeddedVedoMultiViewWidget(QWidget):
    vedo_initialized_signal = pyqtSignal()
    mouse_interaction_started = pyqtSignal()
    mouse_interaction_ended = pyqtSignal()

    def __init__(self, processor_instance,
                 HardwareGridVisualizerClass,
                 Hw3DBarVisualizerClass,
                 parent_main_window,
                 plotter_kwargs=None,
                 initial_ci_width=9.0):
        super().__init__(parent_main_window)
        if plotter_kwargs is None: plotter_kwargs = {}

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setContentsMargins(0,0,0,0)
        self.vedo_canvas = VedoQtCanvas(self)
        self.vlayout.addWidget(self.vedo_canvas)

        plotter_creation_args = {'shape':(1,2), 'sharecam':False}
        if plotter_kwargs: plotter_creation_args.update(plotter_kwargs)
        if 'title' not in plotter_creation_args:
            plotter_creation_args['title'] = "Dental Visualizations"

        self.main_plotter = self.vedo_canvas.GetPlotter(**plotter_creation_args)
        if not self.main_plotter or len(self.main_plotter.renderers) < 2:
            logging.error("EmbeddedVedoMultiViewWidget: Failed to create main Vedo Plotter with 2 sub-renderers."); return

        self._processor_instance = processor_instance
        self._HardwareGridVisualizerClass = HardwareGridVisualizerClass
        self._Hw3DBarVisualizerClass = Hw3DBarVisualizerClass
        self._parent_main_window = parent_main_window
        self._initial_ci_width = initial_ci_width # <-- STORE THE VALUE

        self.grid_visualizer = None
        self.bar_visualizer = None

        QTimer.singleShot(250, self._finish_widget_initialization)

    def _finish_widget_initialization(self):
        logging.info("EmbeddedVedoMultiViewWidget: Starting delayed initialization...")
        if not self.main_plotter:
            logging.error("EmbeddedVedoMultiViewWidget: Main plotter is None during delayed init.")
            return

        # 1. Create Visualizers
        self.grid_visualizer = self._HardwareGridVisualizerClass(self._processor_instance, self.main_plotter, 0)
        self.bar_visualizer = self._Hw3DBarVisualizerClass(self._processor_instance, self.main_plotter, 1)

        # 2. Set Main App Window Refs (if visualizers use them)
        if hasattr(self.grid_visualizer, 'set_main_app_window_ref'):
            self.grid_visualizer.set_main_app_window_ref(self._parent_main_window)
        if hasattr(self.bar_visualizer, 'set_main_app_window_ref'):
            self.bar_visualizer.set_main_app_window_ref(self._parent_main_window)

        # 3. Setup Scenes for Visualizers
        logging.info("EmbeddedVedoMultiViewWidget: Calling setup_scene for grid_visualizer...")
        if hasattr(self.grid_visualizer, 'setup_scene'): self.grid_visualizer.setup_scene()
        
        logging.info("EmbeddedVedoMultiViewWidget: Calling setup_scene for bar_visualizer...")
        if hasattr(self.bar_visualizer, 'setup_scene'): self.bar_visualizer.setup_scene()

        # 4. Setup Mouse Callbacks
        if self.main_plotter:
            # Callback for actor picking logic
            if hasattr(self, '_dispatch_mouse_click'):
                self.main_plotter.add_callback('mouse click', self._dispatch_mouse_click) # For picking actors
                logging.debug("EmbeddedVedoMultiViewWidget: _dispatch_mouse_click callback added for 'mouse click'.")

            # Callbacks for pausing animation during general mouse interaction
            # Use VTK's general interaction events
            if self.main_plotter.interactor: # Get the vtkRenderWindowInteractor
                # StartInteractionEvent is good for when any manipulation (rotate, pan, zoom) begins
                self.main_plotter.interactor.AddObserver("StartInteractionEvent", self._on_vtk_interaction_started)
                # EndInteractionEvent for when it finishes
                self.main_plotter.interactor.AddObserver("EndInteractionEvent", self._on_vtk_interaction_ended)
                logging.debug("EmbeddedVedoMultiViewWidget: VTK Start/EndInteractionEvent observers added.")
            else:
                logging.warning("EmbeddedVedoMultiViewWidget: No interactor found to add Start/EndInteractionEvent observers.")
    
        # 5. Initial Render
        self.Render() 
        logging.info("EmbeddedVedoMultiViewWidget: Delayed initialization complete.")
        self.vedo_initialized_signal.emit()

    def _on_vtk_interaction_started(self, vtk_interactor, event_name):
        # event_name will be "StartInteractionEvent"
        logging.debug(f"VTK Interaction Started ({event_name}), emitting mouse_interaction_started.")
        self.mouse_interaction_started.emit()

    def _on_vtk_interaction_ended(self, vtk_interactor, event_name):
        # event_name will be "EndInteractionEvent"
        logging.debug(f"VTK Interaction Ended ({event_name}), emitting mouse_interaction_ended.")
        self.mouse_interaction_ended.emit()
        
    def _on_any_mouse_button_press(self, event):
        # This callback is for pausing animation during general camera manipulation
        logging.debug(f"Interaction event (Press): {event.name} received, emitting mouse_interaction_started.")
        self.mouse_interaction_started.emit()
        # Do not consume the event or stop propagation if other interactors need it.

    def _on_any_mouse_button_release(self, event):
        # This callback is for resuming animation after general camera manipulation
        logging.debug(f"Interaction event (Release): {event.name} received, emitting mouse_interaction_ended.")
        self.mouse_interaction_ended.emit()

    def _dispatch_mouse_click(self, event): # This is for specific actor picking
        if not event: return
        picked_actor = event.actor
        renderer_index_of_click = getattr(event, 'at', None)
        logging.debug(f"DISPATCH_CLICK (for picking): Actor: {picked_actor.name if picked_actor else 'None'}. Renderer Index: {renderer_index_of_click}.")
        
        if renderer_index_of_click is not None:
            if self.grid_visualizer and renderer_index_of_click == self.grid_visualizer.renderer_index:
                if hasattr(self.grid_visualizer, '_on_mouse_click'): self.grid_visualizer._on_mouse_click(event)
                return
            elif self.bar_visualizer and renderer_index_of_click == self.bar_visualizer.renderer_index:
                if hasattr(self.bar_visualizer, '_on_mouse_click'): self.bar_visualizer._on_mouse_click(event)
                return
        
        # Fallback for general deselect or if click not in a specific renderer with a handler
        original_actor_for_fallback = event.actor 
        event.actor = None 
        if self.grid_visualizer and hasattr(self.grid_visualizer, '_on_mouse_click'): self.grid_visualizer._on_mouse_click(event)
        if self.bar_visualizer and hasattr(self.bar_visualizer, '_on_mouse_click'): self.bar_visualizer._on_mouse_click(event)
        event.actor = original_actor_for_fallback
    
    # ... (update_views, get_frame_as_array, Render, get_grid_visualizer, get_bar_visualizer methods remain the same) ...
    # In main_qt_app.py -> EmbeddedVedoMultiViewWidget

    def update_views(self, timestamp, latest_hardware_flat_data=None, sensitivity=1):
        if not self.grid_visualizer or not self.bar_visualizer:
            return # Visualizers not yet initialized

        # *** ROBUSTNESS CHECK FOR RENDERERS ***
        if not self.main_plotter or len(self.main_plotter.renderers) < 2:
            logging.error("update_views called but plotter/renderers are not ready. Aborting update.")
            return
        # *** END CHECK ***

        if hasattr(self.grid_visualizer, 'animate'):
            self.main_plotter.at(0) 
            self.grid_visualizer.animate(timestamp, latest_hardware_flat_data, sensitivity)
        if hasattr(self.bar_visualizer, 'animate'):
            self.main_plotter.at(1) 
            self.bar_visualizer.animate(timestamp, latest_hardware_flat_data, sensitivity)
        
        self.Render()

    def get_frame_as_array(self, timestamp, latest_hardware_flat_data=None, sensitivity=1):
        if not self.grid_visualizer or not self.bar_visualizer: return None
        self.update_views(timestamp, latest_hardware_flat_data, sensitivity)
        if self.main_plotter and self.main_plotter.window:
            return self.main_plotter.screenshot(asarray=True)
        return None

    def Render(self):
        if hasattr(self.vedo_canvas, 'Render') and self.vedo_canvas.isVisible():
            self.vedo_canvas.Render()
        elif self.main_plotter and self.main_plotter.window:
             self.main_plotter.render()

    def get_grid_visualizer(self): return self.grid_visualizer
    def get_bar_visualizer(self): return self.bar_visualizer


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class MainAppWindow(QMainWindow):
    def __init__(self, processor, hw_data_source=None):
        super().__init__()
        self.processor = processor
        self.hw_data_source = hw_data_source
        self.current_timestamp_idx = 0
        
        # --- Consistent Animation State Variables ---
        self.user_wants_animation_playing = False # User's explicit Play/Pause state
        self.is_temporarily_paused_for_interaction = False # True if mouse interaction paused it
        # --- End Consistent Animation State Variables ---

        self.animation_timer = QTimer(self)
        # self.graph_time_indicator = None # Managed by GraphVisualizerQt
        self.setWindowTitle("Dental Force Visualization Suite (PyQt)")
        self.setGeometry(50, 50, 1800, 960)
        self.initial_graph_teeth = []
        self.currently_graphed_tooth_ids = []
        self.last_animated_timestamp = None
        
        self.output_video_filename="composite_dental_animation.mp4"
        self.canvas_width=1280 
        self.canvas_height=720  
        
        self.live_animation_fps = 15 
        self._animation_timer_interval = int(1000 / self.live_animation_fps) # Store interval
        self.video_output_fps = 10
        self.video_frame_skip = max(1, int(self.live_animation_fps / self.video_output_fps))
        self.frames_rendered_since_last_video_write = 0
        self.video_writer = None 

        self.current_ci_width = 9.0 # Default value in mm

        
        global _main_app_window_instance_for_atexit
        _main_app_window_instance_for_atexit = self
        
        self.graph_qt_canvas = MatplotlibCanvas(self)
        self.graph_visualizer = GraphVisualizerQt(self.processor)
        self.graph_visualizer.set_figure_axes(self.graph_qt_canvas.fig, self.graph_qt_canvas.axes)
        if self.processor.timestamps and self.processor.tooth_ids:
            self.initial_graph_teeth = self.processor.tooth_ids[:2] if len(self.processor.tooth_ids) >= 2 else self.processor.tooth_ids[:1]
            self.currently_graphed_tooth_ids = list(self.initial_graph_teeth) 
            if self.initial_graph_teeth:
                self.graph_visualizer.plot_tooth_lines(self.initial_graph_teeth)
        
        self.vedo_multiview_widget = EmbeddedVedoMultiViewWidget(
            self.processor,
            HardwareGridVisualizerQt,
            Hardware3DBarVisualizerQt, 
            self, 
            plotter_kwargs={'title': "Dental Force Views"},
            initial_ci_width=self.current_ci_width # Pass the initial value

        )
        self.vedo_multiview_widget.vedo_initialized_signal.connect(self.perform_initial_view_update)
        
        # --- Connect signals for interaction pause ---
        self.vedo_multiview_widget.mouse_interaction_started.connect(self.handle_mouse_interaction_started)
        self.vedo_multiview_widget.mouse_interaction_ended.connect(self.handle_mouse_interaction_ended)
        # --- End signal connection ---
        
        self.detailed_info_label = QLabel("Click an element for details or hover over graph points.")
        self.detailed_info_label.setWordWrap(True)
        self.detailed_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detailed_info_label.setStyleSheet("padding: 5px;")

        self._setup_ui()
        self._setup_animation_timer()

    def perform_initial_view_update(self):
        # ... (this method should be as per the last working version that fixed UnboundLocalError) ...
        logging.info("MainAppWindow: Performing initial view update triggered by Vedo widget.")
        latest_data = self.get_latest_hw_data_for_step()
        timestamp_to_use_for_update = 0.0 

        if self.processor.timestamps and len(self.processor.timestamps) > 0:
            first_ts_from_processor = self.processor.timestamps[0]
            self.last_animated_timestamp = first_ts_from_processor
            timestamp_to_use_for_update = first_ts_from_processor
            logging.info(f"MainAppWindow: Initial update using processor timestamp: {timestamp_to_use_for_update:.2f}s")
        elif self.hw_data_source:
            current_time = time.time() 
            self.last_animated_timestamp = current_time
            timestamp_to_use_for_update = current_time
            logging.info(f"MainAppWindow: Initial update using current time for live source: {timestamp_to_use_for_update:.2f}s")
        else:
            logging.warning("MainAppWindow: No processor timestamps and no live hardware source for initial view update.")
        
        if self.vedo_multiview_widget and self.vedo_multiview_widget.bar_visualizer:
             self.vedo_multiview_widget.update_views(timestamp_to_use_for_update, latest_data, 1)
        else:
            logging.warning("MainAppWindow: Vedo visualizers not yet fully available for initial update.")

        if self.graph_visualizer.figure:
            if not self.graph_visualizer.lines and self.currently_graphed_tooth_ids:
                self.graph_visualizer.plot_tooth_lines(self.currently_graphed_tooth_ids)
            self.graph_visualizer.update_time_indicator(timestamp_to_use_for_update)
            self.graph_qt_canvas.draw_idle()
    
    def get_latest_hw_data_for_step(self):
        # ... (as before) ...
        if self.hw_data_source and hasattr(self.hw_data_source, 'running') and self.hw_data_source.running:
            if hasattr(self.hw_data_source, 'get_latest_raw_forces'):
                return self.hw_data_source.get_latest_raw_forces()
        return None
    
    def _initialize_video_writer(self):
        # ... (as before) ...
        if self.video_writer is None or not self.video_writer.isOpened():
            if os.path.exists(self.output_video_filename):
                try: os.remove(self.output_video_filename); logging.info(f"Removed existing: {self.output_video_filename}")
                except Exception as e: logging.warning(f"Could not remove {self.output_video_filename}: {e}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            self.video_writer = cv2.VideoWriter(
                self.output_video_filename, fourcc, 
                float(self.video_output_fps),
                (self.canvas_width, self.canvas_height)
            )
            if not self.video_writer.isOpened():
                logging.error(f"Could not open video writer for {self.output_video_filename}")
                self.video_writer = None
            else: 
                logging.info(f"Video writer opened for {self.output_video_filename} at {self.video_output_fps} FPS.")
        return self.video_writer is not None and self.video_writer.isOpened()
    
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_area_layout = QHBoxLayout()
        top_area_layout.addWidget(self.vedo_multiview_widget, 3)

        # Control panel on the right
        control_panel_layout = QVBoxLayout()
        self.detailed_info_label = QLabel("Click an element for details.")
        self.detailed_info_label.setWordWrap(True)
        self.detailed_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        control_panel_layout.addWidget(self.detailed_info_label)
        
        # --- UI for User to Input Central Incisor Width ---
        arch_controls_layout = QHBoxLayout()
        arch_label = QLabel("CI Width (mm):")
        self.ci_width_input = QLineEdit(str(self.current_ci_width))
        self.ci_width_input.setValidator(QDoubleValidator(5.0, 15.0, 2)) # Min 5mm, Max 15mm, 2 decimals
        self.ci_width_input.setFixedWidth(50)
        
        update_arch_button = QPushButton("Update Arch")
        update_arch_button.clicked.connect(self.on_update_arch_clicked)
        
        arch_controls_layout.addWidget(arch_label)
        arch_controls_layout.addWidget(self.ci_width_input)
        
        control_panel_layout.addLayout(arch_controls_layout)
        control_panel_layout.addWidget(update_arch_button)
        # --- END UI for User Input ---

        control_panel_layout.addStretch() # Pushes controls to the top
        top_area_layout.addLayout(control_panel_layout, 1)

        main_layout.addLayout(top_area_layout, 3)
        main_layout.addWidget(self.graph_qt_canvas, 2)

        # --- Bottom Controls (Play/Pause, Reset View) ---
        # ... (This part is unchanged) ...
        controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play Animation")
        self.play_pause_button.clicked.connect(self.toggle_animation)
        self.reset_3d_view_button = QPushButton("Reset 3D View")
        self.reset_3d_view_button.clicked.connect(self.reset_3d_bar_camera_in_multiview)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.reset_3d_view_button)
        controls_layout.addStretch(1)
        main_layout.addLayout(controls_layout)


    def reset_3d_bar_camera_in_multiview(self):
        # ... (as before) ...
        if self.vedo_multiview_widget and self.vedo_multiview_widget.bar_visualizer and \
           hasattr(self.vedo_multiview_widget.bar_visualizer, 'reset_camera_view'):
            logging.info("Resetting 3D Bar View camera (multiview).")
            self.vedo_multiview_widget.bar_visualizer.reset_camera_view()

    # --- Animation Control Methods with Consistent Naming ---
    def _setup_animation_timer(self):
        self.animation_timer.timeout.connect(self.animation_step)

    def handle_mouse_interaction_started(self):
        if self.user_wants_animation_playing and not self.is_temporarily_paused_for_interaction:
            self.animation_timer.stop()
            self.is_temporarily_paused_for_interaction = True
            logging.debug("Animation updates TEMPORARILY PAUSED for mouse interaction.")

    def handle_mouse_interaction_ended(self):
        if self.user_wants_animation_playing and self.is_temporarily_paused_for_interaction:
            self.animation_timer.start(self._animation_timer_interval)
            self.is_temporarily_paused_for_interaction = False
            logging.debug("Animation updates RESUMED after mouse interaction.")

    def toggle_animation(self):
        if self.user_wants_animation_playing:
            # User wants to PAUSE
            self.animation_timer.stop()
            self.play_pause_button.setText("Play Animation")
            self.user_wants_animation_playing = False
            self.is_temporarily_paused_for_interaction = False 
            logging.info("Animation Paused by user.")
        else:
            # User wants to PLAY
            can_animate = (self.processor.timestamps and len(self.processor.timestamps) > 0) or \
                          (self.hw_data_source and hasattr(self.hw_data_source, 'running') and self.hw_data_source.running)
            if not can_animate:
                logging.warning("Cannot start animation: No data source.")
                return

            if not self._initialize_video_writer():
                logging.warning("Video writer not ready. Animation will play without recording.")
            
            self.user_wants_animation_playing = True
            self.play_pause_button.setText("Pause Animation")
            
            if not self.is_temporarily_paused_for_interaction:
                self.frames_rendered_since_last_video_write = 0 
                self.animation_timer.start(self._animation_timer_interval)
                logging.info(f"Animation Started by user at {self.live_animation_fps} FPS.")
            else:
                logging.info(f"User requested Play, but interaction ongoing. Will start at {self.live_animation_fps} FPS when interaction ends.")
    # --- End Animation Control Methods ---

    def animation_step(self): 
        # ... (this method should be as per your last working version, ensuring it uses the animation state flags correctly if needed for its logic) ...
        current_display_timestamp = 0.0
        latest_hardware_flat_data = None
        sensitivity_from_ui = 1 

        if self.hw_data_source and hasattr(self.hw_data_source, 'running') and self.hw_data_source.running:
            latest_hardware_flat_data = self.get_latest_hw_data_for_step()
            current_display_timestamp = time.time() 
        elif self.processor.timestamps and len(self.processor.timestamps) > 0 :
            if self.current_timestamp_idx >= len(self.processor.timestamps):
                self.current_timestamp_idx = 0 
            current_display_timestamp = self.processor.timestamps[self.current_timestamp_idx]
            latest_hardware_flat_data = self.get_latest_hw_data_for_step() 
        else: 
            self.user_wants_animation_playing = False # Stop if no data
            self.animation_timer.stop()
            self.play_pause_button.setText("Play Animation")
            logging.warning("Animation step: No data source or timestamps. Stopping animation.")
            return

        self.last_animated_timestamp = current_display_timestamp

        if self.vedo_multiview_widget:
             self.vedo_multiview_widget.update_views(current_display_timestamp, latest_hardware_flat_data, sensitivity_from_ui)
        
        graph_ts_to_update = current_display_timestamp
        if self.processor.timestamps and len(self.processor.timestamps) > 0:
            graph_ts_to_update = self.processor.timestamps[self.current_timestamp_idx % len(self.processor.timestamps)]

        if self.graph_visualizer.figure and self.graph_visualizer.ax:
            self.graph_visualizer.update_graph_to_timestamp(graph_ts_to_update, self.currently_graphed_tooth_ids)
            self.graph_visualizer.update_time_indicator(graph_ts_to_update) 
            self.graph_qt_canvas.draw_idle()
        
        self.frames_rendered_since_last_video_write += 1
        if self.video_writer and self.video_writer.isOpened() and \
           self.frames_rendered_since_last_video_write >= self.video_frame_skip:
            
            self.frames_rendered_since_last_video_write = 0 
            frame_vedo = None
            if self.vedo_multiview_widget:
                frame_vedo = self.vedo_multiview_widget.get_frame_as_array(current_display_timestamp, latest_hardware_flat_data, sensitivity_from_ui)
            frame_graph = self.graph_visualizer.get_frame_as_array(graph_ts_to_update, self.currently_graphed_tooth_ids)
            
            canvas=np.zeros((self.canvas_height,self.canvas_width,3),dtype=np.uint8); canvas[:]=(210,210,210)
            h_v=int(self.canvas_height*0.65);w_v=self.canvas_width
            h_g=self.canvas_height-h_v;w_g=self.canvas_width
            y_vedo_start=0; y_graph_start=h_v
            if frame_vedo is not None:
                if frame_vedo.shape[2] == 4: frame_vedo = cv2.cvtColor(frame_vedo, cv2.COLOR_RGBA2BGR)
                elif frame_vedo.shape[2] == 3 and frame_vedo.dtype == np.uint8 : pass # Assume BGR
                elif frame_vedo.shape[2] == 3 : frame_vedo = cv2.cvtColor(frame_vedo, cv2.COLOR_RGB2BGR) # if RGB
                canvas[y_vedo_start : y_vedo_start+h_v, 0:w_v] = cv2.resize(frame_vedo, (w_v,h_v))
            if frame_graph is not None: 
                canvas[y_graph_start : y_graph_start+h_g, 0:w_g] = cv2.resize(frame_graph,(w_g,h_g))
            self.video_writer.write(canvas)
        
        if self.processor.timestamps and len(self.processor.timestamps) > 0 and \
           not (self.hw_data_source and hasattr(self.hw_data_source, 'running') and self.hw_data_source.running):
            self.current_timestamp_idx += 1

    def on_update_arch_clicked(self):
        """
        Reads the CI width from the user input and tells the visualizer to redraw the arch.
        """
        try:
            new_width_str = self.ci_width_input.text()
            new_width = float(new_width_str)
            
            if 5.0 <= new_width <= 15.0:
                self.current_ci_width = new_width
                logging.info(f"User requested arch update. New CI Width: {self.current_ci_width} mm")
                
                # Call the update method on the grid visualizer
                if self.vedo_multiview_widget and self.vedo_multiview_widget.grid_visualizer and \
                   hasattr(self.vedo_multiview_widget.grid_visualizer, 'update_arch_parameters'):
                    
                    self.vedo_multiview_widget.grid_visualizer.update_arch_parameters(
                        user_central_incisor_width=self.current_ci_width
                    )
                else:
                    logging.warning("Grid visualizer not ready or does not support parameter updates.")
            else:
                logging.warning(f"Invalid CI width entered: {new_width}. Must be between 5.0 and 15.0.")
                self.ci_width_input.setText(str(self.current_ci_width)) # Revert
        except ValueError:
            logging.error(f"Invalid input for CI width: '{self.ci_width_input.text()}'")
            self.ci_width_input.setText(str(self.current_ci_width)) # Revert


    def update_graph_on_click(self, sel_tid=None): 
        # ... (as before) ...
        new_ids_to_graph = [sel_tid] if sel_tid is not None else self.initial_graph_teeth
        if set(new_ids_to_graph) != set(self.currently_graphed_tooth_ids) or not self.graph_visualizer.lines:
            self.graph_visualizer.plot_tooth_lines(new_ids_to_graph)
            self.currently_graphed_tooth_ids = list(new_ids_to_graph) 
            current_graph_time_for_update = self.last_animated_timestamp 
            if self.processor.timestamps and len(self.processor.timestamps) > 0:
                 current_graph_time_for_update = self.processor.timestamps[self.current_timestamp_idx % len(self.processor.timestamps)]
            self.graph_visualizer.update_graph_to_timestamp(current_graph_time_for_update, new_ids_to_graph)
            self.graph_visualizer.update_time_indicator(current_graph_time_for_update)
            self.graph_qt_canvas.draw_idle()
            
    def update_detailed_info(self, info_str):
        # ... (as before) ...
        self.detailed_info_label.setText(info_str)

    def closeEvent(self, event):
        logging.info("Main window closing...")
        self.animation_timer.stop() 
        self.user_wants_animation_playing = False # Explicitly set state
        self.is_temporarily_paused_for_interaction = False
        if hasattr(self, 'video_writer') and self.video_writer and self.video_writer.isOpened():
            logging.info("Releasing video writer from MainAppWindow closeEvent.")
            self.video_writer.release()
            self.video_writer = None
        if hasattr(self.vedo_multiview_widget, 'close'): # Check if close method exists
            self.vedo_multiview_widget.close()
        super().closeEvent(event)

if __name__ == '__main__':
    # ... (same __main__ block as before) ...
    app = QApplication(sys.argv)
    logging.info(f"Python version: {sys.version.split()[0]}")
    logging.info(f"Vedo version: {vedo.__version__}")
    if 'vtk' in sys.modules: logging.info(f"VTK version: {vtk.VTK_VERSION}")
    else: logging.warning("VTK module not directly imported in main.")

    sim_reader = SensorDataReader()
    data = sim_reader.simulate_data(duration=5, num_teeth=14, num_sensor_points_per_tooth=4) 
    processor = DataProcessor(data)
    processor.create_force_matrix()

    class DummyHWSource: 
        def __init__(self, num_valid_sensors):
            self.num_valid_sensors = num_valid_sensors; self.running = True; self.iter = 0
        def get_latest_raw_forces(self):
            self.iter +=1
            return [abs( (self.iter*7 + i*13 + int(time.perf_counter()*5)) % 990 ) + 10 for i in range(self.num_valid_sensors)]
        def connect(self): self.running = True; return True
        def disconnect(self): self.running = False

    pa = PointsArray()
    num_valid_hw_cells = sum(1 for r in range(44) for c in range(52) if pa.is_valid(c,r))
    hw_data_source_for_app = DummyHWSource(num_valid_hw_cells)

    if not (processor.timestamps and len(processor.timestamps) > 0) and not hw_data_source_for_app:
        logging.error("No data source. Exiting."); sys.exit(-1)
        
    main_window = MainAppWindow(processor, hw_data_source=hw_data_source_for_app) 
    main_window.show()
    sys.exit(app.exec_())