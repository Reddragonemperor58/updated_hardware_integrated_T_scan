# --- START OF FILE hardware_grid_visualizer_qt.py ---
import numpy as np
from vedo import Text2D, Rectangle, colors, Plotter # Plotter for type hint
import logging
from points_array import PointsArray 
import vtk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HardwareGridVisualizerQt:
    def __init__(self, processor_instance, parent_plotter_instance, renderer_index):
        self.processor = processor_instance 
        self.parent_plotter = parent_plotter_instance
        self.renderer_index = renderer_index
        self.renderer = parent_plotter_instance.renderers[renderer_index]

        self.hw_rows = 44; self.hw_cols = 52
        self.points_array_checker = PointsArray()
        self.max_force_for_scaling = 1000.0 
        
        # --- PERSISTENT ACTORS ---
        self.cell_rect_actors = {} # Dict: {(r, c): vedo.Rectangle}
        self.time_text_actor = None # Still recreated, simple enough
        # ---
        
        self.timestamps = self.processor.timestamps
        self.current_timestamp_idx = 0; self.last_animated_timestamp = None
        self.main_app_window_ref = None

        if not self.renderer: logging.error(f"HwGridViz (R{self.renderer_index}): Renderer not provided."); return
        if self.processor.cleaned_data is None: self.processor.create_force_matrix() # Ensure data for timestamps

    def set_main_app_window_ref(self, main_app_window_instance): # Same
        self.main_app_window_ref = main_app_window_instance

    def setup_scene(self): # Creates persistent actors
        if not self.renderer or not self.parent_plotter: return
        logging.info(f"HwGridViz (R{self.renderer_index}): Setting up scene with persistent actors...")
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera
        cam.ParallelProjectionOn()

        self._create_and_add_grid_rects_once() # This now creates AND ADDS them once

        # Camera fitting after actors are added
        cell_render_size = 0.25 
        grid_render_width = self.hw_cols * cell_render_size
        grid_render_height = self.hw_rows * cell_render_size
        cam.SetFocalPoint(0,0,0) # Assuming grid is centered by _create_and_add_grid_rects_once
        cam.SetPosition(0,0, (grid_render_height/2) / np.tan(np.radians(cam.GetViewAngle()/2)) * 1.5 ) # Basic perspective distance
        # For parallel projection:
        cam.SetPosition(0, 0, 20) # Adjust Z for desired view
        cam.SetViewUp(0, 1, 0)     
        cam.SetParallelScale(grid_render_height / 1.8) 

        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.renderer.SetBackground(0.93, 0.93, 0.97) # Light lavender
        logging.info(f"HwGridViz (R{self.renderer_index}): Scene setup complete.")

    def _create_and_add_grid_rects_once(self):
        if not self.renderer: 
            logging.warning(f"HwGridViz (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Renderer not available for creating grid rects.")
            return
            
        if self.cell_rect_actors: # Clear existing if any
            # Get the vtkActor from each Vedo Rectangle object for removal
            vtk_actors_to_remove = [rect.actor for rect in self.cell_rect_actors.values() if hasattr(rect, 'actor')]
            for act in vtk_actors_to_remove:
                self.renderer.RemoveActor(act) # Remove individual vtkActors
            self.cell_rect_actors.clear()

        cell_size = 0.25; padding = 0.005 
        effective_cell_draw_size = cell_size - padding
        half_draw_size = effective_cell_draw_size / 2.0
        total_grid_visual_width = self.hw_cols * cell_size
        total_grid_visual_height = self.hw_rows * cell_size
        # Center the grid around (0,0) in its local XY plane
        offset_x = -total_grid_visual_width / 2
        offset_y = -total_grid_visual_height / 2 
        
        logging.info(f"HwGridViz: Creating {self.hw_rows*self.hw_cols} potential rectangle objects...")
        
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                cell_center_x = offset_x + (c_idx * cell_size) + cell_size / 2
                cell_center_y = offset_y + ((self.hw_rows - 1 - r_idx) * cell_size) + cell_size / 2 # r=0 is top row
                
                p1x = cell_center_x - half_draw_size
                p1y = cell_center_y - half_draw_size
                p2x = cell_center_x + half_draw_size
                p2y = cell_center_y + half_draw_size
                
                rect = Rectangle((p1x, p1y), (p2x, p2y), c='lightgrey', alpha=0.1)
                rect.lw(0) 
                if not self.points_array_checker.is_valid(c_idx, r_idx):
                    rect.alpha(0) 
                
                self.cell_rect_actors[(r_idx, c_idx)] = rect 
                # --- CORRECTED: Add individual VTK actor ---
                if hasattr(rect, 'actor') and rect.actor: # Ensure it's a Vedo visual object with a vtkActor
                    self.renderer.AddActor(rect.actor)
                # --- END CORRECTION ---
        
        logging.info(f"HwGridViz (R{self.renderer_index if hasattr(self, 'renderer_index') else 'N/A'}): Added {len(self.cell_rect_actors)} cell rectangles to renderer.")
        
    def _value_to_color_hardware(self, value, sensitivity=1): # Same
        mapped_value = (value / sensitivity * 255) // self.max_force_for_scaling; mapped_value = min(255,max(0,int(mapped_value)))
        r,g,b = 200,200,200 
        if mapped_value>204:r=255;g=max(0,int(150-((mapped_value-204)*150/51)));b=0
        elif mapped_value>140:r=int(139+((mapped_value-140)*116/64));g=int((mapped_value-140)*150/64);b=0
        elif mapped_value>76:g=int(255-((mapped_value-76)*155/64));r=int(((mapped_value-76)/64)*100);b=0
        elif mapped_value>12:r=0;g=int(255-((mapped_value-12)*155/64));b=int(100-((mapped_value-12)*50/64))
        return (r/255.0,g/255.0,b/255.0)

    def render_grid_view(self, timestamp, hardware_data_flat_array, sensitivity=1):
        if not self.renderer or not self.parent_plotter: return
        self.parent_plotter.at(self.renderer_index) # Ensure context for Text2D positioning

        # --- Update Time text (still recreated) ---
        if self.time_text_actor: self.renderer.RemoveActor(self.time_text_actor.actor)
        self.time_text_actor = Text2D(f"HW Grid - T: {timestamp:.1f}s", pos="bottom-left", c='k', s=0.7)
        self.renderer.AddActor(self.time_text_actor.actor) # Add new one

        if hardware_data_flat_array is None: 
            # Optionally hide all cell_rect_actors if no data
            for rect_actor in self.cell_rect_actors.values(): rect_actor.alpha(0.05)
            return

        data_idx = 0
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                rect_actor = self.cell_rect_actors.get((r_idx, c_idx))
                if not rect_actor: continue 

                if self.points_array_checker.is_valid(c_idx, r_idx):
                    if data_idx < len(hardware_data_flat_array):
                        value = hardware_data_flat_array[data_idx]
                        color = self._value_to_color_hardware(value, sensitivity)
                        # --- UPDATE EXISTING RECT ACTOR'S PROPERTIES ---
                        rect_actor.color(color).alpha(1.0 if value > 5 else 0.1) 
                        data_idx += 1
                    else: 
                        rect_actor.alpha(0.1).color('lightgrey') # Not enough data, make dim
                # else: Invalid cells were set to alpha(0) in _create_and_add_grid_rects_once and remain so
        
        # No explicit self.renderer.render() here; EmbeddedVedoMultiViewWidget handles it.

    def animate(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1): # Same
        self.last_animated_timestamp = timestamp_to_render
        self.render_grid_view(timestamp_to_render, hardware_data_for_timestamp, sensitivity)

    def get_frame_as_array(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1): # Same
        # ... (This method returns None, main multiview widget screenshots itself)
        if not self.renderer or not self.parent_plotter: return None
        self.parent_plotter.at(self.renderer_index)
        self.render_grid_view(timestamp_to_render, hardware_data_for_timestamp, sensitivity)
        return None # Main multiview widget does the screenshot of the whole window

    # _on_mouse_click would need to be adapted if picking persistent rect_actors
    # self.set_main_app_window_ref(...) remains the same
# --- END OF FILE hardware_grid_visualizer_qt.py ---