# --- START OF FILE hardware_grid_visualizer_qt.py ---
import numpy as np
from vedo import Text2D, Rectangle, colors, Line, Spline, Points, Text3D # Use Text instead of Text3D
import logging
from points_array import PointsArray 
import vtk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareGridVisualizerQt:
    TOOTH_WIDTH_PROPORTIONS = {
        "Central Incisor": 1.00, "Lateral Incisor": 0.78, "Canine": 0.89,
        "1st Premolar": 0.75, "2nd Premolar": 0.70, "1st Molar": 1.20, "2nd Molar": 1.10,
    }
    TOOTH_ORDER = ["Central Incisor", "Lateral Incisor", "Canine", "1st Premolar", "2nd Premolar", "1st Molar", "2nd Molar"]

    def __init__(self, processor_instance, parent_plotter_instance, renderer_index, user_central_incisor_width=9.0):
        self.processor = processor_instance 
        self.parent_plotter = parent_plotter_instance
        self.renderer_index = renderer_index
        self.renderer = parent_plotter_instance.renderers[renderer_index]
        self.hw_rows = 44; self.hw_cols = 52
        self.points_array_checker = PointsArray()
        self.max_force_for_scaling = 1000.0 
        
        self.cell_rect_actors = {} 
        self.time_text_actor = None
        self.segment_boundary_lines = []
        self.arch_boundary_splines = []
        self.segment_control_points_actors = [] # Placeholder for future interactivity
        self.segment_text_actors = [] # Will hold the Text3D labels
        self.segment_polygons = []
        self.segment_cell_map = {}
        
        self.user_ci_width = user_central_incisor_width
        # --- REVISED PARAMETERS FOR BETTER FIT ---
        self.arch_params = {
            'grid_cell_size_mm': 0.7, 
            'width_scale': 0.4,        # Fit 98% of the valid grid width
            'depth_scale': 0.99,         # Arch depth is 50% of the valid grid height
            'anterior_flatness' : 0.2,   # A value around 0.5 gives a good U-shape
            'lingual_curve_offset': 3.5, # Inner curve depth relative to outer curve
            'vertical_offset_factor': -0.09, # Shift arch up/down (+ is up) as a fraction of valid grid height
        }
        # ---
        if not self.renderer: logger.error(f"HwGridViz (R{self.renderer_index}): Renderer not provided."); return
        if self.processor.cleaned_data is None: self.processor.create_force_matrix()

    def setup_scene(self):
        if not self.renderer or not self.parent_plotter: return
        logger.info(f"HwGridViz (R{self.renderer_index}): Setting up scene with segments...")
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera
        cam.ParallelProjectionOn()
        self._create_and_add_grid_rects_once()
        self._rebuild_arch_overlay_actors() # This now creates lines, splines, and text
        self._map_cells_to_segments()
        
        cell_render_size = 0.25 
        grid_render_height = self.hw_rows * cell_render_size
        cam.SetFocalPoint(0,0,0)
        cam.SetPosition(0, 0, 20)
        cam.SetViewUp(0, 1, 0)     
        cam.SetParallelScale(grid_render_height / 1.7) 
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.renderer.SetBackground(0.93, 0.93, 0.97)
        logger.info(f"HwGridViz (R{self.renderer_index}): Scene setup complete.")


    # def _add_segment_actors_to_renderer(self):
    #     # Remove old text actors
    #     for actor in self.segment_text_actors: self.renderer.RemoveActor(actor)
    #     self.segment_text_actors.clear()

    #     # Add lines and splines
    #     all_segment_actors = self.segment_boundary_lines + self.arch_boundary_splines
    #     for actor in all_segment_actors: self.renderer.AddActor(actor)
        
    #     num_segments = len(self.segment_polygons) if hasattr(self, 'segment_polygons') and self.segment_polygons else 14
        
    #     for i in range(num_segments):
    #         # Using Text with billboard=True is the most robust way for text in a 3D scene
    #         # that should always face the camera, even in a 2D projection.
    #         text_actor = Text3D(f"{i+1}", s=0.5, justify='center', c='black')
    #         text_actor.billboard() # This makes it always face the camera
            
    #         self.segment_text_actors.append(text_actor)
    #         self.renderer.AddActor(text_actor)
        
    #     logger.info(f"Created {len(self.segment_text_actors)} Text (billboard) actors for labels.")

    # In hardware_grid_visualizer_qt.py

    def _rebuild_arch_overlay_actors(self):
        """
        Removes all old overlay actors (lines, splines, text) and creates/adds new ones.
        This is the single source of truth for the arch overlay.
        """
        # 1. Remove all existing overlay actors from the scene
        all_overlay_actors = self.segment_boundary_lines + self.arch_boundary_splines + self.segment_text_actors
        if self.parent_plotter and all_overlay_actors:
            self.parent_plotter.at(self.renderer_index)
            self.parent_plotter.remove(all_overlay_actors)
        
        # 2. Clear the internal Python lists
        self.segment_boundary_lines.clear()
        self.arch_boundary_splines.clear()
        self.segment_text_actors.clear()

        # 3. Generate the geometric points for the new arch
        interprox_line_coords, outer_spline_points, inner_spline_points = self._calculate_arch_points()
        
        # 4. Create the new vedo objects (Lines, Splines)
        for p1, p2 in interprox_line_coords:
            line = Line(p1, p2, c='royalblue', lw=2, alpha=0.7)
            self.segment_boundary_lines.append(line)
        if outer_spline_points:
            outer_spline = Spline(outer_spline_points); outer_spline.color('darkblue').linewidth(2).alpha(0.7)
            self.arch_boundary_splines.append(outer_spline)
        if inner_spline_points:
            inner_spline = Spline(inner_spline_points); inner_spline.color('darkblue').linewidth(2).alpha(0.7)
            self.arch_boundary_splines.append(inner_spline)

        # 5. Create the new Text3D actors
        num_segments = len(interprox_line_coords) - 1 if interprox_line_coords else 0
        text_height = 0.5
        for i in range(num_segments):
            text_actor = Text3D(f"{i+1}", s=text_height, justify='center', c='black')
            text_actor.follow_camera()
            self.segment_text_actors.append(text_actor)

        # *** 6. CORRECTED AND ROBUST ACTOR ADDITION LOGIC ***
        new_actors_to_add = self.segment_boundary_lines + self.arch_boundary_splines + self.segment_text_actors
        for vedo_obj in new_actors_to_add:
            if vedo_obj is None:
                continue
            
            # Check if the object has a .actor attribute that should be used
            # This is common for objects that are assemblies or more complex.
            # For Line and Spline, the object itself IS the actor.
            actor_to_add = None
            if hasattr(vedo_obj, 'actor') and vedo_obj.actor is not None:
                actor_to_add = vedo_obj.actor
            else:
                # Assume the object itself is a vtkProp (like Line, Spline, Text3D)
                actor_to_add = vedo_obj
            
            if actor_to_add:
                self.renderer.AddActor(actor_to_add)
            else:
                logger.warning(f"Could not find a valid vtkProp to add for object of type {type(vedo_obj)}")
        # *** END CORRECTION ***

        logger.info(f"Rebuilt arch overlay with {len(new_actors_to_add)} objects.")

    def _create_and_add_grid_rects_once(self):
        # ... (This function remains unchanged) ...
        if not self.renderer: 
            logger.warning(f"HwGridViz (R{self.renderer_index}): Renderer not available for creating grid rects.")
            return
        if self.cell_rect_actors: 
            for act in [rect.actor for rect in self.cell_rect_actors.values() if hasattr(rect, 'actor')]:
                self.renderer.RemoveActor(act)
            self.cell_rect_actors.clear()
        cell_size = 0.25; padding = 0.005 
        total_grid_visual_width = self.hw_cols * cell_size
        total_grid_visual_height = self.hw_rows * cell_size
        offset_x = -total_grid_visual_width / 2
        offset_y = -total_grid_visual_height / 2 
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                cell_center_x = offset_x + (c_idx * cell_size) + cell_size / 2
                cell_center_y = offset_y + ((self.hw_rows - 1 - r_idx) * cell_size) + cell_size / 2 
                p1x, p1y = cell_center_x - (cell_size-padding)/2, cell_center_y - (cell_size-padding)/2
                p2x, p2y = cell_center_x + (cell_size-padding)/2, cell_center_y + (cell_size-padding)/2
                rect = Rectangle((p1x, p1y), (p2x, p2y), c='lightgrey', alpha=0.1)
                rect.lw(0) 
                if not self.points_array_checker.is_valid(c_idx, r_idx): rect.alpha(0) 
                self.cell_rect_actors[(r_idx, c_idx)] = rect 
                if hasattr(rect, 'actor') and rect.actor: self.renderer.AddActor(rect.actor)


    # In hardware_grid_visualizer_qt.py

    def update_arch_parameters(self, user_central_incisor_width=None, arch_params=None):
        logger.info("Updating arch parameters...")
        if user_central_incisor_width is not None:
            self.user_ci_width = user_central_incisor_width
            logger.info(f"  - Central Incisor Width set to: {self.user_ci_width} mm")
        if arch_params is not None:
            self.arch_params.update(arch_params)
            logger.info(f"  - Arch shape parameters updated.")

        self._rebuild_arch_overlay_actors() # This single call now handles removing old and adding new
        self._map_cells_to_segments() # Re-map cells after arch is updated
        
        if self.parent_plotter and self.parent_plotter.qt_widget:
            self.parent_plotter.qt_widget.Render()


    def _calculate_arch_points(self):
        """
        Generates two equidistant, U-shaped curves whose ends connect to the respective
        outer and inner top corners of the valid grid area.
        """
        # 1. Find the bounding box of the VISIBLE/VALID grid cells (remains the same)
        cell_size = 0.25
        grid_offset_x = -(self.hw_cols * cell_size) / 2
        grid_offset_y = -(self.hw_rows * cell_size) / 2
        
        valid_x_coords, valid_y_coords = [], []
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                if self.points_array_checker.is_valid(c_idx, r_idx):
                    cell_center_x = grid_offset_x + (c_idx * cell_size) + cell_size / 2
                    cell_center_y = grid_offset_y + ((self.hw_rows - 1 - r_idx) * cell_size) + cell_size / 2
                    valid_x_coords.append(cell_center_x)
                    valid_y_coords.append(cell_center_y)

        if not valid_x_coords:
            logger.warning("No valid grid cells found to draw arch on.")
            return [], [], []

        min_x_grid, max_x_grid = min(valid_x_coords), max(valid_x_coords)
        min_y_grid, max_y_grid = min(valid_y_coords), max(valid_y_coords)
        valid_grid_width = max_x_grid - min_x_grid
        valid_grid_height = max_y_grid - min_y_grid
        grid_center_x = (min_x_grid + max_x_grid) / 2

        self.valid_grid_bounds = {
            'min_x': min_x_grid, 'max_x': max_x_grid,
            'min_y': min_y_grid, 'max_y': max_y_grid,
            'center_x': grid_center_x, 'center_y': (min_y_grid + max_y_grid) / 2
        }
        
        # 2. Define Arch Dimensions (remains the same)
        arch_a = (valid_grid_width / 2.0) * self.arch_params['width_scale']
        arch_b = arch_a * self.arch_params['depth_scale']
        
        # 3. Define the BASE arch curve function (remains the same)
        def get_base_y(x_relative, semi_major_a, semi_minor_b):
            flat_zone_half_width = semi_major_a * self.arch_params['anterior_flatness']
            if abs(x_relative) <= flat_zone_half_width: return semi_minor_b
            else:
                curve_zone_width = semi_major_a - flat_zone_half_width
                if curve_zone_width <= 0: return semi_minor_b
                x_normalized_for_curve = (abs(x_relative) - flat_zone_half_width) / curve_zone_width
                return semi_minor_b * np.sqrt(max(0, 1 - x_normalized_for_curve**2))

        # 4. Generate points and normals for the main U-shaped part of the arch (remains the same)
        points_outer, points_inner = [], []
        num_spline_points = 200
        base_outer_curve_points = []
        x_spline_start = -arch_a; x_spline_end = arch_a
        for i in range(num_spline_points + 1):
            x = x_spline_start + (x_spline_end - x_spline_start) * i / num_spline_points
            y_base = get_base_y(x, arch_a, arch_b)
            base_outer_curve_points.append(np.array([x, y_base]))
        normals = []
        for i in range(len(base_outer_curve_points)):
            if i == 0: tangent = base_outer_curve_points[i+1] - base_outer_curve_points[i]
            elif i == len(base_outer_curve_points) - 1: tangent = base_outer_curve_points[i] - base_outer_curve_points[i-1]
            else: tangent = base_outer_curve_points[i+1] - base_outer_curve_points[i-1]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 0: tangent /= tangent_norm
            normal = np.array([-tangent[1], tangent[0]])
            if normal[1] < 0: normal *= -1
            normals.append(normal)

        # 5. Create final outer and inner curves by positioning and offsetting (remains the same)
        y_shift = max_y_grid - arch_b + (valid_grid_height * self.arch_params['vertical_offset_factor'])
        offset_distance = self.arch_params['lingual_curve_offset']
        for i in range(len(base_outer_curve_points)):
            point = base_outer_curve_points[i]
            normal = normals[i]
            final_outer_point = np.array([point[0] + grid_center_x, -point[1] + y_shift])
            points_outer.append(final_outer_point.tolist())
            inner_point_base = point + normal * offset_distance
            final_inner_point = np.array([inner_point_base[0] + grid_center_x, -inner_point_base[1] + y_shift])
            points_inner.append(final_inner_point.tolist())

        # 6. Calculate interproximal lines (remains the same)
        interproximal_lines = []
        total_segments = 14
        interproximal_lines.append( (points_outer[0], points_inner[0]) )
        for i in range(1, total_segments):
            point_index = int(i * (num_spline_points / total_segments))
            if point_index >= len(points_outer): continue
            p_outer = points_outer[point_index]
            p_inner = points_inner[point_index]
            interproximal_lines.append( (p_outer, p_inner) )
        interproximal_lines.append( (points_outer[-1], points_inner[-1]) )
        
        # *** 7. CORRECTED: REPLACE ENDPOINTS TO CONNECT TO SEPARATE CORNERS ***
        # Define the target corners for the OUTER curve
        target_top_left_outer = [min_x_grid, max_y_grid]
        target_top_right_outer = [max_x_grid, max_y_grid]

        # Define the target corners for the INNER curve, inset by the offset distance
        # This assumes the ends of the arch are mostly vertical
        target_top_left_inner = [min_x_grid + offset_distance, max_y_grid]
        target_top_right_inner = [max_x_grid - offset_distance, max_y_grid]
        
        # Replace the first point of each curve with its respective top-left corner
        points_outer[0] = target_top_left_inner
        points_inner[0] = target_top_left_outer

        # Replace the last point of each curve with its respective top-right corner
        points_outer[-1] = target_top_right_inner
        points_inner[-1] = target_top_right_outer
        # *** END CORRECTION ***
        
        return interproximal_lines, points_outer, points_inner

    def _create_default_segment_boundaries(self):
        # ... (This function remains unchanged) ...
        self.segment_boundary_lines.clear()
        self.arch_boundary_splines.clear()
        interprox_line_coords, outer_spline_points, inner_spline_points = self._calculate_arch_points()
        for p1, p2 in interprox_line_coords:
            line = Line(p1, p2, c='royalblue', lw=2, alpha=0.7)
            self.segment_boundary_lines.append(line)
        if outer_spline_points:
            outer_spline = Spline(outer_spline_points); outer_spline.color('darkblue').linewidth(2).alpha(0.7)
            self.arch_boundary_splines.append(outer_spline)
        if inner_spline_points:
            inner_spline = Spline(inner_spline_points); inner_spline.color('darkblue').linewidth(2).alpha(0.7)
            self.arch_boundary_splines.append(inner_spline)
        logger.info(f"Created {len(self.segment_boundary_lines)} segment boundary lines and {len(self.arch_boundary_splines)} arch splines.")

    def _add_segment_actors_to_renderer(self):
        # ... (This function remains unchanged) ...
        all_segment_actors = self.segment_boundary_lines + self.arch_boundary_splines
        for actor in all_segment_actors:
            if hasattr(actor, 'actor') and actor.actor: self.renderer.AddActor(actor.actor)
            else: self.renderer.AddActor(actor)
    
    def _value_to_color_hardware(self, value, sensitivity=1):
        # ... (This function remains unchanged) ...
        mapped_value = (value / sensitivity * 255) // self.max_force_for_scaling; mapped_value = min(255,max(0,int(mapped_value)))
        r,g,b = 200,200,200 
        if mapped_value>204:r=255;g=max(0,int(150-((mapped_value-204)*150/51)));b=0
        elif mapped_value>140:r=int(139+((mapped_value-140)*116/64));g=int((mapped_value-140)*150/64);b=0
        elif mapped_value>76:g=int(255-((mapped_value-76)*155/64));r=int(((mapped_value-76)/64)*100);b=0
        elif mapped_value>12:r=0;g=int(255-((mapped_value-12)*155/64));b=int(100-((mapped_value-12)*50/64))
        return (r/255.0,g/255.0,b/255.0)

    # In hardware_grid_visualizer_qt.py, add this new method

    # In hardware_grid_visualizer_qt.py

    def _map_cells_to_segments(self):
        """
        Defines the 14 segment polygons from the arch lines and maps each valid grid cell
        to a segment. Populates self.segment_cell_map.
        Corrected to use .points as a property.
        """
        if not self.segment_boundary_lines or not self.arch_boundary_splines:
            logger.warning("Cannot map cells to segments: Arch boundaries not yet created.")
            return

        # 1. Define the 14 segment polygons from the lines
        self.segment_polygons = []
        
        # Sort the lines by the X-coordinate of their center of mass.
        sorted_lines = sorted(
            self.segment_boundary_lines, 
            key=lambda line: line.center_of_mass()[0]
        )

        for i in range(len(sorted_lines) - 1): # For N lines, this creates N-1 polygons
            left_line = sorted_lines[i]
            right_line = sorted_lines[i+1]
            
            # *** CORRECTED: Access .points as a property, not a method ***
            left_line_points = left_line.points
            right_line_points = right_line.points
            # *** END CORRECTION ***

            # Ensure the points are valid before proceeding
            if left_line_points is None or len(left_line_points) < 2 or \
               right_line_points is None or len(right_line_points) < 2:
                logger.warning(f"Skipping polygon creation due to invalid line points at index {i}.")
                continue

            # Determine which point is "outer" (higher Y value) and "inner" (lower Y value)
            if left_line_points[0][1] > left_line_points[1][1]:
                p_left_outer, p_left_inner = left_line_points[0], left_line_points[1]
            else:
                p_left_inner, p_left_outer = left_line_points[0], left_line_points[1]

            if right_line_points[0][1] > right_line_points[1][1]:
                p_right_outer, p_right_inner = right_line_points[0], right_line_points[1]
            else:
                p_right_inner, p_right_outer = right_line_points[0], right_line_points[1]

            # The 4 corners of the segment polygon
            poly = np.array([p_left_outer, p_right_outer, p_right_inner, p_left_inner])
            self.segment_polygons.append(poly)

        logger.info(f"Defined {len(self.segment_polygons)} segment polygons for mapping.")
        if len(self.segment_polygons) == 0: return

        # 2. Map valid grid cells to these polygons
        self.segment_cell_map = {i: [] for i in range(len(self.segment_polygons))}
        
        cell_size = 0.25
        grid_offset_x = -(self.hw_cols * cell_size) / 2
        grid_offset_y = -(self.hw_rows * cell_size) / 2
        
        from matplotlib.path import Path

        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                if self.points_array_checker.is_valid(c_idx, r_idx):
                    cell_center_x = grid_offset_x + (c_idx * cell_size) + cell_size / 2
                    cell_center_y = grid_offset_y + ((self.hw_rows - 1 - r_idx) * cell_size) + cell_size / 2
                    cell_point = (cell_center_x, cell_center_y)

                    for seg_idx, polygon_points in enumerate(self.segment_polygons):
                        # Use the 2D coordinates (X, Y) for the path
                        path = Path(polygon_points[:, :2])
                        if path.contains_point(cell_point):
                            self.segment_cell_map[seg_idx].append((r_idx, c_idx))
                            break 

        cell_counts = [len(cells) for cells in self.segment_cell_map.values()]
        logger.info(f"Cell mapping complete. Cells per segment: {cell_counts}")

    # In hardware_grid_visualizer_qt.py

    # In hardware_grid_visualizer_qt.py

    # In hardware_grid_visualizer_qt.py

    # In hardware_grid_visualizer_qt.py

    def render_grid_view(self, timestamp, hardware_data_flat_array, sensitivity=1):
        if not self.renderer or not self.parent_plotter: return
        self.parent_plotter.at(self.renderer_index)
        
        # Update time text
        if self.time_text_actor: self.renderer.RemoveActor(self.time_text_actor.actor)
        self.time_text_actor = Text2D(f"HW Grid - T: {timestamp:.1f}s", pos="bottom-left", c='k', s=0.7)
        self.renderer.AddActor(self.time_text_actor.actor)

        # Handle case with no data
        if hardware_data_flat_array is None:
            for rect_actor in self.cell_rect_actors.values(): rect_actor.alpha(0.05)
            for txt_actor in self.segment_text_actors: txt_actor.off()
            return

        # Map flat data array to a 2D grid for easy lookup
        force_grid = np.full((self.hw_rows, self.hw_cols), 0.0)
        data_idx = 0
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                if self.points_array_checker.is_valid(c_idx, r_idx):
                    if data_idx < len(hardware_data_flat_array):
                        force_grid[r_idx, c_idx] = hardware_data_flat_array[data_idx]
                        data_idx += 1

        # Update the color of the grid cells
        for (r_idx, c_idx), rect_actor in self.cell_rect_actors.items():
            if rect_actor.alpha() > 0:
                value = force_grid[r_idx, c_idx]
                if value > 0:
                    rect_actor.color(self._value_to_color_hardware(value, sensitivity)).alpha(1.0 if value > 5 else 0.2)
                else:
                    rect_actor.color('lightgrey').alpha(0.1)

        # Calculate and Display Force Percentages with "Outer Sunburst" Layout
        total_force = np.sum(hardware_data_flat_array)
        
        if total_force < 1.0:
            for txt_actor in self.segment_text_actors: txt_actor.off()
            return
        else:
            for txt_actor in self.segment_text_actors: txt_actor.on()
            
        # We need the points of the boundary splines to do this correctly
        if not self.arch_boundary_splines or len(self.arch_boundary_splines) < 2:
            logger.warning("Arch boundary splines not ready for text placement.")
            return

        outer_spline = self.arch_boundary_splines[0]
        inner_spline = self.arch_boundary_splines[1]
        
        # Ensure they are in the correct order (outer should have lower Y values)
        if outer_spline.center_of_mass()[1] > inner_spline.center_of_mass()[1]:
            outer_spline, inner_spline = inner_spline, outer_spline

        # *** CORRECTED: Access .points as a property ***
        outer_points = outer_spline.points
        inner_points = inner_spline.points
        # *** END CORRECTION ***

        if outer_points is None or inner_points is None:
            logger.warning("Spline points are None, cannot place text.")
            return

        num_spline_points = len(outer_points) -1
        # The number of segments is based on the number of text actors created
        total_segments = len(self.segment_text_actors)

        for seg_idx in range(total_segments):
            if seg_idx >= len(self.segment_polygons): continue # Ensure we don't go out of bounds for polygon map

            segment_force = 0.0
            for r_idx, c_idx in self.segment_cell_map.get(seg_idx, []):
                segment_force += force_grid[r_idx, c_idx]
            
            percentage = (segment_force / total_force * 100.0) if total_force > 0 else 0.0
            
            text_actor = self.segment_text_actors[seg_idx]
            
            if percentage < 0.5:
                text_actor.off()
            else:
                text_actor.on()
                text_string = f"{percentage:.1f}%"
                
                if text_actor.text() != text_string:
                    text_actor.text(text_string)
                
                # "Outer Sunburst" POSITIONING LOGIC
                start_idx = int(seg_idx * (num_spline_points / total_segments))
                end_idx = int((seg_idx + 1) * (num_spline_points / total_segments))

                # Add a check to prevent index out of bounds
                if start_idx >= len(outer_points) or end_idx >= len(outer_points): continue

                mid_point_outer = np.mean(outer_points[start_idx:end_idx+1], axis=0)
                mid_point_inner = np.mean(inner_points[start_idx:end_idx+1], axis=0)
                
                direction_vector = mid_point_outer - mid_point_inner
                norm = np.linalg.norm(direction_vector)
                if norm > 0:
                    direction_vector /= norm

                offset_distance = 1.0 
                text_pos = mid_point_outer + direction_vector * offset_distance
                
                text_actor.pos(text_pos[0], text_pos[1], 0.2)


    def animate(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1):
        self.render_grid_view(timestamp_to_render, hardware_data_for_timestamp, sensitivity)

    def get_frame_as_array(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1):
        if not self.renderer or not self.parent_plotter: return None
        self.parent_plotter.at(self.renderer_index)
        self.render_grid_view(timestamp_to_render, hardware_data_for_timestamp, sensitivity)
        return None 
# --- END OF FILE hardware_grid_visualizer_qt.py ---