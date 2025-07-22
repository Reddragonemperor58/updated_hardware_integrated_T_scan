# --- START OF FILE hardware_grid_visualizer_qt.py ---
import numpy as np
from vedo import Text2D, Rectangle, colors, Line, Spline, Points, Text3D, Polygon, Mesh
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
        self.segment_cell_map = {}

        # All visual overlay actors are stored here
        self.presentation_segment_actors = []
        self.segment_text_actors = []
        
        self.user_ci_width = user_central_incisor_width
        self.mapping_arch_params = { 'grid_cell_size_mm': 0.7, 'width_scale': 0.4, 'depth_scale': 0.99, 'anterior_flatness': 0.2, 'lingual_curve_offset': 3.5, 'vertical_offset_factor': -0.09 }
        self.pseudo_arch_params = {
            'center_x': 0.0, 
            'center_y': -2.0,            # Lower the center for a wider top
            'inner_radius': 5.0,         # Increase inner radius
            'outer_radius': 20.5,         # Increase outer radius for thicker teeth
            'start_angle': 195,          # Widen the arch start angle
            'end_angle': 345,            # Widen the arch end angle
            'gap_angle': 1.5             # Increase the gap between segments
        }
        if not self.renderer: logger.error(f"HwGridViz (R{self.renderer_index}): Renderer not provided."); return

    def setup_scene(self):
        if not self.renderer or not self.parent_plotter: return
        logger.info(f"HwGridViz (R{self.renderer_index}): Setting up scene with presentation arch...")
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera
        cam.ParallelProjectionOn()

        self._create_and_add_grid_rects_once() # Creates invisible rects
        self._rebuild_all_overlays() # This one function does it all
        
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

    # def setup_scene(self):
    #     if not self.renderer or not self.parent_plotter: return
    #     logger.info(f"HwGridViz (R{self.renderer_index}): Setting up scene with segments...")
    #     self.parent_plotter.at(self.renderer_index)
    #     cam = self.parent_plotter.camera
    #     cam.ParallelProjectionOn()

    #     self._create_and_add_grid_rects_once()
        
    #     # This one function now does all the work of creating lines, splines, and text
    #     self._rebuild_arch_overlay_actors() 
        
    #     # Map cells after the polygons have been defined inside _rebuild_arch_overlay_actors
    #     self._map_cells_to_segments()
        
    #     # Camera setup
    #     cell_render_size = 0.25 
    #     grid_render_height = self.hw_rows * cell_render_size
    #     cam.SetFocalPoint(0,0,0)
    #     cam.SetPosition(0, 0, 20)
    #     cam.SetViewUp(0, 1, 0)     
    #     cam.SetParallelScale(grid_render_height / 1.7) 
    #     self.renderer.ResetCamera()
    #     self.renderer.ResetCameraClippingRange()
    #     self.renderer.SetBackground(0.93, 0.93, 0.97)
    #     logger.info(f"HwGridViz (R{self.renderer_index}): Scene setup complete.")


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

    def update_arch_parameters(self, user_central_incisor_width=None):
            logger.info("Updating arch parameters for data mapping...")
            if user_central_incisor_width is not None:
                self.user_ci_width = user_central_incisor_width
            
            # When parameters change, just rebuild everything
            self._rebuild_all_overlays()
            
            if self.parent_plotter and self.parent_plotter.qt_widget:
                self.parent_plotter.qt_widget.Render()

    def _rebuild_all_overlays(self):
        """
        The master function to create mapping geometry, map cells, and create presentation geometry.
        """
        # 1. GENERATE MAPPING GEOMETRY (in memory, not drawn)
        mapping_polygons = self._calculate_mapping_polygons()
        if not mapping_polygons: return

        # 2. MAP CELLS using the mapping geometry
        self._map_cells_to_segments(mapping_polygons)

        # 3. CREATE PRESENTATION ACTORS (the visible arch)
        self._create_presentation_actors(len(mapping_polygons))

    def _calculate_mapping_polygons(self):
        """
        Calculates and returns a list of 14 raw polygon vertex arrays for the data mapping.
        Does NOT create any vedo actors.
        """
        # Find Bounding Box of valid cells
        cell_size=0.25; grid_offset_x=-(self.hw_cols*cell_size)/2; grid_offset_y=-(self.hw_rows*cell_size)/2
        valid_x, valid_y = [], []
        for r in range(self.hw_rows):
            for c in range(self.hw_cols):
                if self.points_array_checker.is_valid(c,r):
                    cx = grid_offset_x + (c*cell_size)+cell_size/2; cy = grid_offset_y + ((self.hw_rows-1-r)*cell_size)+cell_size/2
                    valid_x.append(cx); valid_y.append(cy)
        if not valid_x: return []
        min_x, max_x = min(valid_x), max(valid_x); min_y, max_y = min(valid_y), max(valid_y)
        valid_w = max_x - min_x; valid_h = max_y - min_y
        grid_cx = (min_x + max_x) / 2
        
        # Define Arch Shape
        p = self.mapping_arch_params
        arch_a = (valid_w/2.0)*p['width_scale']; arch_b = arch_a*p['depth_scale']
        def get_base_y(x_rel):
            flat_half_w = arch_a*p['anterior_flatness']
            if abs(x_rel) <= flat_half_w: return arch_b
            else:
                curve_w = arch_a - flat_half_w; 
                if curve_w<=0: return arch_b
                x_norm = (abs(x_rel)-flat_half_w)/curve_w; 
                return arch_b * np.sqrt(max(0,1-x_norm**2))

        # Generate Base Curve and Normals
        num_pts = 200; base_curve_pts = []
        x_start, x_end = -arch_a, arch_a
        for i in range(num_pts + 1):
            x = x_start + (x_end-x_start)*i/num_pts
            base_curve_pts.append(np.array([x, get_base_y(x)]))
        normals = []
        for i in range(len(base_curve_pts)):
            if i==0: t=base_curve_pts[i+1]-base_curve_pts[i]
            elif i==len(base_curve_pts)-1: t=base_curve_pts[i]-base_curve_pts[i-1]
            else: t=base_curve_pts[i+1]-base_curve_pts[i-1]
            if np.linalg.norm(t)>0: t/=np.linalg.norm(t)
            n=np.array([-t[1],t[0]]); 
            if n[1]<0: n*=-1
            normals.append(n)

        # Generate Final Curve Points (Outer and Inner)
        points_outer, points_inner = [], []
        y_shift = max_y - arch_b + (valid_h * p['vertical_offset_factor'])
        for i in range(len(base_curve_pts)):
            pt = base_curve_pts[i]; n = normals[i]
            points_outer.append(np.array([pt[0]+grid_cx, -pt[1]+y_shift]))
            inner_pt = pt + n * p['lingual_curve_offset']
            points_inner.append(np.array([inner_pt[0]+grid_cx, -inner_pt[1]+y_shift]))
        
        # Generate 15 dividing lines to create 14 segments
        num_segments = 14
        dividing_lines = []
        for i in range(num_segments + 1):
            idx = int(i * (num_pts / num_segments))
            if idx >= len(points_outer): idx = len(points_outer) - 1
            dividing_lines.append( (points_outer[idx], points_inner[idx]) )
        
        # Create 14 Polygon vertex arrays
        polygons = []
        for i in range(len(dividing_lines) - 1):
            p_left_outer, p_left_inner = dividing_lines[i]
            p_right_outer, p_right_inner = dividing_lines[i+1]
            polygons.append(np.array([p_left_outer, p_right_outer, p_right_inner, p_left_inner]))
        
        return polygons

    def _map_cells_to_segments(self, mapping_polygons):
        self.mapping_polygons = mapping_polygons
        if not self.mapping_polygons: return
        self.segment_cell_map = {i: [] for i in range(len(self.mapping_polygons))}
        cell_size = 0.25; grid_offset_x = -(self.hw_cols * cell_size) / 2; grid_offset_y = -(self.hw_rows * cell_size) / 2
        from matplotlib.path import Path
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                if self.points_array_checker.is_valid(c_idx, r_idx):
                    cell_center = (grid_offset_x + (c_idx + 0.5) * cell_size, grid_offset_y + ((self.hw_rows - 1 - r_idx) + 0.5) * cell_size)
                    for seg_idx, polygon_points in enumerate(self.mapping_polygons):
                        path = Path(polygon_points[:, :2])
                        if path.contains_point(cell_center):
                            self.segment_cell_map[seg_idx].append((r_idx, c_idx)); break
        cell_counts = [len(cells) for cells in self.segment_cell_map.values()]
        logger.info(f"Cell mapping complete. Cells per segment: {cell_counts}")

    def _create_presentation_actors(self, num_segments):
        """Creates the visible, schematic arch with a specified number of segments."""
        p = self.pseudo_arch_params
        total_angle = p['end_angle'] - p['start_angle']
        total_gap_angle = (num_segments - 1) * p['gap_angle']
        angle_per_segment = (total_angle - total_gap_angle) / num_segments
        text_height = (p['outer_radius'] - p['inner_radius']) * 0.1
        text_color = 'black' # Change from white to black
        font_name = "VictorMono-Bold"

        current_angle = p['start_angle']
        for i in range(num_segments):
            start_rad = np.radians(current_angle); end_rad = np.radians(current_angle + angle_per_segment)
            p1 = [p['center_x'] + p['inner_radius'] * np.cos(start_rad), p['center_y'] + p['inner_radius'] * np.sin(start_rad)]
            p2 = [p['center_x'] + p['outer_radius'] * np.cos(start_rad), p['center_y'] + p['outer_radius'] * np.sin(start_rad)]
            p3 = [p['center_x'] + p['outer_radius'] * np.cos(end_rad),   p['center_y'] + p['outer_radius'] * np.sin(end_rad)]
            p4 = [p['center_x'] + p['inner_radius'] * np.cos(end_rad),   p['center_y'] + p['inner_radius'] * np.sin(end_rad)]
            
            poly_actor = Mesh([[p1, p2, p3, p4], [[0, 1, 2, 3]]]).z(0.05).color('grey').alpha(0.8).linewidth(1).linecolor('black')
            self.presentation_segment_actors.append(poly_actor)
            
            text_actor = Text3D(f"{i+1}", s=text_height, justify='center', c=text_color, font=font_name); text_actor.follow_camera()
            self.segment_text_actors.append(text_actor)
            current_angle += angle_per_segment + p['gap_angle']

        if self.parent_plotter:
            self.parent_plotter.add(self.presentation_segment_actors + self.segment_text_actors)
        logger.info(f"Created {len(self.presentation_segment_actors)} presentation segment actors.")


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
                rect = Rectangle((p1x, p1y), (p2x, p2y), c='lightgrey', alpha=0.0) # MAKE GRID INVISIBLE
                rect.lw(0) 
                # The is_valid check is no longer needed for visibility, but doesn't hurt
                if not self.points_array_checker.is_valid(c_idx, r_idx): rect.alpha(0) 
                
                self.cell_rect_actors[(r_idx, c_idx)] = rect 
                if hasattr(rect, 'actor') and rect.actor: self.renderer.AddActor(rect.actor)


    
    def _update_mapping_geometry(self):
        """Calculates the mapping polygons and then maps cells to them."""
        # 1. Generate the points for the real arch overlay
        interprox_coords, _, _ = self._calculate_arch_points()
        
        # 2. Define the mapping polygons from these lines
        self.mapping_polygons.clear()
        sorted_lines_pts = sorted(interprox_coords, key=lambda p: (p[0][0] + p[1][0]) / 2)
        for i in range(len(sorted_lines_pts) - 1):
            p_left_outer, p_left_inner = sorted_lines_pts[i]
            p_right_outer, p_right_inner = sorted_lines_pts[i+1]
            # Ensure correct winding order for polygon
            poly_points = np.array([p_left_outer, p_right_outer, p_right_inner, p_left_inner])
            self.mapping_polygons.append(poly_points)
        
        # 3. Map the grid cells using these new polygons
        self._map_cells_to_segments()

    

    def _calculate_arch_points(self):
        """
        Generates geometry for the INVISIBLE mapping arch, ensuring it creates
        exactly 15 lines to define 14 tooth segments.
        """
        # ... (Steps 1, 2, 3 for finding bounds and defining the curve are the same) ...
        # ...

        # 4. Generate the points for the two parallel curves (outer and inner)
        points_outer, points_inner = [], []
        num_spline_points = 200
        # ... (The normal vector calculation and curve generation is the same as before) ...
        # ... (This populates points_outer and points_inner lists) ...

        # 5. Calculate exactly 15 interproximal lines to define 14 segments
        interproximal_lines_coords = []
        num_segments = 14
        
        # This loop will run 15 times (for i=0 to 14), creating 15 lines
        # These lines will be the divisions between the 14 segments, plus the two ends.
        for i in range(num_segments + 1):
            point_index = int(i * (num_spline_points / num_segments))
            if point_index >= len(points_outer): point_index = len(points_outer) - 1
            
            p_outer = points_outer[point_index]
            p_inner = points_inner[point_index]
            
            interproximal_lines_coords.append( (p_outer, p_inner) )

        return interproximal_lines_coords, points_outer, points_inner

    

    # def _create_default_segment_boundaries(self):
    #     # ... (This function remains unchanged) ...
    #     self.segment_boundary_lines.clear()
    #     self.arch_boundary_splines.clear()
    #     interprox_line_coords, outer_spline_points, inner_spline_points = self._calculate_arch_points()
    #     for p1, p2 in interprox_line_coords:
    #         line = Line(p1, p2, c='royalblue', lw=2, alpha=0.7)
    #         self.segment_boundary_lines.append(line)
    #     if outer_spline_points:
    #         outer_spline = Spline(outer_spline_points); outer_spline.color('darkblue').linewidth(2).alpha(0.7)
    #         self.arch_boundary_splines.append(outer_spline)
    #     if inner_spline_points:
    #         inner_spline = Spline(inner_spline_points); inner_spline.color('darkblue').linewidth(2).alpha(0.7)
    #         self.arch_boundary_splines.append(inner_spline)
    #     logger.info(f"Created {len(self.segment_boundary_lines)} segment boundary lines and {len(self.arch_boundary_splines)} arch splines.")

    # def _add_segment_actors_to_renderer(self):
    #     # ... (This function remains unchanged) ...
    #     all_segment_actors = self.segment_boundary_lines + self.arch_boundary_splines
    #     for actor in all_segment_actors:
    #         if hasattr(actor, 'actor') and actor.actor: self.renderer.AddActor(actor.actor)
    #         else: self.renderer.AddActor(actor)
    
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

    # In hardware_grid_visualizer_qt.py

    # In hardware_grid_visualizer_qt.py


    # In hardware_grid_visualizer_qt.py

    def render_grid_view(self, timestamp, hardware_data_flat_array, sensitivity=1):
        if not self.renderer or not self.parent_plotter: return
        self.parent_plotter.at(self.renderer_index)
        if self.time_text_actor: self.renderer.RemoveActor(self.time_text_actor.actor)
        self.time_text_actor = Text2D(f"HW Grid - T: {timestamp:.1f}s", pos="bottom-left", c='k', s=0.7)
        self.renderer.AddActor(self.time_text_actor.actor)
        if hardware_data_flat_array is None:
            for txt_actor in self.segment_text_actors: txt_actor.off()
            for poly_actor in self.presentation_segment_actors: poly_actor.color('grey')
            return
        force_grid = np.full((self.hw_rows, self.hw_cols), 0.0); data_idx=0
        for r in range(self.hw_rows):
            for c in range(self.hw_cols):
                if self.points_array_checker.is_valid(c,r) and data_idx < len(hardware_data_flat_array):
                    force_grid[r, c] = hardware_data_flat_array[data_idx]; data_idx += 1
        total_force = np.sum(hardware_data_flat_array)
        if total_force < 1.0:
            for txt_actor in self.segment_text_actors: txt_actor.off()
            for poly_actor in self.presentation_segment_actors: poly_actor.color('grey')
        else:
            for txt_actor in self.segment_text_actors: txt_actor.on()
        for seg_idx in range(len(self.presentation_segment_actors)):
            if seg_idx >= len(self.segment_cell_map): continue
            segment_force, avg_force_val = 0.0, 0.0
            cells_in_segment = self.segment_cell_map.get(seg_idx, [])
            if cells_in_segment:
                for r_idx, c_idx in cells_in_segment: segment_force += force_grid[r_idx, c_idx]
                avg_force_val = segment_force / len(cells_in_segment)
            poly_actor = self.presentation_segment_actors[seg_idx]
            poly_actor.color(self._value_to_color_hardware(avg_force_val, sensitivity))
            percentage = (segment_force / total_force * 100.0) if total_force > 0 else 0.0
            text_actor = self.segment_text_actors[seg_idx]
            if percentage < 0.5: text_actor.off()
            else:
                text_actor.on()
                text_string = f"{percentage:.1f}%"
                if text_actor.text() != text_string: text_actor.text(text_string)
                center_point = poly_actor.center_of_mass()
                text_actor.pos(center_point[0], center_point[1], 0.2)

    def animate(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1):
        self.render_grid_view(timestamp_to_render, hardware_data_for_timestamp, sensitivity)

    def get_frame_as_array(self, timestamp_to_render, hardware_data_for_timestamp=None, sensitivity=1):
        if not self.renderer or not self.parent_plotter: return None
        self.parent_plotter.at(self.renderer_index)
        self.render_grid_view(timestamp_to_render, hardware_data_for_timestamp, sensitivity)
        return None 
# --- END OF FILE hardware_grid_visualizer_qt.py ---