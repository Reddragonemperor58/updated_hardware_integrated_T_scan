import numpy as np
from vedo import Text2D, Rectangle, colors, Line, Spline, Points, Text3D, Mesh, Assembly
import logging
from points_array import PointsArray 
import vtk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareGridVisualizerQt:
    # ... (TOOTH_WIDTH_PROPORTIONS and TOOTH_ORDER are unchanged) ...
    TOOTH_WIDTH_PROPORTIONS = { "Central Incisor": 1.00, "Lateral Incisor": 0.78, "Canine": 0.89, "1st Premolar": 0.75, "2nd Premolar": 0.70, "1st Molar": 1.20, "2nd Molar": 1.10 }
    TOOTH_ORDER = ["Central Incisor", "Lateral Incisor", "Canine", "1st Premolar", "2nd Premolar", "1st Molar", "2nd Molar"]
    
    def __init__(self, processor_instance, parent_plotter_instance, renderer_index, user_central_incisor_width=9.0):
        self.processor = processor_instance 
        self.parent_plotter = parent_plotter_instance
        self.renderer_index = renderer_index
        self.renderer = parent_plotter_instance.renderers[renderer_index]
        self.hw_rows = 44; self.hw_cols = 52
        self.points_array_checker = PointsArray()
        self.max_force_for_scaling = 1000.0 
        
        self.time_text_actor = None
        self.segment_cell_map = {}
        self.mapping_polygons = [] # The invisible polygons for data mapping

        # --- All Presentation Actors ---
        self.presentation_actors = {} 
        self.summary_bar_left = None
        self.summary_bar_right = None
        self.summary_text_left = None
        self.summary_text_right = None
        
        self.user_ci_width = user_central_incisor_width
        self.mapping_arch_params = { 'grid_cell_size_mm': 0.7, 'width_scale': 0.4, 'depth_scale': 0.99, 'anterior_flatness': 0.2, 'lingual_curve_offset': 3.5, 'vertical_offset_factor': -0.09 }

        if not self.renderer: logger.error(f"HwGridViz (R{self.renderer_index}): Renderer not provided."); return

    # In hardware_grid_visualizer_qt.py

    def setup_scene(self):
        if not self.renderer or not self.parent_plotter: return
        logger.info(f"HwGridViz (R{self.renderer_index}): Setting up T-Scan style scene...")
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera
        cam.ParallelProjectionOn()
        
        # We no longer create the visible grid rects
        
        self._rebuild_all_overlays()
        
        # Adjust camera to fit the new layout
        all_actors = [d['shape'] for d in self.presentation_actors.values()]
        if self.summary_bar_left: all_actors.extend([self.summary_bar_left, self.summary_bar_right])
        
        scene_assembly = Assembly(all_actors)
        bounds = scene_assembly.bounds()

        # *** CORRECTED BOUNDS CHECK AND CENTER CALCULATION ***
        if bounds is not None and len(bounds) == 6:
            # Calculate center from the bounds array
            center_x = (bounds[0] + bounds[1]) / 2.0
            center_y = (bounds[2] + bounds[3]) / 2.0
            center_z = (bounds[4] + bounds[5]) / 2.0
            center = (center_x, center_y, center_z)
            
            y_height = bounds[3] - bounds[2] # ymax - ymin
            
            cam.SetParallelScale(y_height * 0.6) # Zoom to fit height with some padding
            cam.SetFocalPoint(center)
            cam.SetPosition(center[0], center[1], center[2] + 30) # Look from the front, further back
        else: # Fallback if no actors were created
            cam.SetParallelScale(10)
            cam.SetFocalPoint(0,0,0)
            cam.SetPosition(0,0,20)
        # *** END CORRECTION ***

        cam.SetViewUp(0, 1, 0)
        self.renderer.ResetCameraClippingRange()
        self.renderer.SetBackground(0.93, 0.93, 0.97)
        logger.info(f"HwGridViz (R{self.renderer_index}): Scene setup complete.")

    def update_arch_parameters(self, user_central_incisor_width=None):
        logger.info("Updating arch parameters for data mapping...")
        if user_central_incisor_width is not None: self.user_ci_width = user_central_incisor_width
        self._rebuild_all_overlays() # Rebuild everything when parameters change
        if self.parent_plotter and self.parent_plotter.qt_widget: self.parent_plotter.qt_widget.Render()

    def _rebuild_all_overlays(self):
        """Master function to create mapping geometry, map cells, and create presentation geometry."""
        self.mapping_polygons = self._calculate_mapping_polygons()
        if not self.mapping_polygons: return
        self._map_cells_to_segments()
        self._create_presentation_actors()

    def _calculate_mapping_polygons(self):
        # ... (This function is the one that produces the correct parallel curves for the data)
        # ... (It should be the same as the last version that worked for data mapping)
        # It MUST return exactly 14 polygon vertex arrays.
        # This simplified version ensures 14 polygons.
        cell_size=0.25; grid_offset_x=-(self.hw_cols*cell_size)/2; grid_offset_y=-(self.hw_rows*cell_size)/2
        valid_x, valid_y = [], []; centers = []
        for r in range(self.hw_rows):
            for c in range(self.hw_cols):
                if self.points_array_checker.is_valid(c,r):
                    cx = grid_offset_x + (c*cell_size)+cell_size/2; cy = grid_offset_y + ((self.hw_rows-1-r)*cell_size)+cell_size/2
                    valid_x.append(cx); valid_y.append(cy); centers.append((cx,cy))
        if not valid_x: return []
        min_x, max_x = min(valid_x), max(valid_x); min_y, max_y = min(valid_y), max(valid_y)
        valid_w = max_x - min_x; valid_h = max_y - min_y; grid_cx = (min_x + max_x) / 2
        p = self.mapping_arch_params
        arch_a = (valid_w/2.0)*p['width_scale']; arch_b = arch_a*p['depth_scale']
        def get_base_y(x_rel):
            flat_half_w = arch_a*p['anterior_flatness']
            if abs(x_rel) <= flat_half_w: return arch_b
            else:
                curve_w = arch_a-flat_half_w;
                if curve_w<=0: return arch_b
                x_norm = (abs(x_rel)-flat_half_w)/curve_w; return arch_b * np.sqrt(max(0,1-x_norm**2))
        num_pts = 200; base_curve_pts = []
        x_start, x_end = -arch_a, arch_a
        for i in range(num_pts+1):
            x = x_start+(x_end-x_start)*i/num_pts
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
        points_outer, points_inner = [], []
        y_shift = max_y - arch_b + (valid_h * p['vertical_offset_factor'])
        for i in range(len(base_curve_pts)):
            pt=base_curve_pts[i]; n=normals[i]
            points_outer.append(np.array([pt[0]+grid_cx, -pt[1]+y_shift]))
            inner_pt = pt + n * p['lingual_curve_offset']
            points_inner.append(np.array([inner_pt[0]+grid_cx, -inner_pt[1]+y_shift]))
        div_lines = []
        for i in range(14+1):
            idx = int(i * (num_pts/14));
            if idx>=len(points_outer): idx=len(points_outer)-1
            div_lines.append((points_outer[idx], points_inner[idx]))
        polygons = []
        for i in range(len(div_lines)-1):
            p_lo, p_li = div_lines[i]; p_ro, p_ri = div_lines[i+1]
            polygons.append(np.array([p_lo, p_ro, p_ri, p_li]))
        return polygons

    def _map_cells_to_segments(self):
        # ... (This function is the same, just using self.mapping_polygons) ...
        if not self.mapping_polygons: return
        self.segment_cell_map = {i:[] for i in range(len(self.mapping_polygons))}
        cell_size = 0.25; grid_offset_x = -(self.hw_cols * cell_size) / 2; grid_offset_y = -(self.hw_rows * cell_size) / 2
        from matplotlib.path import Path
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                if self.points_array_checker.is_valid(c_idx, r_idx):
                    cell_center = (grid_offset_x + (c_idx + 0.5) * cell_size, grid_offset_y + ((self.hw_rows - 1 - r_idx) + 0.5) * cell_size)
                    for seg_idx, polygon_points in enumerate(self.mapping_polygons):
                        path = Path(polygon_points[:, :2])
                        if path.contains_point(cell_center): self.segment_cell_map[seg_idx].append((r_idx, c_idx)); break
        cell_counts = [len(cells) for cells in self.segment_cell_map.values()]
        logger.info(f"Cell mapping complete. Cells per segment: {cell_counts}")

    # In hardware_grid_visualizer_qt.py

    def _create_presentation_actors(self):
        """
        Creates the visible, schematic arch with 16 individual tooth shapes
        using a robust, algorithmic layout.
        """
        if self.parent_plotter:
            all_old_actors = []
            for d in self.presentation_actors.values(): all_old_actors.extend(d.values())
            if self.summary_bar_left: all_old_actors.extend([self.summary_bar_left, self.summary_bar_right, self.summary_text_left, self.summary_text_right])
            self.parent_plotter.remove(all_old_actors)
        self.presentation_actors.clear()

        # --- ALGORITHMIC LAYOUT DEFINITION ---
        # Parameters for the main arch where the teeth are centered
        arch_center_x = 0
        arch_center_y = 1.0
        arch_radius = 8.0
        start_angle_deg = 210
        end_angle_deg = 330
        
        # Parameters for the tooth shapes and text
        base_size_x, base_size_y = 1.4, 1.6 # Width and Height of each tooth segment
        nr_text_size = 0.8
        pct_text_size = 0.6
        pct_text_offset = 1.2 # How far below the shape to place the percentage

        total_angle = end_angle_deg - start_angle_deg
        angle_step = total_angle / (16 - 1) # Calculate angle step for 16 teeth

        # Standard Universal Tooth Numbering (1-16 for top arch)
        for i in range(16):
            tooth_nr = i + 1
            
            # Calculate angle for this tooth
            angle_rad = np.radians(start_angle_deg + i * angle_step)
            
            # Calculate center position on the arch
            x = arch_center_x + arch_radius * np.cos(angle_rad)
            y = arch_center_y + arch_radius * np.sin(angle_rad)
            
            # Rotation angle for the tooth shape to be perpendicular to the arch
            rotation_deg = np.degrees(angle_rad) + 90
            
            # Create the tooth shape (a simple rectangle mesh)
            w = base_size_x / 2.0; h = base_size_y / 2.0
            vertices = [[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]]
            faces = [[0, 1, 2, 3]]
            shape = Mesh([vertices, faces])
            shape.pos(x, y, 0).rotate_z(rotation_deg).color('grey').alpha(0.8).linewidth(1).linecolor('black')
            
            # Create the tooth number text, positioned in the center of the shape
            nr_text = Text3D(f"{tooth_nr}", s=nr_text_size, justify='center', c='black')
            nr_text.pos(x, y, 0.1).follow_camera()
            
            # Create the percentage text, positioned below the shape
            pct_text = Text3D("0.0%", s=pct_text_size, justify='center', c='black')
            # Calculate the position below the shape along its rotated axis
            offset_vec = np.array([0, -pct_text_offset, 0])
            # Rotate the offset vector to match the tooth's orientation
            rad = np.radians(rotation_deg)
            c, s = np.cos(rad), np.sin(rad)
            rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            rotated_offset = rot_matrix.dot(offset_vec)
            pct_pos = np.array([x, y, 0.1]) + rotated_offset
            pct_text.pos(pct_pos).follow_camera()
            
            self.presentation_actors[tooth_nr] = {'shape': shape, 'nr_text': nr_text, 'pct_text': pct_text}
        
        # Create Left/Right Summary Bars (remains the same)
        bar_y = -8.0; bar_h = 1.0; bar_w_full = 8.0; h2 = bar_h/2; w2=bar_w_full/2
        self.summary_bar_left = Mesh([ [[-w2,-h2,0],[w2,-h2,0],[w2,h2,0],[-w2,h2,0]], [[0,1,2,3]] ])
        self.summary_bar_left.pos(-w2-0.2, bar_y, 0).color('green').alpha(0.7).lw(0)
        self.summary_bar_right = Mesh([ [[-w2,-h2,0],[w2,-h2,0],[w2,h2,0],[-w2,h2,0]], [[0,1,2,3]] ])
        self.summary_bar_right.pos(w2+0.2, bar_y, 0).color('firebrick').alpha(0.7).lw(0)
        self.summary_text_left = Text3D("Left: 0.0%", s=0.6, c='white', justify='center').pos(-w2-0.2, bar_y, 0.1).follow_camera()
        self.summary_text_right = Text3D("Right: 0.0%", s=0.6, c='white', justify='center').pos(w2+0.2, bar_y, 0.1).follow_camera()

        if self.parent_plotter:
            actors_to_add = []
            for d in self.presentation_actors.values(): actors_to_add.extend(d.values())
            actors_to_add.extend([self.summary_bar_left, self.summary_bar_right, self.summary_text_left, self.summary_text_right])
            self.parent_plotter.add(actors_to_add)
            
    def render_grid_view(self, timestamp, hardware_data_flat_array, sensitivity=1):
        # ... (time text logic) ...

        if hardware_data_flat_array is None:
            # ... (reset logic) ...
            return

        # *** THIS BLOCK WAS MISSING - IT IS NOW RESTORED ***
        # Map the 1D flat data array to a 2D grid for easy lookup by (row, col)
        force_grid = np.full((self.hw_rows, self.hw_cols), 0.0)
        data_idx = 0
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                if self.points_array_checker.is_valid(c_idx, r_idx):
                    if data_idx < len(hardware_data_flat_array):
                        force_grid[r_idx, c_idx] = hardware_data_flat_array[data_idx]
                        data_idx += 1
        # *** END RESTORED BLOCK ***

        total_force = np.sum(hardware_data_flat_array)
        if total_force < 1.0: 
            # ... (reset logic) ...
            return

        # Calculate force for each of the 14 MAPPED segments
        mapped_segment_forces = []
        for seg_idx in range(len(self.mapping_polygons)):
            segment_force = 0.0
            for r_idx, c_idx in self.segment_cell_map.get(seg_idx, []):
                segment_force += force_grid[r_idx, c_idx] # Now 'force_grid' is defined
            mapped_segment_forces.append(segment_force)
        
        # ... (rest of the function for mapping forces to presentation actors and summary bars) ...
        # Simplified mapping:
        left_forces = mapped_segment_forces[:7]
        right_forces = mapped_segment_forces[7:]
        total_left_force = sum(left_forces)
        total_right_force = sum(right_forces)

        # Map to Patient Right side (Teeth 1-8 in Universal, which we map to screen-left 16-9)
        for i in range(7): # 7 data segments on the left of the map
            tooth_nr = 16 - i # Visual teeth 16, 15, ..., 10
            if tooth_nr in self.presentation_actors:
                force = left_forces[i]; pct = (force/total_force)*100
                cell_count = len(self.segment_cell_map.get(i, [1]))
                avg_force = force / cell_count if cell_count > 0 else 0
                self.presentation_actors[tooth_nr]['shape'].color(self._value_to_color_hardware(avg_force, sensitivity))
                self.presentation_actors[tooth_nr]['pct_text'].text(f"{pct:.1f}%").on() if pct > 0.5 else self.presentation_actors[tooth_nr]['pct_text'].off()

        # Map to Patient Left side (Teeth 9-16 in Universal, screen-right 9-16)
        for i in range(7): # 7 data segments on the right of the map
            tooth_nr = 9 + i # Visual teeth 9, 10, ..., 15
            if tooth_nr in self.presentation_actors:
                force = right_forces[i]; pct = (force/total_force)*100
                cell_count = len(self.segment_cell_map.get(i + 7, [1]))
                avg_force = force / cell_count if cell_count > 0 else 0
                self.presentation_actors[tooth_nr]['shape'].color(self._value_to_color_hardware(avg_force, sensitivity))
                self.presentation_actors[tooth_nr]['pct_text'].text(f"{pct:.1f}%").on() if pct > 0.5 else self.presentation_actors[tooth_nr]['pct_text'].off()
        
        # Update summary bars
        left_pct = (total_left_force/total_force)*100; right_pct = 100 - left_pct
        self.summary_text_left.text(f"Left: {left_pct:.1f}%"); self.summary_text_right.text(f"Right: {right_pct:.1f}%")
        # Scaling the bar itself
        bar_w_full = 8.0
        self.summary_bar_left.scale([left_pct / 100.0, 1, 1]).pos(-bar_w_full/2 + (bar_w_full * left_pct/200.0), -8.0, 0)
        self.summary_bar_right.scale([right_pct / 100.0, 1, 1]).pos(bar_w_full/2 - (bar_w_full * right_pct/200.0), -8.0, 0)