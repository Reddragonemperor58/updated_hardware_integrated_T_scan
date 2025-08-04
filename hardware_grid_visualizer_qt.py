# --- START OF FILE hardware_grid_visualizer_qt.py ---
import numpy as np
from vedo import Text2D, Rectangle, colors, Line, Spline, Points, Text3D, Mesh, Assembly
import logging
from points_array import PointsArray 
import vtk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareGridVisualizerQt:
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
        self.mapping_polygons = [] # The invisible polygons for data mapping
        self.segment_cell_map = {} # Result: {data_idx_0_to_13: [cells]}

        self.presentation_actors = {} # The 16 T-Scan style shapes and their labels
        self.summary_bar_left, self.summary_bar_right = None, None
        self.summary_text_left, self.summary_text_right = None, None
        self.bar_max_width = 7.0
        
        self.user_ci_width = user_central_incisor_width
        self.mapping_arch_params = { # Your tuned params for the real arch
            'grid_cell_size_mm': 0.7, 'width_scale': 0.4, 'depth_scale': 0.99,
            'anterior_flatness': 0.2, 'lingual_curve_offset': 3.5, 'vertical_offset_factor': -0.09,
        }
        if not self.renderer: logger.error("Renderer not provided."); return

    def setup_scene(self):
        if not self.renderer or not self.parent_plotter: return
        logger.info("Setting up T-Scan style scene...")
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera
        cam.ParallelProjectionOn()
        
        self._rebuild_all_components() # One master function
        
        self._fit_camera_to_presentation()
        self.renderer.SetBackground(0.93, 0.93, 0.97)
        logger.info("Scene setup complete.")

    def update_arch_parameters(self, user_central_incisor_width=None):
        if user_central_incisor_width is not None:
            self.user_ci_width = user_central_incisor_width
            logger.info(f"CI Width changed to: {self.user_ci_width} mm. Rebuilding mapping.")
            self._rebuild_all_components()
        if self.parent_plotter and self.parent_plotter.qt_widget: self.parent_plotter.qt_widget.Render()

    def _rebuild_all_components(self):
        """Master function to create mapping geometry, map cells, and create presentation geometry."""
        # 1. Generate the invisible mapping polygons from the real arch shape
        self.mapping_polygons = self._calculate_mapping_polygons()
        if not self.mapping_polygons: return
        # 2. Map the grid cells using these invisible polygons
        self._map_cells_to_segments()
        # 3. Create the visible presentation actors
        self._create_presentation_actors()

    def _calculate_mapping_polygons(self):
        """
        Calculates and returns a list of 14 raw polygon vertex arrays for the data mapping arch.
        This is the corrected version that ensures 14 polygons are created.
        """
        cell_size=0.25; grid_offset_x=-(self.hw_cols*cell_size)/2; grid_offset_y=-(self.hw_rows*cell_size)/2
        valid_x, valid_y = [], []
        for r in range(self.hw_rows):
            for c in range(self.hw_cols):
                if self.points_array_checker.is_valid(c,r):
                    cx=grid_offset_x+(c*cell_size)+cell_size/2; cy=grid_offset_y+((self.hw_rows-1-r)*cell_size)+cell_size/2
                    valid_x.append(cx); valid_y.append(cy)
        if not valid_x: return []
        min_x, max_x = min(valid_x), max(valid_x); min_y, max_y = min(valid_y), max(valid_y)
        valid_w=max_x-min_x; valid_h=max_y-min_y; grid_cx=(min_x+max_x)/2
        p = self.mapping_arch_params
        arch_a=(valid_w/2.0)*p['width_scale']; arch_b=arch_a*p['depth_scale']
        def get_base_y(x_rel):
            flat_half_w=arch_a*p['anterior_flatness']
            if abs(x_rel)<=flat_half_w: return arch_b
            else:
                curve_w=arch_a-flat_half_w;
                if curve_w<=0: return arch_b
                x_norm=(abs(x_rel)-flat_half_w)/curve_w; return arch_b*np.sqrt(max(0,1-x_norm**2))
        num_pts=200; base_curve_pts=[]
        x_start,x_end=-arch_a,arch_a
        for i in range(num_pts+1):
            x=x_start+(x_end-x_start)*i/num_pts; base_curve_pts.append(np.array([x,get_base_y(x)]))
        normals=[]
        for i in range(len(base_curve_pts)):
            if i==0: t=base_curve_pts[i+1]-base_curve_pts[i]
            elif i==len(base_curve_pts)-1: t=base_curve_pts[i]-base_curve_pts[i-1]
            else: t=base_curve_pts[i+1]-base_curve_pts[i-1]
            if np.linalg.norm(t)>0: t/=np.linalg.norm(t)
            n=np.array([-t[1],t[0]]);
            if n[1]<0: n*=-1
            normals.append(n)
        points_outer,points_inner=[],[]
        y_shift=max_y-arch_b+(valid_h*p['vertical_offset_factor'])
        for i in range(len(base_curve_pts)):
            pt=base_curve_pts[i]; n=normals[i]
            points_outer.append(np.array([pt[0]+grid_cx,-pt[1]+y_shift]))
            inner_pt=pt+n*p['lingual_curve_offset']
            points_inner.append(np.array([inner_pt[0]+grid_cx,-inner_pt[1]+y_shift]))
        
        dividing_lines_coords = []
        num_segments = 14
        for i in range(num_segments + 1):
            idx = int(i * (num_pts / num_segments))
            if idx >= len(points_outer): idx = len(points_outer) - 1
            dividing_lines_coords.append((points_outer[idx], points_inner[idx]))
        
        polygons = []
        for i in range(len(dividing_lines_coords) - 1):
            p_lo, p_li = dividing_lines_coords[i]; p_ro, p_ri = dividing_lines_coords[i+1]
            polygons.append(np.array([p_lo, p_ro, p_ri, p_li]))
        logger.info(f"Calculated {len(polygons)} mapping polygons.")
        return polygons

    def _map_cells_to_segments(self):
        if not self.mapping_polygons: return
        self.segment_cell_map = {i:[] for i in range(len(self.mapping_polygons))}
        cell_size = 0.25; grid_offset_x = -(self.hw_cols*cell_size)/2; grid_offset_y = -(self.hw_rows*cell_size)/2
        from matplotlib.path import Path
        for r_idx in range(self.hw_rows):
            for c_idx in range(self.hw_cols):
                if self.points_array_checker.is_valid(c_idx, r_idx):
                    cell_center = (grid_offset_x + (c_idx+0.5)*cell_size, grid_offset_y + ((self.hw_rows-1-r_idx)+0.5)*cell_size)
                    for seg_idx, polygon_points in enumerate(self.mapping_polygons):
                        if Path(polygon_points[:,:2]).contains_point(cell_center):
                            self.segment_cell_map[seg_idx].append((r_idx, c_idx)); break
        cell_counts = [len(cells) for cells in self.segment_cell_map.values()]
        logger.info(f"Cell mapping complete. Cells per data segment (0-13): {cell_counts}")

    # In hardware_grid_visualizer_qt.py

    def _define_explicit_tscan_layout(self):
        """
        Generates layout properties for a fixed set of 14 TEETH, matching the data mapping.
        This is now the single source of truth for the presentation layout.
        """
        layout = {}
        num_teeth = 14 # Hardcode to 14 visual segments

        # These values define the overall size and shape of the arch in rendering units.
        # Tune these to adjust the final look.
        arch_layout_width = 17.0 # Wider arch for more space
        arch_layout_depth = 11.0 # Deeper arch
        
        # This helper function is now local as it's only used here
        def get_arch_positions(nr_teeth, arch_w, arch_d):
            if nr_teeth == 0: return np.array([])
            x_coords = np.linspace(-arch_w / 2, arch_w / 2, nr_teeth)
            k = arch_d / ((arch_w / 2)**2) if arch_w != 0 else 0
            return np.array([[x, arch_d - k * (x**2)] for x in x_coords])

        arch_centers = get_arch_positions(num_teeth, arch_layout_width, arch_layout_depth)

        # Base size for a "standard" tooth in the arch
        avg_spacing_x = arch_layout_width / (num_teeth -1) if num_teeth > 1 else 1.0
        base_cell_w = avg_spacing_x * 0.85 # Use a fixed size for visual consistency
        base_cell_h = base_cell_w * 1.2

        # Create layout for 14 teeth, keyed 0 to 13
        for i in range(num_teeth):
            center_xy = arch_centers[i]
            
            # The keys of the layout dictionary will be 0, 1, 2, ..., 13
            # This will perfectly match the data segment indices.
            layout[i] = {'center': center_xy, 'width': base_cell_w, 'height': base_cell_h}
        return layout
    
    def _get_arch_positions_for_layout(self, num_teeth, arch_width, arch_depth):
        """Helper function from your reference code to generate points on a parabola."""
        if num_teeth == 0: return np.array([])
        x_coords = np.array([0.0]) if num_teeth == 1 else np.linspace(-arch_width / 2, arch_width / 2, num_teeth)
        k = arch_depth / ((arch_width / 2)**2) if arch_width != 0 else 0
        return np.array([[x, arch_depth - k * (x**2), 0] for x in x_coords])

    def _create_presentation_actors(self):
        if self.parent_plotter:
            all_old_actors = []; 
            for d in self.presentation_actors.values(): all_old_actors.extend(d.values())
            if self.summary_bar_left: all_old_actors.extend([self.summary_bar_left, self.summary_bar_right, self.summary_text_left, self.summary_text_right])
            self.parent_plotter.remove(all_old_actors)
        self.presentation_actors.clear()
        
        layout = self._define_explicit_tscan_layout()
        if not layout: return
        
        for tooth_nr, props in layout.items():
            x,y=props['center']; w=props['width']/2.0; h=props['height']/2.0
            shape = Mesh([ [[x-w,y-h,0],[x+w,y-h,0],[x+w,y+h,0],[x-w,y+h,0]], [[0,1,2,3]] ]).color('grey').alpha(0.8).lw(1).lc('black')
            nr_text = Text3D(f"{tooth_nr}", s=h*0.5, justify='center', c='black').pos(x, y+h+0.3, 0.1).follow_camera()
            pct_text = Text3D("0.0%", s=h*0.4, justify='center', c='black').pos(x, y-h-0.3, 0.1).follow_camera()
            self.presentation_actors[tooth_nr] = {'shape': shape, 'nr_text': nr_text, 'pct_text': pct_text}

        min_y = min(p['center'][1]-p['height']/2 for p in layout.values())
        bar_y=min_y-1.5; bar_h=1.0; bar_w=7.0; h2=bar_h/2; w2=bar_w/2
        self.summary_bar_left = Mesh([ [[-w2,-h2,0],[w2,-h2,0],[w2,h2,0],[-w2,h2,0]], [[0,1,2,3]] ]).pos(-w2-0.2, bar_y, 0).color('green').alpha(0.7).lw(0)
        self.summary_bar_right = Mesh([ [[-w2,-h2,0],[w2,-h2,0],[w2,h2,0],[-w2,h2,0]], [[0,1,2,3]] ]).pos(w2+0.2, bar_y, 0).color('firebrick').alpha(0.7).lw(0)
        self.summary_text_left = Text3D("Left: 0.0%", s=0.6, c='white', justify='center').pos(-w2-0.2, bar_y, 0.1).follow_camera()
        self.summary_text_right = Text3D("Right: 0.0%", s=0.6, c='white', justify='center').pos(w2+0.2, bar_y, 0.1).follow_camera()
        
        actors_to_add = []
        for d in self.presentation_actors.values(): actors_to_add.extend([d['shape'],d['nr_text'],d['pct_text']])
        actors_to_add.extend([self.summary_bar_left, self.summary_bar_right, self.summary_text_left, self.summary_text_right])
        self.parent_plotter.add(actors_to_add)

    # In hardware_grid_visualizer_qt.py -> inside the HardwareGridVisualizerQt class

    def _fit_camera_to_presentation(self):
        """
        Calculates the bounds of all visible presentation actors and sets the camera
        to frame them perfectly.
        """
        if not self.presentation_actors or not self.parent_plotter:
            logger.warning("Cannot fit camera: no presentation actors or plotter.")
            # Set a default fallback camera
            cam = self.parent_plotter.camera
            cam.SetParallelScale(10)
            cam.SetFocalPoint(0,0,0)
            cam.SetPosition(0,0,20)
            return

        # Activate the correct renderer for camera manipulation
        self.parent_plotter.at(self.renderer_index)
        cam = self.parent_plotter.camera

        # Collect all actors that should be included in the camera's view
        all_actors_for_bounds = [d['shape'] for d in self.presentation_actors.values()]
        # Also include the summary bars in the bounds calculation
        if self.summary_bar_left: all_actors_for_bounds.extend([self.summary_bar_left, self.summary_bar_right])
        
        if not all_actors_for_bounds:
            logger.warning("No actors found to calculate bounds for camera fitting.")
            return

        # Create a temporary Assembly to easily get the collective bounds of all actors
        scene_assembly = Assembly(all_actors_for_bounds)
        bounds = scene_assembly.bounds()

        if bounds is not None and len(bounds) == 6:
            # Calculate center from the bounds array
            center_x = (bounds[0] + bounds[1]) / 2.0
            center_y = (bounds[2] + bounds[3]) / 2.0
            center = (center_x, center_y, 0) # Focal point will be on the XY plane
            
            # Calculate the required scale to fit everything vertically, with a small padding
            y_height = bounds[3] - bounds[2] # ymax - ymin
            cam.SetParallelScale(y_height * 0.55) # Use 55% of scale to add padding
            
            # Position the camera
            cam.SetFocalPoint(center)
            cam.SetPosition(center[0], center[1], 20) # Look from the front (positive Z)
        else:
            # Fallback if bounds calculation fails
            logger.warning("Failed to calculate bounds for camera fitting. Using default camera settings.")
            cam.SetParallelScale(10)
            cam.SetFocalPoint(0,0,0)
            cam.SetPosition(0,0,20)

        cam.SetViewUp(0, 1, 0) # Y-axis is up
        self.renderer.ResetCameraClippingRange() # Important after setting camera
        
    # In hardware_grid_visualizer_qt.py

    # In hardware_grid_visualizer_qt.py

    # In hardware_grid_visualizer_qt.py

    # In hardware_grid_visualizer_qt.py

    def render_grid_view(self, timestamp, hardware_data_flat_array, sensitivity=1):
        if not self.presentation_actors or not self.summary_bar_left:
            return
            
        self.parent_plotter.at(self.renderer_index)
        
        if self.time_text_actor: self.renderer.RemoveActor(self.time_text_actor.actor)
        self.time_text_actor = Text2D(f"Time: {timestamp:.1f}s", pos="bottom-left", c='k')
        self.renderer.AddActor(self.time_text_actor.actor)

        total_force = np.sum(hardware_data_flat_array) if hardware_data_flat_array is not None else 0.0

        if total_force < 1.0:
            for actors in self.presentation_actors.values():
                actors['shape'].color('grey')
                actors['pct_text'].off()
            self.summary_text_left.text("Left: 0.0%")
            self.summary_text_right.text("Right: 0.0%")
            self.summary_bar_left.scale([0.001, 1, 1])
            self.summary_bar_right.scale([0.001, 1, 1])
            return

        force_grid = np.full((self.hw_rows, self.hw_cols), 0.0); data_idx=0
        for r in range(self.hw_rows):
            for c in range(self.hw_cols):
                if self.points_array_checker.is_valid(c,r) and data_idx < len(hardware_data_flat_array):
                    force_grid[r, c] = hardware_data_flat_array[data_idx]; data_idx += 1
        
        mapped_segment_forces = {i: 0.0 for i in range(14)}
        mapped_segment_avg_force = {i: 0.0 for i in range(14)}
        for seg_idx in range(len(self.mapping_polygons)):
            segment_force = 0.0; cells = self.segment_cell_map.get(seg_idx, [])
            if cells:
                for r_idx, c_idx in cells: segment_force += force_grid[r_idx, c_idx]
                mapped_segment_forces[seg_idx] = segment_force
                mapped_segment_avg_force[seg_idx] = segment_force / len(cells)
        
        total_left_force, total_right_force = 0.0, 0.0
        for seg_idx in range(14):
            if seg_idx not in self.presentation_actors: continue
            force = mapped_segment_forces.get(seg_idx, 0.0)
            avg_force = mapped_segment_avg_force.get(seg_idx, 0.0)
            percentage = (force / total_force) * 100.0
            if seg_idx < 7: total_right_force += force
            else: total_left_force += force
            actors = self.presentation_actors[seg_idx]
            actors['shape'].color(self._value_to_color_hardware(avg_force, sensitivity))
            if percentage > 0.5: actors['pct_text'].on().text(f"{percentage:.1f}%")
            else: actors['pct_text'].off()
        
        # --- CORRECTED SUMMARY BAR UPDATE LOGIC ---
        left_pct = (total_left_force / total_force) * 100.0
        right_pct = (total_right_force / total_force) * 100.0
        
        self.summary_text_left.text(f"Left: {left_pct:.1f}%")
        self.summary_text_right.text(f"Right: {right_pct:.1f}%")

        # Get the layout again to find the lowest point for positioning
        layout = self._define_explicit_tscan_layout()
        if not layout: return # Should not happen if we got this far
        
        min_y = min(p['center'][1]-p['height']/2 for p in layout.values())
        bar_y_center = min_y - 1.5
        bar_gap = 0.4
        
        left_scale_x = max(0.001, left_pct / 100.0)
        new_left_width = self.bar_max_width * left_scale_x
        new_left_center_x = -bar_gap/2 - new_left_width/2
        self.summary_bar_left.scale([left_scale_x, 1, 1]).pos(new_left_center_x, bar_y_center, 0)
        
        right_scale_x = max(0.001, right_pct / 100.0)
        new_right_width = self.bar_max_width * right_scale_x
        new_right_center_x = bar_gap/2 + new_right_width/2
        self.summary_bar_right.scale([right_scale_x, 1, 1]).pos(new_right_center_x, bar_y_center, 0)
        # --- END CORRECTION ---
        
    def animate(self, timestamp, data, sensitivity=1):
        self.render_grid_view(timestamp, data, sensitivity)
        print(self.bar_max_width)

    def _value_to_color_hardware(self, value, sensitivity=1):
        # ... (This function remains unchanged) ...
        mapped_value = (value / sensitivity * 255) // self.max_force_for_scaling; mapped_value = min(255,max(0,int(mapped_value)))
        r,g,b = 200,200,200 
        if mapped_value>204:r=255;g=max(0,int(150-((mapped_value-204)*150/51)));b=0
        elif mapped_value>140:r=int(139+((mapped_value-140)*116/64));g=int((mapped_value-140)*150/64);b=0
        elif mapped_value>76:g=int(255-((mapped_value-76)*155/64));r=int(((mapped_value-76)/64)*100);b=0
        elif mapped_value>12:r=0;g=int(255-((mapped_value-12)*155/64));b=int(100-((mapped_value-12)*50/64))
        return (r/255.0,g/255.0,b/255.0)
