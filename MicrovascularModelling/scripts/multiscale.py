from classes.GeometryClasses import Node, Tree, TissueHandler, Visualizer
from classes.ConfigClass import Config
from classes.MatrixHandlerClasses import MatrixHandler, MatrixSolver
from classes.GetFEMClasses import GetFEMHandler1D, GetFEMHandler3D
from classes.GrowthClasses import GrowthHandler
from itertools import combinations
import sys
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

def main():
    # These parameters contorl elements of the growth strategy
    config_file = "single_growth" # Physical parameters for the simulation
    ratio = 1.4 # Flow volume ratio to target
    growth_case = 165 # Growth case to use (how far the vessel grows before seeking anastomosis)
    pressure_strategy_num = 1#0 # Pressure control strategy to use

    # Load parameters from config file
    config_string = "./config/Cases/"+ config_file +".json"
    config = Config.load_config(config_string)
    config.set_growth_case(growth_case)
    # Overwrite default config parameters for this experiment with the specified ones above
    config.config_access["RUN_PARAMETERS"]["output_path"] = f"./outputs/test3"
    config.config_access["GROWTH_PARAMETERS"]["sprouting_strategy"] = 3
    config.config_access["GROWTH_PARAMETERS"]["pressure_strategy"] = pressure_strategy_num
    config.config_access["RUN_PARAMETERS"]["test_name"] = f"ratio_{ratio}"
    config.config_access["GROWTH_PARAMETERS"]["flow_volume_ratio"] = ratio
    config.setup_output_folder()
    config.setup_case_folder()

    config.logger.log(f"Output Path is {config.test_name}")



    def make_random_tree(num_points=80, cube_size=3e-4, seed=None):
        """
        Generates a connected network inside a cube with:
        - two terminals (inlet/outlet) = farthest pair (degree == 1)
        - every internal node degree in {2,3}
        - adds additional loops while preserving degree <= 3
        """
        if num_points < 3:
            raise ValueError("Need at least 3 points to build a valid internal network.")

        if seed is not None:
            np.random.seed(seed)

        my_tree = Tree()

        # 1) Random positions
        pts = cube_size * np.random.rand(num_points, 3)

        # 2) pick farthest pair as inlet/outlet
        dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        inlet_idx, outlet_idx = np.unravel_index(np.argmax(dmat), dmat.shape)
        if inlet_idx == outlet_idx:
            raise RuntimeError("Failed to pick distinct inlet/outlet.")

        # 3) Create nodes in Tree and bookkeeping
        node_map = {}
        degree = {}
        for i in range(num_points):
            coords = list(pts[i])
            if i == inlet_idx:
                node_map[i] = my_tree.add_inlet(coords, i)
            elif i == outlet_idx:
                node_map[i] = my_tree.add_outlet(coords, i)
            else:
                node_map[i] = my_tree.add_node(coords, i)
            degree[i] = 0

        seg_id = 0
        R = 3e-6

        # 4) Build a backbone path from inlet to outlet (greedy nearest neighbor).
        #    This ensures a chain where internal backbone nodes already have degree=2.
        unvisited = set(range(num_points))
        unvisited.remove(inlet_idx)
        backbone = [inlet_idx]
        current = inlet_idx

        while unvisited:
            # if outlet still unvisited and it's nearest, prefer moving to it when close
            candidates = list(unvisited)
            candidates.sort(key=lambda j: np.linalg.norm(pts[j] - pts[current]))
            next_node = candidates[0]

            # If the nearest is the outlet and we still have other nodes far away,
            # allow continuing until outlet is the last step to maximize backbone coverage.
            if next_node == outlet_idx and len(unvisited) > 1:
                # pick second nearest if exists
                if len(candidates) > 1:
                    next_node = candidates[1]

            backbone.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        # Ensure outlet is at the end of backbone; if not, move it to the end.
        if backbone[-1] != outlet_idx:
            # place outlet at end, preserving order of others
            backbone.remove(outlet_idx)
            backbone.append(outlet_idx)

        # Connect backbone sequentially
        for a, b in zip(backbone[:-1], backbone[1:]):
            my_tree.add_segment(node_map[a], node_map[b], seg_id, radius=R)
            degree[a] += 1
            degree[b] += 1
            seg_id += 1

        # Now backbone nodes: inlet and outlet should be degree 1; internal backbone nodes should be >=2 (they are 2)
        # Quick sanity:
        if degree[inlet_idx] != 1:
            raise RuntimeError(f"Inlet degree != 1 after backbone build: {degree[inlet_idx]}")
        if degree[outlet_idx] != 1:
            raise RuntimeError(f"Outlet degree != 1 after backbone build: {degree[outlet_idx]}")

        # 5) For any non-backbone nodes (should be none because backbone covered all),
        #    we'd attach them with at least two connections; but because backbone greedy used all nodes,
        #    we only have to ensure no internal node is left with degree 1.
        #
        # In edge cases (small N) some nodes may have degree 1 if backbone has single neighbors; fix them.

        # 6) Fix internal nodes that have degree < 2 (excluding inlet/outlet)
        #    Strategy: repeatedly find internal node with degree == 1 and connect it to its nearest node
        #    that has degree < 3 (preferring nodes with degree == 2 to keep them internal).
        changed = True
        max_fix_iters = num_points * 5
        iter_count = 0
        while True:
            iter_count += 1
            if iter_count > max_fix_iters:
                raise RuntimeError("Unable to repair degrees within allowed iterations.")
            low_deg_nodes = [i for i in range(num_points)
                            if i not in (inlet_idx, outlet_idx) and degree[i] < 2]
            if not low_deg_nodes:
                break  # all internal nodes have degree >= 2

            repaired_any = False
            for i in low_deg_nodes:
                # find nearest candidate j != i with degree < 3 (prefer degree==2)
                candidates = [j for j in range(num_points) if j != i and degree[j] < 3]
                if not candidates:
                    # no available neighbors to attach to (should be rare)
                    continue
                # sort by (degree preference, distance) -> prefer connecting to degree==2 then degree==1 then degree==0
                candidates.sort(key=lambda j: (degree[j], np.linalg.norm(pts[j] - pts[i])))
                j = candidates[0]
                # add segment if it doesn't already exist (we don't have segment list; assume add_segment tolerates duplicates or user class checks)
                my_tree.add_segment(node_map[i], node_map[j], seg_id, radius=R)
                degree[i] += 1
                degree[j] += 1
                seg_id += 1
                repaired_any = True

            if not repaired_any:
                # try pairing low-degree nodes together (if possible)
                pairs = []
                ld = [i for i in range(num_points) if i not in (inlet_idx, outlet_idx) and degree[i] < 2]
                if len(ld) < 2:
                    # can't pair; break and will raise below
                    break
                # pair nearest-low-degree with nearest-low-degree
                ld.sort(key=lambda i: np.linalg.norm(pts[i] - pts[inlet_idx]))  # arbitrary stable order
                for a, b in zip(ld[::2], ld[1::2]):
                    if degree[a] < 3 and degree[b] < 3:
                        my_tree.add_segment(node_map[a], node_map[b], seg_id, radius=R)
                        degree[a] += 1
                        degree[b] += 1
                        seg_id += 1
                        repaired_any = True

            if not repaired_any:
                # no repair possible
                break

        # Final check after repairs
        bad_nodes = [i for i in range(num_points)
                    if i not in (inlet_idx, outlet_idx) and (degree[i] < 2 or degree[i] > 3)]
        if bad_nodes:
            raise ValueError(f"Cannot satisfy degree constraints for internal nodes, bad nodes: {bad_nodes}")

        if degree[inlet_idx] != 1 or degree[outlet_idx] != 1:
            raise ValueError(f"Inlet/outlet degrees incorrect: inlet {degree[inlet_idx]}, outlet {degree[outlet_idx]}")

        # 7) Add controlled loops while enforcing degree <= 3 and not touching inlet/outlet
        #    We try pairs by random order and add if both degrees < 3 and not already directly connected.
        pairs = list(combinations(range(num_points), 2))
        np.random.shuffle(pairs)

        # OPTIONAL: limit total segments to keep mesh size reasonable
        max_extra_segments = int(num_points * 2)  # tuneable
        extra_added = 0

        # We need a quick way to avoid duplicate edges: build a set of frozenset node pairs for existing segments.
        # We'll reconstruct it by checking degrees heuristically - but better to ask Tree for segments if available.
        # To be safe, keep an in-memory set and populate it by walking all pairs we explicitly added above.
        existing_edges = set()
        # Populate existing_edges from backbone (we know backbone pairs)
        for a, b in zip(backbone[:-1], backbone[1:]):
            existing_edges.add(frozenset((a, b)))

        # We also added repair edges; we don't track them explicitly earlier, but we tracked seg_id and degrees.
        # To be safe, we will avoid adding an edge if it is already present in existing_edges.

        for i, j in pairs:
            if extra_added >= max_extra_segments:
                break
            if i in (inlet_idx, outlet_idx) or j in (inlet_idx, outlet_idx):
                continue
            if degree[i] >= 3 or degree[j] >= 3:
                continue
            key = frozenset((i, j))
            if key in existing_edges:
                continue
            dist = np.linalg.norm(pts[i] - pts[j])
            # avoid extremely small loops
            if dist < 0.05 * cube_size / max(1.0, np.cbrt(num_points)):
                continue
            # add the loop
            my_tree.add_segment(node_map[i], node_map[j], seg_id, radius=R)
            degree[i] += 1
            degree[j] += 1
            existing_edges.add(key)
            seg_id += 1
            extra_added += 1

        # Final verification before populate_junctions
        for i in range(num_points):
            seg_count = degree[i]
            if i in (inlet_idx, outlet_idx):
                if seg_count != 1:
                    raise ValueError(f"Terminal {i} has degree {seg_count} (must be 1)")
            else:
                if seg_count < 2 or seg_count > 3:
                    raise ValueError(f"Unusual number of segments ({seg_count}) connecting to internal node {i} found")

        my_tree.populate_junctions()
        return my_tree

  

    def vtk_to_custom_tree(vtk_filename, inlet_coord, radius_array_name="radius", debug=True):
        """
        Parses VTP file and constructs the vascular tree with correct flow direction 
        (Inlet -> Outlet).
        """
        
        # 1. Load the VTK file
        print(f"Loading {vtk_filename}...")
        raw_mesh = pv.read(vtk_filename)
        
        # 2. CLEAN THE MESH
        # Merge duplicate points so junctions are connected
        mesh = raw_mesh.clean(tolerance=1e-7)
        
        if radius_array_name not in mesh.point_data:
            raise ValueError(f"Array '{radius_array_name}' not found. Available: {mesh.point_data.keys()}")
        
        radii = mesh.point_data[radius_array_name]
        points = mesh.points
        
        # 3. Identify the Inlet ID
        inlet_id = mesh.find_closest_point(inlet_coord)
        print(f"Inlet identified at Index {inlet_id}")
        
        # 4. Build Topology Graph
        G = nx.Graph()
        G.add_nodes_from(range(mesh.n_points))
        
        lines = mesh.lines
        i = 0
        while i < len(lines):
            n_pts = lines[i]
            segment_indices = lines[i+1 : i+1+n_pts]
            for j in range(len(segment_indices) - 1):
                u, v = segment_indices[j], segment_indices[j+1]
                G.add_edge(u, v)
            i += n_pts + 1
            
        node_degrees = dict(G.degree())

        # --- VISUAL DEBUGGER ---
        if debug:
            print("Opening debug window... Red=Inlet, Blue=Outlet, Green=Junction")
            p = pv.Plotter()
            p.add_mesh(mesh, color="white", opacity=0.3, line_width=2, label="Vessels")
            
            inlet_pts = [points[n] for n in G.nodes() if n == inlet_id]
            outlet_pts = [points[n] for n, d in node_degrees.items() if d == 1 and n != inlet_id]
            junction_pts = [points[n] for n, d in node_degrees.items() if d > 2]
            
            if inlet_pts:
                p.add_mesh(pv.PolyData(inlet_pts), color="red", point_size=20, render_points_as_spheres=True, label="Inlet")
            if outlet_pts:
                p.add_mesh(pv.PolyData(outlet_pts), color="blue", point_size=10, render_points_as_spheres=True, label="Outlets")
            if junction_pts:
                p.add_mesh(pv.PolyData(junction_pts), color="green", point_size=10, render_points_as_spheres=True, label="Bifurcations")

            p.add_legend()
            p.show()
        # -----------------------

        # 5. Initialize Library Tree
        my_tree = Tree()
        vtk_to_lib_map = {}

        # 6. Create Nodes
        # We populate the map first so we can reference them when creating segments
        for i in range(mesh.n_points):
            coord = points[i]
            degree = node_degrees.get(i, 0)
            
            if i == inlet_id:
                node_obj = my_tree.add_inlet(coord, i)
            elif degree == 1:
                node_obj = my_tree.add_outlet(coord, i)
            else:
                # Includes standard segments (degree 2) and bifurcations (degree 3+)
                node_obj = my_tree.add_node(coord, i)
                
            vtk_to_lib_map[i] = node_obj

        # 7. Create Segments (CORRECTED FOR DIRECTION)
        # We use BFS starting from inlet to ensure we always move downstream.
        # u is the current node (upstream), v is the neighbor (downstream).
        segment_counter = 0
        
        # bfs_edges returns an iterator of edges (u, v) in the order of traversal from source
        for u, v in nx.bfs_edges(G, source=inlet_id):
            
            node_u = vtk_to_lib_map[u] # Upstream (Parent)
            node_v = vtk_to_lib_map[v] # Downstream (Child)
            
            # Calculate radius (average of the two points)
            seg_radius = (radii[u] + radii[v]) / 2.0
            
            # Add segment: First arg is Upstream, Second is Downstream
            my_tree.add_segment(node_u, node_v, segment_counter, radius=seg_radius)
            segment_counter += 1

        # 8. Finalize
        my_tree.populate_junctions()
        
        print(f"Success! Tree created with {mesh.n_points} nodes and {segment_counter} segments.")
        return my_tree



    



    

    def connect_artery_vein_trees(file1, file2, inlet1_coord, inlet2_coord, terminal_array="isTerminal", debug=True):
        """
        Connects Artery (Flow: Inlet->Capillary) to Vein (Flow: Capillary->Outlet).
        Ensures correct segment directionality using BFS.
        """
        
        # --- Helper to load and process a mesh ---
        def process_mesh(fname, label):
            print(f"Loading {label} from {fname}...")
            raw = pv.read(fname)
            mesh = raw.clean(tolerance=1e-7) # Merge duplicate points
            
            if terminal_array not in mesh.point_data:
                raise ValueError(f"Array '{terminal_array}' not found in {fname}")
                
            points = mesh.points
            radii = mesh.point_data["radius"] if "radius" in mesh.point_data else np.ones(mesh.n_points)
            term_flags = mesh.point_data[terminal_array]
            
            # Get IDs where isTerminal is roughly 1
            term_ids = [i for i, val in enumerate(term_flags) if val > 0.5]
            
            G = nx.Graph()
            lines = mesh.lines
            i = 0
            while i < len(lines):
                n = lines[i]
                pts = lines[i+1 : i+1+n]
                for k in range(len(pts)-1):
                    G.add_edge(pts[k], pts[k+1])
                i += n + 1
                
            return {
                "mesh": mesh, "graph": G, "points": points, "radii": radii,
                "term_ids": term_ids
            }

        # 1. Load Data
        data1 = process_mesh(file1, "Arterial Tree (File 1)")
        data2 = process_mesh(file2, "Venous Tree (File 2)")
        
        # 2. Match Terminals (Hungarian Algorithm)
        terms1 = data1["term_ids"]
        terms2 = data2["term_ids"]
        
        print(f"Terminals to connect: {len(terms1)} (File 1) vs {len(terms2)} (File 2)")
        
        if len(terms1) != len(terms2):
            raise ValueError(f"Mismatch! {len(terms1)} vs {len(terms2)} terminals.")
        
        coords1 = data1["points"][terms1]
        coords2 = data2["points"][terms2]
        
        cost_matrix = distance_matrix(coords1, coords2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        bridged_nodes = set()
        bridge_pairs = []
        
        for r, c in zip(row_ind, col_ind):
            id1 = terms1[r]
            id2 = terms2[c]
            key1 = (0, id1) # (File Index, Node Index)
            key2 = (1, id2)
            bridged_nodes.add(key1)
            bridged_nodes.add(key2)
            bridge_pairs.append((key1, key2))

        # 3. Identify Inlets/Outlets
        # Artery Root (Start of Flow)
        inlet_id_1 = data1["mesh"].find_closest_point(inlet1_coord)
        global_inlet_key = (0, inlet_id_1)
        
        # Vein Root (End of Flow / Global Outlet)
        inlet_id_2 = data2["mesh"].find_closest_point(inlet2_coord)
        global_outlet_key = (1, inlet_id_2)

        # 4. Build the Tree
        my_tree = Tree()
        node_obj_map = {}
        
        # Mutable integer pointer for unique IDs
        global_id_counter = [0] 

        def add_nodes_from_file(file_idx, data):
            points = data["points"]
            G = data["graph"]
            degrees = dict(G.degree())
            
            for i in range(len(points)):
                key = (file_idx, i)
                coord = points[i]
                
                # GET UNIQUE ID
                current_uid = global_id_counter[0]
                
                if key == global_inlet_key:
                    node_obj_map[key] = my_tree.add_inlet(coord, current_uid)
                    
                elif key == global_outlet_key:
                    node_obj_map[key] = my_tree.add_outlet(coord, current_uid)
                    
                elif key in bridged_nodes:
                    node_obj_map[key] = my_tree.add_node(coord, current_uid)
                    
                elif degrees.get(i, 0) == 1:
                    # Dead ends (side branches that aren't the main inlet/outlet)
                    # For arteries: these are outlets. For veins: these are inlets.
                    # Usually safest to mark as 'outlet' or 'node' depending on your BCs.
                    node_obj_map[key] = my_tree.add_outlet(coord, current_uid)
                    
                else:
                    node_obj_map[key] = my_tree.add_node(coord, current_uid)
                
                global_id_counter[0] += 1

        # Process File 1 then File 2
        add_nodes_from_file(0, data1)
        add_nodes_from_file(1, data2)
        
        print(f"Total Nodes Created: {global_id_counter[0]}")

        # 5. Create Segments (CORRECTED FLOW LOGIC)
        seg_counter = 0
        
        # --- ARTERY SEGMENTS (File 0) ---
        # Flow: Inlet -> Terminals
        # BFS Source: Global Inlet
        G_art = data1["graph"]
        radii_art = data1["radii"]
        
        # BFS guarantees u is closer to inlet, v is further away
        for u, v in nx.bfs_edges(G_art, source=inlet_id_1):
            nu = node_obj_map[(0, u)] # Upstream
            nv = node_obj_map[(0, v)] # Downstream
            
            # r = (radii_art[u] + radii_art[v]) / 2.0
            r = 3e-6
            my_tree.add_segment(nu, nv, seg_counter, radius=r)
            seg_counter += 1

        # --- VEIN SEGMENTS (File 1) ---
        # Flow: Terminals -> Outlet
        # BFS Source: Global Outlet (inlet_id_2)
        # BFS guarantees u is closer to outlet, v is further away (capillaries)
        # THEREFORE: Flow goes v -> u
        G_vein = data2["graph"]
        radii_vein = data2["radii"]
        
        for u, v in nx.bfs_edges(G_vein, source=inlet_id_2):
            # u is closer to outlet (Downstream node in terms of flow)
            # v is further from outlet (Upstream node in terms of flow)
            
            nu = node_obj_map[(1, u)] # Downstream (Target)
            nv = node_obj_map[(1, v)] # Upstream (Source)
            
            # r = (radii_vein[u] + radii_vein[v]) / 2.0
            r = 3e-6
            
            # SWAP: add_segment(upstream, downstream) -> (v, u)
            
            my_tree.add_segment(nv, nu, seg_counter, radius=r)
            seg_counter += 1
                
        # --- BRIDGE SEGMENTS ---
        # Connect Artery Terminal -> Vein Terminal
        for key1, key2 in bridge_pairs:
            nu = node_obj_map[key1] # Artery (Upstream)
            nv = node_obj_map[key2] # Vein (Downstream)
            
            # r1 = data1["radii"][key1[1]]
            # r2 = data2["radii"][key2[1]]
            # r_bridge = (r1+r2)/2.0
            r_bridge = 3e-6
            
            my_tree.add_segment(nu, nv, seg_counter, radius=r_bridge)
            seg_counter += 1

        # 6. Finalize
        # my_tree.apply_maximum_segment_length(5e-6)
        # my_tree.reset_node_and_segment_numbering() 
        my_tree.populate_junctions()
        
        if debug:
            plot_debug(data1, data2, bridge_pairs, inlet_id_1, inlet_id_2)

        return my_tree

    def plot_debug(data1, data2, bridges, i1, i2):
        p = pv.Plotter()
        p.add_mesh(data1["mesh"], color="red", opacity=0.4, label="Artery")
        p.add_mesh(data2["mesh"], color="blue", opacity=0.4, label="Vein")
        
        # Inlet
        p.add_mesh(pv.PolyData(data1["points"][i1]), color="green", point_size=20, 
                render_points_as_spheres=True, label="Global Inlet")
        # Outlet
        p.add_mesh(pv.PolyData(data2["points"][i2]), color="black", point_size=20, 
                render_points_as_spheres=True, label="Global Outlet")
        # Bridges
        for k1, k2 in bridges:
            p1 = data1["points"][k1[1]]
            p2 = data2["points"][k2[1]]
            p.add_mesh(pv.Line(p1, p2), color="yellow", line_width=3)
        
        p.add_legend()
        p.show()



    # Initialize tree structure and apply a standard maximum segment length refinement
    #my_tree = make_random_tree(num_points=10, cube_size=2.5e-4)
    #my_tree = vtk_to_custom_tree("/hpc/ash252/CCO/example_1/ex1_simple3D_1.vtp", [0, 0.000125, 0.000125])
    #my_tree = connect_artery_vein_trees("/hpc/ash252/CCO/example_1/ex1_simple3D_1.vtp", "/hpc/ash252/CCO/example_1/ex1_simple3D_2.vtp", [0, 0.000125, 0.000125], [0.00025, 0.000125, 0.000125], terminal_array="isTerminal")
    my_tree = connect_artery_vein_trees("/hpc/ash252/CCO/example_1/ex1_simple3D_1.vtp", "/hpc/ash252/CCO/example_1/ex1_simple3D_2.vtp", [0, 0.000125, 0.000125], [0, 0.000125 - 2.5e-5, 0.000125], terminal_array="isTerminal")
    # my_tree = connect_artery_vein_trees("/hpc/ash252/CCO/example_1/ex1_simple3D_2.vtp", "/hpc/ash252/CCO/example_1/ex1_simple3D_1.vtp", [0.00025, 0.000125, 0.000125], [0, 0.000125, 0.000125], terminal_array="isTerminal")
    # my_tree = make_tree_baseline()
    # my_tree.apply_maximum_segment_length(5e-6)
   

    # Initialize the tissue domain, setting the physical limits and the mesh resolution
    mesh_cells = config.config_access["3D_CONDITIONS"]["num_cells"] + 1
    tissue_handler = TissueHandler()
    tissue_handler.set_config(config)
    tissue_handler.add_tissue([0,2.5e-4],[0,2.5e-4],[0,2.5e-4],[mesh_cells,mesh_cells,mesh_cells]) # type: ignore
    x,y,z = tissue_handler.current_tissue().tissue_limits()
    Node.set_limits(x,y,z)

    # initialize getfem handlers
    getfem_handler_1D = GetFEMHandler1D.load_config(config)
    getfem_handler_3D = GetFEMHandler3D.load_config(config)

    config.logger.log(my_tree)
    visualizer = Visualizer(config)

    mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
    solver = MatrixSolver(config.logger,mat_handler)
    growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

   

    # Evaluate initial flow volume ratio and set inlet pressure accordingly
    if pressure_strategy_num != 0:
        growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)

    # TIME TRACKERS
    haemodynamic_time = 0
    
    config.logger.log("ENTERING GROWTH LOOP")
    
    mat_handler.reset_system()
    
    # Final solve and save state at end of growth loop
    time_17 = time.time()
    # solver.iterative_solve_fluid_1D(tolerance=1e-8, tolerance_h=1e-8, alpha=0.9, beta=0.7, max_iterations=100)
    solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    time_18 = time.time()
    solver.iterative_solve_oxygen(tolerance=1e-8, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    # time_19 = time.time()
    

    mat_handler.save_tissue(0,vegf=False)
    my_tree.save_from_config(config)
    visualizer = growth_handler.initialize_visualizer()
    mat_handler.add_properties_to_visualizer(visualizer,True,True,False)
    visualizer.save_to_file([0])


    # Update time trackers
    haemodynamic_time += (time_18 - time_17)
    # stats = growth_handler.get_hemo_statistics(0)
    # stats = growth_handler.get_vascular_statistics(0)
    stats = growth_handler.get_hemo_o2_statistics(0)



    config.logger.log("TIME SUMMARY FOR SIMULATION")
    config.logger.log(f"Total haemodynamic time: {haemodynamic_time} seconds")
    config.remove_from_lock_file()
    print(f"resistance: {stats['hydraulic_resist']}")
    # print(f"mean O2 value: {stats['global_oxygen_mean']}")
    print(f"outlet oxygen: {100 * stats['outgoing_oxygen']/stats['incoming_oxygen']}%")
    return

if __name__ == "__main__":
    main()



