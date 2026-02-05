from classes.GeometryClasses import Node, Tree, TissueHandler, Visualizer
from classes.ConfigClass import Config
from classes.MatrixHandlerClasses import MatrixHandler, MatrixSolver
from classes.GetFEMClasses import GetFEMHandler1D, GetFEMHandler3D
from classes.GrowthClasses import GrowthHandler
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
    pressure_strategy_num = 0 # Pressure control strategy to use

    # Load parameters from config file
    config_string = "./config/Cases/"+ config_file +".json"
    config = Config.load_config(config_string)
    config.set_growth_case(growth_case)
    # Overwrite default config parameters for this experiment with the specified ones above
    config.config_access["RUN_PARAMETERS"]["output_path"] = f"./outputs/results"
    config.config_access["GROWTH_PARAMETERS"]["sprouting_strategy"] = 3
    config.config_access["GROWTH_PARAMETERS"]["pressure_strategy"] = pressure_strategy_num
    config.config_access["RUN_PARAMETERS"]["test_name"] = f"ratio_{ratio}"
    config.config_access["GROWTH_PARAMETERS"]["flow_volume_ratio"] = ratio
    config.setup_output_folder()
    config.setup_case_folder()

    config.logger.log(f"Output Path is {config.test_name}")

    def connect_artery_vein_trees(file1, file2, inlet1_coord, inlet2_coord, terminal_array="isTerminal", debug=False):
        """
        Connects Artery (Flow: Inlet->Capillary) to Vein (Flow: Capillary->Outlet).
        Ensures correct segment directionality using BFS.
        """
        
        # --- Helper to load and process a mesh ---
        def process_mesh(fname, label):
            # print(f"Loading {label} from {fname}...")
            raw = pv.read(fname)
            mesh = raw.clean(tolerance=1e-15) # Merge duplicate points
            
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
        
        # print(f"Terminals to connect: {len(terms1)} (File 1) vs {len(terms2)} (File 2)")
        
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
        
        # print(f"Total Nodes Created: {global_id_counter[0]}")

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
            
            r = (radii_art[u] + radii_art[v]) / 2.0
            
            
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
            
            r = (radii_vein[u] + radii_vein[v]) / 2.0
            
            
            # SWAP: add_segment(upstream, downstream) -> (v, u)
            
            my_tree.add_segment(nv, nu, seg_counter, radius=r)
            seg_counter += 1
                
        # --- BRIDGE SEGMENTS ---
        # Connect Artery Terminal -> Vein Terminal
        for key1, key2 in bridge_pairs:
            nu = node_obj_map[key1] # Artery (Upstream)
            nv = node_obj_map[key2] # Vein (Downstream)
            
            r1 = data1["radii"][key1[1]]
            r2 = data2["radii"][key2[1]]
            r_bridge = (r1+r2)/2.0
            
            
            my_tree.add_segment(nu, nv, seg_counter, radius=r_bridge)
            seg_counter += 1

        # 6. Finalize
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

    
    my_tree = connect_artery_vein_trees("../CCO/ex1_simple3D_1.vtp", "../CCO/ex1_simple3D_2.vtp",
                                         [0, 0.000125, 0.000125], [0, 0.000125 - 2.5e-5, 0.000125], terminal_array="isTerminal")
   
   
    # Initialize the tissue domain, setting the physical limits and the mesh resolution
    mesh_cells = config.config_access["3D_CONDITIONS"]["num_cells"] + 1
    tissue_handler = TissueHandler()
    tissue_handler.set_config(config)
    tissue_handler.add_tissue([0,2.5e-4],[0,2.5e-4],[0,2.5e-4],[mesh_cells,mesh_cells,mesh_cells]) 
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
    solver.iterative_solve_fluid_1D(tolerance=1e-8, tolerance_h=1e-8, alpha=0.9, beta=0.7, max_iterations=100)
    # solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    time_18 = time.time()
    # solver.iterative_solve_oxygen(tolerance=1e-8, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    # time_19 = time.time()
    

    mat_handler.save_tissue(0,vegf=False)
    my_tree.save_from_config(config)
    visualizer = growth_handler.initialize_visualizer()
    mat_handler.add_properties_to_visualizer(visualizer,True,False,False)
    visualizer.save_to_file([0])


    # Update time trackers
    haemodynamic_time += (time_18 - time_17)
    stats = growth_handler.get_hemo_statistics(0)
    # stats = growth_handler.get_vascular_statistics(0)
    # stats = growth_handler.get_hemo_o2_statistics(0)



    config.logger.log("TIME SUMMARY FOR SIMULATION")
    config.logger.log(f"Total haemodynamic time: {haemodynamic_time} seconds")
    config.remove_from_lock_file()
    print(f"resistance: {stats['hydraulic_resist']}")
    # print(f"mean O2 value: {stats['global_oxygen_mean']}")
    # print(f"outlet oxygen: {100 * stats['outgoing_oxygen']/stats['incoming_oxygen']}%")
    return

if __name__ == "__main__":
    main()



