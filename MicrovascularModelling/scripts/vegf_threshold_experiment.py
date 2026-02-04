from classes.GeometryClasses import Node, Tree, TissueHandler, Visualizer, GraphRepresentation
from classes.ConfigClass import Config
from classes.MatrixHandlerClasses import MatrixHandler, MatrixSolver
from classes.GetFEMClasses import GetFEMHandler1D, GetFEMHandler3D
from classes.GrowthClasses import GrowthHandler
import sys
import csv
import numpy as np # type: ignore


def main():
    config_file = "single_growth"
    growth_case = 188
    age = 0
    ratio = 0.3

    print(f"Starting Experiment")

    config_string = "./config/Cases/"+ config_file +".json"
    config = Config.load_config(config_string)
    config.set_age_to_load(age)
    config.set_growth_case(growth_case)
    config.config_access["RUN_PARAMETERS"]["output_path"] = "./outputs/VEGF_THRESHOLD_LOWER"
    config.test_name = f"ratio_{ratio}"
    config.setup_output_folder()
    config.setup_case_folder()
    
    config.config_access["GROWTH_PARAMETERS"]["k_up"] = 0.1
    config.config_access["GROWTH_PARAMETERS"]["r_min"] = -1e-6

    mesh_cells = config.config_access["3D_CONDITIONS"]["num_cells"] + 1
    tissue_handler = TissueHandler()
    tissue_handler.set_config(config)
    tissue_handler.add_tissue([0,2.5e-4],[0,2.5e-4],[0,2.5e-4],[mesh_cells,mesh_cells,mesh_cells]) # type: ignore
    x,y,z = tissue_handler.current_tissue().tissue_limits()
    Node.set_limits(x,y,z)


    config.logger.log(f"Output Path is {config.test_name}")

    def make_tree_undersupply():
        my_tree = Tree()
        N0 = my_tree.add_inlet([1.25e-4,1.25e-4,0e-4],0)
        N1 = my_tree.add_outlet([1.25e-4,1.25e-4,2.5e-4],1)
    
        S0 = my_tree.add_segment(N0,N1,0,radius=3e-6)

        my_tree.populate_junctions()
        return my_tree
    
    def generate_alternating_points(min_x, max_x, min_y, max_y, n_generations):
        points = [(0.5 * (max_x - min_x) + min_x, 0.5 * (max_y - min_y) + min_y, 1,-1)]  # Start with center
        

        x_step = 0.25 * (max_x - min_x)
        y_step = 0.25 * (max_y - min_y)
        current_index = 1
        old_points = []
        connectivity = []
        
        for gen in range(n_generations):
            new_points = []
            if gen % 2 == 0:
                # Split along x-direction
                for x, y, num, _ in points:
                    current_index += 1
                    new_points.append((x+x_step, y, current_index, gen))
                    connectivity.append((num,current_index))
                
                    current_index += 1
                    new_points.append((x-x_step, y, current_index, gen))
                    connectivity.append((num,current_index))
                x_step = x_step/2
            else:
                # Split along y-direction
                for x, y, num, _ in points:
                    current_index += 1
                    new_points.append((x, y+y_step, current_index, gen))
                    connectivity.append((num,current_index))

                    current_index += 1
                    new_points.append((x, y-y_step, current_index, gen))
                    connectivity.append((num,current_index))
                y_step = y_step/2

            old_points.extend(points)
            points = new_points

        old_points.extend(points)
        
        return np.array(old_points), np.array(connectivity)

    def make_tree_supply(gen_num):
        my_tree = Tree()
        N0 = my_tree.add_inlet([1.25e-4,1.25e-4,0e-4])
        node_count = 0

        points, connections = generate_alternating_points(0,2.5e-4,0,2.5e-4,gen_num)
        # bifurcations
        for point in points:
            my_tree.add_node([point[0],point[1],0.5e-4])
            node_count += 1

        my_tree.add_segment(0,1,radius=6e-6)
        radii = 5e-6
        radii_degrade_count = 2
        segment_count = 0
        for connection in connections:
            my_tree.add_segment(connection[0],connection[1],radius=radii)
            segment_count += 1
            if segment_count >= radii_degrade_count:
                radii_degrade_count += 2*radii_degrade_count
                radii = max([radii-1e-6,3e-6])

        first_side_nodes = node_count
        # anastomoses
        for point in points:
            my_tree.add_node([point[0],point[1],2.0e-4])
            node_count += 1

        my_tree.add_segment(0,1,radius=6e-6)
        radii = 5e-6
        radii_degrade_count = 2
        segment_count = 0
        for connection in connections:
            my_tree.add_segment(connection[0]+first_side_nodes,connection[1]+first_side_nodes,radius=radii)
            segment_count += 1
            if segment_count >= radii_degrade_count:
                radii_degrade_count += 2*radii_degrade_count
                radii = max([radii-1e-6,3e-6])
        
        N1 = my_tree.add_outlet([1.25e-4,1.25e-4,2.5e-4])
        my_tree.add_segment(first_side_nodes+1,N1,radius=6e-6) 

        # side connections
        # Extract the last column (gen values)
        gen_values = points[:, 3]

        # Find the maximum gen value
        max_gen = np.max(gen_values)

        # Get the ids where gen is equal to max_gen
        ids_with_max_gen = points[gen_values == max_gen, 2].astype(int).tolist()
        
        for id in ids_with_max_gen:
            my_tree.add_segment(id,id+first_side_nodes,radius=3e-6)

        my_tree.populate_junctions()
        return my_tree
    # def make_tree_supply():
    #     my_tree = Tree()
    #     N0 = my_tree.add_inlet([1.25e-4,1.25e-4,0e-4],0)
    #     N1 = my_tree.add_node([1.25e-4,1.25e-4,0.5e-4],1)
    #     N2 = my_tree.add_node([1.75e-4,1.25e-4,0.5e-4],2)
    #     N3 = my_tree.add_node([0.75e-4,1.25e-4,0.5e-4],3)
    #     N4 = my_tree.add_node([1.75e-4,1.75e-4,0.5e-4],4)
    #     N5 = my_tree.add_node([1.75e-4,0.75e-4,0.5e-4],5)
    #     N6 = my_tree.add_node([0.75e-4,1.75e-4,0.5e-4],6)
    #     N7 = my_tree.add_node([0.75e-4,0.75e-4,0.5e-4],7)
    #     N8 = my_tree.add_node([1.75e-4,1.75e-4,2.0e-4],8)
    #     N9 = my_tree.add_node([1.75e-4,0.75e-4,2.0e-4],9)
    #     N10 = my_tree.add_node([0.75e-4,1.75e-4,2.0e-4],10)
    #     N11 = my_tree.add_node([0.75e-4,0.75e-4,2.0e-4],11)
    #     N12 = my_tree.add_node([1.25e-4,1.75e-4,2.0e-4],12)
    #     N13 = my_tree.add_node([1.25e-4,0.75e-4,2.0e-4],13)
    #     N14 = my_tree.add_node([1.25e-4,1.25e-4,2.0e-4],14)
    #     N15 = my_tree.add_outlet([1.25e-4,1.25e-4,2.5e-4],15)
    
    #     S0 = my_tree.add_segment(N0,N1,0,radius=5e-6)
    #     S1 = my_tree.add_segment(N1,N2,1,radius=4e-6)
    #     S2 = my_tree.add_segment(N1,N3,2,radius=4e-6)
    #     S3 = my_tree.add_segment(N2,N4,3,radius=3e-6)
    #     S4 = my_tree.add_segment(N2,N5,4,radius=3e-6)
    #     S5 = my_tree.add_segment(N3,N6,5,radius=3e-6)
    #     S6 = my_tree.add_segment(N3,N7,6,radius=3e-6)
    #     S7 = my_tree.add_segment(N4,N8,7,radius=3e-6)
    #     S8 = my_tree.add_segment(N5,N9,8,radius=3e-6)
    #     S9 = my_tree.add_segment(N6,N10,9,radius=3e-6)
    #     S10 = my_tree.add_segment(N7,N11,10,radius=3e-6)
    #     S11 = my_tree.add_segment(N8,N12,11,radius=3e-6)
    #     S12 = my_tree.add_segment(N10,N12,12,radius=3e-6)
    #     S13 = my_tree.add_segment(N9,N13,13,radius=3e-6)
    #     S14 = my_tree.add_segment(N11,N13,14,radius=3e-6)
    #     S15 = my_tree.add_segment(N12,N14,15,radius=4e-6)
    #     S16 = my_tree.add_segment(N13,N14,16,radius=4e-6)
    #     S17 = my_tree.add_segment(N14,N15,17,radius=5e-6)

    #     my_tree.populate_junctions()
    #     return my_tree

    # def eval_vegf(tree, growth_handler):
    #     polling_rate = 11
    #     used_ids = []
    #     master_points = []
    #     segment_ids = []
    #     for keys in tree.segment_dict.keys():
    #         # Retrieve the node positions
    #         node_1_id = tree.segment_dict[keys].node_1_id
    #         node_2_id = tree.segment_dict[keys].node_2_id

    #         node_1_pos = tree.node_dict[node_1_id].location()
    #         node_2_pos = tree.node_dict[node_2_id].location()

    #         # Generate the points to add to the mesh
    #         local_points = np.linspace(node_1_pos,node_2_pos, polling_rate)

    #         # Add used ids to the used 1D list
    #         if node_1_id in used_ids:
    #             local_points = local_points[1:]
    #         else:
    #             used_ids.append(node_1_id)
    #         if node_2_id in used_ids:
    #             local_points = local_points[:-2]
    #         else:
    #             used_ids.append(node_2_id)

    #         master_points.extend(local_points)
    #         # Store the segment ID for each point
    #         segment_ids.extend([keys] * len(local_points))

    #     location_array = np.array(master_points)
    #     loc_col = np.hsplit(location_array,3)
    #     growth_handler._get_growth_info()
    #     vegf_values = growth_handler.sample_vegf_values(loc_col)

    #     vegf_mean = np.mean(vegf_values)

    #     return vegf_values, vegf_mean
    
    # my_tree = make_tree_undersupply()
    # my_tree.apply_maximum_segment_length(5e-6)

    # getfem_handler_1D = GetFEMHandler1D.load_config(config)
    # getfem_handler_3D = GetFEMHandler3D.load_config(config)

    # mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
    # solver = MatrixSolver(config.logger,mat_handler)
    # growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

    # growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)

    # mat_handler.reset_system()
    # solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    # norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    # solver.iterative_solve_vegf(alpha=1)

    # vegf_values_under, vegf_mean_under = growth_handler.eval_vegf()

    # config.logger.log(f"VEGF VALUES UNDER: {vegf_values_under}")
    # config.logger.log(f"VEGF MEAN UNDER: {vegf_mean_under}")

    # mat_handler.save_tissue(age,vegf=True)
    # visualizer = growth_handler.initialize_visualizer()
    # mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
    # visualizer.save_to_file([0])

    config.config_access["RUN_PARAMETERS"]["output_path"] = "./outputs/VEGF_1_GEN"
    config.test_name = f"ratio_{ratio}"
    config.setup_output_folder()
    config.setup_case_folder()


    my_tree = make_tree_supply(1)
    my_tree.apply_maximum_segment_length(5e-6)

    # graph = GraphRepresentation()
    # graph.build_undirected(my_tree)

    # inlet_ids = my_tree.get_node_ids_inlet()
    # outlet_ids = my_tree.get_node_ids_outlet()
    
    # seg_ids = np.array([my_tree.segment_dict[seg_id].segment_id for seg_id in my_tree.segment_dict])
    # lengths = np.array([my_tree.length(seg_id) for seg_id in seg_ids])

    # edge_lengths = dict(zip(seg_ids, lengths))

    # inlet_distances = np.array(list(graph.shortest_distances_from_sources(inlet_ids,edge_lengths).values()))
    # outlet_distances = np.array(list(graph.shortest_distances_from_sources(outlet_ids,edge_lengths).values()))

    # config.logger.log(inlet_ids)
    # config.logger.log(outlet_ids)
    # config.logger.log(edge_lengths)
    # config.logger.log(inlet_distances)
    # config.logger.log(outlet_distances)
    # config.logger.log(f"inlet_distances shape: {np.shape(inlet_distances)}")
    # config.logger.log(f"outlet_distances shape: {np.shape(outlet_distances)}")



    getfem_handler_1D = GetFEMHandler1D.load_config(config)
    getfem_handler_3D = GetFEMHandler3D.load_config(config)

    mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
    solver = MatrixSolver(config.logger,mat_handler)
    growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

    growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)

    mat_handler.reset_system()
    solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    solver.iterative_solve_vegf(alpha=1)

    vegf_values_supply, vegf_mean_supply = growth_handler.eval_vegf()

    mat_handler.save_tissue(age,vegf=True)
    visualizer = growth_handler.initialize_visualizer()
    mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
    visualizer.save_to_file([0])

    config.logger.log(f"VEGF VALUES SUPPLY: {vegf_values_supply}")
    config.logger.log(f"VEGF MEAN SUPPLY: {vegf_mean_supply}")


    config.config_access["RUN_PARAMETERS"]["output_path"] = "./outputs/VEGF_2_GEN"
    config.test_name = f"ratio_{ratio}"
    config.setup_output_folder()
    config.setup_case_folder()


    my_tree = make_tree_supply(2)
    my_tree.apply_maximum_segment_length(5e-6)

    getfem_handler_1D = GetFEMHandler1D.load_config(config)
    getfem_handler_3D = GetFEMHandler3D.load_config(config)

    mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
    solver = MatrixSolver(config.logger,mat_handler)
    growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

    growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)

    mat_handler.reset_system()
    solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    solver.iterative_solve_vegf(alpha=1)

    vegf_values_supply, vegf_mean_supply = growth_handler.eval_vegf()

    mat_handler.save_tissue(age,vegf=True)
    visualizer = growth_handler.initialize_visualizer()
    mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
    visualizer.save_to_file([0])

    config.logger.log(f"VEGF VALUES SUPPLY: {vegf_values_supply}")
    config.logger.log(f"VEGF MEAN SUPPLY: {vegf_mean_supply}")

    config.config_access["RUN_PARAMETERS"]["output_path"] = "./outputs/VEGF_3_GEN"
    config.test_name = f"ratio_{ratio}"
    config.setup_output_folder()
    config.setup_case_folder()


    my_tree = make_tree_supply(3)
    my_tree.apply_maximum_segment_length(5e-6)

    getfem_handler_1D = GetFEMHandler1D.load_config(config)
    getfem_handler_3D = GetFEMHandler3D.load_config(config)

    mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
    solver = MatrixSolver(config.logger,mat_handler)
    growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

    growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)

    mat_handler.reset_system()
    solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    solver.iterative_solve_vegf(alpha=1)

    vegf_values_supply, vegf_mean_supply = growth_handler.eval_vegf()

    mat_handler.save_tissue(age,vegf=True)
    visualizer = growth_handler.initialize_visualizer()
    mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
    visualizer.save_to_file([0])

    config.logger.log(f"VEGF VALUES SUPPLY: {vegf_values_supply}")
    config.logger.log(f"VEGF MEAN SUPPLY: {vegf_mean_supply}")

    config.config_access["RUN_PARAMETERS"]["output_path"] = "./outputs/VEGF_4_GEN"
    config.test_name = f"ratio_{ratio}"
    config.setup_output_folder()
    config.setup_case_folder()


    my_tree = make_tree_supply(4)
    my_tree.apply_maximum_segment_length(5e-6)

    getfem_handler_1D = GetFEMHandler1D.load_config(config)
    getfem_handler_3D = GetFEMHandler3D.load_config(config)

    mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
    solver = MatrixSolver(config.logger,mat_handler)
    growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

    growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)

    mat_handler.reset_system()
    solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    solver.iterative_solve_vegf(alpha=1)

    vegf_values_supply, vegf_mean_supply = growth_handler.eval_vegf()

    mat_handler.save_tissue(age,vegf=True)
    visualizer = growth_handler.initialize_visualizer()
    mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
    visualizer.save_to_file([0])

    config.logger.log(f"VEGF VALUES SUPPLY: {vegf_values_supply}")
    config.logger.log(f"VEGF MEAN SUPPLY: {vegf_mean_supply}")

    config.config_access["RUN_PARAMETERS"]["output_path"] = "./outputs/VEGF_5_GEN"
    config.test_name = f"ratio_{ratio}"
    config.setup_output_folder()
    config.setup_case_folder()

    my_tree = make_tree_supply(5)
    my_tree.apply_maximum_segment_length(5e-6)

    getfem_handler_1D = GetFEMHandler1D.load_config(config)
    getfem_handler_3D = GetFEMHandler3D.load_config(config)

    mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
    solver = MatrixSolver(config.logger,mat_handler)
    growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

    growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)

    mat_handler.reset_system()
    solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    solver.iterative_solve_vegf(alpha=1)

    vegf_values_supply, vegf_mean_supply = growth_handler.eval_vegf()

    mat_handler.save_tissue(age,vegf=True)
    visualizer = growth_handler.initialize_visualizer()
    mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
    visualizer.save_to_file([0])

    config.logger.log(f"VEGF VALUES SUPPLY: {vegf_values_supply}")
    config.logger.log(f"VEGF MEAN SUPPLY: {vegf_mean_supply}")


    config.config_access["RUN_PARAMETERS"]["output_path"] = "./outputs/VEGF_6_GEN"
    config.test_name = f"ratio_{ratio}"
    config.setup_output_folder()
    config.setup_case_folder()

    my_tree = make_tree_supply(6)
    my_tree.apply_maximum_segment_length(5e-6)

    getfem_handler_1D = GetFEMHandler1D.load_config(config)
    getfem_handler_3D = GetFEMHandler3D.load_config(config)

    mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
    solver = MatrixSolver(config.logger,mat_handler)
    growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

    growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)

    mat_handler.reset_system()
    solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
    norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
    solver.iterative_solve_vegf(alpha=1)

    vegf_values_supply, vegf_mean_supply = growth_handler.eval_vegf()

    mat_handler.save_tissue(age,vegf=True)
    visualizer = growth_handler.initialize_visualizer()
    mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
    visualizer.save_to_file([0])

    config.logger.log(f"VEGF VALUES SUPPLY: {vegf_values_supply}")
    config.logger.log(f"VEGF MEAN SUPPLY: {vegf_mean_supply}")


if __name__ == "__main__":
    main()