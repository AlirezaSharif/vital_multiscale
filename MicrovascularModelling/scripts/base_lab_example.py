from classes.GeometryClasses import Node, Tree, TissueHandler, Visualizer
from classes.ConfigClass import Config
from classes.MatrixHandlerClasses import MatrixHandler, MatrixSolver
from classes.GetFEMClasses import GetFEMHandler1D, GetFEMHandler3D
from classes.GrowthClasses import GrowthHandler
import sys
import time

def main():
    # These parameters contorl elements of the growth strategy
    config_file = "single_growth" # Physical parameters for the simulation
    age = 0 # Starting age of the simulation only no 0 if you want to load from a previous age (requires additional setup)
    ratio = 1.4 # Flow volume ratio to target
    growth_case = 165 # Growth case to use (how far the vessel grows before seeking anastomosis)
    pressure_strategy_num = 1 # Pressure control strategy to use

    # Load parameters from config file
    config_string = "./config/Cases/"+ config_file +".json"
    config = Config.load_config(config_string)
    config.set_age_to_load(age)
    config.set_growth_case(growth_case)

    # Overwrite default config parameters for this experiment with the specified ones above
    config.config_access["RUN_PARAMETERS"]["output_path"] = f"./outputs/base_lab_example"
    config.config_access["GROWTH_PARAMETERS"]["sprouting_strategy"] = 3
    config.config_access["GROWTH_PARAMETERS"]["pressure_strategy"] = pressure_strategy_num
    config.config_access["RUN_PARAMETERS"]["test_name"] = f"ratio_{ratio}"
    config.config_access["GROWTH_PARAMETERS"]["flow_volume_ratio"] = ratio
    config.setup_output_folder()
    config.setup_case_folder()

    config.logger.log(f"Output Path is {config.test_name}")

    # Define a baseline tree structure (This is where you define the initial vascular network)
    def make_tree_baseline():
        my_tree = Tree() # create empty tree

        # Define nodal locations in space giving each an unique ID number
        # Mulitple inlets and outlets can be specified but all inlets and outlets share the same boundary conditions
        N0 = my_tree.add_inlet([0.1e-4,1.25e-4,2.5e-4],0) # specify type as inlet
        N1 = my_tree.add_node([0.1e-4,1.25e-4,0.1e-4],1)
        N2 = my_tree.add_node([2.4e-4,0.1e-4,0.1e-4],2)
        N3 = my_tree.add_outlet([2.4e-4,0.1e-4,2.5e-4],3) # specify type as outlet
        N4 = my_tree.add_node([2.4e-4,2.4e-4,0.1e-4],4)
        N5 = my_tree.add_outlet([2.4e-4,2.4e-4,2.5e-4],5) # specify type as outlet

        # Define segments between nodal IDs giving each an unique ID number and an initial radius
        # For the construction to be valid all nodes must be connected and each node must be connected to at most 3 segments
        S0 = my_tree.add_segment(N0,N1,0,radius=3e-6)
        S1 = my_tree.add_segment(N1,N2,1,radius=3e-6)
        S2 = my_tree.add_segment(N2,N3,2,radius=3e-6)
        S3 = my_tree.add_segment(N1,N4,3,radius=3e-6)
        S4 = my_tree.add_segment(N4,N5,4,radius=3e-6)

        # Finalize tree structure by populating junctions based on segments defined
        my_tree.populate_junctions()
        return my_tree
    
    # Initialize tree structure and apply a standard maximum segment length refinement
    my_tree = make_tree_baseline()
    my_tree.apply_maximum_segment_length(5e-6)
    
    # If the age is not zero load the tree structure from file
    if age != 0:
        my_tree =  Tree.load_from_config(config,True,age)
    
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
    oxygen_time = 0
    vegf_time = 0
    growth_time = 0
    adaptation_time = 0
    saving_time = 0

    config.logger.log("ENTERING GROWTH LOOP")
    num_consecutive_growth_steps = 1
    # Define number of growth loops to perform
    num_loops = 10
    for loops in range(num_loops):

    # # Or Run until a target global oxygen level is reached
    # current_o2 = 0
    # target_o2 = 0.0003 # SET YOUR TARGET GLOBAL OXYGEN LEVEL HERE (mLO2/cm3)
    # o2_wiggle_room = 0.00005 # SET YOUR WIGGLE ROOM HERE
    # max_loops = 1
    # loops = 0
    # while (current_o2 < (target_o2 - o2_wiggle_room) or current_o2 > (target_o2 + o2_wiggle_room)) and loops < max_loops:
        for i in range(num_consecutive_growth_steps):
            # Initial State Evaluation
            mat_handler.reset_system()
            time_1 = time.time()
            solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
            time_2 = time.time()
            norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
            time_3 = time.time()
            solver.iterative_solve_vegf(alpha=1)
            time_4 = time.time()
            mat_handler.save_tissue(age,vegf=True)
            time_5 = time.time()

            # Growth Step
            config.logger.log("STARTING GROWTH STEP")
            growth_handler.set_weighting_eval_metric_acculmulate()
            growth_handler.load_growth_case()
            growth_handler.age = age
            wss_solution = growth_handler.get_wss()
            growth_handler.run_growth_process()
            growth_handler.save_sprout_metrics_to_json()
            growth_handler.get_vessels_from_segments()
            age = growth_handler.age
            time_6 = time.time()

            # Re-evaluate system after growth
            # config.logger.log(my_tree)
            getfem_handler_1D = GetFEMHandler1D.load_config(config)
            getfem_handler_3D = GetFEMHandler3D.load_config(config)
            mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
            solver = MatrixSolver(config.logger,mat_handler)
            growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

            time_7 = time.time()
            solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
            time_8 = time.time()
            norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,very_verbose=False)
            time_9 = time.time()
            solver.iterative_solve_vegf(alpha=1)
            time_10 = time.time()
            mat_handler.save_tissue(age,vegf=True)
            time_11 = time.time()
            new_pressure = growth_handler.run_pressure_control_mechanism(ratio,pressure_strategy_num)

            getfem_handler_1D.set_inlet_pressure(new_pressure)
            ratio = growth_handler.get_flow_volume_ratio()
            time_12 = time.time()
            config.set_age_to_load(age)
            # my_tree.save_from_config(config)
            config.logger.log("ENDING GROWTH STEP")

            # Update time trackers
            haemodynamic_time += (time_2 - time_1) + (time_8 - time_7) + (time_12 - time_11)
            oxygen_time += (time_3 - time_2) + (time_9 - time_8)
            vegf_time += (time_4 - time_3) + (time_10 - time_9)
            growth_time += (time_6 - time_5)
            saving_time += (time_5 - time_4) + (time_11 - time_10)

        mat_handler.reset_system()

        # Adaptation Loop
        adjustment_made = True
        i = 0
        while adjustment_made and i < 8:
            # Perform a 1D fluid solve to update flows and shear stresses
            config.logger.log(f"STARTING ADAPTATION STEP {i}")
            time_13 = time.time()
            solver.iterative_solve_fluid_1D(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100,verbose=False)
            time_14 = time.time()
            # mat_handler.save_to_vtk(blood_1D=True)
            growth_handler.age = age

            # Run the adaptation process
            adjustment_made = growth_handler.run_maintenance_process(verbose=True)
            time_15 = time.time()
            if adjustment_made:
                new_pressure = growth_handler.run_pressure_control_mechanism(ratio,pressure_strategy_num)
            time_16 = time.time()
            age = growth_handler.age
            config.logger.log(f"ENDING ADAPTATION STEP {i}")
            i += 1
            config.logger.log(f"Was there an adjustment: {adjustment_made}")

            # Re-initialize system after adaptation
            getfem_handler_1D = GetFEMHandler1D.load_config(config)
            getfem_handler_3D = GetFEMHandler3D.load_config(config)
            mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
            solver = MatrixSolver(config.logger,mat_handler)
            growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)
            getfem_handler_1D.set_inlet_pressure(new_pressure) # type: ignore
            # Update time trackers
            haemodynamic_time += (time_14 - time_13) + (time_16 - time_15)
            adaptation_time += (time_15 - time_14)

        # Final solve and save state at end of growth loop
        time_17 = time.time()
        solver.iterative_solve_fluid(tolerance=1e-8, tolerance_h=1e-8, alpha=0.9, beta=0.7, max_iterations=100)
        time_18 = time.time()
        norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-8, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
        time_19 = time.time()
        solver.iterative_solve_vegf(alpha=1)
        time_20 = time.time()

        growth_handler.age = age + 1
        mat_handler.save_tissue(age,vegf=True)
        my_tree.save_from_config(config)
        visualizer = growth_handler.initialize_visualizer()
        mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
        visualizer.save_to_file([0])
        time_21 = time.time()

        # Update time trackers
        haemodynamic_time += (time_18 - time_17)
        oxygen_time += (time_19 - time_18)
        vegf_time += (time_20 - time_19)
        saving_time += (time_21 - time_20)
        # mat_handler.oxygen_summary()
        stats = growth_handler.get_vascular_statistics(loops)
        current_o2 = stats["global_oxygen_mean"]
        growth_handler.eval_vegf(loops)
        loops += 1


    config.logger.log("TIME SUMMARY FOR SIMULATION")
    config.logger.log(f"Total haemodynamic time: {haemodynamic_time} seconds")
    config.logger.log(f"Total oxygen time: {oxygen_time} seconds")
    config.logger.log(f"Total VEGF time: {vegf_time} seconds")
    config.logger.log(f"Total growth time: {growth_time} seconds")
    config.logger.log(f"Total adaptation time: {adaptation_time} seconds")
    config.logger.log(f"Total saving time: {saving_time} seconds")
    config.remove_from_lock_file()

    return

if __name__ == "__main__":
    main()



