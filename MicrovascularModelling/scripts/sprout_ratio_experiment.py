from classes.GeometryClasses import Node, Tree, TissueHandler, Visualizer
from classes.ConfigClass import Config
from classes.MatrixHandlerClasses import MatrixHandler, MatrixSolver
from classes.GetFEMClasses import GetFEMHandler1D, GetFEMHandler3D
from classes.GrowthClasses import GrowthHandler
import sys

def main():
    config_file = "single_growth"
    # growth_case = 188
    age = 0
    ratio = round(int(sys.argv[1]) * 0.1,1)
    growth_case = int(sys.argv[2])
    
    config_string = "./config/Cases/"+ config_file +".json"
    config = Config.load_config(config_string)
    config.set_age_to_load(age)
    config.set_growth_case(growth_case)

    config.config_access["RUN_PARAMETERS"]["output_path"] = f"./outputs/SPROUT_RATIO_NEW"
    config.config_access["RUN_PARAMETERS"]["test_name"] = f"ratio_{ratio}"
    config.config_access["GROWTH_PARAMETERS"]["flow_volume_ratio"] = ratio
    config.setup_output_folder()
    config.setup_case_folder()

    config.logger.log(f"RATIO = {ratio}")
    config.logger.log(f"Output Path is {config.test_name}")

    def make_tree_baseline():
        my_tree = Tree()
        N0 = my_tree.add_inlet([0.1e-4,1.25e-4,2.5e-4],0)
        N1 = my_tree.add_node([0.1e-4,1.25e-4,0.1e-4],1)
        N2 = my_tree.add_node([2.4e-4,0.1e-4,0.1e-4],2)
        N3 = my_tree.add_outlet([2.4e-4,0.1e-4,2.5e-4],3)
        N4 = my_tree.add_node([2.4e-4,2.4e-4,0.1e-4],4)
        N5 = my_tree.add_outlet([2.4e-4,2.4e-4,2.5e-4],5)

        S0 = my_tree.add_segment(N0,N1,0,radius=3e-6)
        S1 = my_tree.add_segment(N1,N2,1,radius=3e-6)
        S2 = my_tree.add_segment(N2,N3,2,radius=3e-6)
        S3 = my_tree.add_segment(N1,N4,3,radius=3e-6)
        S4 = my_tree.add_segment(N4,N5,4,radius=3e-6)

        my_tree.populate_junctions()
        return my_tree
    
    my_tree = make_tree_baseline()
    my_tree.apply_maximum_segment_length(5e-6)
    
    # age = 34
    if age != 0:
        my_tree =  Tree.load_from_config(config,True,age)

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

    growth_handler.set_inlet_pressure_for_flow_volume_ratio(ratio)
    max_iter = 1
    config.logger.log("ENTERING GROWTH LOOP")
    num_loops = 10
    for loops in range(num_loops):
        for i in range(max_iter):
            mat_handler.reset_system()
            solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
            norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
            solver.iterative_solve_vegf(alpha=1)
            mat_handler.save_tissue(age,vegf=True)
    
            config.logger.log("STARTING GROWTH STEP")
            
            growth_handler.set_weighting_eval_metric_acculmulate()
            growth_handler.load_growth_case()
            growth_handler.age = age
            wss_solution = growth_handler.get_wss()
            growth_handler.run_growth_process()
            growth_handler.save_sprout_metrics_to_json()
            growth_handler.get_vessels_from_segments()
            age = growth_handler.age
    
            config.logger.log(my_tree)
            getfem_handler_1D = GetFEMHandler1D.load_config(config)
            getfem_handler_3D = GetFEMHandler3D.load_config(config)
            mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
            solver = MatrixSolver(config.logger,mat_handler)
            growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

            solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100)
            norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-7, max_iterations=200,alpha=0.85,very_verbose=False)
            solver.iterative_solve_vegf(alpha=1)
            mat_handler.save_tissue(age,vegf=True)
            new_pressure = growth_handler.run_pressure_control_mechanism(ratio)

            getfem_handler_1D.set_inlet_pressure(new_pressure)
            # ratio = growth_handler.get_flow_volume_ratio()
            
            config.set_age_to_load(age)
            # my_tree.save_from_config(config)
            config.logger.log("ENDING GROWTH STEP")
          
        mat_handler.reset_system()

        adjustment_made = True
        i = 0
        while adjustment_made and i < 8:
            config.logger.log(f"STARTING ADAPTATION STEP {i}")
            solver.iterative_solve_fluid_1D(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100,verbose=False)
            # mat_handler.save_to_vtk(blood_1D=True)
            growth_handler.age = age
            adjustment_made = growth_handler.run_maintenance_process(verbose=True)
            if adjustment_made:
                new_pressure = growth_handler.run_pressure_control_mechanism(ratio)
            age = growth_handler.age
            config.logger.log(f"ENDING ADAPTATION STEP {i}")
            i += 1
            config.logger.log(f"Was there an adjustment: {adjustment_made}")
            getfem_handler_1D = GetFEMHandler1D.load_config(config)
            getfem_handler_3D = GetFEMHandler3D.load_config(config)
            mat_handler = MatrixHandler(my_tree,tissue_handler.current_tissue(),getfem_handler_1D,getfem_handler_3D,config)
            solver = MatrixSolver(config.logger,mat_handler)
            growth_handler = GrowthHandler(my_tree, mat_handler, solver, tissue_handler.current_tissue(), config)

            getfem_handler_1D.set_inlet_pressure(new_pressure) # type: ignore

        solver.iterative_solve_fluid(tolerance=1e-8, tolerance_h=1e-8, alpha=0.9, beta=0.7, max_iterations=100)
        norms, convergence_rates = solver.iterative_solve_oxygen(tolerance=1e-8, max_iterations=200,alpha=0.85,momentum=0.95,very_verbose=False)
        solver.iterative_solve_vegf(alpha=1)

        tissue = tissue_handler.current_tissue()
        tissue.sample_n_subdomains(5,21.6,my_tree,loops,True,False,growth_handler)
        tissue.sample_n_subdomains(5,21.6,my_tree,loops,False,False,growth_handler)

        growth_handler.age = age + 1
        mat_handler.save_tissue(age,vegf=True)
        visualizer = growth_handler.initialize_visualizer()
        mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
        visualizer.save_to_file([0])

        # mat_handler.oxygen_summary()
        stats = growth_handler.get_vascular_statistics(loops)
        growth_handler.eval_vegf(loops)
    # config.logger.log(stats)
    config.remove_from_lock_file()

    config.logger.log(f"RATIO = {ratio}")
    config.logger.log(f"Pressure = {new_pressure}")

    return

if __name__ == "__main__":
    main()



