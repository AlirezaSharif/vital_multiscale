import getfem as gf # type: ignore
import numpy as np # type: ignore
import os
from os.path import join as os_path_join
import copy
import json
import time
import csv
from classes.ConfigClass import Config
from classes.GeometryClasses import Tree, Tissue, Sprout, GraphRepresentation, Visualizer
from classes.MatrixHandlerClasses import MatrixHandler, MatrixSolver
from scipy.spatial.transform import Rotation as R # type: ignore
import pyvista as pv # type: ignore
from typing import List, Dict, Tuple, Union
import math
import matplotlib.pyplot as plt

class GrowthHandler():
    """
    A class used to handle the growth process in response to the existing VEGF field.

    ----------
    Class Attributes
    ----------

    ----------
    Instance Attributes
    ----------
    tree : object
        An object containing a Tree class instance.

    sprout_dict : dictionary
        A dictionary containing Sprout class instances.

    vtx : numpy array
        A numpy array containing the information 

    ----------
    Class Methods
    ----------      

    ----------
    Instance Methods
    ----------  
    save_file(filepath)
        Saves class instance to file

    add_tissue(x_values, y_values, z_values, num_cells, cell_type, tissue_id = None)
        Adds a new Tissue instance to the handler

    grow_older()
        Increment the tissue age

    volume(tissue_id, verbose = False)
        Returns the volume of a tissue instance

    count_tissue()
        Returns the length of the tissue dictionary

    current_tissue()
        Returns the tissue which is currently most recent
    
    """

    def __init__(self, tree:Tree, mat_handler:MatrixHandler, solver:MatrixSolver, tissue:Tissue, config:Config):
        self.tree = tree
        self.tissue = tissue
        self.mim = mat_handler.getfem_handler_3D.mim
        self.getfem_handler_3D = mat_handler.getfem_handler_3D
        self.getfem_handler_1D = mat_handler.getfem_handler_1D
        self.solver = solver
        
        self.filepath = mat_handler.filepath
        self.run_name = mat_handler.run_name

        self.mat_handler = mat_handler

        self.cells_1D = config.config_access["1D_CONDITIONS"]["num_cells"]
        self.logger = config.logger
        _,_,_,_,self.run_name = config.parse_run()
        self.output_path = os.path.join(config.return_filepath("output"),config.test_name)
        self.config = config
        self.use_config = False

        self.l1 = 5e-5
        self.l2 = 0

        self.activate_visualisation()

        self.step_size = 5e-6
        self.sprout_growth_limit = 30e-5
        self.generation = 0
        self.age = 0
        self.sigmoid_center = config.growth_params["x_position"]
        self.sigmoid_steepness = config.growth_params["steepness_value"]
        
        
        self._set_from_config()

        self.reset_phase()
        self.reset_phase2()

        self.set_weighting_eval_metric_max()

    def _set_from_config(self):
        self.l1 = self.config.config_access["GROWTH_PARAMETERS"]["search_radius"]
        self.step_size = self.config.config_access["GROWTH_PARAMETERS"]["step_size"]
        self.sprout_growth_limit = self.config.config_access["GROWTH_PARAMETERS"]["sprout_growth_limit"]
        self.inhibition_range = self.config.config_access["GROWTH_PARAMETERS"]["inhibition_range"]
        self.activation_threshold = self.config.config_access["GROWTH_PARAMETERS"]["activation_threshold"]
        self.stochastic = self.config.config_access["GROWTH_PARAMETERS"]["stochastic"]
        self.sigmoid_center = self.config.config_access["GROWTH_PARAMETERS"]["growth_anastomosis_boundary"]
        self.sigmoid_steepness = self.config.config_access["GROWTH_PARAMETERS"]["boundary_steepness"]
        self.flow_volume_ratio = self.config.config_access["GROWTH_PARAMETERS"]["flow_volume_ratio"]
        self.alpha = self.config.config_access["GROWTH_PARAMETERS"]["size_change_alpha"]
        self.r_min = self.config.config_access["GROWTH_PARAMETERS"]["r_min"]
        self.r_max = self.config.config_access["GROWTH_PARAMETERS"]["r_max"]
        self.r_base = self.config.config_access["GROWTH_PARAMETERS"]["r_base"]
        self.k_x = self.config.config_access["GROWTH_PARAMETERS"]["k_x"]
        self.k_up = self.config.config_access["GROWTH_PARAMETERS"]["k_up"]

        self.sprouting_strategy = self.config.config_access["GROWTH_PARAMETERS"]["sprouting_strategy"]
        self.adaptation_strategy = self.config.config_access["GROWTH_PARAMETERS"]["adaptation_strategy"]
        self.pressure_strategy = self.config.config_access["GROWTH_PARAMETERS"]["pressure_strategy"]

        self.use_config = True

    def _get_growth_info(self):
        self.vtx = self.mat_handler.solution_vegf["vtx"]
        self.vtfem = self.mat_handler.vtfem
        self.otx = self.mat_handler.solution_oxygen["otx"]
        self.otfem = self.mat_handler.vtfem

    def _initialize_statistics(self):
        self.total_num_generated_sprouts = 0
        self.total_num_failed_sprouts = 0
        self.sucessful_sprout_length_list = []
        self.sucessful_sprout_bend_list = []
        self.sucessful_sprout_delta_q_list = []
        self.tree_starting_surface = self.tree.get_network_surface()
        self.starting_oxygen_satisfaction = self.calculate_norm_of_o2_field()

    def load_growth_case(self):
        data = self.config.growth_params
        self.test_name = data["test_name"]
        self.sigmoid_center = data["x_position"]
        self.sigmoid_steepness = data["steepness_value"]

        return 
    
    def set_test_name(self,name:str):
        self.test_name = name
        return

    def activate_visualisation(self):
        self.visualisation_state = True
        return
    
    def deactivate_visualisation(self):
        self.visualisation_state = False
        return

    def set_weighting_eval_metric_max(self):
        self.weighting_evaluation_metric = "max"
        return
    
    def set_weighting_eval_metric_acculmulate(self):
        self.weighting_evaluation_metric = "acculmulate"
        return

    def reset_phase(self):
        self.phase = 0
        self.sprout_growth_amount = 0
        self.plotter = None
        self.first = True
        self.inital_keys = []
        self.initial_geo = []
        self.fused_sprouts_geometry = []

    def reset_phase2(self):
        self.phase2 = 0
        self.growth_vector_dict = {}

    def eval_vegf(self,generation:int=None): # type: ignore
        polling_rate = 11
        used_ids = []
        master_points = []
        segment_ids = []
        sub_dist_inlet = []
        sub_dist_outlet = []
        inlet_ids = self.tree.get_node_ids_inlet()
        outlet_ids = self.tree.get_node_ids_outlet()
        # for node_id in self.tree.node_dict.keys():
        #     node_pos = self.tree.node_dict[node_id].location()
        #     master_points.extend(node_pos)
        
        seg_ids = np.array([self.tree.segment_dict[seg_id].segment_id for seg_id in self.tree.segment_dict])
        lengths = np.array([self.tree.length(seg_id) for seg_id in seg_ids])

        edge_lengths = dict(zip(seg_ids, lengths))

        graph = self._make_connectivity_graph()
        inlet_distances = graph.shortest_distances_from_sources(inlet_ids,edge_lengths)
        outlet_distances = graph.shortest_distances_from_sources(outlet_ids,edge_lengths)
        
        for key in self.tree.segment_dict.keys():
            # Retrieve the node positions
            node_1_id = self.tree.segment_dict[key].node_1_id
            node_2_id = self.tree.segment_dict[key].node_2_id

            node_1_pos = self.tree.node_dict[node_1_id].location()
            node_2_pos = self.tree.node_dict[node_2_id].location()

            # Generate the points to add to the mesh
            local_points = np.linspace(node_1_pos,node_2_pos, polling_rate)
            local_distances_in = np.linspace(inlet_distances[node_1_id],inlet_distances[node_2_id],polling_rate)
            local_distances_out = np.linspace(outlet_distances[node_1_id],outlet_distances[node_2_id],polling_rate)

            # Add used ids to the used 1D list
            if node_1_id in used_ids:
                local_points = local_points[1:]
                local_distances_in = local_distances_in[1:]
                local_distances_out = local_distances_out[1:]
            else:
                used_ids.append(node_1_id)
            if node_2_id in used_ids:
                local_points = local_points[:-2]
                local_distances_in = local_distances_in[:-2]
                local_distances_out = local_distances_out[:-2]
            else:
                used_ids.append(node_2_id)

            master_points.extend(local_points)
            sub_dist_inlet.extend(local_distances_in)
            sub_dist_outlet.extend(local_distances_out)
            # Store the segment ID for each point
            segment_ids.extend([key] * len(local_points))

        

        location_array = np.array(master_points)
        loc_col = np.hsplit(location_array,3)
        self._get_growth_info()
        vegf_values = self.sample_vegf_values(loc_col)

        vegf_mean = np.mean(vegf_values)

        # Specify the output CSV filename
        _, _, _, output_path, test_name = self.config.parse_run() # type: ignore
        filepath = os_path_join(output_path,test_name)
        case = self.config.growth_case # type: ignore
        file_path = "./"+filepath+"/statistic_results/"+f"case_{case}/"
        csv_filename = file_path+"percieved_vegf_values"
        if not generation is None:
            csv_filename += f"_{generation}"
        csv_filename += ".csv"

        # Concatenate all data into a single array
        loc_col = np.array(loc_col) 
        loc_col = loc_col.squeeze(-1).T
        sub_dist_inlet = np.array(sub_dist_inlet)
        sub_dist_outlet = np.array(sub_dist_outlet)

        # self.logger.log(f"loc_col shape: {np.shape(loc_col)}")
        # self.logger.log(f"inlet_distances shape: {np.shape(sub_dist_inlet[:, None])}")
        # self.logger.log(f"outlet_distances shape: {np.shape(sub_dist_outlet[:, None])}")
        # self.logger.log(f"vegf_values shape: {np.shape(vegf_values)}")

        final_array = np.hstack((loc_col, sub_dist_inlet[:, None], sub_dist_outlet[:, None], vegf_values))  # Shape (n, 6)

        # Writing to CSV
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(["X Coordinate","Y Coordinate","Z Coordinate","Distance from Inlet","Distance from Outlet","VEGF Values"])
            
            # Write data
            for x,y,z,inlet_dist,outlet_dist,value in final_array:
                writer.writerow([x,y,z,inlet_dist,outlet_dist,value])

        return vegf_values, vegf_mean

    def sample_nodes(self, polling_rate=11, verbose=False) -> np.array:
        if self.phase != 0:
            raise ValueError("sample_nodes should be the first method called in a sequence. If starting a new growth sequence first use .reset_phase()")

        if verbose:
            self.logger.log("################## BEGINNING NODE SAMPLING ##################")

        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("vt", self.vtfem)
        model.set_variable("vt", self.vtx)

        used_ids = []
        master_points = []
        segment_ids = []

        for keys in self.tree.segment_dict.keys():
            # Retrieve the node positions
            node_1_id = self.tree.segment_dict[keys].node_1_id
            node_2_id = self.tree.segment_dict[keys].node_2_id

            node_1_pos = self.tree.node_dict[node_1_id].location()
            node_2_pos = self.tree.node_dict[node_2_id].location()

            # Generate the points to add to the mesh
            local_points = np.linspace(node_1_pos,node_2_pos, polling_rate)

            # Add used ids to the used 1D list
            if node_1_id in used_ids:
                local_points = local_points[1:]
            else:
                used_ids.append(node_1_id)
            if node_2_id in used_ids:
                local_points = local_points[:-2]
            else:
                used_ids.append(node_2_id)

            master_points.extend(local_points)
            # Store the segment ID for each point
            segment_ids.extend([keys] * len(local_points))
        
        location_array = np.array(master_points)
        loc_col = np.hsplit(location_array,3)

        # location_values = model.interpolation("vt", location_array)
        # location_values = gf.compute_interpolate_on(self.vtfem,self.vtx,location_array)
        if verbose:
            self.logger.log("DISPLAYING LOCATION ARRAY")
            self.logger.log(loc_col)

        score = self.calculate_sprouting_score(loc_col,segment_ids)
        segment_ids = np.array(segment_ids)[:,np.newaxis]
        
        if verbose:
            self.logger.log(f"SPROUTING SCORE DISTRIBUTION")
            self.logger.log(f"Maximum Score = {np.max(score)}")
            self.logger.log(f"Minimum Score = {np.min(score)}")
            self.logger.log(f"Mean Score = {np.mean(score)}")
            self.logger.log(f"Median Score = {np.median(score)}")

        self.samples = np.hstack((loc_col[0],loc_col[1],loc_col[2], score, segment_ids))
        self.phase = 1

        if verbose:
            self.logger.log("DISPLAYING SAMPLES")
            self.logger.log(self.samples)
            self.logger.log("################## ENDING NODE SAMPLING ##################")
            
        return self.samples

    def calculate_sprouting_score(self, location:np.ndarray, segment_ids:np.ndarray, mechanism:int=4, stochastic:bool=False) -> np.ndarray:
        """
        Function to calculate the sprouting score of the vessels
        Mechanism determines the approach for the sprouting strategy
        1: Only VEGF
        2: VEGF with Pressure
        3: VEGF with Oxygen
        4: VEGF with Radius

        :param location: A numpy array containing all of the points in space to be evaluated.
        :param mechanism: An int indicating what strategy to use for evaluating sprout locations.
        :param stochastic: A boolean indicating whether to apply a random effect to the score value.
        :return: A numpy array containing a Score for all of the points in space.
        """
        # This function evalueates the score used to calculate the location of our sprouts.
        # Contibuting components are VEGF Exposure, Oxygen Tension, Stochastic Sensitivity
        if self.use_config is True:
            mechanism = self.sprouting_strategy

        # Sample local vegf values in 3D based on points in the 1D
        location_vegf_values = self.sample_vegf_values(location)
        segment_list = np.array([int(seg_id) for seg_id in segment_ids])

        # Sample local oxygen values in the 3D based on points in the 1D
        location_oxygen_values = self.sample_oxygen_values(location)
        self.oxygen_tension_estimate = self.oxygen_to_venousness(location_oxygen_values)

        if not mechanism  in [1,2,3,4]:
            raise ValueError(f"The parameter mechanism ({mechanism}) must be an in belonging ([1,2,3,4])")
        
        # Pure VEGF strategy
        if mechanism == 1:
            score = location_vegf_values

        # VEGF and Pressure Strategy
        # Scale is based on local pressure biasing to low pressure areas (venous). 
        if mechanism == 2:
            max_P = self.getfem_handler_1D.pressure_inlet
            min_P = self.getfem_handler_1D.pressure_outlet

            tree = self.tree
            haemodynamic_solution = self.mat_handler.solution_haemodynamic
            node_point_ref = self.getfem_handler_1D.node_point_ref

            node_ids = np.array([tree.node_dict[node_key].node_id for node_key in tree.node_dict])
            pressures = np.array([haemodynamic_solution["pvx"][node_point_ref[node_id]] for node_id in node_ids])

            nodes_1 = np.array([tree.segment_dict[seg_id].node_1_id for seg_id in segment_list])
            nodes_2 = np.array([tree.segment_dict[seg_id].node_1_id for seg_id in segment_list])

            pressures_1 = np.array([pressures[node_id] for node_id in nodes_1])
            pressures_2 = np.array([pressures[node_id] for node_id in nodes_2])
            pressure_mean = (pressures_1 + pressures_2)/2

            pressure_scale = (2.5 - (pressure_mean - min_P)/((max_P - min_P))).reshape(-1, 1)
            p_scale_mean = np.mean(pressure_scale)

            score = location_vegf_values * pressure_scale * (1/p_scale_mean)

        # VEGF and Oxygen Strategy
        # Scale is based on the presence of oxygen biasing in an interesting way.
        if mechanism == 3:
            # Calculate a combined score
            o2_scale = 1 - self.oxygen_tension_estimate
            o2_scale_mean = np.mean(o2_scale)
            score = location_vegf_values * o2_scale * (1/o2_scale_mean)

        # VEGF and Radius Strategy
        # Scale is based on the local radius biasing to the main pathway
        if mechanism == 4:
            evaluated_radii = np.array([self.tree.segment_dict[seg_id].radius for seg_id in segment_list])
            all_radii = np.array([self.tree.segment_dict[segment].radius for segment in self.tree.segment_dict])
            r_max_active = max(all_radii)

            radius_scale = (evaluated_radii / r_max_active).reshape(-1, 1)
            r_scale_mean = np.mean(radius_scale)

            score = location_vegf_values * radius_scale * (1/r_scale_mean)
            
        if self.use_config is True:
            stochastic = self.stochastic

        if stochastic:
            # Generate random noise with values between 0.8 and 1.2
            noise = np.random.uniform(0.8, 1.2, size=location_vegf_values.shape)
            score = score * noise # type: ignore

        return score  # type: ignore


    def sample_vegf_values(self, location:np.ndarray) -> np.ndarray:
        Mi = gf.asm_interpolation_matrix(self.vtfem,location)
        location_vegf_values = Mi.mult(self.vtx)
        location_vegf_values = np.array(location_vegf_values[:,np.newaxis])
        return location_vegf_values

    def sample_oxygen_values(self, location:np.ndarray) -> np.ndarray:
        Mi = gf.asm_interpolation_matrix(self.otfem,location)
        location_o2_values = Mi.mult(self.otx)
        location_o2_values = np.array(location_o2_values[:,np.newaxis])
        self.o2_values = location_o2_values
        return location_o2_values

    def calculate_norm_of_o2_field(self) -> float:
        l1_norm = gf.asm('generic', self.mim, 0, 'u', -1, 'u', 1, self.otfem, self.otx)
        # self.logger.log(f"l1 norm {l1_norm}")
        # l2_norm = gf.compute_L2_norm(self.otfem,self.otx,self.mim)
        return l1_norm

    def calculate_global_oxygen_mean(self) -> float:
        # Integral of the concentration field u over the whole mesh (region = -1)
        integral_u = gf.asm('generic',self.mim, 0, 'u', -1, 'u', 1, self.otfem, self.otx)
        domain_volume = gf.asm('generic',self.mim,0,'1',-1)

        mean_o2 = integral_u/domain_volume

        return mean_o2

    def calculate_global_vegf_mean(self) -> float:
        # Integral of the concentration field u over the whole mesh (region = -1)
        integral_u = gf.asm('generic',self.mim, 0, 'u', -1, 'u', 1, self.vtfem, self.vtx)
        domain_volume = gf.asm('generic',self.mim,0,'1',-1)

        mean_VEGF = integral_u/domain_volume

        return mean_VEGF

    def oxygen_to_venousness(self,arr):
        # This function takes an array of evaluated oxygen values and returns an estimate of how venous the the vessel is.
        # Find the maximum values in the array
        self.max_val = np.max(arr)
        
        # Transform and scale the array
        transformed_arr = (arr) / (self.max_val)
        
        # Invert the array so that the highest value becomes 0 and the lowest value becomes 1
        inverted_arr = 1 - transformed_arr
        
        return inverted_arr

    def sample_vegf_gradient(self,location:np.ndarray, verbose=False) -> np.ndarray:
        mesh3D = self.vtfem.mesh()
        mf_du = gf.MeshFem(mesh3D, 1)
        mf_du.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))

        loc_col = np.hsplit(location,3)

        grad_field = gf.compute_gradient(self.vtfem,self.vtx,mf_du)
        gradient = gf.compute_interpolate_on(mf_du, grad_field, loc_col)

        if verbose:
            self.logger.log("################## SAMPLING VEGF GRADIENT ##################")
            self.logger.log("SAMPLING LOCATION")
            self.logger.log(loc_col)
            self.logger.log("LOCAL GRADIENT")
            self.logger.log(gradient)
            self.logger.log("################## SAMPLING VEGF COMPLETE ##################")

        return gradient

    def calculate_attraction_field(self, solver:MatrixSolver,save=True):
        existing_sprout_keys = list(self.sprout_dict.keys())
        existing_inhibitors = self.processed_samples[existing_sprout_keys]
        self.tree.set_and_propagate_inhibition(existing_inhibitors)
        self.x_tree, self.x_sprouts = solver.solve_attractor_field(self.sprout_dict)
        if save:
            self.getfem_handler_3D.save_attraction_field(self.output_path,self.run_name,self.getfem_handler_3D.attractor_element,self.x_tree,self.x_sprouts,self.age)

        return self.x_tree, self.x_sprouts

    def sample_vessel_attractor_gradient(self,location:np.ndarray, x_tree:np.ndarray, verbose=False) -> np.ndarray:
        mesh3D = self.vtfem.mesh()
        mf_du = gf.MeshFem(mesh3D, 1)
        mf_du.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))

        loc_col = np.hsplit(location,3)

        grad_field = gf.compute_gradient(self.vtfem,x_tree,mf_du)
        gradient = gf.compute_interpolate_on(mf_du, grad_field, loc_col)

        if verbose:
            self.logger.log("################## SAMPLING VESSEL ATTRACTOR GRADIENT ##################")
            self.logger.log("SAMPLING LOCATION")
            self.logger.log(loc_col)
            self.logger.log("LOCAL GRADIENT")
            self.logger.log(gradient)
            self.logger.log("################## SAMPLING VESSEL ATTRACTOR COMPLETE ##################")

        return gradient

    def sample_tip_attractor_gradient(self,location:np.ndarray, x_sprouts:dict, sprout_id:int, verbose=False) -> np.ndarray:
        mesh3D = self.vtfem.mesh()
        mf_du = gf.MeshFem(mesh3D, 1)
        mf_du.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))

        loc_col = np.hsplit(location,3)

        sprout_gradient = {}
        other_sprouts = {point_id: gradient for point_id, gradient in x_sprouts.items() if point_id != sprout_id}
        for key in other_sprouts.keys():
            x_sprout = x_sprouts[key]
            grad_field = gf.compute_gradient(self.vtfem,x_sprout,mf_du)
            gradient = gf.compute_interpolate_on(mf_du, grad_field, loc_col)

            if verbose:
                self.logger.log(f"################## SAMPLING TIP {key} ATTRACTOR GRADIENT ##################")
                self.logger.log("SAMPLING LOCATION")
                self.logger.log(loc_col)
                self.logger.log("LOCAL GRADIENT")
                self.logger.log(gradient)
                self.logger.log(f"################## SAMPLING TIP {key} ATTRACTOR COMPLETE ##################")

            sprout_gradient.update({key:gradient})

        return sprout_gradient


    def process_samples(self, samples:Union[List,None]=None, inhibition_range=90e-6, activation_threshold=0.1, verbose=False) -> Tuple[List, int]:
        # Inhibition range is currently set at approximately 2 endothelial cells length
        if self.phase != 1:
            raise ValueError("process_samples requires locations sampled from the 1D space. As such it must be called after sample_nodes().")
        if samples is None:
            samples = self.samples

        if self.use_config is True:
            inhibition_range = self.inhibition_range
            activation_threshold = self.activation_threshold

        # Step 1: Order rows by value (highest to lowest)
        sorted_indices = np.argsort(samples[:, 3])[::-1] # type: ignore
        sorted_array = samples[sorted_indices] # type: ignore

        # Add a 5th column to capture markings
        # Create a column of zeros
        zeros_column = np.zeros((sorted_array.shape[0], 1))

        # Add the zeros_column to the original array along axis=1
        sorted_array = np.hstack((sorted_array, zeros_column))
        # Initialize Inhibition
        passed_array = sorted_array[sorted_array[:, -1] == 1]
        inhibition_dict = self.tree.set_and_propagate_inhibition(passed_array,inhibition_range)
        inhibitor_signal = self.tree.measure_inhibition_signal(sorted_array,passed_array,inhibition_dict,inhibition_range)
        mask = (inhibitor_signal == 1) & (sorted_array[:, -1] != 1)
        sorted_array[mask, -1] = -1


        # Initialize an index to keep track of the top rows
        current_top_row = 0
        num_passed = 0

        if verbose:
            self.logger.log("################## BEGINNING SAMPLE PROCESSING ##################")
            self.logger.log("DISPLAYING SORTED SAMPLES")
            self.logger.log(sorted_array)


        while current_top_row < sorted_array.shape[0] and sorted_array[current_top_row, 3] > activation_threshold: 
            if sorted_array[current_top_row, -1] == 0:
                if self.tree.length(sorted_array[current_top_row,4]) < self.step_size/2:
                    sorted_array[current_top_row, -1] = -1
                else:
                    # Step 2: Add a 5th column with value 1 to the top row
                    sorted_array[current_top_row, -1] = 1

                    # Increment the number of passed points
                    num_passed += 1

                    # Step 3: Propagate the inhibition signal
                    passed_array = sorted_array[sorted_array[:, -1] == 1]
                    # self.logger.log(sorted_array[:, -1] == 1)
                    inhibition_dict = self.tree.set_and_propagate_inhibition(passed_array,inhibition_range)

                    # Step 4: Measure the signal at all points being sampled
                    inhibitor_signal = self.tree.measure_inhibition_signal(sorted_array,passed_array,inhibition_dict,inhibition_range)
                    #self.logger.log(inhibitor_signal)
                    
                    # Step 5: Search in a radius and mark points within the radius with -1 in the 5th row
                    # top_row_position = sorted_array[current_top_row, :3]
                    # distances = np.linalg.norm(sorted_array[:, :3] - top_row_position, axis=1)
                    mask = (inhibitor_signal == 1) & (sorted_array[:, -1] != 1)
                    sorted_array[mask, -1] = -1

            # Move to the next top remaining row
            current_top_row += 1

        # Extract passed rows into a new array
        passed_array = sorted_array[sorted_array[:, -1] == 1]
        self.processed_samples = passed_array
        self.num_passed = num_passed
        self.phase = 2

        if verbose:
            self.logger.log(f"NUM PASSED SAMPLES: {num_passed}")
            self.logger.log("DISPLAYING PASSED SAMPLES")
            self.logger.log(passed_array)
            self.logger.log("################## ENDING SAMPLE PROCESSING ##################")

        # if num_passed == 0:
        #     raise ValueError("There were no locations that passed the process under current conditions.")

        return passed_array, num_passed

    # def _check_inhibition_value(self):

    def make_sprout_bases(self, corrective_step=0.1,location_tol=2.5e-6, verbose=False) -> Dict:
        if self.phase != 2:
            raise ValueError("make_sprout_bases requires a processed array of locations sampled from the 1D space. As such it must be called after process_samples.")

        if verbose:
            self.logger.log("################## MAKING SPROUT BASES ##################")

        self.base_nodes = {}
        a = corrective_step
        
        for sprouts in range(self.num_passed):
            if verbose:
                self.logger.log(f"######## Sprout {sprouts} ########")
            location = self.processed_samples[sprouts,:3]
            segment_id = self.processed_samples[sprouts,4]

            node_1_id = self.tree.segment_dict[segment_id].node_1_id
            node_2_id = self.tree.segment_dict[segment_id].node_2_id
            node_1_loc, node_2_loc = self.tree.get_segment_node_locations(segment_id)

            if verbose:
                self.logger.log(f"Creating Sprout at location: {location}")
                self.logger.log(f"Sprout is on segment {segment_id} with nodes at: {node_1_loc} and {node_2_loc}")

            if np.linalg.norm(np.array(node_1_loc)-np.array(location)) < location_tol:
                if verbose:
                        self.logger.log(f"Sprout on node 1")
                        self.logger.log(f"Bifurcation check: {self.tree.check_node_in_junction(node_1_id)}, Internal Check: {not self.tree.check_node_internal(node_1_id)}")
                if self.tree.check_node_in_junction(node_1_id) or not self.tree.check_node_internal(node_1_id):
                    component1 = [element * (1-a) for element in node_1_loc]
                    component2 = [element * a for element in node_2_loc]
                    corrected_loc = [x + y for x, y in zip(component1, component2)]
                    new_node_id, seg_ids = self.tree.break_segment(segment_id, corrected_loc)
                    self._update_segment_ids_after_breaking(seg_ids)
                    self._update_vessel_dict_info(seg_ids)
                    self.base_nodes.update({sprouts:{"id":new_node_id,"new":True}})
                    self.processed_samples[sprouts,:3] = corrected_loc
                    if verbose:
                        self.logger.log(f"New Node generated at: {corrected_loc}")
                        self.logger.log(f"Length of broken segments: {self.tree.length(seg_ids[0]):.4g} and {self.tree.length(seg_ids[1]):.4g}")
                else:
                    self.base_nodes.update({sprouts:{"id":node_1_id,"new":False}})
                    if verbose:
                        self.logger.log(f"Old Node utilized at: {node_1_loc}")
            elif np.linalg.norm(np.array(node_2_loc)-np.array(location)) < location_tol:
                if verbose:
                        self.logger.log(f"Sprout on node 2")
                        self.logger.log(f"Bifurcation check: {self.tree.check_node_in_junction(node_2_id)}, Internal Check: {not self.tree.check_node_internal(node_2_id)}")
                if self.tree.check_node_in_junction(node_2_id) or not self.tree.check_node_internal(node_2_id):
                    component1 = [element * (1-a) for element in node_2_loc]
                    component2 = [element * a for element in node_1_loc]
                    corrected_loc = [x + y for x, y in zip(component1, component2)]
                    new_node_id, seg_ids = self.tree.break_segment(segment_id, corrected_loc)
                    self._update_segment_ids_after_breaking(seg_ids)
                    self._update_vessel_dict_info(seg_ids)
                    self.base_nodes.update({sprouts:{"id":new_node_id,"new":True}})
                    self.processed_samples[sprouts,:3] = corrected_loc
                    if verbose:
                        self.logger.log(f"New Node generated at: {corrected_loc}")
                        self.logger.log(f"Length of broken segments: {self.tree.length(seg_ids[0]):.4g} and {self.tree.length(seg_ids[1]):.4g}")
                else:
                    self.base_nodes.update({sprouts:{"id":node_2_id,"new":False}})
                    if verbose:
                        self.logger.log(f"Old Node utilized at: {node_2_loc}")
            else:
                new_node_id, seg_ids = self.tree.break_segment(segment_id, location)
                self._update_segment_ids_after_breaking(seg_ids)
                self._update_vessel_dict_info(seg_ids)
                self.base_nodes.update({sprouts:{"id":new_node_id,"new":True}})
                if verbose:
                    self.logger.log(f"Location not near existing nodes.")
                    self.logger.log(f"New Node generated at: {location}")
                    self.logger.log(f"Length of broken segments: {self.tree.length(seg_ids[0]):.4g} and {self.tree.length(seg_ids[1]):.4g}")

        self.phase = 3
        if verbose:
            self.logger.log("################## SPROUT BASES MADE ##################")

        return self.base_nodes

    def _update_segment_ids_after_breaking(self,seg_ids:List):
        original_id, new_id = seg_ids
        indices = [index for index, value in enumerate(self.processed_samples[:,4]) if value == original_id]
        for index in indices:
            point = self.processed_samples[index,:3]
            if self._is_point_on_segment(original_id,point):
                self.processed_samples[index,4] = original_id
            elif self._is_point_on_segment(new_id,point):
                self.processed_samples[index,4] = new_id
                # self.logger.log(f"Point on Segment {original_id}, corrected to Segment {new_id}")
            else:
                raise ValueError("Point not on either segment: Segment break correction error")
        return
        
    def _update_vessel_dict_info(self,seg_ids:list):
        if len(seg_ids)==2:
            id_output = self._find_id_sublist(self.vessel_dict_new,seg_ids[0])
            if not id_output is None:
                key,_ = id_output
                self.vessel_dict_new[key].append(seg_ids[1])
        return
        
            

    def _is_point_on_segment(self,seg_id,point):
        node_1_loc, node_2_loc = self.tree.get_segment_node_locations(seg_id)

        line_vec = node_2_loc - node_1_loc # type: ignore
        line_len = np.linalg.norm(line_vec)

        line_1_vec = node_2_loc - point
        line_1_len = np.linalg.norm(line_1_vec)

        line_2_vec = node_1_loc - point
        line_2_len = np.linalg.norm(line_2_vec)

        # if the sum of the distances between a point and the two ends of the line equals the distance between the two ends of the line then the point is on the line.
        if np.isclose(line_len,line_1_len+line_2_len):
            return True
        else:
            return False
        
        

    def make_sprouts(self, verbose=False) -> Dict:
        if self.phase != 3:
            raise ValueError("make_sprouts requires the establishment of sprout bases. As such it must be called after using make_sprout_bases.")
        self.sprout_dict = {}
        self.sprout_data = {}

        if verbose:
            self.logger.log("################## MAKING SPROUT OBJECTS ##################")

        for sprouts in self.base_nodes.keys():
            loc = self.processed_samples[sprouts,:3]
            oxy_values = self.sample_oxygen_values(loc)
            oxygen_tension = 1 - (oxy_values / (self.max_val))
        
            new_sprout = Sprout(None,self.processed_samples[sprouts,:3],self.processed_samples[sprouts,4],\
                self.tree,self.base_nodes[sprouts]["id"],oxygen_tension)
            self.sprout_dict.update({sprouts:new_sprout})
            self.sprout_data.update({sprouts:{"vegf_acculmulation":[],"growth_weighting":[],"min_dist":[],"macrophage_weighting":[]}})
            self.total_num_generated_sprouts += 1

        self.phase = 4

        if verbose:
            self.logger.log("################## SPROUT OBJECTS MADE ##################")

        return self.sprout_dict
    
    @staticmethod
    def _grad_to_vector_normalized(grad):
        vec = grad.flatten()
        norm = np.linalg.norm(vec)
        if norm < 1e-16:
            return vec
        vec /= norm
        return vec

    # @staticmethod
    def rotate_to_minimum_angle(self, p1, p2, p3, min_angle=130):
        # Define vectors
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        # Normalize the vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate the current angle
        dot_product = np.dot(v1_norm, v2_norm)
        current_angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        # self.logger.log(f"Current Angle is {current_angle}")
        
        # If the current angle is already at least min_angle, no rotation is needed
        if current_angle >= min_angle:
            return p3
        
        # Find the axis of rotation (normal to the plane formed by p1, p2, p3)
        rotation_axis = np.cross(v1, v2)
        rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis
        
        # Calculate the rotation needed to reach min_angle
        required_rotation = min_angle - current_angle
        
        # Rotation matrix for rotating around an arbitrary axis
        def rotation_matrix(axis, angle):
            angle_rad = np.radians(angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            ux, uy, uz = axis
            return np.array([
                [cos_a + ux**2 * (1 - cos_a), ux * uy * (1 - cos_a) - uz * sin_a, ux * uz * (1 - cos_a) + uy * sin_a],
                [uy * ux * (1 - cos_a) + uz * sin_a, cos_a + uy**2 * (1 - cos_a), uy * uz * (1 - cos_a) - ux * sin_a],
                [uz * ux * (1 - cos_a) - uy * sin_a, uz * uy * (1 - cos_a) + ux * sin_a, cos_a + uz**2 * (1 - cos_a)]
            ])
        
        # Apply rotation to vector v2 around p2 by required_rotation degrees
        rotation_mat = rotation_matrix(rotation_axis, required_rotation)
        rotated_v2 = np.dot(rotation_mat, v2)
        
        # Calculate the new position of p3 after rotation
        new_p3 = np.array(p2) + rotated_v2


        # Define vectors
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(new_p3) - np.array(p2)
        
        # Normalize the vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate the current angle
        dot_product = np.dot(v1_norm, v2_norm)
        current_angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        # self.logger.log(f"Angle corrected to {current_angle}")
        
        return new_p3

    @staticmethod
    def _adjust_secondary_vector(primary, secondary, max_angle):
        # Normalize primary and secondary
        primary /= np.linalg.norm(primary)
        secondary /= np.linalg.norm(secondary)

        max_angle_radians = np.radians(max_angle)
        # Calculate the axis of rotation using the cross product of primary and secondary
        axis = np.cross(primary, secondary)
        axis /= np.linalg.norm(axis)  # Normalize the axis vector
        
        # Construct the rotation matrix using the Rodrigues' rotation formula
        I = np.identity(3)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = I + np.sin(max_angle_radians) * K + (1 - np.cos(max_angle_radians)) * np.dot(K, K)
        
        # Rotate the primary vector to get the adjusted secondary vector
        adjusted_secondary = np.dot(rotation_matrix, primary)
        
        return adjusted_secondary

    @staticmethod
    def add_stochastic_noise(input_vector, max_angle_degrees):
        # Normalize the input vector
        input_vector = input_vector / np.linalg.norm(input_vector)

        # Convert the maximum angle to radians
        max_angle = np.radians(max_angle_degrees)

        # Generate random spherical coordinates
        theta = np.arccos(2 * np.random.uniform(0, 1) - 1)
        phi = np.random.uniform(0, 2 * np.pi)

        # Calculate the rotation axis from spherical coordinates
        rotation_axis = np.array([np.sin(theta) * np.cos(phi),
                                np.sin(theta) * np.sin(phi),
                                np.cos(theta)])

        # Generate a random rotation angle within the specified maximum angle
        rotation_angle = np.random.uniform(0, max_angle)

        # Create the rotation matrix for the random rotation
        rotation_matrix = R.from_rotvec(rotation_angle * rotation_axis)

        # Apply the rotation to the input vector
        rotated_vector = rotation_matrix.apply(input_vector)

        # Return the rotated vector
        return rotated_vector



    def grow_sprout(self,sprout_id,stochastic=False,verbose=False):
        if self.phase != 4:
            raise ValueError("grow_sprout requires the establishment of the sprouts. As such it must be called after using make_sprouts.")
        if self.phase2 != 0:
            raise ValueError("grow_sprout is the first step in the secondary loop. Call the sprout growth methods in sequence.")

        if verbose:
            self.logger.log("################## GROWING SPROUT ##################")
            self.logger.log(f"Growing Sprout: {sprout_id}")
        beta_used = False
        gamma_used = False
        anastomotic_force_vector = 0
        normalized_macrophage_vector = 0

        loc = self.sprout_dict[sprout_id].get_tip_loc()
        step_size = self.step_size

        vec = np.array([0.0,0.0,0.0])
        alpha_weight, beta_weight, gamma_weight = self._evaluate_weighting(sprout_id)

        # VEGF contribution to growth step
        vegf_gradient = self.sample_vegf_gradient(loc)
        vec_vegf = self._grad_to_vector_normalized(vegf_gradient)
        
        # self.logger.log(f"vec_vegf = {vec_vegf}")
        alpha_vec = alpha_weight * vec_vegf
        vec += alpha_vec
        

        """ # Attractor Signal Component -> Now replaced with generalized anastomotic force.
        # Vessel attractant contribution to growth step
        # UTILIZE VESSEL ATTRACTOR GRADIENT ALONGSIDE GENERALIZED ANASTOMOTIC
        if beta_weight >= 0.2:
            vessel_gradient = self.sample_vessel_attractor_gradient(loc,self.x_tree)
            vec_vessel = self._grad_to_vector_normalized(vessel_gradient)
            vec += beta_weight * vec_vessel
        

        # # Other tip cells attractant contribution to growth step
        # tip_gradients = self.sample_tip_attractor_gradient(loc,self.x_sprouts,sprout_id)
        # if len(tip_gradients.keys()) > 0:
        #     gamma_weight = 1/len(tip_gradients.keys())
        #     for key in tip_gradients.keys():
        #         vec_tip = self._grad_to_vector_normalized(tip_gradients[key])
        #         # self.logger.log(f"vec_tip = {vec_tip}")
        #         vec += gamma_weight * vec_tip """

        min_dist = np.inf
        if len(self.sprout_dict) > 1:
            anastomotic_force_vector, dist = self.generalized_anastomotic_force(sprout_id)
            beta_vec = beta_weight * anastomotic_force_vector
            vec += beta_vec
            beta_used = True
            if dist < min_dist:
                min_dist = dist
        elif verbose:
            self.logger.log("Generalized anastomotic force excluded due to no other tips")

        # Macrophage Component Contribution.
        if self.sprout_dict[sprout_id].check_has_macrophage():
            self.macrophage_reevaluation(sprout_id)
        if not self.sprout_dict[sprout_id].check_has_macrophage():
            self.macrophage_connection_finder(sprout_id)
        if self.sprout_dict[sprout_id].check_has_macrophage():
            normalized_macrophage_vector, dist_to_target = self.macrophage_action_contribution(sprout_id)
            gamma_vec = gamma_weight * normalized_macrophage_vector
            vec += gamma_vec
            gamma_used = True

            if dist_to_target < min_dist:
                min_dist = dist_to_target
        

        # normalize vector
        vec /= np.linalg.norm(vec)


        if verbose:
            self.logger.log("TOTAL GRADIENT VECTOR")
            self.logger.log(vec)
            self.logger.log("VEGF COMPONENT")
            self.logger.log(alpha_weight * vec_vegf)
            if beta_used:
                self.logger.log("GENERALIZED ANASTOMOTIC COMPONENT")
                self.logger.log(beta_weight * anastomotic_force_vector)
            if gamma_used:
                self.logger.log("MACROPHAGE OVERRIDE COMPONENT")
                self.logger.log(gamma_weight * normalized_macrophage_vector)

        # PROBLEM HERE?
        # if self.sprout_dict[sprout_id].get_sprout_age() > 1:
        #     angle, primary_vector, base_case = self.sprout_dict[sprout_id].calculate_angle(vec)
        #     if verbose:
        #         self.logger.log(f"Angle of segments to gradient: {angle}")

        #     max_angle = 80
        #     stochastic_angle = 5
        #     if angle < max_angle:
        #         if verbose:
        #             self.logger.log(f"Angle adjusted to: {max_angle}")
        #         vec = self._adjust_secondary_vector(primary_vector,vec,max_angle)

            # if stochastic:
            #     random_factor = np.random.random()
            #     random_angle = 2*stochastic_angle*random_factor - stochastic_angle
            #     if verbose:
            #         self.logger.log(f"Applying stochastic adjustment of: {random_angle}")
            #     vec = self.add_stochastic_noise(primary_vector,stochastic_angle)

        step_loc = loc + step_size*vec

        if self.sprout_dict[sprout_id].tip_node >= 2:
            # self.logger.log(f"sprout_id = {sprout_id}, p1 ID = {self.sprout_dict[sprout_id].tip_node-1}")
            p1 = self.sprout_dict[sprout_id].snail_trail_dict[self.sprout_dict[sprout_id].tip_node-1]
            final_loc = self.rotate_to_minimum_angle(p1,loc,step_loc,min_angle=40)
            self.sprout_dict[sprout_id].add_step(final_loc)
        else:
            self.sprout_dict[sprout_id].add_step(step_loc)
        # Update Sprout-Sprout macrophage location reference
        # This is technically in the wrong location as it should be updated for other sprouts after a growth instance.
        if self.sprout_dict[sprout_id].check_has_macrophage():
            if self.sprout_dict[sprout_id].macrophage_connection["type"] == "Sprout":
                target_id = self.sprout_dict[sprout_id].macrophage_connection["id"]
                self.sprout_dict[target_id].macrophage_connection["position"] += 1

        current_sprout_growth_dict = {}
        current_sprout_growth_dict.update({"alpha":alpha_vec})

        if beta_used is True:
            current_sprout_growth_dict.update({"beta":beta_vec}) # type: ignore
        if gamma_used is True:
            current_sprout_growth_dict.update({"gamma":gamma_vec}) # type: ignore
        
        self.growth_vector_dict.update({sprout_id:current_sprout_growth_dict})
        self.sprout_data[sprout_id]["min_dist"].append(min_dist)


        if verbose:
            self.logger.log("################## SPROUT GROWN ##################")

        return step_loc

    def _evaluate_weighting(self, sprout_id, expected_length:float=150e-6, verbose:bool=False):
        locs = []
        for loc in self.sprout_dict[sprout_id].snail_trail_dict.values():
            locs.append(loc)
        locs = np.array(locs)
        # self.logger.log(locs)
        locs = np.hsplit(locs,3)
        vegf_values = self.sample_vegf_values(locs)
        if self.weighting_evaluation_metric == "max":
            sigmoid_pos = max(vegf_values)
            self.sprout_data[sprout_id]["vegf_acculmulation"].append(sigmoid_pos)
            beta = self._sigmoid_weight(sigmoid_pos,self.sigmoid_center,self.sigmoid_steepness/4)
        elif self.weighting_evaluation_metric == "acculmulate":
            expected_steps = expected_length/self.step_size
            scale = 1/expected_steps
            sigmoid_pos = sum(vegf_values)
            self.sprout_data[sprout_id]["vegf_acculmulation"].append(sigmoid_pos)
            beta = self._sigmoid_weight(sigmoid_pos,self.sigmoid_center/scale,self.sigmoid_steepness*scale/4)
        else:
            raise ValueError(f"Supported Evaluation metrics are 'max' and 'accumulate' provide metric was: {self.weighting_evaluation_metric}")
        
        alpha = 1-beta
        # Bias to promote macrophage dominance.
        gamma = 2*beta

        self.sprout_data[sprout_id]["growth_weighting"].append(alpha)
        # beta = self._sigmoid_weight(max_val,x=2,y=3.5,k=3)
        # gamma = self._sigmoid_weight(max_val,x=3,y=5,k=6)
        if verbose:
            self.logger.log(f"Sprout {sprout_id} has alpha={alpha}, beta={beta}, gamma={gamma}")
        # # self.logger.log(vegf_values)
        # if all(value <= 2.0 for value in vegf_values):
        #     alpha = 1.0
        #     beta = 0.0
        #     gamma = 0.0
        #     self.logger.log("Weighting on VEGF")
        # elif all(value <= 3.0 for value in vegf_values):
        #     alpha = 1.0
        #     beta = 0.5
        #     gamma = 0.0
        #     self.logger.log("Weighting on Middle")
        # else:
        #     alpha = 0.8
        #     beta = 1.0
        #     gamma = 0.5
        #     self.logger.log("Weighting on Anastomosis")

        return alpha, beta, gamma

    @staticmethod
    def _sigmoid_weight(score, x, k=5):
        """
        Calculate weight using a sigmoid function.

        Parameters:
        - score (float): The score value for which to calculate the weight.
        - x (float): The score value at the centre of the transition range.
        - k (float): Steepness of the sigmoid curve. Higher values make the transition more abrupt.

        Returns:
        - weight (float): The calculated weight between 0 and 1.
        """
        return 1 / (1 + np.exp(-k * (score - x)))


    def generalized_anastomotic_force(self, sprout_id:int):
        """TIPS ONLY AT THE MOMENT"""
        # This is our generalized anastomotic force that exists between a tip and other tips and the segments
        # This is intended to give rise to the behaviour that sprouts tend to grow towards the arterial side of the system.
        # Get information about current sprout
        current_tip_loc = self.sprout_dict[sprout_id].get_tip_loc()
        current_tension = self.sprout_dict[sprout_id].oxygen_tension_score.reshape(-1)

        # Create dummy vector
        anastomotic_force_vector = np.zeros([3])
        
        min_dist = np.inf
        # Iterate through all sprouts
        for keys in self.sprout_dict.keys():
            # Ignore current sprout
            if (keys != sprout_id):
                # Get information about other sprout
                other_tip_loc = self.sprout_dict[keys].get_tip_loc()
                other_tension = self.sprout_dict[keys].oxygen_tension_score.reshape(-1)
                # Calculate direction vector from current tip to other tips
                vec = other_tip_loc - current_tip_loc
                # self.logger.log(f"vec = {vec}")
                # Calculate distance between tips
                dist = np.linalg.norm(vec)
                if dist < min_dist:
                    min_dist = dist
                # Calculate oxygen tension difference
                o2_tension_diff = np.abs(other_tension-current_tension)
                # Calculate contribution: dq*(pi-px)/(|pi-px|^2)
                contribution = o2_tension_diff * vec / (dist*dist)
                anastomotic_force_vector += contribution
                if math.isnan(contribution[0]):
                    self.logger.log(f"Current Sprout Info: {self.sprout_dict[sprout_id]}")
                    self.logger.log(f"Other Sprout Info: {self.sprout_dict[keys]}")
                    raise ValueError(f"Error by NaN Anastomosis: loc1: {current_tip_loc}, loc2: {other_tip_loc}, tension1: {current_tension}, tension2:{other_tension}, dist: {dist}")


        # normalize the anastomotic force vector
        anastomotic_force_vector /= np.linalg.norm(anastomotic_force_vector)

        return anastomotic_force_vector, min_dist

    def macrophage_connection_finder(self, sprout_id:int, verbose=False):
        # This function checks if there are any possible macrophage connections for the specified sprout
        # If there is already a connection return immediately
        if self.sprout_dict[sprout_id].check_has_macrophage():
            return
        # Get the location of the tip of the current sprout
        if verbose:
            self.logger.log(f"Attempting to generate a Macrophage Connection for sprout {sprout_id}")
        tip_location = self.sprout_dict[sprout_id].get_tip_loc()
        # Specify some relevant parameters
        filopodia_length = 40e-6 #  Longest >100 μm doi: 10.1083/jcb.200302047
        macrophage_size = 20e-6
        # Initialize variables
        min_dist = np.inf
        candidate_sprout = None
        candidate_point = None
        # Iterate through all sprouts
        for keys in self.sprout_dict.keys():
            # Ignore current sprout and sprouts that already have macrophage connections
            if (keys != sprout_id) and (not self.sprout_dict[keys].check_has_macrophage()):
                # self.logger.log(f"Current Sprout = {sprout_id}")
                # self.logger.log(f"Checking Sprout = {keys}")
                # Find the closest point on the sprout
                closest_point, closest_distance = self.sprout_dict[keys].in_search_radius(filopodia_length,macrophage_size,tip_location)
                # Check if any points in search radius
                if not closest_point is None:
                    # Compare to current minimum distance
                    if closest_distance < min_dist:
                        # Overwrite the current candidate for best connection
                        closest_distance = min_dist
                        candidate_sprout = keys
                        candidate_point = closest_point
        
        # Check if any candidates passed the test
        if not candidate_sprout is None:
            # Add a sprout type macrophage connection to the current sprout based on the candidate
            self.sprout_dict[sprout_id].add_macrophage_connection("Sprout",candidate_sprout,candidate_point)
            if verbose:
                self.logger.log(f"Connection created between Sprout {sprout_id} and Sprout {candidate_sprout}")
                self.logger.log(f"Candidate Sprout tip id {self.sprout_dict[candidate_sprout].tip_node} and Target point {candidate_point}")
            # mirror this connection to the other sprout.
            self.macrophage_mirror_sprout_connection(sprout_id)
            return

        candidate_segment = None
        # If no sprout candidates passed we check for segment candidates
        for keys in self.tree.segment_dict.keys():
            # check for connections by distance
            closest_point, closest_distance = self.tree.in_search_radius(filopodia_length,macrophage_size,tip_location,keys)
            # Check if any points in search radius
            if not closest_point is None:
                # Check if point is subject to inhibition
                inhibited_by_base = self.check_for_inhibited_range(self.sprout_dict[sprout_id].get_base_loc(),closest_point,8e-5)
                if not inhibited_by_base:
                    # Compare to current minimum distance
                    if closest_distance < min_dist:
                        # Overwrite the current candidate for best connection
                        closest_distance = min_dist
                        candidate_segment = keys
                        candidate_point = closest_point

        # Check if any candidates passed the test
        if not candidate_segment is None:
            # Add a segment type macrophage connection to the current sprout based on the candidate
            self.sprout_dict[sprout_id].add_macrophage_connection("Segment",candidate_segment,candidate_point)
            if verbose:
                self.logger.log(f"Connection created between sprout {sprout_id} and segment {candidate_segment}")
            return

        return

    def macrophage_reevaluation(self, sprout_id:int, verbose=False):
        if not self.sprout_dict[sprout_id].check_has_macrophage():
            if verbose:
                self.logger.log(f"Sprout {sprout_id} attempted to reevaluate nonexistent macrophage")
            return

        tip_location = self.sprout_dict[sprout_id].get_tip_loc()
        marcophage_target = self.sprout_dict[sprout_id].macrophage_connection
        if marcophage_target["type"] == "Sprout":
            target_sprout = self.sprout_dict[marcophage_target["id"]]
            target_loc = target_sprout.snail_trail_dict[marcophage_target["position"]]
        elif marcophage_target["type"] == "Segment":
            # self.logger.log(f"fractional pos: {marcophage_target['position']}")
            target_loc = marcophage_target["position"]
            # target_loc = self.tree.get_fractional_segment_position(marcophage_target["id"],marcophage_target["position"])
        else:
            raise ValueError(f"Unsupported Macrophage target type ({marcophage_target['type']})")

        macrophage_vector = target_loc - tip_location
        dist_to_target = np.linalg.norm(macrophage_vector)

        maximum_macrophage_length = 120e-6
        if dist_to_target >= maximum_macrophage_length:
            if verbose:
                self.logger.log(f"Macrophage attached to Sprout {sprout_id} removed due to excess length")
            if marcophage_target["type"] == "Sprout":
                target_sprout.remove_macrophage_connection() # type: ignore
            self.sprout_dict[sprout_id].remove_macrophage_connection()

        return

    def macrophage_action_contribution(self, sprout_id:int, verbose=False) -> Tuple[List,float]:
        if not self.sprout_dict[sprout_id].check_has_macrophage():
            raise ValueError(f"Sprout {sprout_id} attempted to calculate macrophage action without a macrophage connection")
        tip_location = self.sprout_dict[sprout_id].get_tip_loc()
        marcophage_target = self.sprout_dict[sprout_id].macrophage_connection
        if marcophage_target["type"] == "Sprout":
            target_sprout = self.sprout_dict[marcophage_target["id"]]
            target_loc = target_sprout.snail_trail_dict[marcophage_target["position"]]
        elif marcophage_target["type"] == "Segment":
            # self.logger.log(f"fractional pos: {marcophage_target['position']}")
            target_loc = marcophage_target["position"]
            # target_loc = self.tree.get_fractional_segment_position(marcophage_target["id"],marcophage_target["position"])
        else:
            raise ValueError(f"Unsupported Macrophage target type ({marcophage_target['type']})")

        macrophage_vector = target_loc - tip_location
        dist_to_target = np.linalg.norm(macrophage_vector)
        normalized_macrophage_vector = macrophage_vector / dist_to_target
        
        if verbose:
            self.logger.log(f"Normalized macrophage direction vector for sprout {sprout_id} is: {normalized_macrophage_vector}")
            self.logger.log(f"Current Bridged length of Macrophage is: {dist_to_target}")

        if math.isnan(normalized_macrophage_vector[0]):
            raise ValueError(f"Error by NaN Macrophage: loc1: {target_loc}, loc2: {tip_location}, dist: {dist_to_target}")


        return normalized_macrophage_vector, dist_to_target

    def macrophage_mirror_sprout_connection(self,sprout_id:int, verbose=False):
        # This function is used when a macrophage connection is made from one sprout to another
        # This enforces the macrophage connection on the second sprout
        # Retrieve pertinent information
        macrophage_connection = self.sprout_dict[sprout_id].macrophage_connection
        sprout_pos = self.sprout_dict[sprout_id].tip_node
        target_sprout_id = macrophage_connection["id"]
        target_sprout_pos = macrophage_connection["position"]

        ###### THIS IS A LIKELY NON BIOLOGICAL ACTION
        # Cut target sprout to ensure tip-tip pulling connection
        
        if target_sprout_pos < self.sprout_dict[target_sprout_id].tip_node:
            if verbose:
                self.logger.log(f"Target sprout of length: {self.sprout_dict[target_sprout_id].tip_node} cut to {target_sprout_pos} for tip-tip reasons")
            self.sprout_dict[target_sprout_id].sever_sprout(target_sprout_pos)
        self.sprout_dict[target_sprout_id].add_macrophage_connection("Sprout",sprout_id,sprout_pos)

        if verbose:
            self.logger.log(f"Mirroring Connection")
            self.logger.log(f"Connection created between Sprout {target_sprout_id} and Sprout {sprout_id}")
            self.logger.log(f"Candidate Sprout tip id {self.sprout_dict[sprout_id].tip_node} and Target point {sprout_pos}")
        
        return

    def check_for_tip_to_sprout(self, verbose=False):
        tips = []
        sprout_ids = []
        for key, sprout in self.sprout_dict.items():
            tips.append(sprout.get_tip_loc())
            sprout_ids.append(key)

        distances = np.zeros([len(self.sprout_dict), len(self.sprout_dict)])
        pairs = []

        for i, tip in enumerate(tips):
            current_sprout_id = sprout_ids[i]
            for j, (other_sprout_id, sprout) in enumerate(self.sprout_dict.items()):
                if other_sprout_id != current_sprout_id:
                    if verbose:
                        self.logger.log("SHADOW DICT")
                        self.logger.log(sprout.snail_trail_dict)
                    matching_point, distance = sprout.in_search_radius(self.l1, self.l2, tip)
                    if matching_point is not None and distance != 0:
                        pairs.append([current_sprout_id, matching_point, other_sprout_id])
                        if verbose:
                            self.logger.log("DISTANCES")
                            self.logger.log(f"[{i}{j}]")
                            self.logger.log(distances)
                        distances[i, j] = distance

        return pairs, distances

    def check_for_tip_to_segment(self,inhibition_radius=8e-5,corrective_step=0.4):
        """
        This function compares the sprout dictionary against the segment dictionary.
        In doing so it generates a list of pairings between the tip of each sprout to each segment on the tree 
        and a matrix containing entries associated with each pairing that indicates the distance of that pairing.

        :param inhibition_radius: A float indicating a range from the base of the sprout for which pairs are invalid.
        :param corrective_step: A float between 0-0.5 indicating the ratio at which the closest point should snap to the node of the segment
        :return: A list of lists containing the pairwise information between each sprout and valid segments and an array of pairwise distance values.

        """
        tips = []
        bases = []
        sprout_ids = []
        for keys in self.sprout_dict.keys():
            tips.append(self.sprout_dict[keys].get_tip_loc())
            bases.append(self.sprout_dict[keys].get_base_loc())
            sprout_ids.append(keys)

        key_list = list(self.sprout_dict.keys())
        segment_keys = list(self.tree.segment_dict.keys())
        key_to_index = {key: i for i, key in enumerate(segment_keys)}
        
        a = corrective_step
        distances = np.zeros([len(self.sprout_dict.keys()),len(self.tree.segment_dict.keys())])
        pairs = []
        for i, tip in enumerate(tips):
            for keys in self.tree.segment_dict.keys():
                matching_point, distance = self.tree.in_search_radius(self.l1,self.l2,tip,keys)
                if not matching_point is None:
                    inhibited_by_base = self.check_for_inhibited_range(bases[i],matching_point,inhibition_radius)
                else:
                    inhibited_by_base = False
                if (not matching_point is None) and (not inhibited_by_base):
                    # here we make a correction step to make sure that the candidate is not already saturated
                    node_1_id = self.tree.segment_dict[keys].node_1_id
                    node_2_id = self.tree.segment_dict[keys].node_2_id
                    node_1_loc, node_2_loc = self.tree.get_segment_node_locations(keys)

                    if np.allclose(matching_point, node_1_loc,1e-8):
                        if self.tree.check_node_in_junction(node_1_id) or not self.tree.check_node_internal(node_1_id):
                            component1 = [element * (1-a) for element in node_1_loc]
                            component2 = [element * a for element in node_2_loc]
                            matching_point = [x + y for x, y in zip(component1, component2)]
                    elif np.allclose(matching_point, node_2_loc,1e-8):
                        if self.tree.check_node_in_junction(node_2_id) or not self.tree.check_node_internal(node_2_id):
                            component1 = [element * (1-a) for element in node_2_loc]
                            component2 = [element * a for element in node_1_loc]
                            matching_point = [x + y for x, y in zip(component1, component2)]

                    # Double Check to make sure that the matching point is on the segment in question.
                    if not self.is_point_on_segment(node_1_loc,node_2_loc,matching_point):
                        dist1 = np.linalg.norm(np.array(matching_point)-np.array(node_1_loc))
                        dist2 = np.linalg.norm(np.array(matching_point)-np.array(node_1_loc))
                        if dist1 < dist2:
                            matching_point = node_1_loc
                        else:
                            matching_point = node_2_loc

                    pairs.append([sprout_ids[i],matching_point,keys])
                    # self.logger.log([sprout_ids[i],matching_point,keys])
                    # self.logger.log([node_1_loc, node_2_loc])
                    distances[key_list.index(sprout_ids[i]), key_to_index[keys]] = distance

        return pairs, distances, key_to_index
    
    @staticmethod
    def is_point_on_segment(x1, x2, x3, tol=1e-8):
        # Vector from x1 to x2
        x1_to_x2 = x2 - x1
        # Vector from x1 to x3
        x1_to_x3 = x3 - x1
        
        # Check if x1 to x3 is collinear with x1 to x2 using cross product
        cross_product = np.cross(x1_to_x2, x1_to_x3)
        if not np.allclose(cross_product, 0, atol=tol):
            return False
        
        # Check if x3 lies within the bounds of the segment [x1, x2]
        dot_product = np.dot(x1_to_x3, x1_to_x2)
        if dot_product < 0 or dot_product > np.dot(x1_to_x2, x1_to_x2):
            return False
        
        return True

    def check_for_inhibited_range(self,base,point,inhbition_radius,verbose=False):
        if verbose:
            self.logger.log(f"POINT = {point}")
            self.logger.log(f"POINT = {base}")

        dist = np.linalg.norm(point - base)
        if dist < inhbition_radius:
            return True

        return False


    def run_anastomosis_candidate_checks(self, verbose=False):
        if self.phase2 != 1:
            raise ValueError("run_anastomosis_candidate_checks is the second step in the secondary loop. Call the sprout growth methods in sequence.")

        pairs_sprout, distances_sprout = self.check_for_tip_to_sprout()
        pairs_segment, distances_segment, key_to_index = self.check_for_tip_to_segment()

        if verbose:
            self.logger.log("DISTANCES SPROUT")
            self.logger.log(distances_sprout)
            self.logger.log("DISTANCES SEGMENT")
            self.logger.log(distances_segment)

        # Check if both lists are empty
        if not pairs_sprout and not pairs_segment:
            return False

        # Create a dictionary to group pairs by their first number
        grouped_pairs = {}

        # Group pairs from sprouts
        for pair in pairs_sprout:
            tip, _, _ = pair
            if tip not in grouped_pairs:
                grouped_pairs[tip] = []
            grouped_pairs[tip].append((pair, "sprouts"))  # Indicate source of pair
        
        # Group pairs from segments
        for pair in pairs_segment:
            tip, _, _ = pair
            if tip not in grouped_pairs:
                grouped_pairs[tip] = []
            grouped_pairs[tip].append((pair, "segments"))  # Indicate source of pair
        
        # Initialize variables to track pairs with shortest distance
        shortest_distances = {}
        key_list = list(self.sprout_dict.keys())

        # Iterate through grouped pairs
        for first_num, pairs in grouped_pairs.items():
            shortest_distance = float('inf')
            shortest_pair = None
            
            # Find the pair with the shortest distance for the current group
            for pair in pairs:
                if verbose:
                    self.logger.log(f"Key_list: {key_list}")
                    self.logger.log(f"Pair:: {pair}")
                    self.logger.log(f"Sprout-Sprout: {pair in pairs_sprout}")
                    self.logger.log(f"distances_sprout: {distances_sprout.shape}")
                    self.logger.log(f"distances_segment: {distances_segment.shape}")
                    self.logger.log(f"first_idx: {key_list.index(pair[0][0])}")
                    if pair in pairs_sprout:
                        self.logger.log(f"second_idx: {key_list.index(pair[0][2])}")
                    else:
                        self.logger.log(f"second_idx: {pair[0][2]}")
                distance = distances_sprout[key_list.index(pair[0][0]), key_list.index(pair[0][2])] if pair in pairs_sprout else distances_segment[key_list.index(pair[0][0]), key_to_index[pair[0][2]]]
                if distance < shortest_distance:
                    if verbose:
                        self.logger.log(f"DISTANCES")
                        self.logger.log(distance)
                    shortest_distance = distance
                    shortest_pair = pair
            
            # Store the shortest pair for the current group
            if not shortest_pair is None:
                shortest_distances[first_num] = shortest_pair

        # Now shortest_distances dictionary contains the shortest pairs for each group
        if verbose:
            self.logger.log(f"SHORTEST PAIR IN ANASTOMOSIS PROCESS")
            self.logger.log(shortest_distances)
        self.anastomosis_pairs = shortest_distances
        

        if not self.anastomosis_pairs is None:
            self.phase2 = 2
            return True
        else: 
            self.phase2 = 1
            return False

    def resolve_anastomosis_process(self,tip_info,use_existing_radii=False,verbose=False):
        if self.phase2 != 2:
            raise ValueError("resolve_anastomosis_process is the third step in the secondary loop. Call the sprout growth methods in sequence.")

        if verbose:
            self.logger.log(f"ENTERING ANASTOMOSIS PROCESS FOR: {tip_info}")
            self.logger.log(f"TIP INFO TYPE: {tip_info[-1]}")

        if tip_info[-1] == "sprouts":
            ######################################################################################
            # THIS IMPLEMENTATION KEEPS THE SECONDARY SPROUT. THIS MAY NOT BE IDEAL OR PHYSICAL
            ######################################################################################
            primary_sprout_id = tip_info[0][0] 
            matching_point = tip_info[0][1] 
            target_sprout_id = tip_info[0][2]

            if verbose: 
                self.logger.log(f"ATTACHING TIP TO SPROUT")
                self.logger.log(f"TIP INFO: {tip_info[0]}")

            # Check to make sure that the sprout hasnt already been used.
            if primary_sprout_id in self.sprout_dict.keys():
                # Check to make sure that the sprouts don't have identical characteristics.
                if math.isclose(self.sprout_dict[primary_sprout_id].oxygen_tension_score,self.sprout_dict[target_sprout_id].oxygen_tension_score,rel_tol=0.05):
                    self.sprout_dict.pop(primary_sprout_id)
                    self.sprout_dict.pop(target_sprout_id)
                    self.sprouts_to_remove.append(primary_sprout_id)
                    self.sprouts_to_remove.append(target_sprout_id)
                    self.total_num_failed_sprouts += 2
                    self.logger.log(f"Killed Sprouts {primary_sprout_id} and {target_sprout_id} for bad flow characteristic.")
                else:
                    # self.logger.log(f"ATTACHING SPROUT THROUGH SPROUT")
                    # self.logger.log(self.sprout_dict[primary_sprout_id])
                    # Cut the target sprout to facilitate the anastomosis
                    dangling_points = self.sprout_dict[target_sprout_id].sever_sprout(matching_point)

                    # Attach the primary and target sprouts
                    tip_primary_id, primary_seg_ids = self.tree.attach_sprout(self.sprout_dict[primary_sprout_id],use_existing_radii)
                    tip_target_id, target_seg_ids = self.tree.attach_sprout(self.sprout_dict[target_sprout_id],use_existing_radii,reverse=True)

                    primary_seg = primary_seg_ids[-1]
                    self.sprout_to_segment_id_list.extend(primary_seg_ids)
                    # self._add_new_segments_to_geometry_for_plotting(primary_seg_ids)
                    # self._add_new_segments_to_geometry_for_vtk(primary_seg_ids)

                    if len(target_seg_ids) != 0:
                        target_seg = target_seg_ids[-1]
                        self.sprout_to_segment_id_list.extend(target_seg_ids)
                        # self._add_new_segments_to_geometry_for_plotting(target_seg_ids)
                        # self._add_new_segments_to_geometry_for_vtk(target_seg_ids)
                    else:
                        target_seg = primary_seg

                    if verbose:
                        self.logger.log(f"TIP PRIMARY ID: {tip_primary_id}")
                        self.logger.log(f"TIP TARGET ID: {tip_target_id}")

                    # Add the final connecting segment
                    if use_existing_radii:
                        R1 = self.tree.segment_dict[primary_seg].radius
                        R2 = self.tree.segment_dict[target_seg].radius
                        anastomosis_id = self.tree.add_segment_with_max_length(tip_primary_id,tip_target_id,self.step_size,radius=(R1+R2)/2)
                    else:
                        anastomosis_id = self.tree.add_segment_with_max_length(tip_primary_id,tip_target_id,self.step_size)

                    self.sprout_to_segment_id_list.extend(anastomosis_id)
                    # self._add_new_segments_to_geometry_for_plotting([anastomosis_id])
                    # self._add_new_segments_to_geometry_for_vtk([anastomosis_id])

                    if verbose:
                        self.logger.log(f"NEWLY ADDED SEG_IDS SPROUT-SPROUT")
                        self.logger.log(primary_seg_ids)
                        self.logger.log(target_seg_ids)
                        self.logger.log(anastomosis_id)
                        # Remove the sprouts from the iteration process.
                        self.logger.log(f"Primary sprout id: {primary_sprout_id}, Target sprout id: {target_sprout_id}")

                    primary_tension = self.sprout_dict[primary_sprout_id].oxygen_tension_score
                    target_tension = self.sprout_dict[target_sprout_id].oxygen_tension_score
                    tension_diff = np.abs(target_tension-primary_tension)

                    bending = self.sprout_dict[primary_sprout_id].calculate_bending()
                    self.sucessful_sprout_bend_list.append(bending)
                    self.sucessful_sprout_length_list.append(self.sprout_dict[primary_sprout_id].get_sprout_age()*self.step_size)
                    self.sucessful_sprout_delta_q_list.append(tension_diff)
                    self.sprout_dict.pop(primary_sprout_id)
                    self.sprouts_to_remove.append(primary_sprout_id)

                    bending = self.sprout_dict[target_sprout_id].calculate_bending()
                    self.sucessful_sprout_bend_list.append(bending)
                    self.sucessful_sprout_length_list.append(self.sprout_dict[target_sprout_id].get_sprout_age()*self.step_size)
                    self.sucessful_sprout_delta_q_list.append(tension_diff)
                    self.sprout_dict.pop(target_sprout_id)
                    self.sprouts_to_remove.append(target_sprout_id)

            for keys in self.sprout_dict.keys():
                if self.sprout_dict[keys].check_has_macrophage():
                    if self.sprout_dict[keys].macrophage_connection["type"] == "Sprout":
                        if self.sprout_dict[keys].macrophage_connection["id"] in [primary_sprout_id, target_sprout_id]:
                            self.sprout_dict[keys].remove_macrophage_connection()

            if verbose:
                self.logger.log(f"Remaining sprouts: {self.sprout_dict.keys()}")

            self.anastomosis_pairs = {key: value for key, value in self.anastomosis_pairs.items() if (value[0][2] != primary_sprout_id) and value[-1] == "sprouts"}
            self.anastomosis_pairs = {key: value for key, value in self.anastomosis_pairs.items() if (value[0][2] != target_sprout_id) and value[-1] == "sprouts"}

            if verbose:
                self.logger.log(f"REMAINING LEN OF SPROUT DICT: {len(self.sprout_dict)}")

            # Reintroduce any dangling sprouts to the system: IF YOU WANT TO CUT INSTEAD REMOVE BENEATH THIS
            # if len(dangling_points) > 0:
            #     dangling_sprout = Sprout(None,self.tree.node_dict[tip_target_id].location(),anastomosis_id,self.tree,tip_target_id)
            #     for point in dangling_points:
            #         dangling_sprout.add_step(point)
                    
            # self.sprout_dict.update({target_sprout_id:dangling_sprout})

            self.phase2 = 1
            return 

        elif tip_info[-1] == "segments":
            primary_sprout_id = tip_info[0][0] 
            matching_point = tip_info[0][1] 
            segment_id = tip_info[0][2]

            # self.logger.log(f"ATTACHING SPROUT THROUGH SEGMENT")
            # self.logger.log(self.sprout_dict[primary_sprout_id])

            # Attach the primary sprout
            tip_primary_id, primary_seg_ids = self.tree.attach_sprout(self.sprout_dict[primary_sprout_id],use_existing_radii)

            primary_seg = primary_seg_ids[-1]
            self.sprout_to_segment_id_list.extend(primary_seg_ids)
            # self._add_new_segments_to_geometry_for_plotting(primary_seg_ids)
            # self._add_new_segments_to_geometry_for_vtk(primary_seg_ids)

            # Add the final connecting segment
            if verbose:
                self.logger.log(f"ATTACHING TIP TO SEGMENT")
                self.logger.log(f"TIP PRIMARY ID: {tip_primary_id}")
                self.logger.log(f"MATCHING POINT ID: {tuple(matching_point)}")
                self.logger.log(f"Segment ID: {segment_id}")
                node_1_loc, node_2_loc = self.tree.get_segment_node_locations(segment_id)
                self.logger.log(f"Node 1: {node_1_loc}")
                self.logger.log(f"Node 2: {node_2_loc}")

            node_1_id = self.tree.segment_dict[segment_id].node_1_id
            node_2_id = self.tree.segment_dict[segment_id].node_2_id
            node_1_loc, node_2_loc = self.tree.get_segment_node_locations(segment_id)
            length = self.tree.length(segment_id)
            if np.linalg.norm(np.array(node_1_loc)-np.array(tuple(matching_point))) < length/3 and not self.tree.check_node_in_junction(node_1_id):
                connecting_id = node_1_id
            elif np.linalg.norm(np.array(node_2_loc)-np.array(tuple(matching_point))) < length/3 and not self.tree.check_node_in_junction(node_2_id):
                connecting_id = node_2_id
            else:
                connecting_id, new_seg_ids = self.tree.break_segment(segment_id,matching_point)
                self._update_vessel_dict_info(new_seg_ids)

            if use_existing_radii:
                R1 = self.tree.segment_dict[primary_seg].radius
                R2 = self.tree.segment_dict[segment_id].radius
                anastomosis_id = self.tree.add_segment_with_max_length(tip_primary_id,connecting_id,self.step_size,radius=(R1+R2)/2)
            else:
                anastomosis_id = self.tree.add_segment_with_max_length(tip_primary_id,connecting_id,self.step_size)

            # Note the new junction information
            new_junction_id = self.tree.get_junction_on_node(connecting_id)
            if verbose:
                self.logger.log(f"NODE BEING CONNECTED TO: {connecting_id}")
                self.logger.log(f"ASSOCIATED JUNCTION ID: {new_junction_id}")

            if new_junction_id != False:
                self.tree.junction_dict[new_junction_id].segment_3_id = anastomosis_id[-1]
            else:
                if True:
                    self.logger.log(f"No existing Junction on Node ({connecting_id}) creating new Junction")
                seg_ids,_ = self.tree.get_segment_ids_on_node(connecting_id)
                if len(seg_ids) != 3:
                    raise ValueError(f"Failure to create new Junction on on Node {connecting_id} due to unexpected number of segments {len(seg_ids)} for a Sprout-Segment Anastomosis")
                new_junction_id = self.tree.add_junction(connecting_id,seg_ids[0],seg_ids[1],seg_ids[2])

            self.sprout_to_segment_id_list.extend(anastomosis_id)
            # self._add_new_segments_to_geometry_for_plotting([anastomosis_id])
            # self._add_new_segments_to_geometry_for_vtk([anastomosis_id])

            if verbose:
                self.logger.log(f"NEWLY ADDED SEG_IDS SPROUT-SEGMENT")
                self.logger.log(primary_seg_ids)
                self.logger.log(anastomosis_id)

            # Remove the sprouts from the iteration process.
            bending = self.sprout_dict[primary_sprout_id].calculate_bending()
            primary_tension = self.sprout_dict[primary_sprout_id].oxygen_tension_score
            target_o2 = self.sample_oxygen_values(matching_point)
            target_tension = target_o2/self.max_val
            tension_diff = np.abs(target_tension-primary_tension)

            self.sucessful_sprout_bend_list.append(bending)
            self.sucessful_sprout_length_list.append(self.sprout_dict[primary_sprout_id].get_sprout_age()*self.step_size)
            self.sucessful_sprout_delta_q_list.append(tension_diff)
            self.sprout_dict.pop(primary_sprout_id)
            self.sprouts_to_remove.append(primary_sprout_id)
            
            for keys in self.sprout_dict.keys():
                    if self.sprout_dict[keys].check_has_macrophage():
                        if self.sprout_dict[keys].macrophage_connection["type"] == "Sprout":
                            if self.sprout_dict[keys].macrophage_connection["id"] in [primary_sprout_id]:
                                self.sprout_dict[keys].remove_macrophage_connection()

            self.anastomosis_pairs = {key: value for key, value in self.anastomosis_pairs.items() if (value[0][2] != primary_sprout_id) and value[-1] == "sprouts"}

            if verbose:
                self.logger.log(f"REMAINING LEN OF SPROUT DICT: {len(self.sprout_dict)}")
        self.phase2 = 1
        return

    def kill_out_of_bounds(self, verbose=False):
        # Create a list of keys to remove
        keys_to_remove = []

        # Iterate over the dictionary and identify keys to remove
        for keys in self.sprout_dict.keys():
            tip_loc = self.sprout_dict[keys].get_tip_loc()
            if not self.tissue.check_point_in_bounds(tip_loc):
                keys_to_remove.append(keys)

        for sprout_id in keys_to_remove:
            base_loc = self.sprout_dict[sprout_id].get_base_loc()
            self.sprout_dict.pop(sprout_id)
            self.sprouts_to_remove.append(sprout_id)
            self.total_num_failed_sprouts += 1
            for keys in self.sprout_dict.keys():
                    if self.sprout_dict[keys].check_has_macrophage():
                        if self.sprout_dict[keys].macrophage_connection["type"] == "Sprout":
                            if self.sprout_dict[keys].macrophage_connection["id"] in [sprout_id]:
                                self.sprout_dict[keys].remove_macrophage_connection()
            
            # Find the index of the entry with matching values
            index_to_remove = None
            for i, entry in enumerate(self.processed_samples):
                if verbose:
                    self.logger.log(entry[:3])
                    self.logger.log(base_loc)
                if np.isclose(entry[:3], base_loc).all():
                    index_to_remove = i
                    break

            # Check if a matching entry was found
            if index_to_remove is not None:
                # Remove the entry at the identified index
                np.delete(self.processed_samples,index_to_remove, axis=0)
            else:
                raise ValueError(f"No base inhibition point matching removed sprout {i}") # type: ignore

            if verbose:
                self.logger.log(f"Sprout {sprout_id} killed for being out of bounds")
                self.logger.log(f"Tip location was {tip_loc}") # type: ignore
                self.logger.log(f"Box is {self.tissue.bottom_corner()}, {self.tissue.top_corner()}")
                self.logger.log(f"Number of Sprouts remaining: {len(self.sprout_dict.keys())}")

        return
    
    def find_new_inlet_pressure(self,initial_volume:float,initial_flow:float,verbose:bool=True):
        """
        Function to find the appropriate inlet pressure for the system so as to match the current luminal volume

        :param initial_volume: A float indicating the luminal volume of the system before the last adjustment
        :param initial_volume: A float indicating the total inlet flow of the system before the last adjustment
        :return: A boolean indicating if any adjustments were made on this iteration.
        """
        # Get the Original length weighted wss mean
        original_pressure = copy.copy(self.getfem_handler_1D.pressure_inlet)
        outlet_pressure = self.getfem_handler_1D.pressure_outlet
        pressure_drop = original_pressure - outlet_pressure

        total_incoming_flow = initial_flow

        if verbose:
            self.logger.log(f"################## Finding New Inlet Pressure Through Flow/Volume Constraint ##################")
            self.logger.log(f"Initial Volume: {initial_volume:.4g}")
            self.logger.log(f"Initial Flow: {total_incoming_flow:.4g}")
            self.logger.log(f"Initial Pressure Drop: {pressure_drop:.4g}")

        new_volume = self.tree.get_network_volume()
        # new haemodynamic solution
        self.getfem_handler_1D.recalculate_vector = True
        self.solver.iterative_solve_fluid_1D(tolerance=1e-8, tolerance_h=1e-8, alpha=0.9, beta=0.7, max_iterations=200)
        inlet_id_list = self.tree.get_node_ids_inlet()
        new_incoming_flow = 0
        for inlet_id in inlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            new_incoming_flow += mean_flow

        original_flow_volume_ratio = total_incoming_flow / initial_volume
        new_flow_volume_ratio = new_incoming_flow / new_volume

        new_original_ratio_ratio = original_flow_volume_ratio/new_flow_volume_ratio
        new_pressure = pressure_drop * new_original_ratio_ratio + outlet_pressure

        if verbose:
            self.logger.log(f"")
            self.logger.log(f"New Volume: {new_volume:.4g}")
            self.logger.log(f"New Flow: {new_incoming_flow:.4g}")
            self.logger.log(f"Original Ratio = {original_flow_volume_ratio:.4g}, New Ratio = {new_flow_volume_ratio:.4g}")
            self.logger.log(f"")
            self.logger.log(f"Corrected Pressure Drop: {new_pressure-outlet_pressure:.4g}")

        self.getfem_handler_1D.set_inlet_pressure(new_pressure)
        self.getfem_handler_1D.recalculate_vector = True
        self.solver.iterative_solve_fluid_1D(tolerance=1e-8, tolerance_h=1e-8, alpha=0.9, beta=0.7, max_iterations=200)
        inlet_id_list = self.tree.get_node_ids_inlet()
        corrected_incoming_flow = 0
        for inlet_id in inlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            corrected_incoming_flow += mean_flow

        corrected_flow_volume_ratio = corrected_incoming_flow / new_volume
        if verbose:
            self.logger.log(f"Corrected Flow: {corrected_incoming_flow:.4g}")
            self.logger.log(f"Original Ratio = {original_flow_volume_ratio:.4g}, Corrected Ratio = {corrected_flow_volume_ratio:.4g}")

        if new_pressure != original_pressure:
            result = True
        else:
            result = False

        return result
    
    def run_pressure_control_mechanism(self, target_ratio:float=1., mechanism:int=None): # type: ignore
        """
        Function to set the appropriate inlet pressure for the system.
        The exact process used depends on the specified mechanism

        :param target_ratio: A float indicating the target ratio between inlet flow and the luminal volume of the system,required for some mechanisms
        :param mechanism: A int indicating the desired pressure control mechanism
        :return: A float value for the new inlet pressure.
        """
        if self.use_config is True:
            mechanism = self.pressure_strategy

        if not mechanism  in [0,1,2,3]:
            raise ValueError(f"The parameter mechanism ({mechanism}) must be an int belonging to ([1,2,3])")
        
        if mechanism == 0:
            new_pressure = self.getfem_handler_1D.pressure_inlet_nominal
        if mechanism == 1:
            new_pressure = self.set_inlet_pressure_for_flow_volume_ratio(target_ratio)
        if mechanism == 2:
            new_pressure = self.set_inlet_pressure_for_metabolic_need()
        if mechanism == 3:
            new_pressure = self.set_inlet_pressure_for_metabolic_and_flow(target_ratio)

        return new_pressure # type: ignore
    
    def get_flow_volume_ratio(self):
        """
        Function to calculate the current inlet flow / luminal volume ratio. Rquired to update system for metabolic need.

        :return: A float value for the current ratio.
        """
        # Get the current Ratio
        self.solver.iterative_solve_fluid_1D(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100,verbose=False,very_verbose=False)
        volume = self.tree.get_network_volume()
        inlet_id_list = self.tree.get_node_ids_inlet()
        net_incoming_flow = 0
        for inlet_id in inlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            net_incoming_flow += mean_flow

        flow_volume_ratio = net_incoming_flow / volume

        return flow_volume_ratio


    def set_inlet_pressure_for_flow_volume_ratio(self,target_ratio:float,verbose:bool=True):
        """
        Function to set the appropriate inlet pressure for the system so as achieve the target flow / luminal volume ratio.

        :param target_ratio: A float indicating the target ratio between inlet flow and the luminal volume of the system
        :param verbose: A boolean indicating if information in this method should be logged.
        :return: A float value for the new inlet pressure.
        """
        # Get the current Ratio
        # if self.use_config is True:
        #     target_ratio = self.flow_volume_ratio

        flow_volume_ratio = self.get_flow_volume_ratio()
        
        ratio_ratio_to_update_pressure = target_ratio/flow_volume_ratio
        pressure_drop = self.getfem_handler_1D.pressure_inlet - self.getfem_handler_1D.pressure_outlet
        new_pressure = pressure_drop * ratio_ratio_to_update_pressure + self.getfem_handler_1D.pressure_outlet

        if verbose:
            self.logger.log(f"")
            self.logger.log(f"Target Ratio = {target_ratio:.4g}, New Ratio = {flow_volume_ratio:.4g}")
            self.logger.log(f"")
            self.logger.log(f"Initial Pressure Drop: {pressure_drop:.4g}")
            self.logger.log(f"Corrected Pressure Drop: {new_pressure-self.getfem_handler_1D.pressure_outlet:.4g}")

        self.getfem_handler_1D.set_inlet_pressure(new_pressure)
        self.getfem_handler_1D.recalculate_vector = True

        ## EVERYTHING BELOW THIS IS VERIFICATION NOT ACTUALLY NEEDED FOR FUNCTIONALITY.
        # self.solver.iterative_solve_fluid_1D(tolerance=1e-8, tolerance_h=1e-8, alpha=0.9, beta=0.7, max_iterations=200)
        # inlet_id_list = self.tree.get_node_ids_inlet()
        # corrected_incoming_flow = 0
        # for inlet_id in inlet_id_list:
        #     segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
        #     segment_id = segment_ids[0]
        #     area = self.tree.area(segment_id)
        #     start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
        #     end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

        #     start_flow = area*start_velocity
        #     end_flow = area*end_velocity
        #     mean_flow = np.mean([start_flow,end_flow])
        #     corrected_incoming_flow += mean_flow

        # corrected_flow_volume_ratio = corrected_incoming_flow / volume
        # self.logger.log(f"Corrected Flow: {corrected_incoming_flow:.4g}")
        # self.logger.log(f"Target Ratio = {target_ratio:.4g}, Corrected Ratio = {corrected_flow_volume_ratio:.4g}")

        return new_pressure
    
    def set_inlet_pressure_for_metabolic_and_flow(self,target_ratio:float,verbose:bool=False):
        """
        Function to set the appropriate inlet pressure for the system so as to satisfy the percieved metabolic need of the system and the flow-volume ratio.

        :param target_ratio: A float indicating the target ratio between inlet flow and the luminal volume of the system
        :param verbose: A boolean indicating if information in this method should be logged.
        :return: A float value for the new inlet pressure.
        """
        # if self.use_config is True:
        #     target_ratio = self.flow_volume_ratio

        self.logger.log(f"Solving system to calculate metabolic Need")
        self.mat_handler.reset_system()
        self.solver.iterative_solve_fluid(tolerance=1e-7, tolerance_h=1e-7, alpha=0.9, beta=0.7, max_iterations=100,verbose=False,very_verbose=False)
        norms, convergence_rates = self.solver.iterative_solve_oxygen(tolerance=1e-8, max_iterations=200,alpha=0.85,verbose=False,very_verbose=False)
        self.solver.iterative_solve_vegf(alpha=1,verbose=False)
        self._get_growth_info()

        polling_rate = 11
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("vt", self.vtfem)
        model.set_variable("vt", self.vtx)

        used_ids = []
        master_points = []
        segment_ids = []

        for keys in self.tree.segment_dict.keys():
            # Retrieve the node positions
            node_1_id = self.tree.segment_dict[keys].node_1_id
            node_2_id = self.tree.segment_dict[keys].node_2_id

            node_1_pos = self.tree.node_dict[node_1_id].location()
            node_2_pos = self.tree.node_dict[node_2_id].location()

            # Generate the points to add to the mesh
            local_points = np.linspace(node_1_pos,node_2_pos, polling_rate)

            # Add used ids to the used 1D list
            if node_1_id in used_ids:
                local_points = local_points[1:]
            else:
                used_ids.append(node_1_id)
            if node_2_id in used_ids:
                local_points = local_points[:-2]
            else:
                used_ids.append(node_2_id)

            master_points.extend(local_points)
            # Store the segment ID for each point
            segment_ids.extend([keys] * len(local_points))
        
        location_array = np.array(master_points)
        loc_col = np.hsplit(location_array,3)
        vegf_values = self.sample_vegf_values(loc_col)

        vegf_mean = np.mean(vegf_values)

        # Calculate the inlet pressure scaling factor based on the VEGF mean
        x_min = 0.5
        x_max = 4.2
        x = vegf_mean
        y_min = 0.8
        y_max = 1.5
        p = 0.6
        
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        log_x = np.log10(x)
        normalized = (log_x - log_min) / (log_max - log_min)
        y = y_min + (y_max - y_min) * (normalized ** p)

        vegf_inlet_scale = y
        self.logger.log(f"Detected VEGF value mean = {vegf_mean}")
        self.logger.log(f"VEGF pressure ratio scale = {vegf_inlet_scale}")

        # Modify the flow-volume ratio by the VEGF scale 
        target_ratio = target_ratio*vegf_inlet_scale

        # Get the current Ratio
        
        volume = self.tree.get_network_volume()
        inlet_id_list = self.tree.get_node_ids_inlet()
        net_incoming_flow = 0
        for inlet_id in inlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            net_incoming_flow += mean_flow

        flow_volume_ratio = net_incoming_flow / volume
        
        ratio_ratio_to_update_pressure = target_ratio/flow_volume_ratio
        pressure_drop = self.getfem_handler_1D.pressure_inlet - self.getfem_handler_1D.pressure_outlet
        new_pressure = pressure_drop * ratio_ratio_to_update_pressure + self.getfem_handler_1D.pressure_outlet

        if verbose:
            self.logger.log(f"")
            self.logger.log(f"Volume: {volume:.4g}")
            self.logger.log(f"Flow: {net_incoming_flow:.4g}")
            self.logger.log(f"Target Ratio = {target_ratio:.4g}, New Ratio = {flow_volume_ratio:.4g}")
            self.logger.log(f"")
            self.logger.log(f"Initial Pressure Drop: {pressure_drop:.4g}")
            self.logger.log(f"Corrected Pressure Drop: {new_pressure-self.getfem_handler_1D.pressure_outlet:.4g}")

        self.getfem_handler_1D.set_inlet_pressure(new_pressure)
        self.getfem_handler_1D.recalculate_vector = True

        return new_pressure
    
    def set_inlet_pressure_for_metabolic_need(self,verbose:bool=False):
        """
        Function to set the appropriate inlet pressure for the system so as to satisfy the percieved metabolic need of the system.

        :param target_ratio: A float indicating the target ratio between inlet flow and the luminal volume of the system
        :param verbose: A boolean indicating if information in this method should be logged.
        :return: A float value for the new inlet pressure.
        """
        self.logger.log(f"Solving system to calculate metabolic Need")
        self._get_growth_info()

        polling_rate = 11
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("vt", self.vtfem)
        model.set_variable("vt", self.vtx)

        used_ids = []
        master_points = []
        segment_ids = []

        for keys in self.tree.segment_dict.keys():
            # Retrieve the node positions
            node_1_id = self.tree.segment_dict[keys].node_1_id
            node_2_id = self.tree.segment_dict[keys].node_2_id

            node_1_pos = self.tree.node_dict[node_1_id].location()
            node_2_pos = self.tree.node_dict[node_2_id].location()

            # Generate the points to add to the mesh
            local_points = np.linspace(node_1_pos,node_2_pos, polling_rate)

            # Add used ids to the used 1D list
            if node_1_id in used_ids:
                local_points = local_points[1:]
            else:
                used_ids.append(node_1_id)
            if node_2_id in used_ids:
                local_points = local_points[:-2]
            else:
                used_ids.append(node_2_id)

            master_points.extend(local_points)
            # Store the segment ID for each point
            segment_ids.extend([keys] * len(local_points))
        
        location_array = np.array(master_points)
        loc_col = np.hsplit(location_array,3)
        vegf_values = self.sample_vegf_values(loc_col)

        vegf_mean = np.mean(vegf_values)

        # Calculate the inlet pressure scaling factor based on the VEGF mean
        x_min = 0.5
        x_max = 4.2
        x = vegf_mean
        y_min = 0.6
        y_max = 6
        p = 0.6
        
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        log_x = np.log10(x)
        normalized = (log_x - log_min) / (log_max - log_min)
        y = y_min + (y_max - y_min) * (normalized ** p)

        vegf_inlet_scale = y
        p_in_nominal = self.getfem_handler_1D.pressure_inlet_nominal
        p_out = self.getfem_handler_1D.pressure_outlet

        p_drop = (p_in_nominal - p_out)*vegf_inlet_scale
        new_pressure = p_out+p_drop

        self.logger.log(f"Detected VEGF value mean = {vegf_mean}")
        self.logger.log(f"VEGF pressure ratio scale = {vegf_inlet_scale}")

        self.getfem_handler_1D.set_inlet_pressure(new_pressure)
        self.getfem_handler_1D.recalculate_vector = True

        return new_pressure

    def save_sprout_metrics_to_json(self):
        """ This function takes information collected about the characteristics of our sprouts and saves them to a json file """
        information_dict = {}
        information_dict.update({"Number of Sprouts Generated":self.total_num_generated_sprouts})
        information_dict.update({"Number of Sprouts Failed":self.total_num_failed_sprouts})
        information_dict.update({"Number of Sprouts Passed":self.total_num_generated_sprouts-self.total_num_failed_sprouts})
        information_dict.update({"Bending of Sprouts":self.sucessful_sprout_bend_list})
        information_dict.update({"Length of Sprouts":self.sucessful_sprout_length_list})
        information_dict.update({"Oxygen Tension of Sprouts":self.sucessful_sprout_delta_q_list})
        information_dict.update({"Starting Surface":self.tree_starting_surface})
        information_dict.update({"Ending Surface":self.tree.get_network_surface()})
        information_dict.update({"Starting Oxygen Level":self.starting_oxygen_satisfaction})

        # Convert NumPy arrays within lists to lists
        for key, value in information_dict.items():
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, np.ndarray):
                        information_dict[key][i] = item.tolist()

        # Define the file path where you want to save the JSON file
        case = self.config.growth_case
        file_path = self.filepath+"/test_metrics/"+f"case_{case}/metrics_for_age_{self.age}"+".json"

        # Write the dictionary to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(information_dict, json_file)
        return

    def initialize_growth(self) -> Visualizer:
        self.process = 'growth'
        self.reset_phase()
        self._get_growth_info()
        self._initialize_statistics()
        self.vessel_dict_base,_ = self.get_vessels_from_segments()
        self.vessel_dict_new = copy.deepcopy(self.vessel_dict_base)
        # self._initialize_geometry_constructs_for_plotting()
        # self._initialize_geometry_for_vtk()
        self.sprout_to_segment_id_list = []
        self.sprouts_to_remove = []
        visualizer = self.initialize_visualizer()
        self.establish_visualizer_base_line(visualizer)
        
        self.sample_nodes()
        self.process_samples()
        self.make_sprout_bases()
        self.make_sprouts()
        self.past_first_growth = False
        self.l1 = self.step_size * 3

        self.visualize_sprout_growth(visualizer,f"growth_{self.age}.vtk")
        self.mat_handler.save_tissue(self.age,vegf=True)
        self.age += 1
        # self.visualize_geometry(True)
        return visualizer
    
    def initialize_visualizer(self) -> Visualizer:
        visualizer = Visualizer(config=self.config)
        case = self.config.growth_case
        filepath = self.filepath+"/test_metrics/"+f"case_{case}"
        visualizer.set_save_path(filepath)
        file_string = f"growth_{self.age}.vtk"
        visualizer.set_file_name(file_string)
        return visualizer

    def run_growth_process(self):
        visualizer = self.initialize_growth()
        growth_time = []
        visualizer_time = []
        total_start = time.time()
        while len(self.sprout_dict) > 0:
            start_time = time.time()
            growth_visual_time = self.take_growth_step(visualizer,self.visualisation_state)
            end_time = time.time()
            growth_time.append(end_time-start_time)
            visualizer_time.append(growth_visual_time)
            #self.l1 += self.step_size/3
            self.age += 1
        if self.visualisation_state is True:
            start_time = time.time()
            self.growth_vector_dict = {}
            self.visualize_sprout_growth(visualizer,f"growth_{self.age}.vtk")
            self.mat_handler.save_tissue(self.age,vegf=True)
            end_time = time.time()
            visualizer_time.append(end_time-start_time)
        # inlet_p = self.find_new_inlet_pressure(inital_wss_solution, inital_lengths)
        self.generation += 1
        self.reset_phase2()
        self.tree._set_node_and_segement_numbers_by_bfs()
        # self.visualize_geometry(visualizer,f"pre_maximum_length_test.vtk")
        # self.tree.apply_maximum_segment_length(self.step_size)
        self.visualize_geometry(visualizer,f"pre_populate_test.vtk")
        self.tree.populate_junctions()
        self.mat_handler.reset_system()
        
        total_end = time.time()

        self.logger.log(f"Time for Growth = {sum(growth_time)-sum(visualizer_time)}")
        # self.logger.log(f"Time per Growth = {growth_time}")
        self.logger.log(f"Time for Visualization = {sum(visualizer_time)}")
        # self.logger.log(f"Time per Visualization = {visualizer_time}")
        self.logger.log(f"Total Time = {total_end-total_start}")

    def take_growth_step(self,visualizer:Visualizer,visualize:bool=False):
        self.reset_phase2()
        # self.calculate_attraction_field(self.solver)
        for i in self.sprout_dict.keys():
            self.grow_sprout(i)
        self.phase2 += 1
        

        visualization_time = 0.0

        if visualize == True:
            # self.visualize_geometry(True)
            start_time = time.time()
            self.visualize_sprout_growth(visualizer,f"growth_{self.age}.vtk")
            self.mat_handler.save_tissue(self.age,vegf=True)
            end_time = time.time()
            visualization_time = end_time-start_time
        
        self.kill_out_of_bounds()
        if self.past_first_growth == True:
            resolve_anastomosis = True
            i = 0
            while resolve_anastomosis:
                i+=1
                resolve_anastomosis = self.run_anastomosis_candidate_checks()
                # self.logger.log(f"CHECK num {i}")
                if resolve_anastomosis:
                    # while len(self.anastomosis_pairs) > 0:
                    pair = self.anastomosis_pairs.popitem()[-1]
                    # self.logger.log(f"Pair = {pair}")
                    self.resolve_anastomosis_process(pair)
                    # self.tree._set_node_and_segement_numbers_by_bfs()
        
        self.past_first_growth = True
        self.sprout_growth_amount += self.step_size
        if self.sprout_growth_amount >= self.sprout_growth_limit:
            self.logger.log(F"MAXIMUM SPROUT GROWTH LIMIT REACHED")
            self.total_num_failed_sprouts += len(self.sprout_dict.keys())
            self.sprout_dict = {}

        return visualization_time

    def initialize_adaptation(self):
        self.process = 'adaptation'
        setup_time = []
        pressure_time = []
        velocity_time = []
        haematocrit_time = []
        

        start_time = time.time()
        visualizer = self.initialize_visualizer()
        # visualizer.add_tree_to_visualizer(self.tree)
        end_time = time.time()
        setup_time.append(end_time-start_time)
        # self.logger.log(f"Time for Initial Setup = {sum(setup_time)}")
        # else:
        #     visualizer = inherited_visualizer
        #     for segment in self.tree.segment_dict.keys():
        #         start_time = time.time()
        #         visualizer.apply_numbering_to_tree_mesh(segment,self.tree)
        #         end_time = time.time()
        #         setup_time.append(end_time-start_time)
        #     # self.logger.log(f"Time for Inherited Setup = {sum(setup_time)}")
        #     # self.logger.log(f"Mean time per segment = {np.mean(setup_time)}")
        
        # for segment in self.tree.segment_dict.keys():
        #     start_time = time.time()
        #     visualizer.apply_pressure_info_to_tree_mesh(self.mat_handler.solution_haemodynamic,segment,self.tree,self.getfem_handler_1D.node_point_ref)
        #     end_time = time.time()
        #     pressure_time.append(end_time-start_time)
            
        #     start_time = time.time()
        #     visualizer.apply_velocity_info_to_tree_mesh(self.mat_handler.solution_haemodynamic,segment,self.tree)
        #     end_time = time.time()
        #     velocity_time.append(end_time-start_time)
            
        #     start_time = time.time()
        #     visualizer.apply_haematocrit_info_to_tree_mesh(self.mat_handler.solution_haematocrit,segment,self.tree)
        #     end_time = time.time()
        #     haematocrit_time.append(end_time-start_time)
            
        # self.logger.log(f"Time for Pressure Setup = {sum(pressure_time)}")
        # self.logger.log(f"Mean time per segment = {np.mean(pressure_time)}")
        # self.logger.log(f"Time for Velocity Setup = {sum(velocity_time)}")
        # self.logger.log(f"Mean time per segment = {np.mean(velocity_time)}")
        # self.logger.log(f"Time for Haematocrit Setup = {sum(haematocrit_time)}")
        # self.logger.log(f"Mean time per segment = {np.mean(haematocrit_time)}")

        
        sum_of_times = sum(pressure_time)+sum(velocity_time)+sum(haematocrit_time)+sum(setup_time)
        self.logger.log(f"Total Visualisation Time = {sum_of_times}")
        self.logger.log(f"Mean time per Segment = {sum_of_times / self.tree.count_segments()}")
        # self._initialize_geometry_constructs_for_plotting()
        # self._initialize_geometry_for_vtk()
        return visualizer
    
    def run_maintenance_process(self,do_radial:bool=True,do_pruning:bool=True,do_adjust:bool=False,visualize:bool=True,verbose:bool=False):
        """
        Function to run through the vessel network maintainence processes
        This function will calculate wall shear stress, make adjustments to vessel radius, prune vessels, and adjust node positions

        :param do_pruning: A boolean that controls the activation of the pruning process.
        :param do_adjust: A boolean that controls the activation of the node position adjustment.
        :return: A boolean indicating if any adjustments were made on this iteration.
        """

        visualizer = self.initialize_adaptation()
        radius_done = False
        adjust_done = False
        pruning_done = False
        times = np.zeros(4)
        # retrieve information about the system
        start_time = time.time()
        self.wss_dict = self.get_wss()
        end_time = time.time()
        times[0] = end_time-start_time
        if verbose:
            self.logger.log(f"Time to get wss: {times[0]}")
        
        if visualize == True:
            start_time = time.time()
            # visualizer.update_tree_radii(self.tree)
            # visualizer.remove_pruned_segments(pruned_segment_ids)
            mesh = visualizer.create_full_mesh_for_adaptation(self.tree,self.mat_handler.solution_haemodynamic,\
                                                              self.mat_handler.solution_haematocrit,self.getfem_handler_1D.node_point_ref)
            visualizer.save_specific_mesh_to_file(mesh)
            # visualizer.save_to_file([0])
            end_time = time.time()
            times[3] = end_time-start_time
            if verbose:
                self.logger.log(f"Time to update Visualizer: {times[3]}")

        # Evaluate the pressure drop required to keep all vessels
        self.evaluate_pressure_requirements(self.wss_dict)

        # arteriogenesis or regression based on wss
        if do_radial:
            start_time = time.time()
            radius_done = self.perform_radial_adjustments(self.wss_dict)
            end_time = time.time()
            times[1] = end_time-start_time
            if verbose:
                self.logger.log(f"Time to do radial adjustments: {times[1]}")

        if do_adjust:
            # make microadjustments in junction angle and curvature
            tension_vectors = self.calculate_tensions()
            adjust_done = self.adjust_positions(tension_vectors)

        # Pruning should be the last mechanism done since it alters tree connectivity
        if do_pruning:
            start_time = time.time()
            # prune the pathways
            pruning_done, pruned_segment_ids = self.perform_pruning_routine_recursive(2.75e-6)
            end_time = time.time()
            times[2] = end_time-start_time
            if verbose:
                self.logger.log(f"Time to check for pruning: {times[2]}")

        

            # self._initialize_geometry_constructs_for_plotting()
            # self._initialize_geometry_for_vtk()
            # self.visualize_geometry(False, True, wss=True)
            # self.save_geo_to_vtk()
            self.logger.log(f"Saving Adapted Growth at age {self.age}")

        # If a segment was pruned then the mat_handler needs to be reset due to changed connectivity
        if pruning_done:
            self.mat_handler.reset_system()

        self.age += 1
        
        if verbose:
            self.logger.log(f"Total Adaptation Time = {sum(times)}")


        return radius_done or adjust_done or pruning_done
    
    def evaluate_pressure_requirements(self, wss_dict:dict, wss_lower:float=0.6, wss_upper:float=4.0, plot:bool=True):
        """
        This function evaluates amount of pressure that would be needed in order to justify keeping individual vessels
        This is done by looking at the wss and total pressure drop associated with each vessel and then scaling that pressure drop such that the wss reaches a threshold. 

        :param wss_dict: A dictionary containing the wss values of all segments
        :param wss_threshold: A float indicating the wss threshold for a vessel to survive
        :param visualize: A boolean indicating whether to save a graph of the required pressure distributions.
        """
        associated_vessels, vessel_adjacency = self.get_vessels_from_segments()
        
        # Dictionary to hold the mean WSS for each vessel
        vessel_wss = {}
        vessel_radii = {}
        vessel_length = {}
        
        # Calculate the length weighted mean WSS for each vessel
        for vessel_id, segment_ids in associated_vessels.items():
            total_wss = 0
            total_length = 0
            total_radii = 0
            for segment_id in segment_ids:
                length = self.tree.length(segment_id)
                total_wss += wss_dict[segment_id] * length
                total_length += length
                total_radii += self.tree.segment_dict[segment_id].radius * length
            
            mean_wss = total_wss / total_length
            vessel_wss[vessel_id] = mean_wss
            
            mean_radii = total_radii / total_length
            vessel_radii[vessel_id] = mean_radii

            vessel_length[vessel_id] = total_length
        
        p_inlet_current = self.getfem_handler_1D.pressure_inlet
        p_outlet = self.getfem_handler_1D.pressure_outlet
        p_drop = p_inlet_current-p_outlet

        p_inlet_lower_dict = {}
        p_inlet_upper_dict = {}
        ratio_wss_and_threshold_dict = {}
        for vessel_id in vessel_wss.keys():
            ratio_wss_and_threshold_dict[vessel_id] = wss_lower / vessel_wss[vessel_id]
            p_inlet_lower_dict[vessel_id] = p_drop * ratio_wss_and_threshold_dict[vessel_id] + p_outlet

        for vessel_id in vessel_wss.keys():
            ratio_wss_and_threshold_dict[vessel_id] = wss_upper / vessel_wss[vessel_id]
            p_inlet_upper_dict[vessel_id] = p_drop * ratio_wss_and_threshold_dict[vessel_id] + p_outlet

        # Extract radii and pressure values in the same order based on keys
        radius_values = [vessel_radii[vessel_id]*1e6 for vessel_id in vessel_radii.keys()]
        pressure_values = [p_inlet_lower_dict[vessel_id] for vessel_id in p_inlet_lower_dict.keys()]
        pressure_upper_values = [p_inlet_upper_dict[vessel_id] for vessel_id in p_inlet_upper_dict.keys()]
        # self.logger.log(f"Current Inlet Pressure: {p_inlet_current}")
        # self.logger.log(f"Mean Required Inlet Pressure: {np.mean(pressure_values)}")
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            # Create the plot
            axes[0, 0].scatter(radius_values, pressure_values)
            axes[0, 0].axhline(y=p_inlet_current, color="red", linestyle="--", label=f"Current Inlet Pressure = {p_inlet_current}")
            axes[0, 0].set_xlabel("Radius in um")
            axes[0, 0].set_ylabel("Pressure in Pa")
            axes[0, 0].set_ylim(bottom=p_outlet,top=12000)
            axes[0, 0].set_xlim(left=2e-6,right=20e-6)
            title_string = f"Required Inlet Pressure to maintain a WSS of {wss_lower} by Vessel Radius"
            axes[0, 0].set_title(title_string)

            # Create the plot
            axes[0, 1].scatter(radius_values, pressure_upper_values)
            axes[0, 1].axhline(y=p_inlet_current, color="red", linestyle="--", label=f"Current Inlet Pressure = {p_inlet_current}")
            axes[0, 1].set_xlabel("Radius in um")
            axes[0, 1].set_ylabel("Pressure in Pa")
            axes[0, 1].set_ylim(bottom=p_outlet,top=12000)
            axes[0, 1].set_xlim(left=2e-6,right=20e-6)
            title_string = f"Required Inlet Pressure to maintain a WSS of {wss_upper} by Vessel Radius"
            axes[0, 1].set_title(title_string)

            wss_values = [vessel_wss[vessel_id] for vessel_id in vessel_wss.keys()]

            # Create the plot
            axes[1, 0].hist(wss_values, bins=20, linewidth=0.5, edgecolor="white")
            axes[1, 0].set_title('Histogram of WSS values')
            axes[1, 0].set_xlabel('WSS')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(x=wss_lower, color="red", linestyle="--")
            axes[1, 0].axvline(x=wss_upper, color="red", linestyle="--")

            # Create the plot
            axes[1, 1].scatter(radius_values, wss_values)
            axes[1, 1].set_xlabel("Radius in um")
            axes[1, 1].set_ylabel("WSS")
            axes[1, 1].set_ylim(bottom=0)
            axes[1, 1].set_xlim(left=2e-6,right=20e-6)
            title_string = f"WSS by Vessel Radius"
            axes[1, 1].axhline(y=wss_lower, color="red", linestyle="--")
            axes[1, 1].axhline(y=wss_upper, color="red", linestyle="--")
            axes[1, 1].set_title(title_string)

            case = self.config.growth_case
            filepath = self.filepath+"/test_metrics/"+f"case_{case}/"+f"required_pressure_{self.age}.png"
            # Save the plot to a file
            plt.tight_layout()
            plt.savefig(filepath)
            plt.clf()
            plt.close()

        return


    def get_lengths(self):
        """
        This function retrieves the segment length in every Segment ID

        :return: A dictionary with the segment length by Segemnt ID {segment_id:segment_length}.
        """
        length_dict = {}
        for segment_id in self.tree.segment_dict.keys():
            length = self.tree.length(segment_id)
            length_dict.update({segment_id:length})

        return length_dict

    def get_wss(self,verbose=False):
        """
        This function calculates the wall shear stress in every Segment ID

        :return: A dictionary with wall shear stress scores by Segment ID {segment_id:wss_score}.
        """
        wss_dict = {}
        for segment_id in self.tree.segment_dict.keys():
            wss_in_segment = self.calculate_wss_in_segment_u(segment_id,verbose)
            wss_dict.update({segment_id:wss_in_segment})

        return wss_dict

    def calculate_wss_in_segment_p(self, segment_id:int,verbose=False) -> float:
        """
        This function calculates the wall shear stress in a particular Segment ID
        The forumla used for WSS in a segment is: wss = (R*ΔP)/(2*l) 
        Where R is the vessel radius, ΔP is the pressure differential, l is the length of the vessel

        :param segment_id: An integer representing the ID of the segment in the tree
        :param verbose: A boolean indicating if information in this method should be logged.
        :return: Calculated wall shear stress value.
        """
        # Formula for wss: tw = D dP / (4L)
        radii = self.tree.segment_dict[segment_id].radius
        length = self.tree.length(segment_id)

        # Get node ids associated with the segment in the tree representation
        node_1_id = self.tree.segment_dict[segment_id].node_1_id
        node_2_id = self.tree.segment_dict[segment_id].node_2_id

        # Get the PID associated with the node IDs
        pid1 = self.getfem_handler_1D.node_point_ref[node_1_id]
        pid2 = self.getfem_handler_1D.node_point_ref[node_2_id]

        # self.logger.log(f"pid1 = {pid1}, pid2 = {pid2}")

        # # Get the coordinates associated with those node ids
        # node_1_pos = self.tree.node_dict[node_1_id].location()
        # node_2_pos = self.tree.node_dict[node_2_id].location()

        # self.logger.log(self.mat_handler.pvfem)

        # # Get the mesh point ids associated with the coordinates
        # pid1 = self.mat_handler.pvfem.mesh.pid_from_coords(node_1_pos)
        # pid2 = self.mat_handler.pvfem.mesh.pid_from_coords(node_2_pos)

        # Access the value of the solution at the points on the mesh
        # INTERPOLATION MAY BE NECESSARY HERE
        pressure1 = self.mat_handler.solution_haemodynamic["pvx"][pid1]
        pressure2 = self.mat_handler.solution_haemodynamic["pvx"][pid2]

        # Stress does not care about pressure direction
        wss = abs(radii * (pressure1-pressure2) / (2*length))
        if verbose:
            self.logger.log(f"Segment {segment_id} has wss: {wss} due to radii: {radii}, pressure: {abs(pressure1-pressure2)}, and length {length}")

        return wss
    
    def calculate_wss_in_segment_u(self, segment_id:int,verbose=False) -> float:
        """
        This function calculates the wall shear stress in a particular Segment ID
        The forumla used for WSS in a segment is: wss = 2*mu*u_bar/R
        Where R is the vessel radius, mu is the visosity, u_bar is the mean fluid velocity

        :param segment_id: An integer representing the ID of the segment in the tree
        :param verbose: A boolean indicating if information in this method should be logged.
        :return: Calculated wall shear stress value.
        """
        # Formula for wss: tw = 2*mu*u_bar/R
        radii = self.tree.segment_dict[segment_id].radius
        
        start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
        end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

        mean_velocity = (start_velocity+end_velocity)/2

        H_start = self.mat_handler.solution_haematocrit["hx"][segment_id+1][0]
        H_end = self.mat_handler.solution_haematocrit["hx"][segment_id+1][-1]
        temperature = 37

        # VISCOSITY FORMULA
        #basic establishing coefficients
        H = np.mean([H_start,H_end])
        diameter = 2 * radii * 1e6
        #chain of viscosity calculations
        viscosity_water_0 = 1.808 #centipoise viscosity at 0 degrees celcius
        viscosity_ref = 1.8*viscosity_water_0/(1+0.0337*temperature+0.00022*temperature*temperature) #reference viscosity adjusting from water to blood
        #viscosity_nominal = viscosity_ref*(6*np.exp(-0.085*diameter)+3.2-2.44*np.exp(-0.06*np.power(diameter, 0.645))) #value of a nominal viscosity with H=0.45
        viscosity_nominal_ref = (6*np.exp(-0.085*diameter)+3.2-2.44*np.exp(-0.06*np.power(diameter, 0.645)))
        C11 = (0.8+np.exp(-0.075*diameter)) #C is a diameter dependent parameter
        C12 = (-1+1/(1+np.power(diameter, 12)*1e-11))
        C1 = C11*C12
        C2 = 1/(1+np.power(diameter, 12)*1e-11)
        C = C1 + C2
        
        #V1 = (viscosity_nominal/viscosity_ref)-1
        V1 = (viscosity_nominal_ref)-1
        # self.logger.log(f"H: {H}, C: {C}")
        V2_top = np.power(1-H,C)-1
        if H > 1 or H < 0:
            self.logger.log(f"Haematocrit outside valid range in viscosity calc H = {H}")
        V2_bot = np.power(1-0.45,C)-1
        V2 = (V2_top)/(V2_bot)
        V3 = np.power((diameter/(diameter-1.1)),2)
        
        relative_viscosity = ((1+V1*V2*V3)*V3)
        viscosity_vessel = viscosity_ref*relative_viscosity
        viscosity_vessel *= 1e-3

        # Stress does not care about pressure direction
        wss = abs((4 * viscosity_vessel * mean_velocity) / (radii))
        if verbose:
            self.logger.log(f"Segment {segment_id} has wss: {wss} due to radii: {radii}, viscosity: {viscosity_vessel}, and mean velocity {abs(mean_velocity)}")

        return wss


    def plot_wss_values(self, wss_solution, verbose=False):
        """
        This function creates statistical plots of the the wall shear stress solution

        :param wss_solution: A wss solution calculated using GrowthHandler.get_wss()
        :param verbose: A boolean indicating if information in this method should be logged.
        """
        scores = list(wss_solution.values())

        # Step 2: Calculate statistics
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        quartiles = np.percentile(scores, [25, 50, 75])
        lower_quartile = quartiles[0]
        upper_quartile = quartiles[2]

        # Step 3: Print statistics
        if verbose:
            self.logger.log(f"Mean: {mean_score}")
            self.logger.log(f"Median: {median_score}")
            self.logger.log(f"Lower Quartile: {lower_quartile}")
            self.logger.log(f"Upper Quartile: {upper_quartile}")

        # Step 4: Plot the scores and save to file
        # Histogram
        fig, ax = plt.subplots(1,2)
        ax[0].hist(scores, bins=8, linewidth=0.5, edgecolor="white")
        ax[0].set_title('Histogram of Scores')
        ax[0].set_xlabel('Scores')
        ax[0].set_ylabel('Frequency')

        # Boxplot
        ax[1].boxplot(scores, vert=False)
        ax[1].set_title('Boxplot of Scores')
        ax[1].set_xlabel('Scores')

        # Save the plots to files
        plt.savefig(f'scores_plots_{self.age}.png')  # Save both plots together
        # plt.savefig('histogram.png')   # Alternatively, save histogram separately
        # plt.savefig('boxplot.png')     # Alternatively, save boxplot separately

        # Clear the figure to avoid overlap if running the script multiple times
        plt.clf()
        return

    def perform_radial_adjustments(self, wss_solution:dict, lower_bound:float=0.75, upper_bound:float=1, wss_base:float=1, delta_t:float=2e-7, verbose:bool=False):
        """
        This function serves to adjust the radius of segments in the tree based on a wall shear stress solution
        If the shear stress is below a lower bound radius will be stepped down, if it is above an upper bound it will be stepped up

        :param wss_solution: A dictionary with node IDs as keys and 3D vectors as values.
        :param lower_bound: Emperical threshold value to reduce radius.
        :param upper_bound: Emperical threshold value to increase radius.
        :param standard_radial_step_size: Fixed step size value for changing the radius of the vessels.
        :return: A boolean indicating if any 
        """
        if self.use_config is True:
            alpha = self.alpha
            r_min = self.r_min
            r_max = self.r_max
            base_radius = self.r_base
            k_x = self.k_x / delta_t
            k_up = self.k_up

            mechanism = self.adaptation_strategy
        else:
            alpha = 4e-6
            r_min = 0.2e-6
            r_max = 30e-6
            base_radius = 3e-6
            k_x = 0.015 / delta_t
            k_up = 0.1
            mechanism = 2

        venous_weight = 5000
        associated_vessels, vessel_adjacency = self.get_vessels_from_segments()
        radius_changed_flag = False

        # Mechanism 1 = Pure WSS
        # Mechanism 2 = WSS and Rubber Band
        # Mechanism 3 = WSS, RB, and Venous Bias

        if mechanism == 2 or mechanism == 3:
            rubber_band = True
        else:
            rubber_band = False

        if mechanism == 3:
            venous_expansion = True
        else:
            venous_expansion = False
        
        # Dictionaries to hold the mean values for each vessel
        vessel_wss = {}
        vessel_radii = {}
        vessel_length = {}

        if venous_expansion:
            vessel_venous_expansion = {}
            venous_expansion_force = self.calculate_downstream_vessel_expansion_force(base_radius,venous_weight)
        
        # Calculate the length weighted mean WSS for each vessel
        for vessel_id, segment_ids in associated_vessels.items():
            total_wss = 0
            total_length = 0
            total_radii = 0
            total_venous_expansion = 0
            for segment_id in segment_ids:
                length = self.tree.length(segment_id)
                total_wss += wss_solution[segment_id] * length
                total_length += length
                total_radii += self.tree.segment_dict[segment_id].radius * length
                if venous_expansion:
                    total_venous_expansion += venous_expansion_force[segment_id] * length # type: ignore
            
            mean_wss = total_wss / total_length
            vessel_wss[vessel_id] = mean_wss
            
            mean_radii = total_radii / total_length
            vessel_radii[vessel_id] = mean_radii

            vessel_length[vessel_id] = total_length
            if venous_expansion:
                mean_venous_expansion = total_venous_expansion / total_length
                vessel_venous_expansion[vessel_id] = mean_venous_expansion # type: ignore

        # Calculate the vessel adjacency force
        
        # Adjust the radius of segments based on the mean WSS of their vessel
        for vessel_id, segment_ids in associated_vessels.items():
            mean_wss = vessel_wss[vessel_id]
            first_for_vessel = True
            for segment_id in segment_ids:
                if first_for_vessel is True:
                    segment = self.tree.segment_dict[segment_id]
                    radius = segment.radius
                    k_down = np.exp(-alpha * (1 / (radius - r_min/2)))
                    delta_r_wss = self.calculate_radius_change_force_for_wss(mean_wss,radius,k_down,k_up,lower_bound,upper_bound,wss_base)
                    if venous_expansion:
                        delta_r_venous = vessel_venous_expansion[vessel_id] * radius # type: ignore
                    delta_r_resist = self.calculate_radius_change_resist_force_tapered(radius,base_radius,k_x)

                    # delta_r_resist should only ever oppose change from other sources.
                    # Check if rx opposes dx
                    if rubber_band:
                        if venous_expansion:
                            if (delta_r_wss+delta_r_venous) * delta_r_resist < 0:  # type: ignore # dx and rx are in opposite directions
                                # effective_r_resist = min(abs(delta_r_resist), abs(delta_r_wss+delta_r_venous)) * (-1 if delta_r_resist < 0 else 1)  # type: ignore # Limit rx to oppose dx
                                effective_r_resist = delta_r_resist
                            else:
                                effective_r_resist = 0  # rx does not apply if it's in the same direction as dx
                                
                            r_n_plus_1 = radius + delta_t * (delta_r_wss + delta_r_venous + effective_r_resist) # type: ignore
                        else:
                            if delta_r_wss * delta_r_resist < 0:  # dx and rx are in opposite directions
                                effective_r_resist = delta_r_resist
                                # effective_r_resist = min(abs(delta_r_resist), abs(delta_r_wss)) * (-1 if delta_r_resist < 0 else 1)  # Limit rx to oppose dx
                            else:
                                effective_r_resist = 0  # rx does not apply if it's in the same direction as dx

                            r_n_plus_1 = radius + delta_t * (delta_r_wss + effective_r_resist)
                    else:
                        r_n_plus_1 = radius + delta_t * (delta_r_wss)

                    # Calculate the percentage difference
                    percentage_difference = (abs(r_n_plus_1 - radius) / abs(radius)) * 100
                    insignificant_change_flag = percentage_difference < 1
                    if verbose:
                        self.logger.log(f"### VESSEL {vessel_id} ###")
                        self.logger.log(f"Current Radii = {radius:.4g}")
                        self.logger.log(f"Radius Change due to WSS: {delta_r_wss:.4g}")
                        if venous_expansion:
                            self.logger.log(f"Radius Change due to Venous: {delta_r_venous:.4g}") # type: ignore
                        self.logger.log(f"Radius Force due to Resistance: {delta_r_resist:.4g}")
                        if rubber_band:
                            self.logger.log(f"Radius Change due to Resistance: {effective_r_resist:.4g}") # type: ignore
                        self.logger.log(f"Percentage Difference = {percentage_difference:.3g}%")

                    first_for_vessel = False
                
                if not insignificant_change_flag: # type: ignore
                    self.tree.segment_dict[segment_id].radius = r_n_plus_1 # type: ignore
                    radius_changed_flag = True


        return radius_changed_flag
    
    def calculate_radius_change_force_for_wss(self, mean_wss:float, vessel_radius:float, k_down:float, k_up:float, lower_bound:float=0.6, upper_bound:float=4, wss_base:float=1):
        """
        Calculate the radius change that would result from the wss the vessel is exposed to

        :param mean_wss: The current wss value in the system.
        :param vessel_radius: The radius of the vessel being considered
        :param k_down: A static rate controlling parameter for shrinkage.
        :param k_up: A static rate controlling parameter for growth.
        :param lower_bound: The lower bound for wss below which shrinkage occurs.
        :param lower_bound: The upper bound for wss above which growth occurs.
        :param wss_base: The base wss that we are trying to correct towards.
        :param delta_t: A step size controlling parameter.
        :return: The radius change resulting from wss.    
        """
        if mean_wss <= lower_bound:
            r_wss_scale = - (k_down * (mean_wss - wss_base) + vessel_radius) / vessel_radius
            delta_r_wss = k_down * (mean_wss - wss_base)

        elif mean_wss >= upper_bound:
            r_wss_scale = (k_up * (mean_wss - wss_base) + vessel_radius) / vessel_radius
            delta_r_wss = k_up * (mean_wss - wss_base)

        else:
            delta_r_wss = 0
        
        return delta_r_wss
    
    def calculate_radius_change_resist_force(self,vessel_radius:float, base_radius:float, k_x:float= 0.05):
        """
        Calculate the radius change resistance force pulling towards a standard base vessel radius
        This force only ever acts to oppose other changes. 

        :param vessel_radius: The radius of the vessel being considered
        :param base_radius: The base radius value for vessels
        :param k_x: The spring constant associated with the resisting force.
        :return: The radius change resulting from resisting radius change.    
        """
        x = base_radius - vessel_radius
        delta_r_res = k_x * x
        return delta_r_res
    
    def calculate_radius_change_resist_force_tapered(self,vessel_radius:float, base_radius:float, k_x:float= 0.05):
        """
        Calculate the radius change resistance force pulling towards a standard base vessel radius
        This force only ever acts to oppose other changes. 

        :param vessel_radius: The radius of the vessel being considered
        :param base_radius: The base radius value for vessels
        :param k_x: The spring constant associated with the resisting force.
        :return: The radius change resulting from resisting radius change.    
        """ 
        x = base_radius - vessel_radius
        # Apply exponential decay to the resistance force for larger x values
        # Decay factor set such that resistance drops to 30% at abs(x) = 6e-6 
        decay_factor = 200000
        decay = np.exp(decay_factor * abs(x)) if x < 0 else 1
        delta_r_res = k_x * x * decay
        return delta_r_res
    
    def calculate_downstream_vessel_expansion_force(self,base_radius:float=6e-6,weight:float=0.1):
        """
        Calculate the radius change force that operates on the venous side to assist small vessel survival

        :param base_radius: A value for the base radius under which a vessel is considered small 
        :param weight: A weighting term that scales the effect of this force
        :return: The radius change force that operates on the venous side
        """

        # Extract tree and solution information into a workable form.
        haemodynamic_solution = self.mat_handler.solution_haemodynamic
        node_point_ref = self.getfem_handler_1D.node_point_ref
        tree = self.tree
        node_ids = np.array([tree.node_dict[node_key].node_id for node_key in tree.node_dict])

        pressures = np.array([haemodynamic_solution["pvx"][node_point_ref[node_id]] for node_id in node_ids])
        node_1_ids = np.array([tree.segment_dict[seg_id].node_1_id for seg_id in tree.segment_dict])
        node_2_ids = np.array([tree.segment_dict[seg_id].node_2_id for seg_id in tree.segment_dict])
        pressure_means = (pressures[node_1_ids] + pressures[node_2_ids])/2
        pressure_max = self.getfem_handler_1D.pressure_inlet
        pressure_min = self.getfem_handler_1D.pressure_outlet

        # Form a scale for each segment based on the pressure value relative to the pressure range. 
        # Step 1: Obtain a score for the degree of venousness based on pressure value
        av_estimate = 1 - (pressure_means - pressure_min) / (pressure_max - pressure_min)
        
        # Step 2: Scores under 0.5 are arterial sided and not subject to this behaviour
        adjusted = av_estimate - 0.5
        adjusted = np.maximum(adjusted, 0)
        
        # Step 3: rescale the effect to between 0 and 1
        av_scale = adjusted * 2

        # Calculate the small vessel desire value.
        # This is done by calculating the volume of all vessels under base radius and comparing that to the volume they would have had with base radius.
        seg_ids = np.array([tree.segment_dict[seg_id].segment_id for seg_id in tree.segment_dict])
        radii = np.array([tree.segment_dict[seg_id].radius for seg_id in seg_ids])
        lengths = np.array([tree.length(seg_id) for seg_id in seg_ids])

        # Step 1: Filter small vessels
        small_vessel_mask = radii < base_radius
        small_radii = radii[small_vessel_mask]
        small_lengths = lengths[small_vessel_mask]

        # Step 2: Calculate actual and base volumes
        actual_volumes = np.pi * small_radii**2 * small_lengths
        base_volumes = np.pi * base_radius**2 * small_lengths

        # Step 3: Compute total volumes
        total_actual_volume = np.sum(actual_volumes)
        total_base_volume = np.sum(base_volumes)

        # Step 4: Calculate desire value
        desire_value = total_actual_volume / total_base_volume if total_base_volume != 0 else 0
        desire_value = 1-desire_value

        # Calculate the effect the venous expansion force would have on all vessels.
        venous_expansion_force = base_radius * av_scale * desire_value * weight
        self.logger.log(venous_expansion_force)

        return venous_expansion_force
    
    def adjust_raidius_up(self,vessel_id:int,vessel_dict,k_up:float=0.2,delta_t:float=1e-6,mean_wss:float=10,wss_base:float=5):
        """
        Helper function to perform the increasing radius adjustment based on the wss value

        :param vessel_id: A unique id number representing the vessel.
        :param vessel_dict: A dictionary containing the segment ID's associated with the vessel ID {vessel_id:list_of_seg_ids}.
        :param k_up: A static rate controlling parameter.
        :param delta_t: A step size controlling parameter.
        :param mean_wss: The current wss value in the system.
        :param wss_base: The base wss that we are trying to correct towards.
        """
        if mean_wss < wss_base:
            raise ValueError(f"Adjusting radius up only works with a wss value ({mean_wss}) greater than wss base ({wss_base})")
        seg_ids = vessel_dict[vessel_id]
        for segment_id in seg_ids:
            segment = self.tree.segment_dict[segment_id]
            radius = segment.radius
            r_n_plus_1 = k_up * delta_t * (mean_wss - wss_base) + radius
            segment.radius = r_n_plus_1

        self.mat_handler.set_force_geometry(True)

        return
    
    def adjust_raidius_down(self,vessel_id:int,vessel_dict,alpha:float,r_min:float,delta_t:float,mean_wss:float,wss_base:float):
        """
        Helper function to perform the decreasing radius adjustment based on the wss value

        :param vessel_id: A unique id number representing the vessel.
        :param vessel_dict: A dictionary containing the segment ID's associated with the vessel ID {vessel_id:list_of_seg_ids}.
        :param alpha: A rate controlling parameter for the exponential decay approaching r_min,
        :param r_min: A minimum raidus intended to limit the rate of vessel size reduction as it approaches a minimum.
        :param delta_t: A step size controlling parameter.
        :param mean_wss: The current wss value in the system.
        :param wss_base: The base wss that we are trying to correct towards.
        """
        if mean_wss > wss_base:
            raise ValueError(f"Adjusting radius down only works with a wss value ({mean_wss}) less than wss base ({wss_base})")
        seg_ids = vessel_dict[vessel_id]
        for segment_id in seg_ids:
            segment = self.tree.segment_dict[segment_id]
            radius = segment.radius
            k_down = np.exp(-alpha * (1 / (radius - r_min)))
            r_n_plus_1 = k_down * delta_t * (mean_wss - wss_base) + radius
            segment.radius = r_n_plus_1

        self.mat_handler.set_force_geometry(True)

        return

    def check_and_mark_vital_pathways(self):
        """
        Function to make a graph representation of the tree and then evaluate where the critical connections that if broken would disconnect the graph are.

        :return: A dictionary with keys being segment ids and values being a boolean stating if the segment is critical to the structure.
        """
        graph = self._make_connectivity_graph()
        critical_connections = graph.get_critical_connections()
        
        return critical_connections


    def _make_connectivity_graph(self) -> GraphRepresentation:
        """
        Function to build a connectivity graph of the tree

        :return: A graph representation of the tree.
        """
        graph = GraphRepresentation()
        graph.build_undirected(self.tree)
        return graph

    def check_for_pruning(self, pruning_threshold:float) -> dict:
        """
        This function creates a boolean mapping for every Segment ID based on the radius of the segment
        This boolean mapping is intended to be used to identify which segments should be pruned

        :param pruning_threshold: A float value indicating the threshold for the boolean map
        :return: A dictionary with Segment IDs as keys and Booleans as values.
        """
        pruning_dict = {}
        for segment_id in self.tree.segment_dict.keys():
            radius = self.tree.segment_dict[segment_id].radius
            if radius <= pruning_threshold:
                pruning_dict.update({segment_id:True})
            else:
                pruning_dict.update({segment_id:False})
            
        return pruning_dict

    def perform_pruning_routine(self, pruning_dict:dict, vital_pathway_dict:dict):
        """
        This function takes information about pruning candidates and vital pathways in the tree
        It then modifies the tree by removing segments from the tree in they are not vital and marked for pruning.
        if a segment ID is marked for pruning then all other segment ID's in its vessel segment will also be pruned 

        :param pruning_dict: A dictionary with Segment IDs as keys and Booleans as values.
        :param vital_pathway_dict: A dictionary with Segment IDs as keys and Booleans as values.
        :return: A boolean that indicates if any segments were pruned.
        """
        associated_vessels,_ = self.get_vessels_from_segments()
        pruned_segment_ids = []
        segment_pruned = False
        for segment_id in pruning_dict.keys():
            segment_is_prunable = pruning_dict[segment_id]
            segment_is_vital = vital_pathway_dict[segment_id]
            if (segment_is_prunable and not segment_is_vital) and (not segment_id in pruned_segment_ids):
                sublist_info = self._find_id_sublist(associated_vessels,segment_id)
                if sublist_info is None:
                    raise ValueError(f"Segment ID: {segment_id} selected for pruning not found in vessel groupings")
                else:
                    vessel_id, segment_ids_in_vessel = sublist_info
                    pruned_vessel_seg_ids = self.prune_vessel(vessel_id,associated_vessels)
                    self.logger.log(f"Vessel {vessel_id} containing segments {segment_ids_in_vessel} pruned.")
                    pruned_segment_ids.extend(pruned_vessel_seg_ids)
                    segment_pruned = True


        return segment_pruned, pruned_segment_ids
    
    def set_vessel_to_0(self, vessel_id:int, vessel_dict:dict):
        """
        NON_FUNCTIONAL
        This function takes a vessel ID and a dictionary specifying which segments are in which vessel and then sets the radius of those vessels to 0
        This function has no vital pathway information and render the tree unusuable if utilised incorrectly.
        Additionally this function instructs the tree to recalculate all junction information.

        :param vessel_id: A unique id number representing the vessel.
        :param vessel_dict: A dictionary containing the segment ID's associated with the vessel ID {vessel_id:list_of_seg_ids}.
        :return: A list of the segment IDs which were pruned
        """
        seg_id_list = vessel_dict[vessel_id]
        for seg_id in seg_id_list:
            self.tree.segment_dict[seg_id].radius = 0

        # self.tree.reset_junctions()
        # self.tree.populate_junctions()
        return seg_id_list
    
    def prune_vessel(self, vessel_id:int, vessel_dict:dict):
        """
        This function takes a vessel ID and a dictionary specifying which segments are in which vessel and then removes those segments
        This function has no vital pathway information and render the tree unusuable if utilised incorrectly.
        Additionally this function instructs the tree to recalculate all junction information.

        :param vessel_id: A unique id number representing the vessel.
        :param vessel_dict: A dictionary containing the segment ID's associated with the vessel ID {vessel_id:list_of_seg_ids}.
        :return: A list of the segment IDs which were pruned
        """
        seg_id_list = vessel_dict[vessel_id]
        for seg_id in seg_id_list:
            self.tree.remove_segment(seg_id)

        self.tree.reset_junctions()
        self.tree.populate_junctions()
        return seg_id_list
    
    def perform_pruning_routine_recursive(self, pruning_threshold:float,verbose=False) -> Tuple[bool,List[int]]:
        """
        This function takes a pruning treshold and recursively removes the smalled vessl under the threshold.
        It does this by claculating connectivity and effective radii of the connected segments and then pruning them.
        if a segment ID is marked for pruning then all other segment ID's in its vessel segment will also be pruned 

        :param pruning_threshold: A float value specifiying the maximum pruning threshold.
        :param verbose: A boolean indicating if extra information should be printed.
        :return: A boolean that indicates if any segments were pruned and a list of Pruned segments.
        """
        if verbose:
            self.logger.log(f"Entering Pruning Routine")
        
        pruned_segment_ids = []
        segments_pruned_in_recursion = None
        segment_pruned = False
        associated_vessels,_ = self.get_vessels_from_segments()
        vital_pathway_dict = self.check_and_mark_vital_pathways()
        vessel_radii_dict = {}
        for vessel_id in associated_vessels.keys():
            segment_ids_in_vessel = associated_vessels[vessel_id]
            total_length = 0
            weighted_radii_contribution = 0
            for segment_id in segment_ids_in_vessel:
                length = self.tree.length(segment_id)
                radius = self.tree.segment_dict[segment_id].radius

                total_length += length
                weighted_radii_contribution += radius*length

            mean_vessel_radii = weighted_radii_contribution / total_length
            if verbose:
                self.logger.log(f"Vessel {vessel_id} has radii: {mean_vessel_radii}")
            for segment_id in segment_ids_in_vessel:
                self.tree.segment_dict[segment_id].radius = mean_vessel_radii

            if mean_vessel_radii > 1e-12:
                vessel_radii_dict.update({vessel_id:mean_vessel_radii})
        
        min_id = min(vessel_radii_dict, key=vessel_radii_dict.get) # type: ignore
        skip_for_vital_id_flag = any(vital_pathway_dict[key] for key in associated_vessels[min_id]) 
        # If vessel is marked as a vital pathway do not prune it
        while skip_for_vital_id_flag:
            self.logger.log(f"Vessel {min_id} passed over for pruning to being vital.")
            vessel_radii_dict.pop(min_id)
            min_id = min(vessel_radii_dict, key=vessel_radii_dict.get) # type: ignore
            skip_for_vital_id_flag = any(vital_pathway_dict[key] for key in associated_vessels[min_id]) 

        self.logger.log(f"Smallest Vessel ({min_id}) has radii: {vessel_radii_dict[min_id]}, Pruning Threshold is: {pruning_threshold}")
        self.logger.log(f"Vessel Pruned is: {vessel_radii_dict[min_id] < pruning_threshold}")
        

        if vessel_radii_dict[min_id] < pruning_threshold:
            pruned_vessel_seg_ids = self.prune_vessel(min_id,associated_vessels)
            # pruned_vessel_seg_ids = self.set_vessel_to_0(min_id,associated_vessels)
            pruned_segment_ids.extend(pruned_vessel_seg_ids)
            segment_pruned = True
            self.logger.log(f"Vessel {min_id} containing segments {pruned_vessel_seg_ids} pruned.")
            # self.tree._set_node_and_segement_numbers_by_bfs()
            if verbose:
                self.logger.log(f"Entering Recursion")
            _,segments_pruned_in_recursion = self.perform_pruning_routine_recursive(pruning_threshold)
        if not segments_pruned_in_recursion is None:
            pruned_segment_ids.extend(segments_pruned_in_recursion)

        # Final Fusing of Radii for vessels.
        associated_vessels,_ = self.get_vessels_from_segments()
        vessel_radii_dict = {}
        for vessel_id in associated_vessels.keys():
            segment_ids_in_vessel = associated_vessels[vessel_id]
            total_length = 0
            weighted_radii_contribution = 0
            for segment_id in segment_ids_in_vessel:
                length = self.tree.length(segment_id)
                radius = self.tree.segment_dict[segment_id].radius

                total_length += length
                weighted_radii_contribution += radius*length

            mean_vessel_radii = weighted_radii_contribution / total_length
            if verbose:
                self.logger.log(f"Vessel {vessel_id} has radii: {mean_vessel_radii}")
            for segment_id in segment_ids_in_vessel:
                self.tree.segment_dict[segment_id].radius = mean_vessel_radii

        return segment_pruned, pruned_segment_ids
    
    def clean_zero_radii_vessels(self,verbose=False) -> Tuple[bool,List[int]]:
        """
        This function takes a pruning treshold and recursively removes the smalled vessl under the threshold.
        It does this by claculating connectivity and effective radii of the connected segments and then pruning them.
        if a segment ID is marked for pruning then all other segment ID's in its vessel segment will also be pruned 

        :param pruning_threshold: A float value specifiying the maximum pruning threshold.
        :param verbose: A boolean indicating if extra information should be printed.
        :return: A boolean that indicates if any segments were pruned and a list of Pruned segments.
        """
        if verbose:
            self.logger.log(f"Entering clean up")
        
        pruning_threshold = 1e-12
        pruned_segment_ids = []
        segments_pruned_in_recursion = None
        segment_pruned = False
        associated_vessels,_ = self.get_vessels_from_segments()
        vital_pathway_dict = self.check_and_mark_vital_pathways()
        vessel_radii_dict = {}
        for vessel_id in associated_vessels.keys():
            segment_ids_in_vessel = associated_vessels[vessel_id]
            total_length = 0
            weighted_radii_contribution = 0
            for segment_id in segment_ids_in_vessel:
                length = self.tree.length(segment_id)
                radius = self.tree.segment_dict[segment_id].radius

                total_length += length
                weighted_radii_contribution += radius*length

            mean_vessel_radii = weighted_radii_contribution / total_length
            if verbose:
                self.logger.log(f"Vessel {vessel_id} has radii: {mean_vessel_radii}")
            for segment_id in segment_ids_in_vessel:
                self.tree.segment_dict[segment_id].radius = mean_vessel_radii

            vessel_radii_dict.update({vessel_id:mean_vessel_radii})
        
        min_id = min(vessel_radii_dict, key=vessel_radii_dict.get) # type: ignore
        skip_for_vital_id_flag = any(vital_pathway_dict[key] for key in associated_vessels[min_id]) 
        # If vessel is marked as a vital pathway do not prune it
        while skip_for_vital_id_flag:
            self.logger.log(f"Vessel {min_id} passed over for pruning to being vital.")
            vessel_radii_dict.pop(min_id)
            min_id = min(vessel_radii_dict, key=vessel_radii_dict.get) # type: ignore
            skip_for_vital_id_flag = any(vital_pathway_dict[key] for key in associated_vessels[min_id]) 

        self.logger.log(f"Smallest Vessel ({min_id}) has radii: {vessel_radii_dict[min_id]}, Pruning Threshold is: {pruning_threshold}")
        self.logger.log(f"Vessel Pruned is: {vessel_radii_dict[min_id] < pruning_threshold}")
        
        if vessel_radii_dict[min_id] < pruning_threshold:
            pruned_vessel_seg_ids = self.prune_vessel(min_id,associated_vessels)
            pruned_segment_ids.extend(pruned_vessel_seg_ids)
            segment_pruned = True
            self.logger.log(f"Vessel {min_id} containing segments {pruned_vessel_seg_ids} pruned.")
            # self.tree._set_node_and_segement_numbers_by_bfs()
            if verbose:
                self.logger.log(f"Entering Recursion")
            _,segments_pruned_in_recursion = self.perform_pruning_routine_recursive(pruning_threshold)
        if not segments_pruned_in_recursion is None:
            pruned_segment_ids.extend(segments_pruned_in_recursion)

        # Final Fusing of Radii for vessels.
        associated_vessels,_ = self.get_vessels_from_segments()
        vessel_radii_dict = {}
        for vessel_id in associated_vessels.keys():
            segment_ids_in_vessel = associated_vessels[vessel_id]
            total_length = 0
            weighted_radii_contribution = 0
            for segment_id in segment_ids_in_vessel:
                length = self.tree.length(segment_id)
                radius = self.tree.segment_dict[segment_id].radius

                total_length += length
                weighted_radii_contribution += radius*length

            mean_vessel_radii = weighted_radii_contribution / total_length
            if verbose:
                self.logger.log(f"Vessel {vessel_id} has radii: {mean_vessel_radii}")
            for segment_id in segment_ids_in_vessel:
                self.tree.segment_dict[segment_id].radius = mean_vessel_radii

        return segment_pruned, pruned_segment_ids
    
    def _find_id_sublist(self, dictionary:dict, target_id:int) -> Union[list,None]:
        """
        This function serves to find if a particular segment ID is in a dictionary containing lists of IDs.

        :return: A list containing the dictionary key and value containing the target [key,value] or None
        """
        for key in dictionary.keys():
            if target_id in dictionary[key]:
                return [key, dictionary[key]] 
            
        return None  # Return None if the target ID is not found in any sublist
    
    def calculate_tensions(self) -> dict:
        """
        This function serves to calculate the vectors at each node that pull the node to position that best facilitates flow.
        The principle is that the system is most efficient when y junctions are 120 degrees and simple junctions are straight.
        Each segment connected to a node has a characteristic direction vector away from that node and is weighted by diameter.

        :return: A dictionary with node IDs as keys and 3D vectors as values.
        """

        internal_node_ids = self.tree.get_node_ids_internal()
        local_tension_vector_dictionary = {}

        for node_id in internal_node_ids:
            associated_segment_ids, side_of_segment = self.tree.get_segment_ids_on_node(node_id)
            diameters = []
            tangent_versors = []
            for i, segment_id in enumerate(associated_segment_ids):
                diameters.append(2*self.tree.segment_dict[segment_id].radius)
                # tangent versors are calculated as side 2 - side 1 and we want the vector away from the node. 
                if side_of_segment[i] == 1:
                    tangent_versors.append(np.array(self.tree.get_tangent_versor(segment_id)))
                else:
                    tangent_versors.append(-np.array(self.tree.get_tangent_versor(segment_id)))

            result_vector = np.array([0, 0, 0])
            for diameter, vector in zip(diameters,tangent_versors):
                result_vector += diameter*vector
            
            result_vector = result_vector/sum(diameters) 

            local_tension_vector_dictionary.update({node_id,result_vector})
        
        return local_tension_vector_dictionary
    
    def adjust_positions(self, local_tension_vectors:dict, threshold:float=0.05, max_speed:float=1e-6):
        """
        This function serves to adjust the position of each internal node based on the calculated direction of tension
        Movement will not occur unless the magnitude of the vector exceeds a threshold
        Movement size will be limited by maximum migration speed and will be proportional to |vector| - threshold

        :param local_tension_vectors: A dictionary with node IDs as keys and 3D vectors as values.
        :param threshold: Emperical threshold value to prevent excessive migration.
        :param max_speed: Emperical migration speed value to prevent excessive migration.
        :return: A boolean indicaing if any positions were adjusted.
        """
        position_adjust_flag = False

        num_nodes = 0
        num_moved = 0
        for node_id in local_tension_vectors.keys():
            vector = local_tension_vectors[node_id]
            vector_size = np.linalg.norm(vector)
            if vector_size > threshold:
                unit_vector = vector / np.linalg.norm(vector)
                movement_vector = max_speed*(vector-threshold*unit_vector)
                self.tree.node_dict[node_id].move_node(movement_vector)
                self.logger.log(f"Moved node {node_id} by {movement_vector}")
                num_moved += 1
            num_nodes += 1
        
        percent = 100*num_moved/num_nodes
        self.logger.log(f"Percentage of Nodes moved is {percent}")

        if num_moved > 0:
            position_adjust_flag = True

        return position_adjust_flag

    def get_vessels_from_segments(self) -> tuple[dict,dict]:
        """
        This function converts the tree into a graph structure and then retrieves the segment IDs associated with each individual vessel segment
        
        :return: A tuple containing:
                - associated_vessels_dict: A dictionary mapping each vessel ID to its corresponding segment {vessel_id:list of segment ids}.
                - adjacency_dict: A dictionary mapping each vessel ID to a set of adjacent vessel IDs {vessel_id:list of vessel ids}..
        """
        graph = GraphRepresentation()
        graph.build_undirected(self.tree)
        associated_vessels_dict, adjacency_dict = graph.get_vessels()
    

        # self.logger.log(f"List of connected segments")
        # self.logger.log(vessel_list)

        return associated_vessels_dict, adjacency_dict
    
    def _initialize_vessel_dict(self):
        """
        This function uses get_vessel_from_segments t the tree into a graph structure and then retrieves the segment IDs associated with each individual vessel segment
        
        :return: A Dictionary that contains lists of segment IDs associated with each vessel segment {vessel_id:list of segment ids}.
        """

    def establish_visualizer_base_line(self, visualizer:Visualizer):
        """
        This function serves to populate the visualizer with inital tree information.
        This initial information includes the haemodynamic, haematocrit, and oxygen solutions.

        :param visualizer: A Visualizer object that stores and handles visualization.
        :return: The updated Visualizer object.
        """
        uv_sol = self.mat_handler.solution_haemodynamic
        h_sol = self.mat_handler.solution_haematocrit
        o_sol = self.mat_handler.solution_oxygen
        n_p_ref = self.getfem_handler_1D.node_point_ref
        visualizer.create_full_mesh_for_growth(self.tree,uv_sol,h_sol,o_sol,n_p_ref)
        # self.mat_handler.add_properties_to_visualizer(visualizer,True,True,True)
        return visualizer
    
    def visualize_geometry(self, visualizer:Visualizer, file_name:str):
        """
        This function serves to populate the visualizer with inital tree information.
        This initial information includes only the tree geometry.

        :param visualizer: A Visualizer object that stores and handles visualization.
        :param file_name: A string indicating the file name of resulting visualization.
        :return: The updated Visualizer object.
        """
        visualizer.create_full_geometry_mesh(self.tree)
        visualizer.set_file_name(file_name)
        visualizer.save_to_file([0])

        return visualizer

    def visualize_sprout_growth(self, visualizer:Visualizer, file_name:str, save:bool=True, display:bool=False):
        """
        This function serves to populate the visualizer with sprout growth information and save or display the results.
        This function only populates the sprouts and growth vectors.

        :param visualizer: A Visualizer object that stores and handles visualization.
        :param file_name: A string indicating the file name of resulting visualization.
        :param save: A boolean indication if the visualization should be saved.
        :param display: A boolean indication if the visualization should be displayed.
        :return: The updated Visualizer object.
        """
        # visualizer.clear_mesh_layer(3)
        # visualizer.clear_mesh_layer(4)
        # visualizer.clear_mesh_layer(5)
        # visualizer.add_specific_segments_to_visualizer(self.tree,self.sprout_to_segment_id_list) 
        # visualizer.update_sprout_dict_to_visualizer(self.sprout_dict)
        # visualizer.add_growth_directions_to_visualizer(self.sprout_dict,self.growth_vector_dict)
        visualizer.update_mesh_for_growth(self.tree,self.sprout_dict,self.sprout_to_segment_id_list,self.growth_vector_dict)

        visualizer.set_file_name(file_name)

        if save:
            visualizer.save_to_file([0,1,2,3,4,5,6])

        if display:
            visualizer.display_mesh([0,1,2,3,4,5,6],"Radius")
            
        # for sprout_to_remove in self.sprouts_to_remove:
        #     visualizer.clear_mesh_object(2,sprout_to_remove)
        
        # self.sprouts_to_remove = []
        # self.sprout_to_segment_id_list = []
        
        return
    
    def get_vascular_statistics(self,generation:int=None,save=True): # type: ignore
        """
        This function retrieves a set of information about the tree that is valuable for evaluating it in a statistical sense

        :param: An optional int specifying the generation number on the file name
        :return: A dictionary containing the information for vascular statistics
        """
        vessel_radii = {}
        vessel_length = {}
        vessel_oxygen = {}
        vessel_flow = {}
        associated_vessels,_ = self.get_vessels_from_segments()
        self._get_growth_info()
        oxygen_solution = self.mat_handler.solution_oxygen["ovx"]
        node_point_ref = self.getfem_handler_1D.node_point_ref
        
        # Calculate the statistics for each vessel
        num_vessels = 0
        for vessel_id, segment_ids in associated_vessels.items():
            total_length = 0
            total_radii = 0
            total_oxygen = 0
            total_flow = 0
            for segment_id in segment_ids:
                length = self.tree.length(segment_id)
                total_length += length
                total_radii += self.tree.segment_dict[segment_id].radius * length
                id1, id2 = self.tree.get_node_ids_on_segment(segment_id)
                node_ids = [id1,id2]
                oxygens = np.array([oxygen_solution[node_point_ref[node_id]] for node_id in node_ids])
                total_oxygen = np.mean(oxygens)*length

                start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
                end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]
                area = self.tree.area(segment_id)
                start_flow = area*start_velocity
                end_flow = area*end_velocity
                flows = np.mean([start_flow,end_flow])
                total_flow += flows*length

            
            mean_radii = total_radii / total_length
            vessel_radii[vessel_id] = mean_radii

            mean_oxygen = total_oxygen / total_length
            vessel_oxygen[vessel_id] = mean_oxygen

            mean_flows = total_flow / total_length
            vessel_flow[vessel_id] = mean_flows
            
            vessel_length[vessel_id] = total_length
            
            num_vessels += 1

        inlet_id_list = self.tree.get_node_ids_inlet()
        outlet_id_list = self.tree.get_node_ids_outlet()
        
        num_terminals = len(inlet_id_list)+len(outlet_id_list)
        num_bifurcations = (2*num_vessels-num_terminals)/3
        
        luminal_volume = self.tree.get_network_volume()
        tissue_volume = self.tissue.get_tissue_volume()

        vascular_densisty = luminal_volume / tissue_volume

        # Calculate the equivalent hydraulic resistance, R = (Pin - Pout) / Q
        
        incoming_flow = 0
        for inlet_id in inlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            incoming_flow += mean_flow

        
        outgoing_flow = 0
        outgoing_o2 = 0
        for outlet_id in outlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(outlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            outgoing_flow += mean_flow

            outlet_pid = self.getfem_handler_1D.node_point_ref[outlet_id]
            O2_value = self.mat_handler.solution_oxygen["ovx"][outlet_pid]
            outgoing_o2 += O2_value*mean_flow

        

        delta_p = self.getfem_handler_1D.pressure_inlet - self.getfem_handler_1D.pressure_outlet

        resistance = delta_p / incoming_flow
        solubility_O2 = self.config.config_access["1D_CONDITIONS"]["solubility_O2"]
        P_O2_in = self.config.config_access["1D_CONDITIONS"]["P_O2_in"]
        oxygen_concentration = solubility_O2 * P_O2_in
        oxygen_flow = incoming_flow*oxygen_concentration
        global_o2_mean = self.calculate_global_oxygen_mean()
        global_VEGF_mean = self.calculate_global_vegf_mean()

        statistical_dict = {"radii":list(vessel_radii.values()),\
                            "lengths":list(vessel_length.values()),\
                            "oxygens":list(vessel_oxygen.values()),\
                            "flows":list(vessel_flow.values()),\
                            "vascular_density":[luminal_volume,tissue_volume,vascular_densisty],\
                            "num_vessels":num_vessels,\
                            "num_bifurcations":num_bifurcations,\
                            "inlet_pressure":self.getfem_handler_1D.pressure_inlet,\
                            "outlet_pressure":self.getfem_handler_1D.pressure_outlet,\
                            "hydraulic_resist":resistance,\
                            "incoming_flow":incoming_flow,\
                            "outgoing_flow":outgoing_flow,\
                            "incoming_oxygen":oxygen_flow,\
                            "outgoing_oxygen":outgoing_o2,\
                            "global_oxygen_mean":global_o2_mean,\
                            "global_VEGF_mean":global_VEGF_mean,\
                            "age":self.age}
        
        if save:
            # Convert NumPy arrays within lists to lists
            for key, value in statistical_dict.items():
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, np.ndarray):
                            statistical_dict[key][i] = item.tolist()

            # Define the file path where you want to save the JSON file
            case = self.config.growth_case
            file_path = "./"+self.filepath+"/statistic_results/"+f"case_{case}/vascular_statistics"
            if not generation is None:
                file_path += f"_{generation}"
            file_path += ".json"

            # Write the dictionary to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(statistical_dict, json_file)
        
        return statistical_dict
    
    def get_hemo_o2_statistics(self,generation:int=None,save=True): # type: ignore
        """
        This function retrieves a set of information about the tree that is valuable for evaluating it in a statistical sense

        :param: An optional int specifying the generation number on the file name
        :return: A dictionary containing the information for vascular statistics
        """
        vessel_radii = {}
        vessel_length = {}
        vessel_oxygen = {}
        vessel_flow = {}
        associated_vessels,_ = self.get_vessels_from_segments()
        # self._get_growth_info()
        oxygen_solution = self.mat_handler.solution_oxygen["ovx"]
        node_point_ref = self.getfem_handler_1D.node_point_ref
        
        # Calculate the statistics for each vessel
        num_vessels = 0
        for vessel_id, segment_ids in associated_vessels.items():
            total_length = 0
            total_radii = 0
            total_oxygen = 0
            total_flow = 0
            for segment_id in segment_ids:
                length = self.tree.length(segment_id)
                total_length += length
                total_radii += self.tree.segment_dict[segment_id].radius * length
                id1, id2 = self.tree.get_node_ids_on_segment(segment_id)
                node_ids = [id1,id2]
                oxygens = np.array([oxygen_solution[node_point_ref[node_id]] for node_id in node_ids])
                total_oxygen = np.mean(oxygens)*length

                start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
                end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]
                area = self.tree.area(segment_id)
                start_flow = area*start_velocity
                end_flow = area*end_velocity
                flows = np.mean([start_flow,end_flow])
                total_flow += flows*length

            
            mean_radii = total_radii / total_length
            vessel_radii[vessel_id] = mean_radii

            mean_oxygen = total_oxygen / total_length
            vessel_oxygen[vessel_id] = mean_oxygen

            mean_flows = total_flow / total_length
            vessel_flow[vessel_id] = mean_flows
            
            vessel_length[vessel_id] = total_length
            
            num_vessels += 1

        inlet_id_list = self.tree.get_node_ids_inlet()
        outlet_id_list = self.tree.get_node_ids_outlet()
        
        num_terminals = len(inlet_id_list)+len(outlet_id_list)
        num_bifurcations = (2*num_vessels-num_terminals)/3
        
        luminal_volume = self.tree.get_network_volume()
        tissue_volume = self.tissue.get_tissue_volume()

        vascular_densisty = luminal_volume / tissue_volume

        # Calculate the equivalent hydraulic resistance, R = (Pin - Pout) / Q
        
        incoming_flow = 0
        for inlet_id in inlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            incoming_flow += mean_flow

        
        outgoing_flow = 0
        outgoing_o2 = 0
        for outlet_id in outlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(outlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            outgoing_flow += mean_flow

            outlet_pid = self.getfem_handler_1D.node_point_ref[outlet_id]
            O2_value = self.mat_handler.solution_oxygen["ovx"][outlet_pid]
            outgoing_o2 += O2_value*mean_flow

        

        delta_p = self.getfem_handler_1D.pressure_inlet - self.getfem_handler_1D.pressure_outlet

        resistance = delta_p / incoming_flow
        solubility_O2 = self.config.config_access["1D_CONDITIONS"]["solubility_O2"]
        P_O2_in = self.config.config_access["1D_CONDITIONS"]["P_O2_in"]
        oxygen_concentration = solubility_O2 * P_O2_in
        oxygen_flow = incoming_flow*oxygen_concentration
        #global_o2_mean = self.calculate_global_oxygen_mean()
        # global_VEGF_mean = self.calculate_global_vegf_mean()

        statistical_dict = {"radii":list(vessel_radii.values()),\
                            "lengths":list(vessel_length.values()),\
                            "oxygens":list(vessel_oxygen.values()),\
                            "flows":list(vessel_flow.values()),\
                            "vascular_density":[luminal_volume,tissue_volume,vascular_densisty],\
                            "num_vessels":num_vessels,\
                            "num_bifurcations":num_bifurcations,\
                            "inlet_pressure":self.getfem_handler_1D.pressure_inlet,\
                            "outlet_pressure":self.getfem_handler_1D.pressure_outlet,\
                            "hydraulic_resist":resistance,\
                            "incoming_flow":incoming_flow,\
                            "outgoing_flow":outgoing_flow,\
                            "incoming_oxygen":oxygen_flow,\
                            "outgoing_oxygen":outgoing_o2}
        
        if save:
            # Convert NumPy arrays within lists to lists
            for key, value in statistical_dict.items():
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, np.ndarray):
                            statistical_dict[key][i] = item.tolist()

            # Define the file path where you want to save the JSON file
            case = self.config.growth_case
            file_path = "./"+self.filepath+"/statistic_results/"+f"case_{case}/vascular_statistics"
            if not generation is None:
                file_path += f"_{generation}"
            file_path += ".json"

            # Write the dictionary to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(statistical_dict, json_file)
        
        return statistical_dict

    def get_hemo_statistics(self,generation:int=None,save=True): # type: ignore
        """
        This function retrieves a set of information about the tree that is valuable for evaluating it in a statistical sense

        :param: An optional int specifying the generation number on the file name
        :return: A dictionary containing the information for vascular statistics
        """
        vessel_radii = {}
        vessel_length = {}
        vessel_oxygen = {}
        vessel_flow = {}
        associated_vessels,_ = self.get_vessels_from_segments()
        # self._get_growth_info()
        # oxygen_solution = self.mat_handler.solution_oxygen["ovx"]
        # node_point_ref = self.getfem_handler_1D.node_point_ref
        
        # Calculate the statistics for each vessel
        num_vessels = 0
        for vessel_id, segment_ids in associated_vessels.items():
            total_length = 0
            total_radii = 0
            total_oxygen = 0
            total_flow = 0
            for segment_id in segment_ids:
                length = self.tree.length(segment_id)
                total_length += length
                total_radii += self.tree.segment_dict[segment_id].radius * length
                id1, id2 = self.tree.get_node_ids_on_segment(segment_id)
                node_ids = [id1,id2]
                # oxygens = np.array([oxygen_solution[node_point_ref[node_id]] for node_id in node_ids])
                # total_oxygen = np.mean(oxygens)*length

                start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
                end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]
                area = self.tree.area(segment_id)
                start_flow = area*start_velocity
                end_flow = area*end_velocity
                flows = np.mean([start_flow,end_flow])
                total_flow += flows*length

            
            mean_radii = total_radii / total_length
            vessel_radii[vessel_id] = mean_radii

            mean_oxygen = total_oxygen / total_length
            vessel_oxygen[vessel_id] = mean_oxygen

            mean_flows = total_flow / total_length
            vessel_flow[vessel_id] = mean_flows
            
            vessel_length[vessel_id] = total_length
            
            num_vessels += 1

        inlet_id_list = self.tree.get_node_ids_inlet()
        outlet_id_list = self.tree.get_node_ids_outlet()
        
        num_terminals = len(inlet_id_list)+len(outlet_id_list)
        num_bifurcations = (2*num_vessels-num_terminals)/3
        
        luminal_volume = self.tree.get_network_volume()
        tissue_volume = self.tissue.get_tissue_volume()

        vascular_densisty = luminal_volume / tissue_volume

        # Calculate the equivalent hydraulic resistance, R = (Pin - Pout) / Q
        
        incoming_flow = 0
        for inlet_id in inlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            incoming_flow += mean_flow

        
        outgoing_flow = 0
        outgoing_o2 = 0
        for outlet_id in outlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(outlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.mat_handler.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            outgoing_flow += mean_flow

            # outlet_pid = self.getfem_handler_1D.node_point_ref[outlet_id]
            # O2_value = self.mat_handler.solution_oxygen["ovx"][outlet_pid]
            # outgoing_o2 += O2_value*mean_flow

        

        delta_p = self.getfem_handler_1D.pressure_inlet - self.getfem_handler_1D.pressure_outlet

        resistance = delta_p / incoming_flow
        # solubility_O2 = self.config.config_access["1D_CONDITIONS"]["solubility_O2"]
        # P_O2_in = self.config.config_access["1D_CONDITIONS"]["P_O2_in"]
        # oxygen_concentration = solubility_O2 * P_O2_in
        # oxygen_flow = incoming_flow*oxygen_concentration
        # global_o2_mean = self.calculate_global_oxygen_mean()
        # global_VEGF_mean = self.calculate_global_vegf_mean()

        statistical_dict = {"radii":list(vessel_radii.values()),\
                            "lengths":list(vessel_length.values()),\
                            "flows":list(vessel_flow.values()),\
                            "vascular_density":[luminal_volume,tissue_volume,vascular_densisty],\
                            "num_vessels":num_vessels,\
                            "num_bifurcations":num_bifurcations,\
                            "inlet_pressure":self.getfem_handler_1D.pressure_inlet,\
                            "outlet_pressure":self.getfem_handler_1D.pressure_outlet,\
                            "hydraulic_resist":resistance,\
                            "incoming_flow":incoming_flow,\
                            "outgoing_flow":outgoing_flow}
        
        if save:
            # Convert NumPy arrays within lists to lists
            for key, value in statistical_dict.items():
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, np.ndarray):
                            statistical_dict[key][i] = item.tolist()

            # Define the file path where you want to save the JSON file
            case = self.config.growth_case
            file_path = "./"+self.filepath+"/statistic_results/"+f"case_{case}/vascular_statistics"
            if not generation is None:
                file_path += f"_{generation}"
            file_path += ".json"

            # Write the dictionary to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(statistical_dict, json_file)
        
        return statistical_dict       






        






        
        






        
