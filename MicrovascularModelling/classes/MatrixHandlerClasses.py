from cmath import nan
from classes.GeometryClasses import Tree, Tissue, Visualizer
from classes.GetFEMClasses import GetFEMHandler1D,GetFEMHandler3D
from classes.ConfigClass import Config, Logger
import json
import copy
import numpy as np # type: ignore
from petsc4py import PETSc # type: ignore
import scipy.sparse as sp # type: ignore
import pyvista as pv # type: ignore
import time
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union


class MatrixHandler(Tree):
    """
    A class used to handle the PETSc Matrix assemble processes for the 1D geometry.

    ----------
    Class Attributes
    ----------
    __INTIAL_GUESS_VALUES: dict
        A set of values associated with the initial solution guesses

    ----------
    Instance Attributes
    ----------
    tree : object
        An object containing the tree geometry

    A : numpy array
        A numpy array containing the the matrix for the haemodynamic system

    Ah : numpy array
        A numpy array containing the the matrix for the haematocrit system

    Ao : numpy array
        A numpy array containing the the matrix for the oxygen system

    b : numpy array
        A numpy array containing the the forcing vector for the haemodynamic system

    bh : numpy array
        A numpy array containing the the forcing vector for the haematocrit system

    bo : numpy array
        A numpy array containing the the forcing vector for the oxygen system

    solution_haemodynamic : dict of numpy arrays
        A dictionary of numpy arrays containing the the solution vectors for the haemodynamic system

    solution_haematocrit : dict of numpy arrays
        A dictionary of numpy arrays containing the the solution vectors for the haematocrit system

    solution_oxygen : dict of numpy arrays
        A dictionary of numpy arrays containing the the solution vectors for the oxygen system

    getfem_handler1D : object
        An object containing a class instance of the GetFEMHandler1D class

    getfem_handler3D : object
        An object containing a class instance of the GetFEMHandler3D class

    martix_type : string
        A string specifying what type of matrix is currently being handled

    ----------
    Class Methods
    ----------  
    increment_count(amount=0)
        Adds or subtracts from the __SUBMATRIX_COUNT Attribute

    ----------
    Instance Methods
    ----------  
    full_matrix_size()
        Returns the size of the full matrix A
    
    full_vector_size()
        Returns the length of the full vector b

    build()
        Returns the full matrix A and vector b, for the specified case

    feedback_solution(x, relaxation)
        Updates the solution information based on the vector x produced by the solver

    set_haemodynamic()
        Sets the matrix_type to Haemodynamic

    set_haematocrit()
        Sets the matrix_type to Haematocrit

    set_oxygen()
        Sets the matrix_type to Oxygen

    active_array()
        Returns the built array of the currently set type

    active_vec()
        Returns the built vector of the currently set type

    visualize_pressure()
        Plots the pressure along the tree geometry

    visualize_velocity()
        Plots the velocity along the tree geometry

    visualize_haematocrit()
        Plots the haematocrit along the tree geometry

    visualize_oxygen()
        Plots the oxygen along the tree geometry

    visualize_all()
        Plots the pressure, velocity, haematocrit, and oxygen along the tree geometry

    """



    def __init__(self, tree:Tree, tissue:Tissue, getfem_handler1D:GetFEMHandler1D, getfem_handler3D:GetFEMHandler3D,\
                config:Config,cylinder_test=False):
        self.getfem_handler_1D = getfem_handler1D
        self.getfem_handler_3D = getfem_handler3D
        self.set_haemodynamic()
        self.tree = tree
        self.tissue = tissue

        cylinder_test, new_mesh, mesh_path, output_path, test_name = config.parse_run()
        self.mesh_path = mesh_path
        self.set_save_details(output_path, test_name)
        self.config = config
        self.new_mesh = new_mesh
        self.cylinder_test = cylinder_test
        self.mesh3D = self.getfem_handler_3D.make_mesh(new_mesh,self.mesh_path,tissue,cylinder_test)
        self.getfem_handler_3D.make_haemodynamic_elements(mesh3D=self.mesh3D)
        if self.new_mesh == True:
            self.new_mesh = False

        self.initial_3D_haemodynamic_completed = False

        self.reset_system()


        self.tolerance_scale = 1

        self.logger = config.logger

    def build_1D(self, verbose=False):
        if self.matrix_type == "Haemodynamic":
            if self.fedback_haematocrit:
                if self.force_recalculate:
                    start_time = time.time()
                    hx = self.solution_haematocrit["hx"]
                    self.Av, self.bv, self.pvfem, self.uvfem_dict, self.mesh1D, x = self.getfem_handler_1D.build_monolithic_vessel(\
                        self.tree, hx, recalculate_matrix=True)
                    end_time = time.time()
                    self.set_force_recalculate(False)
                    # self.logger.log(f"1D Force Recalc")
                    if verbose:
                        self.logger.log(f"Time for 1D build with Haematocrit Forced Recalc: {end_time-start_time}")
                else:
                    start_time = time.time()
                    hx = self.solution_haematocrit["hx"]
                    self.Av, self.bv, self.pvfem, self.uvfem_dict, self.mesh1D, x = self.getfem_handler_1D.build_monolithic_vessel(\
                        self.tree,hx,mesh1D=self.mesh1D, recalculate_matrix=False)
                    end_time = time.time()
                    # self.logger.log(f"1D Not Force Recalc")
                    if verbose:
                        self.logger.log(f"Time for 1D build with Haematocrit: {end_time-start_time}")
            else:
                start_time = time.time()
                self.Av, self.bv, self.pvfem, self.uvfem_dict, self.mesh1D, x = self.getfem_handler_1D.build_monolithic_vessel(\
                    self.tree)
                end_time = time.time()
                # self.logger.log(f"1D no Haematocrit")
                if verbose:
                    self.logger.log(f"Time for 1D build Naked: {end_time-start_time}")

            self.A = self.Av
            self.b = self.bv
            return
        else:
            raise ValueError(f"Handler should be set to 'Haemodynamic'")

    def build(self, verbose=False):
        if self.matrix_type == "Haemodynamic":
            if self.fedback_haematocrit:
                if self.force_recalculate:
                    start_time = time.time()
                    hx = self.solution_haematocrit["hx"]
                    self.Av, self.bv, self.pvfem, self.uvfem_dict, self.mesh1D, x = self.getfem_handler_1D.build_monolithic_vessel(\
                        self.tree,hx,mesh1D=self.mesh1D, recalculate_matrix=True)
                    end_time = time.time()
                    self.set_force_recalculate(False)
                    if verbose:
                        self.logger.log(f"Time for 1D build with Haematocrit Forced Recalc: {end_time-start_time}")
                else:
                    start_time = time.time()
                    hx = self.solution_haematocrit["hx"]
                    self.Av, self.bv, self.pvfem, self.uvfem_dict, self.mesh1D, x = self.getfem_handler_1D.build_monolithic_vessel(\
                        self.tree,hx,mesh1D=self.mesh1D, recalculate_matrix=False)
                    end_time = time.time()
                    if verbose:
                        self.logger.log(f"Time for 1D build with Haematocrit: {end_time-start_time}")
            else:
                start_time = time.time()
                self.Av, self.bv, self.pvfem, self.uvfem_dict, self.mesh1D, x = self.getfem_handler_1D.build_monolithic_vessel(\
                    self.tree)
                end_time = time.time()
                if verbose:
                    self.logger.log(f"Time for 1D build Naked: {end_time-start_time}")
                
            if not self.initial_3D_haemodynamic_completed:
                if self.fedback_haemodynamic:
                    start_time = time.time()
                    ptx = self.solution_haemodynamic["ptx"]
                    At, bt, mimt, self.ptfem, self.utfem = self.getfem_handler_3D.build_monolithic_tissue(\
                        self.tree,self.tissue,self.mesh3D,"ROBIN",ptx,self.cylinder_test)
                    end_time = time.time()
                    if verbose:
                        self.logger.log(f"Time for 3D build with P: {end_time-start_time}")
                    
                else:
                    start_time = time.time()
                    At, bt, mimt, self.ptfem, self.utfem = self.getfem_handler_3D.build_monolithic_tissue(\
                        self.tree,self.tissue,self.mesh3D,"ROBIN",None,self.cylinder_test)
                    end_time = time.time()
                    if verbose:
                        self.logger.log(f"Time for 3D build Naked: {end_time-start_time}")
                    
                self.At = At
                self.bt = bt
                self.initial_3D_haemodynamic_completed = True

            mimv = self.getfem_handler_1D.mim
            if not self.fedback_haemodynamic or not hasattr(self, 'Bvv'):
                start_time = time.time()
                Mbar, Mlin = self.getfem_handler_3D._build_auxillary_matricies(self.mesh1D, self.pvfem, self.ptfem, self.tree,self.tissue)
                self.Bvv, self.Bvt, self.Btv, self.Btt = self.getfem_handler_3D._build_exchange_matricies(\
                    mimv, self.pvfem, Mbar, Mlin, self.mesh1D, self.tree)
                end_time = time.time()
                if verbose:
                    self.logger.log(f"Time for Interpolation: {end_time-start_time}")
            
            start_time = time.time()
            if self.fedback_haemodynamic and hasattr(self, 'A'):
                del self.A
                self.A = self.getfem_handler_3D._build_final_MAT(self.At, self.Av, self.Bvv, self.Bvt, self.Btv, self.Btt)
            else:
                self.A = self.getfem_handler_3D._build_final_MAT(self.At, self.Av, self.Bvv, self.Bvt, self.Btv, self.Btt)
            

            self.b = self.getfem_handler_3D._build_final_VEC(self.bt, self.bv, self.Bvv, self.Btv)
            end_time = time.time()
            if verbose:
                self.logger.log(f"Time for Assembly: {end_time-start_time}")

            return

        if self.matrix_type == "Haematocrit":
            if self.fedback_haemodynamic:
                start_time = time.time()
                self.capped_haematocrit = False
                h_vector = self.algorithmic_haematocrit()
                end_time = time.time()
                if verbose:
                    self.logger.log(f"Time for Haematocrit Solution: {end_time-start_time}")
            else:
                raise ValueError("A 1D velocity solution must be fed back to the handler before the haematocrit system can be built")
        
            return h_vector

        if self.matrix_type == "Oxygen":
            if self.fedback_haematocrit and self.fedback_haemodynamic:
                hx = self.solution_haematocrit["hx"]
                uvx = self.solution_haemodynamic["uvx"]
                utx = self.solution_haemodynamic["utx"]
            else:
                raise ValueError("The oxygen submatricies require a velocity solution in 1D and 3D and a haematocrit solution in 1D")  

            if self.fedback_oxygen:
                ovx = self.solution_oxygen["ovx"]
                Ao, bo, self.ovfem, xo = self.getfem_handler_1D.build_monolithic_oxygen(\
                    self.tree,self.mesh1D,uvx,hx,ovx,output="Spmat",verbose=False)
            else:
                Ao, bo, self.ovfem, xo = self.getfem_handler_1D.build_monolithic_oxygen(\
                    self.tree,self.mesh1D,uvx,hx,output="Spmat",verbose=False)
                
            if self.fedback_oxygen:
                otx = self.solution_oxygen["otx"]
                Ato, bto, mimt, self.otfem = self.getfem_handler_3D.build_monolithic_oxygen_tissue(\
                    self.mesh3D,utx,otx,self.cylinder_test)   
            else:
                Ato, bto, mimt, self.otfem = self.getfem_handler_3D.build_monolithic_oxygen_tissue(\
                    self.mesh3D,utx,None,self.cylinder_test)   

            mimv = self.getfem_handler_1D.mim
            if not self.fedback_oxygen or not hasattr(self, 'Bvvo'):
                Mbar, Mlin = self.getfem_handler_3D._build_auxillary_matricies(\
                    self.mesh1D, self.ovfem, self.otfem, self.tree, self.tissue)
                self.Bvvo, self.Bvto, self.Btvo, self.Btto = self.getfem_handler_3D._build_oxygen_exchange(\
                    mimv, self.ovfem, Mbar, Mlin, self.mesh1D, self.tree)
                

            self.Ao = self.getfem_handler_3D._build_final_oxygen_MAT(Ato, Ao, self.Bvvo, self.Bvto, self.Btvo, self.Btto)
            self.bo = self.getfem_handler_3D._build_final_oxygen_VEC(bto, bo)

            return

        if self.matrix_type == "VEGF":
            if self.fedback_oxygen and self.fedback_haemodynamic:
                otx = self.solution_oxygen["otx"]
                utx = self.solution_haemodynamic["utx"]
            else:
                raise ValueError("The VEGF matricies require a Oxygen and Velocity solution in 3D")  

            Atv, btv, mimt, self.vtfem = self.getfem_handler_3D.build_monolithic_vegf_tissue(\
                self.mesh3D,utx,otx,self.cylinder_test)
            self.Av = Atv
            self.bv = btv

            return
    
    def get_baseline(self) -> Tuple[float,float]:
        """
        Function to retrieve the baseline information used to update the flow-volume constraint.

        :return: A Tuple containing two floats, [Tree Volume, Total Incoming Flow]. 
        """

        volume = copy.deepcopy(self.tree.get_network_volume())
        inlet_id_list = self.tree.get_node_ids_inlet()
        total_incoming_flow = 0
        for inlet_id in inlet_id_list:
            segment_ids,_ = self.tree.get_segment_ids_on_node(inlet_id)
            segment_id = segment_ids[0]
            area = self.tree.area(segment_id)
            start_velocity = self.solution_haemodynamic["uvx"][segment_id+1][0]
            end_velocity = self.solution_haemodynamic["uvx"][segment_id+1][-1]

            start_flow = area*start_velocity
            end_flow = area*end_velocity
            mean_flow = np.mean([start_flow,end_flow])
            total_incoming_flow += mean_flow

        flow = copy.deepcopy(total_incoming_flow)

        return volume, flow
    
    def reset_feedback(self):
        if self.matrix_type == "Haemodynamic":
            self.fedback_haemodynamic = False
            self.solution_haemodynamic = {}
        if self.matrix_type == "Haematocrit":
            self.fedback_haematocrit = False
            self.solution_haematocrit = {}
        if self.matrix_type == "Oxygen":
            self.fedback_oxygen = False
            self.solution_oxygen = {}
        if self.matrix_type == "VEGF":
            self.fedback_vegf = False
            self.solution_vegf = {}

    def set_force_recalculate(self,to_recalculate:bool=True):
        self.force_recalculate = to_recalculate
    
    def set_force_geometry(self,to_recalculate:bool=True):
        self.getfem_handler_1D.recalculate_geometry = to_recalculate

    def feedback_solution_1D(self, x:np.ndarray, relaxation = 1., verbose=False):
        if not (relaxation > 0 and relaxation <= 1):
            raise ValueError(f"Relaxation should take a value in the range (0,1], Relaxation was {relaxation}")

        if self.matrix_type == "Haemodynamic":
            # initialize difference norms
            step_diff_uvx = 0
            step_diff_pvx = 0

            # preset degrees of freedom
            offset = 0
            pvdofs = self.pvfem.nbdof()

            # if no solution stored create dummy solution
            if not self.fedback_haemodynamic:
                self.solution_haemodynamic.update({"uvx":{}})
                uvdofs = 0
                for keys in self.uvfem_dict:
                    uvdofs = self.uvfem_dict[keys].nbdof()
                    self.solution_haemodynamic["uvx"].update({keys.item():np.zeros(uvdofs)})
                self.solution_haemodynamic.update({"pvx":np.zeros(pvdofs)})
            
            # extract components of the solution vector x
            uvx_new = {}
            for keys in self.uvfem_dict:
                uvdofs = self.uvfem_dict[keys].nbdof()
                uvx_new.update({keys.item():x[offset:offset+uvdofs].copy()})
                # self.logger.log(f"uv_solution = {x[utdofs+ptdofs+offset:utdofs+ptdofs+offset+uvdofs]}")
                offset += uvdofs
            pvx_new = x[offset:].copy()
            # self.logger.log(f"pv_solution = {pvx_new}")

            # calculate the step difference norms if not first step
            if self.fedback_haemodynamic:
                for keys in self.uvfem_dict:
                    step_diff_uvx += np.linalg.norm((relaxation*uvx_new[keys]+(1-relaxation)*self.solution_haemodynamic["uvx"][keys]) - self.solution_haemodynamic["uvx"][keys]) \
                                    / np.linalg.norm(self.solution_haemodynamic["uvx"][keys])
                step_diff_pvx += np.linalg.norm((relaxation*pvx_new+(1-relaxation)*np.array(self.solution_haemodynamic["pvx"])) - self.solution_haemodynamic["pvx"]) \
                                / np.linalg.norm(self.solution_haemodynamic["pvx"])
            # if first step create dummy values
            else:
                step_diff_uvx = 10000
                step_diff_pvx = 10000


            # set the stored solution to the input solution and apply relaxation
            for keys in self.uvfem_dict:
                self.solution_haemodynamic["uvx"][keys] = relaxation*uvx_new[keys] + (1-relaxation)*self.solution_haemodynamic["uvx"][keys]
            self.solution_haemodynamic["pvx"] = relaxation*pvx_new + (1-relaxation)*np.array(self.solution_haemodynamic["pvx"])

            # change feedback state
            self.fedback_haemodynamic = True

            if verbose:
                self.logger.log("##################  U SOLUTION  #####################")
                for segment in self.uvfem_dict:
                    self.logger.log(f"U for Segment {segment}:")
                    self.logger.log(self.solution_haemodynamic["uvx"][segment])
                self.logger.log("##################  U SOLUTION  #####################")


            # return the values of the step difference norms
            return step_diff_uvx, step_diff_pvx
        
    def feedback_solution(self, x:np.ndarray, relaxation = 1., verbose=False):
        if not (relaxation > 0 and relaxation <= 1):
            raise ValueError(f"Relaxation should take a value in the range (0,1], Relaxation was {relaxation}")

        if self.matrix_type == "Haemodynamic":
            # initialize difference norms
            step_diff_utx = 0
            step_diff_ptx = 0
            step_diff_uvx = 0
            step_diff_pvx = 0

            # preset degrees of freedom
            utdofs = self.utfem.nbdof()
            ptdofs = self.ptfem.nbdof()
            offset = 0
            pvdofs = self.pvfem.nbdof()

            # if no solution stored create dummy solution
            if not self.fedback_haemodynamic:
                self.solution_haemodynamic.update({"utx":np.zeros(utdofs)})
                self.solution_haemodynamic.update({"ptx":np.zeros(ptdofs)})
                self.solution_haemodynamic.update({"uvx":{}})
                uvdofs = 0
                for keys in self.uvfem_dict:
                    uvdofs = self.uvfem_dict[keys].nbdof()
                    self.solution_haemodynamic["uvx"].update({keys.item():np.zeros(uvdofs)})
                self.solution_haemodynamic.update({"pvx":np.zeros(pvdofs)})
            
            # extract components of the solution vector x
            utx_new = x[:utdofs].copy()
            ptx_new = x[utdofs:utdofs+ptdofs].copy()
            uvx_new = {}
            for keys in self.uvfem_dict:
                uvdofs = self.uvfem_dict[keys].nbdof()
                uvx_new.update({keys.item():x[utdofs+ptdofs+offset:utdofs+ptdofs+offset+uvdofs].copy()})
                # self.logger.log(f"uv_solution = {x[utdofs+ptdofs+offset:utdofs+ptdofs+offset+uvdofs]}")
                offset += uvdofs
            pvx_new = x[utdofs+ptdofs+offset:].copy()
            # self.logger.log(f"pv_solution = {pvx_new}")

            # calculate the step difference norms if not first step
            if self.fedback_haemodynamic:
                step_diff_utx += np.linalg.norm((relaxation*utx_new+(1-relaxation)*np.array(self.solution_haemodynamic["utx"])) - self.solution_haemodynamic["utx"]) \
                                / np.linalg.norm(self.solution_haemodynamic["utx"])
                step_diff_ptx += np.linalg.norm((relaxation*ptx_new+(1-relaxation)*np.array(self.solution_haemodynamic["ptx"])) - self.solution_haemodynamic["ptx"]) \
                                / np.linalg.norm(self.solution_haemodynamic["ptx"])
                for keys in self.uvfem_dict:
                    step_diff_uvx += np.linalg.norm((relaxation*uvx_new[keys]+(1-relaxation)*self.solution_haemodynamic["uvx"][keys]) - self.solution_haemodynamic["uvx"][keys]) \
                                    / np.linalg.norm(self.solution_haemodynamic["uvx"][keys])
                step_diff_pvx += np.linalg.norm((relaxation*pvx_new+(1-relaxation)*np.array(self.solution_haemodynamic["pvx"])) - self.solution_haemodynamic["pvx"]) \
                                / np.linalg.norm(self.solution_haemodynamic["pvx"])
            # if first step create dummy values
            else:
                step_diff_utx = 10000
                step_diff_ptx = 10000
                step_diff_uvx = 10000
                step_diff_pvx = 10000


            # set the stored solution to the input solution and apply relaxation
            self.solution_haemodynamic["utx"] = relaxation*utx_new + (1-relaxation)*np.array(self.solution_haemodynamic["utx"])
            self.solution_haemodynamic["ptx"] = relaxation*ptx_new + (1-relaxation)*np.array(self.solution_haemodynamic["ptx"])
            for keys in self.uvfem_dict:
                self.solution_haemodynamic["uvx"][keys] = relaxation*uvx_new[keys] + (1-relaxation)*self.solution_haemodynamic["uvx"][keys]
            self.solution_haemodynamic["pvx"] = relaxation*pvx_new + (1-relaxation)*np.array(self.solution_haemodynamic["pvx"])

            # change feedback state
            self.fedback_haemodynamic = True

            if verbose:
                self.logger.log("##################  U SOLUTION  #####################")
                for segment in self.uvfem_dict:
                    self.logger.log(f"U for Segment {segment}:")
                    self.logger.log(self.solution_haemodynamic["uvx"][segment])
                self.logger.log("##################  U SOLUTION  #####################")


            # return the values of the step difference norms
            return step_diff_utx, step_diff_ptx, step_diff_uvx, step_diff_pvx

        elif self.matrix_type == "Haematocrit":
            # initialize difference norm
            step_diff_hx = 0

            # preset degrees of freedom 
            offset = 0

            # if no solution stored, create dummy solution
            if not self.fedback_haematocrit:
                self.solution_haematocrit.update({"hx":{}})
                for keys in self.hvfem_dict:
                    hxdofs = self.hvfem_dict[keys].nbdof()
                    self.solution_haematocrit["hx"].update({keys:np.full(hxdofs,0.45)})

            # extract components of the solution vector x
            hx_new = {}
            haematocrit_hack = False
            for keys in self.hvfem_dict:
                hxdofs = self.hvfem_dict[keys].nbdof()
                local_vector = x[offset:offset+hxdofs]
                if haematocrit_hack:
                    local_vector[:] = local_vector[:2].mean()

                hx_new.update({keys:local_vector})
                # self.logger.log(f"hx_solution = {x[offset:offset+hxdofs]}")
                offset += hxdofs

            # calculate the step difference norm
            if self.fedback_haematocrit:
                for keys in self.hvfem_dict:
                    step_diff_hx += np.linalg.norm((relaxation * hx_new[keys] + (1 - relaxation) * self.solution_haematocrit["hx"][keys]) - self.solution_haematocrit["hx"][keys]) \
                                    / np.linalg.norm(self.solution_haematocrit["hx"][keys])
            # if first step create dummy values
            else:
                step_diff_hx = 10000

            # set the stored solution to the input solution and apply relaxation
            for segment in self.hvfem_dict:
                self.solution_haematocrit["hx"][segment] = relaxation * hx_new[segment] + (1 - relaxation) * self.solution_haematocrit["hx"][segment]

            # change feedback state
            self.fedback_haematocrit = True

            if verbose:
                self.logger.log("##################  H SOLUTION  #####################")
                for segment in self.hvfem_dict:
                    self.logger.log(f"H for Segment {segment}:")
                    self.logger.log(self.solution_haematocrit["hx"][segment])
                self.logger.log("##################  H SOLUTION  #####################")

            # return the value of the step difference norm
            return step_diff_hx

        elif self.matrix_type == "Oxygen":
            # initialize difference norms
            step_diff_otx = 0
            step_diff_ovx = 0

            # preset degrees of freedom
            otdofs = self.otfem.nbdof()
            ovdofs = self.ovfem.nbdof()

            # if no solution stored, create dummy solution
            if not self.fedback_oxygen:
                self.solution_oxygen["otx"] = np.zeros(otdofs)
                self.solution_oxygen["ovx"] = np.zeros(ovdofs)

            # extract components of the solution vector x
            otx_new = x[:otdofs]
            ovx_new = x[otdofs:otdofs+ovdofs]

            # calculate the step difference norms
            if self.fedback_oxygen:
                otx_diff = np.linalg.norm((relaxation * otx_new + (1 - relaxation) * np.array(self.solution_oxygen["otx"])) - self.solution_oxygen["otx"])
                otx_norm = np.linalg.norm(self.solution_oxygen["otx"])
                step_diff_otx += otx_diff / otx_norm if otx_norm > 0 else 10000

                ovx_diff = np.linalg.norm((relaxation * ovx_new + (1 - relaxation) * np.array(self.solution_oxygen["ovx"])) - self.solution_oxygen["ovx"])
                ovx_norm = np.linalg.norm(self.solution_oxygen["ovx"])
                step_diff_ovx += ovx_diff / ovx_norm if ovx_norm > 0 else 10000

            # if first step create dummy values
            else:
                step_diff_otx = 10000
                step_diff_ovx = 10000

            # set the stored solution to the input solution and apply relaxation
            self.solution_oxygen["otx"] = relaxation * otx_new + (1 - relaxation) * np.array(self.solution_oxygen["otx"])
            self.solution_oxygen["ovx"] = relaxation * ovx_new + (1 - relaxation) * np.array(self.solution_oxygen["ovx"])

            # change feedback state
            self.fedback_oxygen = True

            # return the values of the step difference norms
            return step_diff_otx, step_diff_ovx
        
        elif self.matrix_type == "VEGF":
            # initialize difference norms
            step_diff_vtx = 0

            # preset degrees of freedom
            vtdofs = self.vtfem.nbdof()

            # if no solution stored, create dummy solution
            if not self.fedback_vegf:
                self.solution_vegf["vtx"] = np.zeros(vtdofs)

            # extract components of the solution vector x
            vtx_new = x[:]

            # calculate the step difference norms
            if self.fedback_oxygen:
                vtx_diff = np.linalg.norm((relaxation * vtx_new + (1 - relaxation) * np.array(self.solution_vegf["vtx"])) - self.solution_vegf["vtx"])
                vtx_norm = np.linalg.norm(self.solution_vegf["vtx"])
                step_diff_vtx += vtx_diff / vtx_norm if vtx_norm > 0 else 10000

            # if first step create dummy values
            else:
                step_diff_otx = 10000

            # set the stored solution to the input solution and apply relaxation
            self.solution_vegf["vtx"] = relaxation * vtx_new + (1 - relaxation) * np.array(self.solution_vegf["vtx"])

            # change feedback state
            self.fedback_vegf = True

            # return the values of the step difference norms
            return step_diff_vtx

    def algorithmic_haematocrit(self):
        if self.fedback_haemodynamic:
            if self.fedback_haematocrit:
                uvx = self.solution_haemodynamic["uvx"]
                hx = self.solution_haematocrit["hx"]
                xh,self.hvfem_dict,self.capped_haematocrit = self.getfem_handler_1D.algorithmic_haematocrit(self.tree,self.mesh1D,uvx,hx)
            else:
                uvx = self.solution_haemodynamic["uvx"]
                xh,self.hvfem_dict,self.capped_haematocrit = self.getfem_handler_1D.algorithmic_haematocrit(self.tree,self.mesh1D,uvx,None)
        else:
            raise ValueError("A 1D velocity solution must be fed back to the handler before the haematocrit system can be built")
        
        return xh

    def build_post_process_Reynolds(self):
        if not self.fedback_haematocrit and not self.fedback_haemodynamic:
            raise ValueError("Post_processing requires a haemodynamic and a haematocrit solution")

        re_MAT, re_VEC, self.re_element = self.getfem_handler_1D.post_process_Reynolds(self.tree,self.mesh1D,self.solution_haemodynamic["uvx"])
        
        return re_MAT,re_VEC

    def build_post_process_wss(self):
        if not self.fedback_haematocrit and not self.fedback_haemodynamic:
            raise ValueError("Post_processing requires a haemodynamic and a haematocrit solution")

        wss_MAT, wss_VEC, self.wss_element = self.getfem_handler_1D.post_process_wss(self.tree,self.mesh1D,self.solution_haemodynamic["uvx"])

        return wss_MAT,wss_VEC

    def feedback_post_process(self, x:np.ndarray, dictionary_key:str):
        self.solution_post_process.update({dictionary_key:x})

    def build_vessel_attractor_field(self,sprout_dict:dict):
        self.mesh1D,_ = self.getfem_handler_1D.build_1D_mesh(self.tree)
        A_base,b_base,mim,attractor_element = self.getfem_handler_3D.build_base_attractor_field(self.mesh3D)
        RHS_TREE_AUG = self.getfem_handler_3D._build_vessel_inhibition_signal(self.mesh1D,self.tree,self.tissue)
        RHS_SPROUT_AUG = {}
        for key in sprout_dict.keys():
            RHS_SPROUT_AUG.update({key:self.getfem_handler_3D._build_tip_cell_signal(sprout_dict[key])})

        return A_base,b_base,RHS_TREE_AUG,RHS_SPROUT_AUG

    def set_haemodynamic(self):
        self.matrix_type = "Haemodynamic"

    def set_haematocrit(self):
        self.matrix_type = "Haematocrit"

    def set_oxygen(self):
        self.matrix_type = "Oxygen"
        self.getfem_handler_1D.recalculate_O2_matrix = True
        self.getfem_handler_3D.recalculate_O2_matrix = True

    def set_vegf(self):
        self.matrix_type = "VEGF"

    def active_array(self):
        if self.matrix_type == "Haemodynamic":
            return self.A
        if self.matrix_type == "Haematocrit":
            raise ValueError(f"Haematocrit is solved algorithmically not with matricies.")
        if self.matrix_type == "Oxygen":
            return self.Ao
        if self.matrix_type == "VEGF":
            return self.Av
    
    def active_vec(self) -> np.ndarray:
        if self.matrix_type == "Haemodynamic":
            return self.b
        if self.matrix_type == "Haematocrit":
            raise ValueError(f"Haematocrit is solved algorithmically not with matricies.")
        if self.matrix_type == "Oxygen":
            return self.bo
        if self.matrix_type == "VEGF":
            return self.bv

    def set_save_details(self,filepath:str,run_name:str):
        self.filepath = os.path.join(filepath,run_name)#+"/"+run_name
        os.makedirs(self.filepath, exist_ok=True)
        self.run_name = run_name

    def save_to_vtk(self,iteration:Union[int,None]=None, blood_1D:bool=False, blood_3D:bool=False, oxygen:bool=False, vegf:bool=False):
        """
        Function to Save the results stored in the matrix handler to .vtk files
        This function takes boolean inputs on what results to save, by defaults all are false and nothing is saved.
        Additionally it overwrites the contents of results that inform the requested saved info.
        ie: vegf requires oxygen, oxygen requires blood 1D and 3D, blood 3D requires blood 1D

        :param iteration: An optional integer that appends an iteration number to saved filename
        :param blood_1D: A boolean that indicates whether to save the 1D pressure, velocity, and haematocrit results.
        :param blood_3D: A boolean that indicates whether to save the 3D pressure, velocity results.
        :param oxygen: A boolean that indicates whether to save the 1D and 3D Oxygen results.
        :param vegf: A boolean that indicates whether to save the 3D VEGF results.
        """
        visualizer = Visualizer(self.config)
        save_string = self.filepath+"/tree_results"
        visualizer.set_save_path(save_string)
        file_string = f"tree_solution"
        if iteration is not None:
            file_string += f"_iteration_{iteration}"
        file_string += f".vtk"
        visualizer.set_file_name(file_string)
        if blood_1D or blood_3D or oxygen or vegf:
            self.add_properties_to_visualizer(visualizer,True,True)
            # self.config.reset_output_folder_by_index([1,2])
            self.getfem_handler_1D.save_haemodynamic_1D(self.filepath,self.run_name,\
                self.uvfem_dict,self.solution_haemodynamic["uvx"],self.pvfem,self.solution_haemodynamic["pvx"],\
                self.hvfem_dict,self.solution_haematocrit["hx"],iteration)
            
        if blood_3D or oxygen or vegf:
            # self.config.reset_output_folder_by_index([0])
            self.getfem_handler_3D.save_haemodynamic_3D(self.filepath,self.run_name,\
            self.utfem,self.solution_haemodynamic["utx"],self.ptfem,self.solution_haemodynamic["ptx"],\
                iteration)
            
        if oxygen or vegf:
            visualizer.reset_mesh_dict()
            self.add_properties_to_visualizer(visualizer,True,True,True)
            self.getfem_handler_3D.save_oxygen(self.filepath,self.run_name,\
            self.utfem,self.solution_haemodynamic["utx"],self.ptfem,self.solution_haemodynamic["ptx"],\
                self.uvfem_dict,self.solution_haemodynamic["uvx"],self.pvfem,self.solution_haemodynamic["pvx"],\
                self.hvfem_dict,self.solution_haematocrit["hx"],\
                    self.otfem,self.solution_oxygen["otx"],self.ovfem,self.solution_oxygen["ovx"],iteration)
            
        if vegf:
            self.getfem_handler_3D.save_vegf(self.filepath,self.run_name,\
            self.utfem,self.solution_haemodynamic["utx"],self.ptfem,self.solution_haemodynamic["ptx"],\
                self.uvfem_dict,self.solution_haemodynamic["uvx"],self.pvfem,self.solution_haemodynamic["pvx"],\
                self.hvfem_dict,self.solution_haematocrit["hx"],\
                    self.otfem,self.solution_oxygen["otx"],self.ovfem,self.solution_oxygen["ovx"],\
                        self.vtfem, self.solution_vegf["vtx"], self.tree, iteration)
        
        if blood_1D or blood_3D or oxygen or vegf:
            visualizer.save_to_file([0])

        return
    
    def save_tissue(self,iteration:Union[int,None]=None, blood_3D:bool=False, oxygen:bool=False, vegf:bool=False):
        """
        Function to save the results for the tissue stored in the matrix handler to .vtk files
        This function takes boolean inputs on what results to save, by defaults all are false and nothing is saved.
        Additionally it overwrites the contents of results that inform the requested saved info.
        ie: vegf requires oxygen, oxygen requires blood 1D and 3D, blood 3D requires blood 1D

        :param iteration: An optional integer that appends an iteration number to saved filename
        :param blood_3D: A boolean that indicates whether to save the 3D pressure, velocity results.
        :param oxygen: A boolean that indicates whether to save the 1D and 3D Oxygen results.
        :param vegf: A boolean that indicates whether to save the 3D VEGF results.
        """
        case = self.config.growth_case
        filepath = self.filepath+"/tissue_results/"+f"case_{case}"

        if vegf:
            self.getfem_handler_3D.save_vegf(filepath,self.run_name,\
            self.utfem,self.solution_haemodynamic["utx"],self.ptfem,self.solution_haemodynamic["ptx"],\
                self.uvfem_dict,self.solution_haemodynamic["uvx"],self.pvfem,self.solution_haemodynamic["pvx"],\
                self.hvfem_dict,self.solution_haematocrit["hx"],\
                    self.otfem,self.solution_oxygen["otx"],self.ovfem,self.solution_oxygen["ovx"],\
                        self.vtfem, self.solution_vegf["vtx"], self.tree, iteration)    
            return
        
        if blood_3D:
            # self.config.reset_output_folder_by_index([0])
            self.getfem_handler_3D.save_haemodynamic_3D(filepath,self.run_name,\
            self.utfem,self.solution_haemodynamic["utx"],self.ptfem,self.solution_haemodynamic["ptx"],\
                iteration)
            return
            
        if oxygen:
            self.getfem_handler_3D.save_oxygen(filepath,self.run_name,\
            self.utfem,self.solution_haemodynamic["utx"],self.ptfem,self.solution_haemodynamic["ptx"],\
                self.uvfem_dict,self.solution_haemodynamic["uvx"],self.pvfem,self.solution_haemodynamic["pvx"],\
                self.hvfem_dict,self.solution_haematocrit["hx"],\
                    self.otfem,self.solution_oxygen["otx"],self.ovfem,self.solution_oxygen["ovx"],iteration)
            return

        return

    
    def add_properties_to_visualizer(self,visualizer:Visualizer,haemodynamic=False,haematocrit=False,oxygen=False):
        self.logger.log(f"Adding Tree to Visualizer")
        visualizer.add_tree_to_visualizer(self.tree)

        mesh = self.mesh1D
        save_string_mesh = self.filepath+"/1D_mesh"
        self.logger.log(f"Saving 1D mesh to {save_string_mesh}")
        mesh.save(save_string_mesh)

        save_string_dict = self.filepath+"/Node_Point_Reference.json"
        with open(save_string_dict, 'w') as file:
            json.dump(self.getfem_handler_1D.node_point_ref, file, default=self._custom_serializer2)

        
        if haemodynamic:
            self.logger.log(f"Adding Haemodynamics to Visualizer")
            if not self.fedback_haemodynamic:
                raise ValueError(f"Attempted to add pressure and velocity to the vtk without a haemodynamic solution")
            for segment in self.tree.segment_dict.keys():
                visualizer.apply_numbering_to_tree_mesh(segment,self.tree)
                visualizer.apply_pressure_info_to_tree_mesh(self.solution_haemodynamic,segment,self.tree,self.getfem_handler_1D.node_point_ref)
                visualizer.apply_velocity_info_to_tree_mesh(self.solution_haemodynamic,segment,self.tree)

        if haematocrit:
            self.logger.log(f"Adding Haematocrit to Visualizer")
            if not self.fedback_haematocrit:
                raise ValueError(f"Attempted to add haematocrit to the vtk without a haematocrit solution")
            for segment in self.tree.segment_dict.keys():
                visualizer.apply_haematocrit_info_to_tree_mesh(self.solution_haematocrit,segment,self.tree)
                
        if oxygen:
            self.logger.log(f"Adding Oxygen to Visualizer")
            if not self.fedback_oxygen:
                raise ValueError(f"Attempted to add oxygen to the vtk without an oxygen solution")
            for segment in self.tree.segment_dict.keys():
                visualizer.apply_oxygen_info_to_tree_mesh(self.solution_oxygen,segment,self.tree,self.getfem_handler_1D.node_point_ref)
        return
            

    def save_to_json(self,haemodynamic=False,haematocrit=False,oxygen=False,vegf=False):
        if haemodynamic:
            # Specify the file path to save the JSON data
            file_path = self.filepath+"/haemodynamic_data_"+self.run_name+".json"

            # Save the dictionary as JSON
            with open(file_path, "w") as json_file:
                json.dump(
                    {self._custom_serializer2(key): self._custom_serializer(value) for key, value in self.solution_haemodynamic.items()},
                    json_file, cls=NumpyArrayEncoder)

        if haematocrit:
            # Specify the file path to save the JSON data
            file_path = self.filepath+"/haematocrit_data_"+self.run_name+".json"

            # Save the dictionary as JSON
            with open(file_path, "w") as json_file:
                json.dump(
                    {self._custom_serializer2(key): self._custom_serializer(value) for key, value in self.solution_haematocrit.items()},
                    json_file, cls=NumpyArrayEncoder)

        if oxygen:
            # Specify the file path to save the JSON data
            file_path = self.filepath+"/oxygen_data_"+self.run_name+".json"

            # Save the dictionary as JSON
            with open(file_path, "w") as json_file:
                json.dump(
                    {self._custom_serializer2(key): self._custom_serializer(value) for key, value in self.solution_oxygen.items()},
                    json_file, cls=NumpyArrayEncoder)

        if vegf:
            # Specify the file path to save the JSON data
            file_path = self.filepath+"/vegf_data_"+self.run_name+".json"

            # Save the dictionary as JSON
            with open(file_path, "w") as json_file:
                json.dump(
                    {self._custom_serializer2(key): self._custom_serializer(value) for key, value in self.solution_vegf.items()},
                    json_file, cls=NumpyArrayEncoder)
        
        return

    def save_post_process(self):
        self.getfem_handler_1D.save_post_process_vtk(self.filepath,self.run_name,\
            self.wss_element,self.solution_post_process["WSS"],self.re_element,self.solution_post_process["Reynolds"])
        
        return
    
    @staticmethod
    def _custom_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        return obj

    @staticmethod
    def _custom_serializer2(key):
        if isinstance(key, np.int32):
            return int(key.item())  # Convert np.int32 to Python int using item()
        return key

    @staticmethod
    def _custom_deserializer(obj):
        if isinstance(obj, list):
            return np.array(obj)  # Convert list to NumPy array
        if isinstance(obj, dict):
            deserialized_dict = {}
            for key, value in obj.items():
                # Check if the key can be converted to an integer
                try:
                    key = int(key)
                except ValueError:
                    pass
                deserialized_dict[key] = np.array(value) if isinstance(value, list) else value
            return deserialized_dict
        return obj

    def load_from_json(self,haemodynamic=False,haematocrit=False,oxygen=False,vegf=False):
        current_state = self.matrix_type
        self.new_mesh = False
        if haemodynamic:
            self.set_haemodynamic()
            self.mesh1D, self.getfem_handler_1D.node_point_ref = self.getfem_handler_1D.build_1D_mesh(self.tree)
            _, self.pvfem, self.uvfem_dict = self.getfem_handler_1D.build_fluid_elements(self.mesh1D)
            self.getfem_handler_1D.mim = _
            self.getfem_handler_1D.p_element = self.pvfem
            self.getfem_handler_1D.v_element_dict = self.uvfem_dict
            self.getfem_handler_1D.inlet, self.getfem_handler_1D.outlet, self.getfem_handler_1D.junc = \
                self.getfem_handler_1D._mark_junctions_in_mesh_new(self.mesh1D,self.tree,self.getfem_handler_1D.node_point_ref,False)
            self.ptfem,self.utfem ,self.mim, self.mesh3D = self.getfem_handler_3D.make_haemodynamic_elements(mesh=self.mesh_path)

            file_path = self.filepath+"/haemodynamic_data_"+self.run_name+".json"

            with open(file_path, "r") as json_file:
                self.solution_haemodynamic = {
                    key: self._custom_deserializer(value)
                    for key, value in json.load(json_file).items()
                }
            
            self.fedback_haemodynamic = True

        if haematocrit:
            self.set_haematocrit()
            self.hvfem_dict = self.getfem_handler_1D.make_haematocrit_elements(self.tree,self.mesh1D)

            file_path = self.filepath+"/haematocrit_data_"+self.run_name+".json"
            with open(file_path, "r") as json_file:
                self.solution_haematocrit = {
                    key: self._custom_deserializer(value)
                    for key, value in json.load(json_file).items()
                }
            
            self.fedback_haematocrit = True

        if oxygen:
            self.set_oxygen()
            self.ovfem = self.getfem_handler_1D.make_oxygen_elements(self.mesh1D)
            self.otfem = self.getfem_handler_3D.make_oxygen_elements(self.mesh3D)

            file_path = self.filepath+"/oxygen_data_"+self.run_name+".json"
            with open(file_path, "r") as json_file:
                self.solution_oxygen = {
                    key: self._custom_deserializer(value)
                    for key, value in json.load(json_file).items()
                }

            self.fedback_oxygen = True

        if vegf:
            self.set_vegf()
            self.vtfem = self.getfem_handler_3D.make_vegf_elements(self.mesh3D)

            file_path = self.filepath+"/vegf_data_"+self.run_name+".json"
            with open(file_path, "r") as json_file:
                self.solution_vegf = {
                    key: self._custom_deserializer(value)
                    for key, value in json.load(json_file).items()
                }

            self.fedback_vegf = True
            
        self.matrix_type = current_state
        self.set_force_recalculate(True)

    def reset_system(self):
        self.tree._set_node_and_segement_numbers_by_bfs()
        self.tree.reset_junctions()
        self.tree.populate_junctions()
        self.set_force_recalculate(True)

        self.solution_haemodynamic = {}
        self.solution_haematocrit = {}
        self.solution_oxygen = {}
        self.solution_vegf = {}
        self.solution_post_process = {}

        self.fedback_haemodynamic = False
        self.fedback_haematocrit = False
        self.fedback_oxygen = False
        self.fedback_vegf = False
        
        return


    def oxygen_summary(self):
        solubility_O2 =3.1e-5
        half_saturation = 38
        hill_exponent = 3
        kappa2 = (solubility_O2 * half_saturation)**hill_exponent

        inlet_ids = self.tree.get_node_ids_inlet()

        num_of_inlets = 0
        total_inlet_O2 = 0
        for inlet_id in inlet_ids:
            inlet_pid = self.getfem_handler_1D.node_point_ref[inlet_id]
            O2_value = self.solution_oxygen["ovx"][inlet_pid]
            num_of_inlets += 1
            total_inlet_O2 += O2_value

        outlet_ids = self.tree.get_node_ids_outlet()

        num_of_outlets = 0
        total_outlet_O2 = 0
        for outlet_id in outlet_ids:
            outlet_pid = self.getfem_handler_1D.node_point_ref[outlet_id]
            O2_value = self.solution_oxygen["ovx"][outlet_pid]
            num_of_outlets += 1
            total_outlet_O2 += O2_value

        mean_inlet_O2 = total_inlet_O2/num_of_inlets
        mean_outlet_O2 = total_outlet_O2/num_of_outlets

        self.logger.log("#########################################################")
        self.logger.log(f"O2 Partial Pressure (mmHG) in the inlets is {mean_inlet_O2/solubility_O2}")
        self.logger.log(f"O2 Partial Pressure (mmHG) in the outlets is {mean_outlet_O2/solubility_O2}")
        self.logger.log(f"Oxygen Proportion at the Outlets is {mean_outlet_O2/mean_inlet_O2}")
        self.logger.log("#########################################################")


        # for keys in self.function_spaces:
            
        #     H = self.function_spaces[keys]["hx"].x.array[0]

        #     self.logger.log("#########################################################")
        #     self.logger.log("Free oxygen concentration in segment ", keys, "is:")
        #     oxygen = self.function_spaces[keys]["ox"].x.array
        #     self.logger.log(oxygen)

        #     self.logger.log("O2 Partial Pressure (mmHG) in segment ", keys, "is:")
        #     self.logger.log(oxygen/solubility_O2)

        #     self.logger.log("Bound oxygen concentration in segment ", keys, "is:")
        #     bound_oxygen = 0.5 * H * (np.power(oxygen,3)/(np.power(oxygen,3) + kappa2))
        #     self.logger.log(bound_oxygen)

        #     self.logger.log("Total oxygen mass flux in segment ", keys, "is:")
        #     area = self.tree.segment_dict[keys].area()
        #     velocity = self.function_spaces[keys]["ux"].x.array[0]
        #     mass_flux = area * velocity * (oxygen+bound_oxygen)
        #     self.logger.log(mass_flux)
        #     self.logger.log("#########################################################")

        return


    
class MatrixSolver(MatrixHandler):
    """
    A class used to solve the problem: A x = b. 
    To obtain the solution vector: x.

    ----------
    Class Attributes
    ----------
    __TOL: int
        A tolerance value indicating the cessation of an iterative solve

    __MAXITER: int
        A limit on the maximum number of iterations that will be performed when doing an iterative solve

    ----------
    Instance Attributes
    ----------
    _mat_handler: MatrixHandler Object
        A MatrixHandler Object which will be called and manipulated to produce solutions. 

    ----------
    Class Methods
    ----------  
    None
    
    ----------
    Instance Methods
    ----------  
    solve_haemodynamic()
        Returns normalized difference of the solution compared to the previous step 
        for each component of the haemodynamic problem

    solve_haematocrit()
        Returns normalized difference of the solution compared to the previous step 
        for each component of the haematocrit problem

    solve_oxygen()
        Returns normalized difference of the solution compared to the previous step 
        for each component of the oxygen problem

    picard_solve()
        Returns a solution vector x for the problem after a series of picard iterations

    """    

    __TOL = 1.0E-6        # tolerance
    __MAXITER = 100       # max no of iterations allowed

    def __init__(self, logger:Logger, mat_handler:MatrixHandler):
        self.logger = logger
        self._mat_handler = mat_handler
        self.ksp = self._make_solver()

    @staticmethod
    def _make_solver() -> PETSc.KSP:
        # Set up the KSP solver
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        ksp.setType("preonly")
        return ksp

    @staticmethod
    def _Vector_to_Petsc(np_vec:np.ndarray) -> PETSc.Vec:
        len = np_vec.shape[0]
        vec = PETSc.Vec()
        vec.create(comm=PETSc.COMM_WORLD)
        vec.setSizes(len)
        vec.setFromOptions()
        vec.set(0)
        vec.assemble()
        for i in range(len):
            vec.setValue(i, np_vec[i])

        vec.assemble()
        return vec

    def _Spmat_to_Petsc2(self,spmat) -> PETSc.Mat:
        m, n = spmat.size()
        csc_pointer, rows = spmat.csc_ind()
        csc_vals = spmat.csc_val()

        csr_pointer, cols, csr_vals = self.csr_csc_converter(m,n,csc_pointer,rows,csc_vals)
        triplets = self.csc_to_triplets(csc_pointer,rows,csc_vals)
        # print(triplets)

        mat = PETSc.Mat().createDense([m,n],comm=PETSc.COMM_WORLD)
        mat.setPreallocationCSR([csr_pointer, cols, csr_vals])
        # mat.setValues(triplets[0],triplets[1],triplets[2],addv=False)

        # mat = PETSc.Mat().createAIJ([m,n],csr=[csr_pointer, cols, csr_vals],comm=PETSc.COMM_WORLD)

        mat.assemble()

        return mat

    def _Spmat_to_Petsc(self,spmat) -> PETSc.Mat:
        m, n = spmat.size()
        csc_pointer, rows = spmat.csc_ind()
        csc_vals = spmat.csc_val()
        # csr_pointer, cols, csr_vals = self.csr_csc_converter(m,n,csc_pointer,rows,csc_vals)

        mat = PETSc.Mat().createAIJ([m,n],csr=[csc_pointer, rows, csc_vals],comm=PETSc.COMM_WORLD)

        mat.assemble()
        mat.transpose()


        return mat

    @staticmethod
    def csr_to_triplets(Ap, Aj, Ax):
        triplets = []

        for row in range(len(Ap) - 1):
            for jj in range(Ap[row], Ap[row + 1]):
                col = Aj[jj]
                value = Ax[jj]
                triplets.append([row, col, value])

        return triplets

    @staticmethod
    def csc_to_triplets(Bp, Bi, Bx):
        triplets = [[],[],[]]

        for col in range(len(Bp) - 1):
            for ii in range(Bp[col], Bp[col + 1]):
                row = Bi[ii]
                value = Bx[ii]
                triplets[0].append(row)
                triplets[1].append(col)
                triplets[2].append(value)

        return triplets


    @staticmethod
    def csr_csc_converter(n_row:int, n_col:int, Ap:np.ndarray, Aj:np.ndarray, Ax:np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        # This function converts from the csr format to the csc format. This is the equivalent operation the csc->csr.
        nnz = Ap[n_row]

        # Compute number of non-zero entries per column of A
        Bp = [0] * n_col

        for n in range(nnz):
            Bp[Aj[n]] += 1

        # Cumsum the nnz per column to get Bp[]
        cumsum = 0
        for col in range(n_col):
            temp = Bp[col]
            Bp[col] = cumsum
            cumsum += temp
        Bp.append(nnz)

        Bi = [0] * nnz
        Bx = [0] * nnz

        for row in range(n_row):
            for jj in range(Ap[row], Ap[row + 1]):
                col = Aj[jj]
                dest = Bp[col]

                Bi[dest] = row
                Bx[dest] = Ax[jj]

                Bp[col] += 1

        # Reset Bp to store the starting index of each column in the CSC format
        last = 0
        for col in range(n_col + 1):
            temp = Bp[col]
            Bp[col] = last
            last = temp

        return Bp, Bi, Bx

    @staticmethod
    def _Petsc_to_Numpy(Vec:PETSc.Vec) -> np.ndarray:
        np_vector = Vec.getArray()
        return np_vector

    def _getAbx(self) -> Tuple[PETSc.Mat,PETSc.Vec,PETSc.Vec]:
        A = self._Spmat_to_Petsc(self._mat_handler.active_array())
        b = self._Vector_to_Petsc(self._mat_handler.active_vec())
        x = PETSc.Vec()
        x.create(comm=PETSc.COMM_WORLD)
        x.setSizes(b.getSize())
        x.setFromOptions()
        x.set(0)
        x.assemble()

        return A,b,x

    def solve_haemodynamic(self, relaxation=1.0, verbose=False) -> Tuple[float,float,float,float]:
        # Set the system to haemodynamic and get the A,b,x matricies
        self._mat_handler.set_haemodynamic()
        build_time1 = time.time()
        self._mat_handler.build()
        build_time2 = build_time1-time.time()
        if verbose:
            self.logger.log(f"Time to build Matrix: {-build_time2}")

        matrix_time1 = time.time()
        A,b,x = self._getAbx()
        matrix_time2 = matrix_time1-time.time()
        if verbose:
            self.logger.log(f"Time to convert to PETSc: {-matrix_time2}")

        # Set up the KSP solver
        self.ksp.setOperators(A)  # Set the matrix for the solver
        # Create the preconditioner object
        pc = self.ksp.getPC()

        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

        solve_time1 = time.time()
        self.ksp.solve(b,x)
        solve_time2 = solve_time1-time.time()
        if verbose:
            self.logger.log(f"Time to solve system: {-solve_time2}")

        solution_vector = self._Petsc_to_Numpy(x)
        
        step_diff_utx,step_diff_ptx,step_diff_uvx,step_diff_pvx = self._mat_handler.feedback_solution(solution_vector,relaxation)  # type: ignore

        if False:
            self.logger.log("P_SOLUTION")
            np.set_printoptions(precision=0, suppress=True)
            self.logger.log(self._mat_handler.solution_haemodynamic["pvx"])
            np.set_printoptions(precision=8, suppress=False)

        # Reset the KSP object before reusing it
        self.ksp.reset()
        A.destroy()
        b.destroy()
        x.destroy()
    
        return step_diff_utx,step_diff_ptx,step_diff_uvx,step_diff_pvx

    def solve_haemodynamic_1D(self, relaxation=1.0, verbose=False) -> Tuple[float,float]:
        # Set the system to haemodynamic and get the A,b,x matricies
        self._mat_handler.set_haemodynamic()
        build_time1 = time.time()
        self._mat_handler.build_1D(verbose)
        build_time2 = build_time1-time.time()
        if verbose:
            self.logger.log(f"Time to build Matrix: {-build_time2}")

        matrix_time1 = time.time()
        A,b,x = self._getAbx()
        matrix_time2 = matrix_time1-time.time()
        if verbose:
            self.logger.log(f"Time to convert to PETSc: {-matrix_time2}")

        # Set up the KSP solver
        self.ksp.setOperators(A)  # Set the matrix for the solver
        # Create the preconditioner object
        pc = self.ksp.getPC()

        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

        solve_time1 = time.time()
        self.ksp.solve(b,x)
        solve_time2 = solve_time1-time.time()
        if verbose:
            self.logger.log(f"Time to solve system: {-solve_time2}")

        solution_vector = self._Petsc_to_Numpy(x)
        
        step_diff_uvx,step_diff_pvx = self._mat_handler.feedback_solution_1D(solution_vector,relaxation) # type:ignore

        if False:
            self.logger.log("P_SOLUTION")
            np.set_printoptions(precision=0, suppress=True)
            self.logger.log(self._mat_handler.solution_haemodynamic["pvx"])
            np.set_printoptions(precision=8, suppress=False)

        # Reset the KSP object before reusing it
        self.ksp.reset()
        A.destroy()
        b.destroy()
        x.destroy()
    
        return step_diff_uvx,step_diff_pvx # type:ignore
    
    def solve_haematocrit(self, relaxation:float=1., verbose=False) -> Tuple[float,bool]:
        self._mat_handler.set_haematocrit()

        solution_vector = self._mat_handler.build()
        step_diff_hx = self._mat_handler.feedback_solution(solution_vector,relaxation)

        if verbose:
            self.logger.log("H SOLUTION")
            self.logger.log(solution_vector)

        # ksp.destroy()
        # A.destroy()
        # b.destroy()
        # x.destroy()
        
        return step_diff_hx, self._mat_handler.capped_haematocrit # type: ignore

    def solve_oxygen(self, relaxation=1.0, verbose=False, visualize=False) -> Tuple[float,float]:
        self._mat_handler.set_oxygen()
        
        build_time1 = time.time()
        self._mat_handler.build()
        build_time2 = build_time1-time.time()
        if verbose:
            self.logger.log(f"Time to build Matrix: {-build_time2}")

        matrix_time1 = time.time()
        A,b,x = self._getAbx()
        matrix_time2 = matrix_time1-time.time()
        if verbose:
            self.logger.log(f"Time to convert to PETSc: {-matrix_time2}")

        if visualize:
            matrix = self._mat_handler.active_array().full() # type: ignore
            plt.figure(figsize=(8, 6))
            plt.imshow(matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar()  # Add a colorbar to the side
            plt.title('Matrix Visualization')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            plt.show()

        # Set up the KSP solver
        self.ksp.setOperators(A)
        # Create the preconditioner object
        pc = self.ksp.getPC()

        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

        solve_time1 = time.time()
        self.ksp.solve(b,x)
        solve_time2 = solve_time1-time.time()
        if verbose:
            self.logger.log(f"Time to solve system: {-solve_time2}")


        solution_vector = self._Petsc_to_Numpy(x)
        step_diff_otx,step_diff_ovx = self._mat_handler.feedback_solution(solution_vector,relaxation) # type: ignore

        # Reset the KSP object before reusing it
        self.ksp.reset()
        A.destroy()
        b.destroy()
        x.destroy()
        
        
        return step_diff_otx,step_diff_ovx

    def solve_vegf(self, relaxation=1.0, verbose=False, visualize=False) -> int:
        self._mat_handler.set_vegf()
        
        build_time1 = time.time()
        self._mat_handler.build()
        build_time2 = build_time1-time.time()
        if verbose:
            self.logger.log(f"Time to build Matrix: {-build_time2}")


        matrix_time1 = time.time()
        A,b,x = self._getAbx()
        matrix_time2 = matrix_time1-time.time()
        if verbose:
            self.logger.log(f"Time to convert to PETSc: {-matrix_time2}")

        if visualize:
            matrix = self._mat_handler.active_array().full() # type: ignore
            plt.figure(figsize=(8, 6))
            plt.imshow(matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar()  # Add a colorbar to the side
            plt.title('Matrix Visualization')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            plt.show()

        # Set up the KSP solver
        self.ksp.setOperators(A)
        # Create the preconditioner object
        pc = self.ksp.getPC()

        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

        solve_time1 = time.time()
        self.ksp.solve(b,x)
        solve_time2 = solve_time1-time.time()
        if verbose:
            self.logger.log(f"Time to solve system: {-solve_time2}")

        solution_vector = self._Petsc_to_Numpy(x)
        step_diff_vtx = self._mat_handler.feedback_solution(solution_vector,relaxation)

        A.destroy()
        b.destroy()
        x.destroy()
        # Reset the KSP object before reusing it
        self.ksp.reset()
        
        return step_diff_vtx # type: ignore

    def iterative_solve_fluid(self, tolerance=1e-6, tolerance_h=1e-6, max_iterations = 100, alpha=1.0, beta=1.0, momentum=0.95, save=False, save_interval=10, save_specific:List[int]=[], momentum_period=10, verbose=True, very_verbose=False) -> Tuple[float,List[float]]:
        iteration = 0
        times_alpha_reduced = 0
        times_beta_reduced = 0
        convergence_rates_alpha = []
        convergence_rates_beta = []
        flag_history = []
        start_time = time.time()
        iteration_limit = max_iterations*2

        if verbose or very_verbose:
            self.logger.log("###############################################################")
            self.logger.log("Beginning Picard for Haemodynamics and Haematocrit:")
            self.logger.log(f"  Maximum Iterations = {max_iterations}")
            self.logger.log(f"  Haemodynamic Tolerance = {tolerance}")
            self.logger.log(f"  Haemodynamic Relaxation = {alpha}")
            self.logger.log(f"  Haematocrit Tolerance = {tolerance_h}")
            self.logger.log(f"  Haematocrit Relaxation = {beta}")
            self.logger.log("###############################################################")

        
        while iteration < max_iterations:
            if not very_verbose:
                self.logger.log(f"Current Iteration: {iteration}", True)
            # Perform the haemodynamic solve and get the normalized difference
            norm_diff_utx, norm_diff_ptx, norm_diff_uvx, norm_diff_pvx = self.solve_haemodynamic(alpha)
            norm_diff_hx, flag_cap = self.solve_haematocrit(beta)
            flag_history.append(flag_cap)

            # Compute the sum of the normalized differences
            sum_norm_diff = norm_diff_utx + norm_diff_ptx + norm_diff_uvx + norm_diff_pvx

            # Compute the convergence rate
            convergence_rate_alpha = sum_norm_diff / previous_update_norm if iteration > 0 else np.inf # type: ignore
            convergence_rate_beta = norm_diff_hx / previous_update_hx if iteration > 0 else np.inf # type: ignore

            # Store the summary statistics for each iteration
            if iteration > 0:
                convergence_rates_alpha.append(convergence_rate_alpha)
                convergence_rates_beta.append(convergence_rate_beta)

            # Print statistics for the current iteration
            if very_verbose:
                self.logger.log("###############################################################")
                self.logger.log(f"Iteration {iteration + 1}:")
                self.logger.log(f"  norm_diff_utx = {norm_diff_utx:.6g}")
                self.logger.log(f"  norm_diff_ptx = {norm_diff_ptx:.6g}")
                self.logger.log(f"  norm_diff_uvx = {norm_diff_uvx:.6g}")
                self.logger.log(f"  norm_diff_pvx = {norm_diff_pvx:.6g}")
                self.logger.log(f"  Norm of Haemodynamic Update = {sum_norm_diff:.6g}")
                self.logger.log(f"  Convergence Rate Haemodynamic = {convergence_rate_alpha:.6g}")
                self.logger.log(f"  Norm of Haematocrit Update = {norm_diff_hx:.6g}")
                self.logger.log(f"  Convergence Rate Haematocrit = {convergence_rate_beta:.6g}")
                self.logger.log("###############################################################")

            # Check if the end condition is satisfied
            # if sum_norm_diff <= tolerance and norm_diff_hx <= tolerance_h:
            if sum_norm_diff <= tolerance:
                break

            # Check if the current iteration is in the save_specific list
            if iteration in save_specific:
                self._mat_handler.save_to_vtk(iteration, True,True,False,False)

            # Check if the current iteration is a multiple of save_interval
            if save:
                if save_interval > 0 and iteration % save_interval == 0:
                    self._mat_handler.save_to_vtk(iteration, True,True,False,False)
            
            # Check if system is converging and modulate convergence rate if not
            if iteration % 10 == 0 and iteration != 0:
                trigger = 0
                if any(value > 1 for value in convergence_rates_alpha[-10:]) and not any(flag_history[-10:]):
                    alpha *= momentum
                    tolerance *= momentum
                    trigger = 1
                    times_alpha_reduced += 1
                if any(value > 1 for value in convergence_rates_beta[-10:]) and not any(flag_history[-10:]):
                    beta *= momentum
                    tolerance_h *= momentum
                    trigger = 1
                    times_beta_reduced += 1
                if trigger == 0:
                    if sum(flag_history[-momentum_period:]) > (momentum_period/2):
                        alpha *= momentum
                        beta *= momentum
                        tolerance *= momentum
                        trigger = 1
                        times_alpha_reduced += 1
                        times_beta_reduced += 1
                if trigger == 1:
                    max_iterations /= momentum

                if verbose or very_verbose:
                    self.logger.log("###############################################################")
                    self.logger.log("Evaluating Momentum:")
                    self.logger.log(f"  Haemodynamic Convergence: {[f'{x:.3g}' for x in convergence_rates_alpha[-momentum_period:]]}")
                    self.logger.log(f"  Haematocrit Convergence: {[f'{x:.3g}' for x in convergence_rates_beta[-momentum_period:]]}")
                    self.logger.log(f"  Haematocrit Capping: {flag_history[-momentum_period:]:}")
                    self.logger.log(f"  Should momentum be reduced: {trigger==1}")
                    self.logger.log(f"  Tolerance to reach: {tolerance:.3g}")
                    self.logger.log(f"  Current Haemodynamic Error: {sum_norm_diff:.3g}")
                    self.logger.log(f"  Current Haematocrit Error: {norm_diff_hx:.3g}")
                    self.logger.log("###############################################################")

            # Update the iteration count and previous values
            iteration += 1
            previous_update_norm = sum_norm_diff
            previous_update_hx = norm_diff_hx

            if iteration > iteration_limit:
                self.logger.log(f"Iteration Limit Reached")
                break

        # Compute the summary statistics for the entire iterative solve
        average_convergence_rate_alpha = np.mean(convergence_rates_alpha)
        average_convergence_rate_beta = np.mean(convergence_rates_beta)
        total_time = time.time() - start_time

        # Print the summary statistics
        if verbose or very_verbose:
            self.logger.log("###############################################################")
            self.logger.log("Summary:")
            self.logger.log(f"  Iterations: {iteration + 1}")
            self.logger.log(f"  Norm of Haemodynamic Update = {sum_norm_diff:.6g}") # type: ignore
            self.logger.log(f"  Norm of Haematocrit Update = {norm_diff_hx:.6g}") # type: ignore
            self.logger.log(f"  Average Convergence Rate Haemodynamic: {average_convergence_rate_alpha:.6g}")
            self.logger.log(f"  Average Convergence Rate Haematocrit: {average_convergence_rate_beta:.6g}")
            self.logger.log(f"  Total Computational Time: {total_time:.6g} seconds")
            self.logger.log(f"  Haemodynamic Momentum reduced: {times_alpha_reduced} times")
            self.logger.log(f"  Haematocrit Momentum reduced: {times_beta_reduced} times")
            self.logger.log("###############################################################")

        return sum_norm_diff, convergence_rates_alpha # type: ignore

    def iterative_solve_fluid_1D(self, tolerance=1e-6, tolerance_h=1e-6, max_iterations = 100, alpha=1.0, beta=1.0, momentum=0.95, save=False, save_interval=10, save_specific:List[int]=[], momentum_period=10, verbose=True, very_verbose=False) -> Tuple[float,List[float]]:
        iteration = 0
        times_alpha_reduced = 0
        times_beta_reduced = 0
        convergence_rates_alpha = []
        convergence_rates_beta = []
        flag_history = []
        start_time = time.time()
        iteration_limit = max_iterations*2

        if verbose or verbose:
            self.logger.log("###############################################################")
            self.logger.log("Beginning Picard for Haemodynamics and Haematocrit:")
            self.logger.log(f"  Maximum Iterations = {max_iterations}")
            self.logger.log(f"  Haemodynamic Tolerance = {tolerance}")
            self.logger.log(f"  Haemodynamic Relaxation = {alpha}")
            self.logger.log(f"  Haematocrit Tolerance = {tolerance_h}")
            self.logger.log(f"  Haematocrit Relaxation = {beta}")
            self.logger.log("###############################################################")

        
        while iteration < max_iterations:
            if not very_verbose:
                self.logger.log(f"Current Iteration: {iteration}", True)
            # Perform the haemodynamic solve and get the normalized difference
            norm_diff_uvx, norm_diff_pvx = self.solve_haemodynamic_1D(alpha)
            
            norm_diff_hx, flag_cap = self.solve_haematocrit(beta)
            flag_history.append(flag_cap)

            # Compute the sum of the normalized differences
            sum_norm_diff =  norm_diff_uvx + norm_diff_pvx

            # Compute the convergence rate
            convergence_rate_alpha = sum_norm_diff / previous_update_norm if iteration > 0 else np.inf  # type: ignore
            convergence_rate_beta = norm_diff_hx / previous_update_hx if iteration > 0 else np.inf # type: ignore

            # Store the summary statistics for each iteration
            if iteration > 0:
                convergence_rates_alpha.append(convergence_rate_alpha)
                convergence_rates_beta.append(convergence_rate_beta)

            # Print statistics for the current iteration
            if very_verbose:
                self.logger.log("###############################################################")
                self.logger.log(f"Iteration {iteration + 1}:")
                self.logger.log(f"  norm_diff_uvx = {norm_diff_uvx:.6g}")
                self.logger.log(f"  norm_diff_pvx = {norm_diff_pvx:.6g}")
                self.logger.log(f"  Norm of Haemodynamic Update = {sum_norm_diff:.6g}")
                self.logger.log(f"  Convergence Rate Haemodynamic = {convergence_rate_alpha:.6g}")
                self.logger.log(f"  Norm of Haematocrit Update = {norm_diff_hx:.6g}")
                self.logger.log(f"  Convergence Rate Haematocrit = {convergence_rate_beta:.6g}")
                self.logger.log("###############################################################")

            # Check if the end condition is satisfied
            # if sum_norm_diff <= tolerance and norm_diff_hx <= tolerance_h:
            if sum_norm_diff <= tolerance:
                break

            # Check if the current iteration is in the save_specific list
            if iteration in save_specific:
                self._mat_handler.save_to_vtk(iteration, True,False,False,False)

            # Check if the current iteration is a multiple of save_interval
            if save:
                if save_interval > 0 and iteration % save_interval == 0:
                    self._mat_handler.save_to_vtk(iteration, True,False,False,False)

            # Check if system is converging and modulate convergence rate if not
            if iteration % momentum_period == 0 and iteration != 0:
                trigger = 0
                if any(value > 1 for value in convergence_rates_alpha[-momentum_period:]) and not any(flag_history[-momentum_period:]):
                    alpha *= momentum
                    tolerance *= momentum
                    trigger = 1
                    times_alpha_reduced += 1
                if any(value > 1 for value in convergence_rates_beta[-momentum_period:]) and not any(flag_history[-momentum_period:]):
                    beta *= momentum
                    tolerance_h *= momentum
                    trigger = 1
                    times_beta_reduced += 1
                if trigger == 0:
                    if sum(flag_history[-momentum_period:]) > (momentum_period/2):
                        alpha *= momentum
                        beta *= momentum
                        tolerance *= momentum
                        trigger = 1
                        times_alpha_reduced += 1
                        times_beta_reduced += 1
                if trigger == 1:
                    max_iterations /= momentum

                if verbose or very_verbose:
                    self.logger.log("###############################################################")
                    self.logger.log("Evaluating Momentum:")
                    self.logger.log(f"  Haemodynamic Convergence: {[f'{x:.3g}' for x in convergence_rates_alpha[-momentum_period:]]}")
                    self.logger.log(f"  Haemodynamic Convergence: {[f'{x:.3g}' for x in convergence_rates_beta[-momentum_period:]]}")
                    self.logger.log(f"  Haematocrit Capping: {flag_history[-momentum_period:]:}")
                    self.logger.log(f"  Should momentum be reduced: {trigger==1}")
                    self.logger.log(f"  Tolerance to reach: {tolerance:.3g}")
                    self.logger.log(f"  Current Haemodynamic Error: {sum_norm_diff:.3g}")
                    self.logger.log(f"  Current Haemodynamic Error: {norm_diff_hx:.3g}")
                    self.logger.log("###############################################################")


            # Update the iteration count and previous values
            iteration += 1
            previous_update_norm = sum_norm_diff
            previous_update_hx = norm_diff_hx
            
            if iteration > iteration_limit:
                self.logger.log(f"Iteration Limit Reached")
                break


        # Compute the summary statistics for the entire iterative solve
        average_convergence_rate_alpha = np.mean(convergence_rates_alpha)
        average_convergence_rate_beta = np.mean(convergence_rates_beta)
        total_time = time.time() - start_time

        # Print the summary statistics
        if verbose or very_verbose:
            self.logger.log("###############################################################")
            self.logger.log("Summary:")
            self.logger.log(f"  Iterations: {iteration + 1}")
            self.logger.log(f"  norm_diff_uvx = {norm_diff_uvx:.6g}") # type: ignore
            self.logger.log(f"  norm_diff_pvx = {norm_diff_pvx:.6g}") # type: ignore
            self.logger.log(f"  Norm of Haemodynamic Update = {sum_norm_diff:.6g}") # type: ignore
            self.logger.log(f"  Average Convergence Rate Haemodynamic: {average_convergence_rate_alpha:.6g}")
            self.logger.log(f"  Average Convergence Rate Haematocrit: {average_convergence_rate_beta:.6g}")
            self.logger.log(f"  Total Computational Time: {total_time:.6g} seconds")
            self.logger.log(f"  Haemodynamic Momentum reduced: {times_alpha_reduced} times")
            self.logger.log(f"  Haematocrit Momentum reduced: {times_beta_reduced} times")
            self.logger.log("###############################################################")

        return sum_norm_diff, convergence_rates_alpha # type: ignore

    def iterative_solve_oxygen(self, tolerance=1e-6, max_iterations = 100, alpha=1.0, momentum=0.95, save=False, save_interval=10, save_specific:List[int]=[], momentum_period:int=10, verbose:bool=True, very_verbose:bool=False) -> Tuple[float,List[float]]:
        iteration = 0
        times_alpha_reduced = 0
        sum_norm_diff = []
        convergence_rates = []
        start_time = time.time()

        if verbose or very_verbose:
            self.logger.log("###############################################################")
            self.logger.log("Beginning Picard for Oxygen:")
            self.logger.log(f"  Maximum Iterations = {max_iterations}")
            self.logger.log(f"  Oxygen Tolerance = {tolerance}")
            self.logger.log(f"  Oxygen Relaxation Rate = {alpha}")
            self.logger.log("###############################################################")

        while iteration < max_iterations:
            if not very_verbose:
                self.logger.log(f"Current Iteration: {iteration}", True)
            # Perform the haemodynamic solve and get the normalized difference
            norm_diff_otx, norm_diff_ovx = self.solve_oxygen(alpha)
            # Compute the sum of the normalized differences
            sum_norm_diff.append(norm_diff_otx + norm_diff_ovx)

             # Set the Decaying Diffusivitiy term using the 1D error term
            self._mat_handler.getfem_handler_1D.set_decaying_diffusivity(min(sum_norm_diff),iteration)

            # Compute the convergence rate
            convergence_rate = sum_norm_diff[-1] / previous_update_norm if iteration > 0 else np.inf # type: ignore

            # Store the summary statistics for each iteration
            if iteration > 0:
                convergence_rates.append(convergence_rate)

            # Print statistics for the current iteration
            if very_verbose:
                self.logger.log("###############################################################")
                self.logger.log(f"Iteration {iteration + 1}:")
                self.logger.log(f"  norm_diff_otx = {norm_diff_otx:.6g}")
                self.logger.log(f"  norm_diff_ovx = {norm_diff_ovx:.6g}")
                self.logger.log(f"  Norm of Oxygen Update = {sum_norm_diff[-1]:.6g}")
                self.logger.log(f"  Convergence Rate Oxygen = {convergence_rate:.6g}")
                self.logger.log("###############################################################")

            # Check if the end condition is satisfied
            if sum_norm_diff[-1] <= tolerance:
                break

            # Check if the current iteration is in the save_specific list
            if iteration in save_specific:
                self._mat_handler.save_to_vtk(iteration, False,False,True,False)

            # Check if the current iteration is a multiple of save_interval
            if save:
                if save_interval > 0 and iteration % save_interval == 0:
                    self._mat_handler.save_to_vtk(iteration, False,False,True,False)

            # Check if system is converging and modulate convergence rate if not
            if iteration % momentum_period == 0 and iteration != 0:
                trigger = 0
                if any(value > 1 for value in convergence_rates[-momentum_period:]):
                    alpha *= momentum
                    tolerance *= momentum
                    max_iterations /= momentum
                    times_alpha_reduced += 1
                    trigger = 1

                if verbose or very_verbose:
                    self.logger.log("###############################################################")
                    self.logger.log("Evaluating Momentum:")
                    self.logger.log(f"  Oxygen Convergence: {[f'{x:.3g}' for x in convergence_rates[-momentum_period:]]}")
                    self.logger.log(f"  Should momentum be reduced: {trigger==1}")
                    self.logger.log(f"  Tolerance to reach: {tolerance:.3g}")
                    self.logger.log(f"  Current Oxygen Error: {sum_norm_diff[-1]:.3g}")
                    self.logger.log("###############################################################")


            # Update the iteration count and previous values
            iteration += 1
            previous_update_norm = sum_norm_diff[-1]

            


        if iteration == max_iterations:
            self._mat_handler.run_name += "_FAILED_OXYGEN"

        # Compute the summary statistics for the entire iterative solve
        average_convergence_rate = np.mean(convergence_rates)
        total_time = time.time() - start_time

        # Print the summary statistics
        if verbose or very_verbose:         
            self.logger.log("###############################################################")
            self.logger.log("Summary:")
            self.logger.log(f"  Iterations: {iteration + 1}")
            self.logger.log(f"  norm_diff_otx = {norm_diff_otx:.6g}") # type: ignore
            self.logger.log(f"  norm_diff_ovx = {norm_diff_ovx:.6g}") # type: ignore
            self.logger.log(f"  Average Convergence Rate Oxygen: {average_convergence_rate:.6g}")
            self.logger.log(f"  Total Computational Time: {total_time:.6g} seconds")
            self.logger.log(f"  Momentum reduced: {times_alpha_reduced} times")
            self.logger.log("###############################################################")

        return sum_norm_diff, convergence_rates # type: ignore


    def iterative_solve_vegf(self, tolerance=1e-6, max_iterations = 100, alpha=1, save=False, save_interval=10, save_specific:List[int]=[], verbose=True)-> Tuple[float,List[float]]:
        iteration = 0
        sum_norm_diff = []
        convergence_rates = []
        start_time = time.time()
        
        if verbose:
            self.logger.log("###############################################################")
            self.logger.log("Beginning Picard for VEGF:")
            self.logger.log(f"  Maximum Iterations = {max_iterations}")
            self.logger.log(f"  VEGF Tolerance = {tolerance}")
            self.logger.log(f"  VEGF Relaxation Rate = {alpha}")
            self.logger.log("###############################################################")

        while iteration < max_iterations:
            # Perform the haemodynamic solve and get the normalized difference
            norm_diff_vtx = self.solve_vegf(alpha)

            # Compute the sum of the normalized differences
            sum_norm_diff.append(norm_diff_vtx)

            # Compute the convergence rate
            convergence_rate = sum_norm_diff[-1] / previous_update_norm if iteration > 0 else np.inf # type: ignore

            # Store the summary statistics for each iteration
            if iteration > 0:
                convergence_rates.append(convergence_rate)

            # Print statistics for the current iteration
            if verbose:
                self.logger.log("###############################################################")
                self.logger.log(f"Iteration {iteration + 1}:")
                self.logger.log(f"  norm_diff_vtx = {norm_diff_vtx:.6g}")
                self.logger.log(f"  Norm of VEGF Update = {sum_norm_diff[-1]:.6g}")
                self.logger.log(f"  Convergence Rate VEGF = {convergence_rate:.6g}")
                self.logger.log("###############################################################")

            # Check if the end condition is satisfied
            if sum_norm_diff[-1] <= tolerance:
                break

            # Check if the current iteration is in the save_specific list
            if iteration in save_specific:
                self._mat_handler.save_to_vtk(iteration, False,False,False,True)

            # Check if the current iteration is a multiple of save_interval
            if save:
                if save_interval > 0 and iteration % save_interval == 0:
                    self._mat_handler.save_to_vtk(iteration, False,False,False,True)

            # Update the iteration count and previous values
            iteration += 1
            previous_update_norm = sum_norm_diff[-1]

        if iteration == max_iterations:
            self._mat_handler.run_name += "_FAILED"

        # Compute the summary statistics for the entire iterative solve
        average_convergence_rate = np.mean(convergence_rates)
        total_time = time.time() - start_time

        # Print the summary statistics
        if verbose:
            self.logger.log("###############################################################")
            self.logger.log("Summary:")
            self.logger.log(f"  Iterations: {iteration + 1}")
            self.logger.log(f"  Average Convergence Rate VEGF: {average_convergence_rate:.6g}")
            self.logger.log(f"  Total Computational Time: {total_time:.6g} seconds")
            self.logger.log("###############################################################")

        return sum_norm_diff, convergence_rates # type: ignore

    def solve_post_processing(self, Reynolds=True, WSS=True):
        if Reynolds:
            A, b = self._mat_handler.build_post_process_Reynolds()

            A = self._Spmat_to_Petsc(A)
            b = self._Vector_to_Petsc(b)
            x = PETSc.Vec()
            x.create(comm=PETSc.COMM_WORLD)
            x.setSizes(b.getSize())
            x.setFromOptions()
            x.set(0)
            x.assemble()

            # Set up the KSP solver
            self.ksp.setOperators(A)  # Set the matrix for the solver
            
            # Create the preconditioner object
            pc = self.ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")

            self.ksp.solve(b,x)

            solution_vector_re = self._Petsc_to_Numpy(x)
            self._mat_handler.feedback_post_process(solution_vector_re,"Reynolds")

            A.destroy()
            b.destroy()
            x.destroy()
            # Reset the KSP object before reusing it
            self.ksp.reset()
            

        if WSS:
            A, b = self._mat_handler.build_post_process_wss()

            A = self._Spmat_to_Petsc(A)
            b = self._Vector_to_Petsc(b)
            x = PETSc.Vec()
            x.create(comm=PETSc.COMM_WORLD)
            x.setSizes(b.getSize())
            x.setFromOptions()
            x.set(0)
            x.assemble()

            # Set up the KSP solver
            self.ksp.setOperators(A)  # Set the matrix for the solver
            # Create the preconditioner object
            pc = self.ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")

            self.ksp.solve(b,x)

            solution_vector_wss = self._Petsc_to_Numpy(x)
            self._mat_handler.feedback_post_process(solution_vector_wss,"WSS")

            A.destroy()
            b.destroy()
            x.destroy()
            # Reset the KSP object before reusing it
            self.ksp.reset()

        return

    def solve_attractor_field(self,sprout_dict:dict):
        start_time = time.time()
        A_base,b_base,RHS_TREE_AUG,RHS_SPROUT_AUG = self._mat_handler.build_vessel_attractor_field(sprout_dict)

        A = self._Spmat_to_Petsc(A_base)
        b = self._Vector_to_Petsc(b_base-RHS_TREE_AUG)
        x = PETSc.Vec()
        x.create(comm=PETSc.COMM_WORLD)
        x.setSizes(b.getSize())
        x.setFromOptions()
        x.set(0)
        x.assemble()

        # Set up the KSP solver
        self.ksp.setOperators(A)  # Set the matrix for the solver
        # Create the preconditioner object
        pc = self.ksp.getPC()

        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

        self.ksp.solve(b,x)

        solution_vector_tree = self._Petsc_to_Numpy(x)

        A.destroy()
        b.destroy()
        x.destroy()

        solution_vectors_sprout = {}
        for key in RHS_SPROUT_AUG.keys():
            b = self._Vector_to_Petsc(b_base-RHS_SPROUT_AUG[key])
            x = PETSc.Vec()
            x.create(comm=PETSc.COMM_WORLD)
            x.setSizes(b.getSize())
            x.setFromOptions()
            x.set(0)
            x.assemble()

            self.ksp.solve(b,x)
            solution_vectors_sprout.update({key:self._Petsc_to_Numpy(x)})
            b.destroy()
            x.destroy()
        
        # Reset the KSP object before reusing it
        self.ksp.reset()
        total_time = time.time() - start_time

        # Print the summary statistics
        self.logger.log("###############################################################")
        self.logger.log("Summary For Attractor Field:")
        self.logger.log(f"  Vessel Attraction [ Min: {np.min(solution_vector_tree):.6g}, Max: {np.max(solution_vector_tree):.6g}  ]")
        self.logger.log(f"  Total Number of Tips: {len(RHS_SPROUT_AUG.keys())}")
        for key in solution_vectors_sprout.keys():
            self.logger.log(f"  Tip {key} Attraction [ Min: {np.min(solution_vectors_sprout[key]):.6g}, Max: {np.max(solution_vectors_sprout[key]):.6g}  ]")
        self.logger.log(f"  Total Computational Time: {total_time:.6g} seconds")
        self.logger.log("###############################################################")

        return solution_vector_tree, solution_vectors_sprout


    def finalize(self):
        PETSc._finalize()

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj): # type: ignore
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to a Python list
        return json.JSONEncoder.default(self, obj)







    
