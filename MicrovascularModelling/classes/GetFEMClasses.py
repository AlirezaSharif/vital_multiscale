from cmath import nan
from classes.ConfigClass import Config, Logger
from classes.GeometryClasses import Tree, Tissue, GraphRepresentation, Sprout
import numpy as np # type: ignore
import time
from scipy.special import logit, expit # type: ignore
from petsc4py import PETSc # type: ignore
import getfem as gf # type: ignore
import pyvista as pyvista # type: ignore
from typing import List, Dict, Tuple, Union

class GetFEMHandler1D(Tree):
    """
    A class used to handle the Fenics implementation for the generation of segment specific submatricies.

    ----------
    Class Attributes
    ----------
    None

    ----------
    Instance Attributes
    ----------
    pressure_inlet : int
        An integer stating the pressure boundary condition at the inlet

    pressure_outlet : int
        An integer stating the pressure boundary condition at the inlet

    haematocrit_inlet: int
        An integer stating the Haematocrit boundary condition at the inlet

    __OXYGEN_BOUNDARIES: ---
        --- stating the Oxygen boundary condition

    num_cells : int
        An integer defining the number of elements used to generate the submatrix for each segment    

    ----------
    Class Methods
    ----------  
    load_config(filename)
        Returns a class instance from file

    ----------
    Instance Methods
    ----------  
    build_1D_mesh(tree)
        Returns a 1D GetFEM Mesh object where each segment is represented by a region with ID matching the segment
        
    build_fluid_elements(self,mesh):
        Returns the mesh, integration measure and the pressure and velocity elements

    build_monolithic_vessel(my_tree, h_solution=None, output='Spmat', verbose=False)
        Returns the Matrix, forcing vector, finite elements, and mesh for the 1D haemodynamic problem (Pressure and Velocity)

    build_monolithic_haematocrit(my_tree,mesh1D,u_solution,h_solution=None, output="Spmat",verbose=False)
        Returns the Matrix, forcing vector, and finite elements for the 1D haematocrit problem (Haematocrit)
        
    build_monolithic_oxygen(my_tree,mesh1D,uv_solution,h_solution,o_solution=Nonde, output="Spmat,verbose=False)
        Returns the Matrix, forcing vector, and finite elements for the 1D oxygen problem (Oxygen)

    GetFEM_to_PETSc(SpMat)
        Returns the PETSc matrix equivalent to the input GetFEM format

    haemodynamic_submatrix(segment, node_list[segment.node_1_id], node_list[segment.node_2_id], h_solution_partial)
        Returns a segment specific submatrix and subvector for the haemodynamics (Pressure and Velocity)

    haematocrit_submatrix(segment, node_list[segment.node_1_id], node_list[segment.node_2_id], u_solution_partial)
        Returns a segment specific submatrix and subvector for the Haematocrit dynamics

    oxygen_submatrix(segment, node_list[segment.node_1_id], node_list[segment.node_2_id], h_solution_partial, u_solution_partial)
        Returns a segment specific submatrix and subvector for the Oxygen dymanics

    
    """

    def __init__(self, p_inlet:float, p_outlet:float, h_inlet:float, wall_hydraulic_conductivity:float, num_cells:int,\
        kappa1:float, solubility_O2:float, diffusivity_O2:float, half_saturation:float, hill_exponent:float, P_O2_in:float,\
            P_O2_out:float, beta_ov:float, logger:Logger):
        self.logger = logger
        self.pressure_inlet_nominal = p_inlet * 133.32
        self.pressure_outlet = p_outlet * 133.32
        self.pressure_inlet = self.set_inlet_pressure(0.4*self.pressure_inlet_nominal+0.6*self.pressure_outlet)
        self.haematocrit_inlet = h_inlet
        self.wall_hydraulic_conductivity = wall_hydraulic_conductivity
        self.num_cells = num_cells
        self.kappa1 = kappa1
        self.solubility_O2 = solubility_O2
        self.diffusivity_O2 = diffusivity_O2
        self.half_saturation = half_saturation
        self.hill_exponent = hill_exponent
        self.P_O2_in = P_O2_in
        self.P_O2_out = P_O2_out
        self.beta_ov = beta_ov
        self.kappa2 = (solubility_O2 * half_saturation)**hill_exponent
        self.recalculate_matrix = True
        self.recalculate_vector = True
        self.recalculate_geometry = True
        self.recalculate_O2_matrix = True
        self.decaying_diffusivity = 1e-6

    @classmethod
    def load_config(cls, config:Config):
        data = config.config_access["1D_CONDITIONS"]
        return cls(data["pressure_inlet"],data["pressure_outlet"],data["haematocrit_inlet"],\
            data["wall_hydraulic_conductivity"],data["num_cells"],data["kappa1"],data["solubility_O2"],\
                data["diffusivity_O2"],data["half_saturation"],data["hill_exponent"],data["P_O2_in"],data["P_O2_out"],\
                    data["beta_ov"],config.logger)

    def set_inlet_pressure(self, P_in:float, verbose:bool=True) -> float:
        P_abs_max = 120*133.32 # 120mmHg (heart output Pressure)
        P_max = self.pressure_inlet_nominal
        P_min = self.pressure_outlet + 1

        P_in = max(P_min, P_in)
        self.pressure_inlet = P_in
        if P_in > P_max:
            if verbose:
                self.logger.log(f"Warning!: Inlet Pressure Greater than specified Nominal Inlet Pressure")
                self.logger.log(f"P_in = {P_in}, P_max = {P_max}")
        if P_in > P_abs_max:
            if verbose:
                self.logger.log(f"Warning!: Inlet Pressure Greater than Allowable Heart Maximum")
                self.logger.log(f"P_in = {P_in}, P_abs_max = {P_abs_max}")
                P_in = P_abs_max
        
        self.logger.log(f"Inlet Pressure set to {P_in} Pa.")

        self.recalculate_vector = True

        return P_in

    def build_1D_mesh(self, tree:Tree, verbose=False) -> Tuple[gf.Mesh,Dict]:
        node_point_ref = {}
        segment_convex_id_ref = {}
        GT = gf.GeoTrans("GT_PK(1,1)")
        mesh = gf.Mesh('empty', 3)
        # self.logger.log(f"Generating 1D Mesh, num_cells = {self.num_cells}")

        # Iterate through the segments of the tree and make connectivity maps
        for keys in tree.segment_dict.keys():
            node_1_id = tree.segment_dict[keys].node_1_id
            node_2_id = tree.segment_dict[keys].node_2_id
            node_1_pos = tree.node_dict[node_1_id].location()
            node_2_pos = tree.node_dict[node_2_id].location()

            # Generate pusedo nodes in order to increase the cell count in 1D
            # adjusting the connectivity as appropriate
            if self.num_cells > 10000:
                base_cell_size = 10e-5
                adjusted_cell_size = base_cell_size / self.num_cells
                length = tree.length(keys)

                # If the length of the segment is shorter than the interval, ensure first and second point
                num_intervals = max([int(length // adjusted_cell_size),1])
                points_in_seg = np.linspace(node_1_pos,node_2_pos, num_intervals+1)

            else:
                points_in_seg = np.array([node_1_pos,node_2_pos])
            
            # Add the points to the mesh and create convexes between them.
            convexes_in_seg = []
            for i in range(len(points_in_seg)-1):
                convex_id = mesh.add_convex(GT,np.transpose([points_in_seg[i],points_in_seg[i+1]]))
                convexes_in_seg.append(convex_id)
                if i == 0:
                    first_pid = mesh.pid_from_cvid(convex_id)[0][0]
                    node_point_ref.update({node_1_id:first_pid})

            second_pid = mesh.pid_from_cvid(convex_id)[0][1] # type: ignore
            node_point_ref.update({node_2_id:second_pid})
            segment_convex_id_ref.update({keys:convexes_in_seg})
            mesh.set_region(keys+1, np.transpose(convexes_in_seg))

        if verbose:
            self.logger.log(f"TOTAL CONVEXES IN MESH: {len(mesh.cvid())}")
            self.logger.log(f"CVID: {mesh.cvid()}")

        if verbose:
            mesh.display()
            self.logger.log(mesh)
            self.logger.log(node_point_ref)

        mesh.save("mesh1D.txt")
        self.segment_convex_id_ref = segment_convex_id_ref

        return mesh, node_point_ref

    @staticmethod
    def _point_already_exists(new_point, existing_points, tolerance=1e-10):
        for idx, point in enumerate(existing_points):
            if np.linalg.norm(point - new_point) < tolerance:
                return True, idx
        return False, -1


    def build_fluid_elements(self, mesh):
        mim = gf.MeshIm(mesh, gf.Integ('IM_GAUSS1D(6)'))  
        p_element = gf.MeshFem(mesh)
        p_element.set_fem(gf.Fem('FEM_PK(1,1)'))
        # This calculation needs to match with the selected FEM above
        self.p_dofs = mesh.nbpts()

        v_element_dict = {}
        self.u_dof_ref = {}
        self.fluid_viscosity = {}
        total = 0

        for i in mesh.regions():
            temp_element = gf.MeshFem(mesh)
            convex_IDs = mesh.region(i)
            temp_element.set_fem(gf.Fem('FEM_PK(1,2)'), convex_IDs[0])
            v_element_dict.update({i:temp_element})
            pts = mesh.pid_in_regions(i)
            # This calculation needs to match with the selected FEM above
            self.u_dof_ref.update({i:2*len(pts)-1})
            total += 2*len(pts)-1

        self.u_dof_ref.update({"total":total})
            
        return mim, p_element, v_element_dict

    def _count_dofs(self) -> Tuple[int,int,int]:
        dofs_in_vessel = self.p_dofs + self.u_dof_ref["total"]
        return dofs_in_vessel, self.u_dof_ref["total"], self.p_dofs

    def _count_offset_h(self, key:int) -> int:
        _h_dofs = 0
        for keys in range(1,key+1):
            _h_dofs += self.h_dofs_ref[keys]
        return _h_dofs

    def _count_offset(self, key:int) -> int:
        _u_dofs = 0
        for keys in range(1,key+1):
            _u_dofs += self.u_dof_ref[keys]
        return _u_dofs

    def build_monolithic_vessel(self, tree:Tree, h_solution:Dict=None, mesh1D=None, recalculate_matrix=False, verbose=False) ->Tuple[gf.Spmat,np.ndarray, gf.MeshFem, Dict, np.ndarray, np.ndarray]: # type: ignore
        gf.util_trace_level(level=1)

        self.recalculate_matrix = recalculate_matrix
        if mesh1D is None:
            start_time = time.time()
            # Build the mesh for the vessels if necessary
            mesh1D, self.node_point_ref = self.build_1D_mesh(tree)
            # Define the integration method and elements on the mesh
            self.mim, self.p_element, self.v_element_dict = self.build_fluid_elements(mesh1D)
            # Create the h_elements in advance
            self.make_haematocrit_elements(tree, mesh1D)
            if verbose:
                self.logger.log("NEW MESH HAS BEEN GENERATED")
            self.recalculate_matrix = True
            end_time = time.time()
            if verbose:
                self.logger.log(f"Time to make mesh: {end_time-start_time}")

        
        # self.logger.log(self.p_element)
          
        #self.logger.log(mesh1D)
        # Construct the monolithic matrix system
        
        start_time = time.time()
        dof_total, u_tot, p_dofs = self._count_dofs()
        Full_MAT = gf.Spmat("empty", dof_total, dof_total)
        if self.recalculate_matrix is True:
            self.Static_MAT = gf.Spmat("empty", dof_total, dof_total)
            self.Full_VEC = np.zeros(dof_total)
        x = np.zeros(dof_total)
        end_time = time.time()
        if verbose:
            self.logger.log(f"Time to Initialize: {end_time-start_time}")

        # Iterate through each segment to obtain its local self contribution
        shift = 0
        start_time = time.time()
        for keys in tree.segment_dict.keys():
            u_dofs = self.u_dof_ref[keys+1]

            Mvvi = self._build_Mvvi(keys, tree, h_solution=h_solution)
            # Add the Mvvi component to the monolithic matrix
            Full_MAT.add(range(shift,shift+u_dofs),range(shift,shift+u_dofs),Mvvi)

            # This part only needs to trigger when the 1D mesh as been recalculated
            if self.recalculate_matrix is True or self.recalculate_geometry is True:
                Dvvi = self._build_Dvvi(keys, tree)
                # Add the Dvvi component to the monolithic matrix
                self.Static_MAT.add(range(shift,shift+u_dofs),range(u_tot,u_tot+p_dofs),-Dvvi)
                # Add the DvviT component to the monolithic matrix
                Dvvi.transpose()
                self.Static_MAT.add(range(u_tot,u_tot+p_dofs),range(shift,shift+u_dofs),Dvvi)
            
            shift += u_dofs

        end_time = time.time()
        if verbose:
            self.logger.log(f"Time to do segment contributions: {end_time-start_time}")
        # This part only needs to trigger when the 1D mesh as been recalculated
        big_time = time.time()
        if self.recalculate_matrix is True:
            start_time = time.time()
            self.inlet, self.outlet, self.junc = self._mark_junctions_in_mesh_new(mesh1D,tree,self.node_point_ref)
            end_time = time.time()
            if verbose:
                self.logger.log(f"Time to do junction marking: {end_time-start_time}")
        
        if self.recalculate_matrix is True or self.recalculate_geometry is True:
            start_time = time.time()
            Jvv_empty = gf.Spmat("empty", p_dofs, u_tot)
            Jvv = self._build_Jvv_contributions(tree,Jvv_empty)
            self.Static_MAT.add(range(u_tot,u_tot+p_dofs),range(0,u_tot), Jvv)
            Jvv.transpose()
            self.Static_MAT.add(range(0,u_tot),range(u_tot,u_tot+p_dofs), -Jvv)
            end_time = time.time()
            if verbose:
                self.logger.log(f"Time to do junction contributions: {end_time-start_time}")

        if self.recalculate_matrix is True or self.recalculate_vector is True or self.recalculate_geometry is True:
            start_time = time.time()
            Bu_vector = np.zeros(u_tot)
            Bu_vector = self._build_boundary_contributions(tree,Bu_vector)
            end_time = time.time()
            if verbose:
                self.logger.log(f"Time to do boundary contributions: {end_time-start_time}")

            self.Full_VEC[:u_tot] = Bu_vector 
            px = np.empty(p_dofs)
            ux = np.empty(u_tot)   

            end_time = time.time()
            if verbose:
                self.logger.log(f"Time to do all junction operations: {end_time-big_time}")
            # if verbose:
            #     self.logger.log(mesh1D)
            #     for key in self.v_element_dict.keys():
            #         self.logger.log(self.v_element_dict[key])
            
        Full_MAT.add(range(0,dof_total), range(0,dof_total), self.Static_MAT)
        self.recalculate_matrix = False
        self.recalculate_vector = False
        self.recalculate_geometry = False
        if verbose:
            self.Static_MAT.display()
        return Full_MAT, self.Full_VEC, self.p_element, self.v_element_dict, mesh1D, x

    def _build_Mvvi(self, segment_id:int, tree:Tree, h_solution:Union[dict,None]=None, verbose=False):
        R = tree.segment_dict[segment_id].radius
        area = tree.segment_dict[segment_id].area()

        if h_solution == None:
            # Use inlet haematocrit eveywhere
            mu_v = self.pries_viscosity(self.haematocrit_inlet, R*1e6) * 1e-3
        else:
            # Use previous solution everywhere
            segment_solution = h_solution[segment_id+1]
            self.fluid_viscosity.update({segment_id+1:np.empty_like(segment_solution)})

            for i, h_val in enumerate(segment_solution):
                self.fluid_viscosity[segment_id+1][i] = self.pries_viscosity(h_val, R*1e6) * 1e-3
            mu_v = np.mean(self.fluid_viscosity[segment_id+1])

        if verbose:
            self.logger.log(f"Presumed solution:")
            self.logger.log(f"Segment {segment_id}")
            self.logger.log(f"Viscosity {mu_v}")

        scale_term = 8*(mu_v/(R*R)*area)

        Mvvi = gf.asm_mass_matrix(self.mim, self.v_element_dict[segment_id+1],self.v_element_dict[segment_id+1],segment_id+1)
        Mvvi.scale(scale_term)
        
        return Mvvi

    def _build_Dvvi(self, segment_id:int, tree:Tree):
        area = tree.segment_dict[segment_id].area()
        lx, ly, lz = tree.get_tangent_versor(segment_id)

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("p", self.p_element)
        model.add_fem_variable("u", self.v_element_dict[segment_id+1])

        model.add_initialized_data("area", [area])
        model.add_initialized_data("lx", [lx])
        model.add_initialized_data("ly", [ly])
        model.add_initialized_data("lz", [lz])

        duTest_ds = "(Grad_Test_u(1).[lx,ly,lz](1)+Grad_Test_u(2).[lx,ly,lz](2)+Grad_Test_u(3).[lx,ly,lz](3))"
        pv_expression = f"area*p.{duTest_ds}"
        Dvvi = gf.asm_generic(self.mim, 2, pv_expression, segment_id+1, model, 'select_output', 'u','p')
        #self.logger.log(Dvvi)

        model.clear()

        return Dvvi

    def pries_viscosity(self, haematocrit_discharge, radius, temperature=37):
        #basic establishing coefficients
        H = haematocrit_discharge
        diameter = 2 * radius
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

        return viscosity_vessel

    def _mark_junctions_in_mesh_new(self, mesh1D, tree:Tree, node_point_ref:Dict, verbose=False):
        gf.util_trace_level(level=1)
        # outflow is from the junction out
        # inflow is into the junction
        #find the frist empty non-segment region id
        fer = tree.count_segments()+1
        nb_extrema = 0
        inflow_pids = {}
        outflow_pids = {}
        nb_junctions = 0
        junction_pids = {}
        nb_trivial = 0
        junction_type_dict = {1:"extrema",2:"trivial",3:"non-trival"}
        branch_region = 0

        for node in node_point_ref.keys():
            PID = node_point_ref[node]
            faces_on_PID = mesh1D.faces_from_pid(PID)
            junction_type = len(faces_on_PID[0])
            first_on_junction = True

            if verbose:
                self.logger.log(f"##################################################")
                self.logger.log(f"Current PID is {PID}")
                self.logger.log(f"Faces on PID are: {faces_on_PID}")
                self.logger.log(f"Junction Type is: {junction_type_dict[junction_type]}")

            if junction_type == 1:
                # extrema junction
                nb_extrema += 1
                # get cvid and face for the PID
                cvid = faces_on_PID[0][0]
                face = faces_on_PID[1][0]
                
                # add the extrema as its own region
                mesh1D.set_region(fer, [cvid,face])
                if verbose:
                    self.logger.log(f"Generating New Extrema Region {fer} on [{cvid,face}]")
                # record the region associated with the extrema and the branch the convex is associated with
                for keys in tree.segment_dict.keys():
                    if cvid in mesh1D.region(keys+1)[0]:
                        branch_region = keys+1
                        break

                # Check if inflow or outflow
                node_for_pid = [k for k, v in node_point_ref.items() if v == PID][0]
                node_type = tree.node_dict[node_for_pid].node_type()

                if node_type == "Inlet":
                    inflow_pids.update({PID:[fer,branch_region]})
                    fer += 1
                elif node_type == "Outlet":
                    outflow_pids.update({PID:[fer,branch_region]})
                    fer += 1

            elif junction_type == 2:
                # trivial junction
                nb_trivial += 1 
                for i in range(len(faces_on_PID[0])):
                    # get cvid and face for the PID
                    cvid = faces_on_PID[0][i]
                    face = face = faces_on_PID[1][i]
                    if first_on_junction:
                        # add the junction as its own region
                        mesh1D.set_region(fer, [cvid,face])
                        if verbose:
                            self.logger.log(f"Generating New Trivial Region {fer} on [{cvid,face}]")
                        junction_pids.update({PID:{"region":fer, "branches":[]}})
                        fer += 1
                        first_on_junction = False

                    for keys in tree.segment_dict.keys():
                        if cvid in mesh1D.region(keys+1)[0]:
                            branch_region = keys+1
                            break
                    if face == 0:
                        junction_pids[PID]["branches"].append(branch_region)
                    elif face == 1:
                        junction_pids[PID]["branches"].append(-branch_region)
                    else:
                        raise ValueError(f"Unexpected face ID on PID {PID}, CVID {cvid}, Face {face}")
                
            elif junction_type == 3:
                # non trivial junciton
                nb_junctions += 1
                for i in range(len(faces_on_PID[0])):
                    # get cvid and face for the PID
                    cvid = faces_on_PID[0][i]
                    face = face = faces_on_PID[1][i]
                    if first_on_junction:
                        # add the junction as its own region
                        mesh1D.set_region(fer, [cvid,face])
                        if verbose:
                            self.logger.log(f"Generating New Non Trivial Region {fer} on [{cvid},{face}]")
                        junction_pids.update({PID:{"region":fer, "branches":[]}})
                        fer += 1
                        first_on_junction = False

                    for keys in tree.segment_dict.keys():
                        if cvid in mesh1D.region(keys+1)[0]:
                            branch_region = keys+1
                            break
                    if face == 0:
                        junction_pids[PID]["branches"].append(branch_region)
                    elif face == 1:
                        junction_pids[PID]["branches"].append(-branch_region)
                    else:
                        raise ValueError(f"Unexpected face ID on PID {PID}, CVID {cvid}, Face {face}")
            
        if verbose:
            self.logger.log("THIS IS THE JUNCTION AND EXTREMA MARKING INFORMATION")
            self.logger.log(f"Inflow pids = {inflow_pids}")
            self.logger.log(f"Outflow pids = {outflow_pids}")
            self.logger.log(f"Junction pids = {junction_pids}")
            self.logger.log(mesh1D)

            # self.logger.log("Region double check")
            # for rid in mesh1D.regions():
            #     cvfid = mesh1D.region(rid)
            #     pids = mesh1D.pid_in_faces(cvfid)
            #     for pid in pids:
            #         self.logger.log(f"Region {rid} contains Point {pid}")

        return inflow_pids, outflow_pids, junction_pids

    # def _mark_junctions_in_mesh_old(self, mesh1D, tree:Tree, node_point_ref:Dict, verbose=False):
    #     gf.util_trace_level(level=1)
    #     # outflow is from the junction out
    #     # inflow is into the junction
    #     #find the frist empty non-segment region id
    #     fer = tree.count_segments()+1
    #     nb_extrema = 0
    #     inflow_pids = {}
    #     outflow_pids = {}
    #     nb_junctions = 0
    #     junction_pids = {}
    #     nb_trivial = 0

    #     # iterate through convexes
    #     convex_ids = mesh1D.cvid()
    #     for cvid in convex_ids:
    #         if verbose:
    #             self.logger.log(f"##################################################")
    #             self.logger.log(f"Current Convex is {cvid}")
    #             self.logger.log(f"PID from CVID is: {mesh1D.pid_from_cvid(cvid)}")
    #         # get the points associated with the convex
    #         p0 = mesh1D.pid_from_cvid(cvid)[0][0]
    #         p1 = mesh1D.pid_from_cvid(cvid)[0][1]

    #         # check p0 for junctions
    #         convexes_on_p0 = mesh1D.cvid_from_pid(p0, share=True)
    #         convexes_on_p0_size = convexes_on_p0.size

    #         if verbose:
    #             self.logger.log(f"PID for p0 = {p0}")
    #             self.logger.log(f"Convexes on p0 = {convexes_on_p0.size}")

    #         # presumed extrema
    #         if convexes_on_p0_size == 1:
    #             nb_extrema += 1
    #             # add the extrema as its own region
    #             mesh1D.set_region(fer, [cvid,1])
    #             if verbose:
    #                 self.logger.log(f"Generating New Extrema Region {fer} on [{cvid},{1}]")
    #             # record the region associated with the extrema and the branch the convex is associated with
    #             for keys in tree.segment_dict.keys():
    #                 if cvid in mesh1D.region(keys+1)[0]:
    #                     branch_region = keys+1
    #                     break

    #             # Check if inflow or outflow
    #             node_for_pid = [k for k, v in node_point_ref.items() if v == p0][0]
    #             node_type = tree.node_dict[node_for_pid].node_type()

    #             if node_type == "Inlet":
    #                 inflow_pids.update({p0:[fer,branch_region]})
    #                 fer += 1
    #             elif node_type == "Outlet":
    #                 outflow_pids.update({p0:[fer,branch_region]})
    #                 fer += 1
    #             else:
    #                 raise ValueError("Geometry comprehension error, node at p0 identified as extrema not specified as inlet or outlet")

    #         # trivial inflow junction
    #         elif convexes_on_p0_size == 2:
    #             # find the first branch region in which the point appears
    #             for keys in tree.segment_dict.keys():
    #                 if cvid in mesh1D.region(keys+1)[0]:
    #                     branch_region = keys+1
    #                     break

    #             # check for triviality of convex
    #             cv1 = mesh1D.cvid_from_pid(p0, share=True)[0]
    #             cv2 = mesh1D.cvid_from_pid(p0, share=True)[1]

    #             if cvid == cv1:
    #                 first_cv = cv1
    #                 other_cv = cv2
    #             else:
    #                 first_cv = cv2
    #                 other_cv = cv1

    #             is_trivial = first_cv in mesh1D.region(branch_region) and other_cv not in mesh1D.region(branch_region)
    #             if verbose:
    #                 self.logger.log(f"First cv = {first_cv}, other cv = {other_cv}, trivial? = {is_trivial}")
    #             if is_trivial:
    #                 if not p0 in junction_pids.keys():
    #                     nb_trivial += 1
    #                     mesh1D.set_region(fer, [cvid,1])
    #                     if verbose:
    #                         self.logger.log(f"Generating New Trivial Region {fer} on [{cvid},{1}]")
    #                     junction_pids.update({p0:{"region":fer, "branches":[]}})
    #                     fer += 1
    #                 else:
    #                     if verbose:
    #                         self.logger.log(f"Did not generate new region due to p0 already recorded")
    #                         self.logger.log(junction_pids[p0])
    #                 junction_pids[p0]["branches"].append(-branch_region)

    #         # non-trivial inflow junction
    #         elif convexes_on_p0_size == 3:
    #             if not p0 in junction_pids.keys():
    #                 nb_junctions += 1
    #                 # add the junction as its own region
    #                 mesh1D.set_region(fer, [cvid,1])
    #                 if verbose:
    #                     self.logger.log(f"Generating New Non Trivial Region {fer} on [{cvid},{1}]")
    #                 junction_pids.update({p0:{"region":fer, "branches":[]}})
    #                 fer += 1
    #             else:
    #                 if verbose:
    #                     self.logger.log(f"Did not generate new region due to p0 already recorded")
    #                     self.logger.log(junction_pids[p0])

    #             for keys in tree.segment_dict.keys():
    #                 if cvid in mesh1D.region(keys+1)[0]:
    #                     branch_region = keys+1
    #                     break
                
    #             junction_pids[p0]["branches"].append(-branch_region)
    #         else:
    #             raise ValueError(f"Unexpected Number of Convexes ({convexes_on_p0_size}) on PID {p0}")
            
    #         # MOVING TO OTHER PID
    #         convexes_on_p1 = mesh1D.cvid_from_pid(p1, share=True)
    #         convexes_on_p1_size = convexes_on_p1.size

    #         if verbose:
    #             self.logger.log(f"PID for p1 = {p1}")
    #             self.logger.log(f"Convexes on p1 = {convexes_on_p1.size}")

            
    #         # presumed outflow extrema
    #         if convexes_on_p1_size == 1:
    #             nb_extrema += 1
    #             # add the extrema as its own region
    #             mesh1D.set_region(fer, [cvid,0])
    #             if verbose:
    #                 self.logger.log(f"Generating New Extrema Region {fer} on [{cvid},{0}]")
    #             # record the region associated with the extrema and the branch the convex is associated with
    #             for keys in tree.segment_dict.keys():
    #                 if cvid in mesh1D.region(keys+1)[0]:
    #                     branch_region = keys+1
    #                     break

    #             # Check if inflow or outflow
    #             node_for_pid = [k for k, v in node_point_ref.items() if v == p1][0]
    #             node_type = tree.node_dict[node_for_pid].node_type()

    #             if node_type == "Inlet":
    #                 inflow_pids.update({p1:[fer,branch_region]})
    #                 fer += 1
    #             elif node_type == "Outlet":
    #                 outflow_pids.update({p1:[fer,branch_region]})
    #                 fer += 1
    #             else:
    #                 raise ValueError("Geometry comprehension error, node at p1 identified as extrema not specified as inlet or outlet")

    #         # trivial outflow junction
    #         elif convexes_on_p1_size == 2:
    #             if verbose:
    #                 self.logger.log(f"Entering trivial junction")
                
    #             # find the first branch region in which the point appears
    #             for keys in tree.segment_dict.keys():
    #                 if cvid in mesh1D.region(keys+1)[0]:
    #                     branch_region = keys+1
    #                     break

    #             # check for triviality of convex
    #             cv1 = mesh1D.cvid_from_pid(p1, share=True)[0]
    #             cv2 = mesh1D.cvid_from_pid(p1, share=True)[1]

    #             if cvid == cv1:
    #                 first_cv = cv1
    #                 other_cv = cv2
    #             else:
    #                 first_cv = cv2
    #                 other_cv = cv1

    #             is_trivial = first_cv in mesh1D.region(branch_region) and other_cv not in mesh1D.region(branch_region)
    #             if verbose:
                    
    #                 self.logger.log(f"First cv = {first_cv}, other cv = {other_cv}, trivial? = {is_trivial}")
    #             if is_trivial:
    #                 if not p1 in junction_pids.keys():
    #                     nb_trivial += 1
    #                     mesh1D.set_region(fer, [cvid,0])
    #                     if verbose:
    #                         self.logger.log(f"Generating New Trivial Region {fer} on [{cvid},{0}]")
    #                     junction_pids.update({p1:{"region":fer, "branches":[]}})
    #                     fer += 1
    #                 else:
    #                     if verbose:
    #                         self.logger.log(f"Did not generate new region due to p1 already recorded")
    #                         self.logger.log(junction_pids[p1])
    #                 junction_pids[p1]["branches"].append(branch_region)

                

    #         # non-trivial inflow junction
    #         elif convexes_on_p1_size == 3:
    #             if not p1 in junction_pids.keys():
    #                 nb_junctions += 1
    #                 # add the junction as its own region
    #                 mesh1D.set_region(fer, [cvid,0])
    #                 if verbose:
    #                     self.logger.log(f"Generating New Non Trivial Region {fer} on [{cvid},{0}]")
    #                 junction_pids.update({p1:{"region":fer, "branches":[]}})
    #                 fer += 1
    #             else:
    #                 if verbose:
    #                     self.logger.log(f"Did not generate new region due to p1 already recorded")
    #                     self.logger.log(junction_pids[p1])
    #             for keys in tree.segment_dict.keys():
    #                 if cvid in mesh1D.region(keys+1)[0]:
    #                     branch_region = keys+1
    #                     break
                
    #             junction_pids[p1]["branches"].append(branch_region)

    #         else:
    #             raise ValueError(f"Unexpected Number of Convexes ({convexes_on_p1_size}) on PID {p1}")
            
    #         if verbose:
    #             self.logger.log(f"##################################################")
            
    #     if verbose:
    #         self.logger.log("THIS IS THE JUNCTION AND EXTREMA MARKING INFORMATION")
    #         self.logger.log(f"Inflow pids = {inflow_pids}")
    #         self.logger.log(f"Outflow pids = {outflow_pids}")
    #         self.logger.log(f"Junction pids = {junction_pids}")
    #         self.logger.log(mesh1D)

    #         # self.logger.log("Region double check")
    #         # for rid in mesh1D.regions():
    #         #     cvfid = mesh1D.region(rid)
    #         #     pids = mesh1D.pid_in_faces(cvfid)
    #         #     for pid in pids:
    #         #         self.logger.log(f"Region {rid} contains Point {pid}")

    #     return inflow_pids, outflow_pids, junction_pids
    
    def _build_Jvv_contributions(self, tree:Tree, Mat, verbose=False):
        if verbose:
            self.logger.log(f"Applying junction contributions")
        # loop through all branches
        offset = 0
        dofs = self.p_dofs
        for keys in tree.segment_dict.keys():
            area_i = tree.segment_dict[keys].area()
            # loop through all junctions
            for junc_ids in self.junc.keys():
                # iterate through the branches associated with the junction
                for branch in self.junc[junc_ids]['branches']:
                    # limit the junction effects to the current segment
                    if abs(branch) == keys+1:
                        model=gf.Model('real') # real or complex space.
                        model.add_fem_variable("p", self.p_element)
                        p_basis = gf.asm("generic", self.mim, 1, 'p', self.junc[junc_ids]["region"], model)
                        model.clear()

                        # for i in range(100000000000):
                        # print(f"iteration {i}")
                        row = 0
                        found = False
                        while (not found) and (row < dofs):
                            found = (1.0 - p_basis[row] < 1e-6)
                            if not found:
                                row += 1
                
                        # branches which are -ve are outflow, branches which are +ve are inflow
                        # outflow is from the junction out
                        # inflow is into the junction
                        if branch < 0:
                            # outflow contributions
                            #col1 = (keys)*self.v_element_dict[keys+1].nbdof()
                            col = offset
                            if verbose:
                                self.logger.log(f"Row is: {row}")
                                self.logger.log(f"Col is: {col}")
                                self.logger.log(f"Applying outflow associated with junction: {junc_ids} Area is: {area_i}")
                            Mat.add(row, col, area_i)
                            
                        if branch > 0:
                            # inflow contributions
                            #col1 = (keys+1)*self.v_element_dict[keys+1].nbdof()-1
                            col = offset + self.u_dof_ref[keys+1]-1
                            if verbose:
                                self.logger.log(f"Row is: {row}")
                                self.logger.log(f"Col is: {col}")
                                self.logger.log(f"Applying inflow associated with junction: {junc_ids} Area is: {area_i}")
                            Mat.add(row, col, -area_i)
                
            offset += self.u_dof_ref[keys+1]
        if verbose:
            self.logger.log(Mat)
        return Mat

    def _build_boundary_contributions(self, tree:Tree, Vec:np.ndarray, verbose=False) -> np.ndarray:
        # loop through all inlet boundaries
        for inlet_pid in self.inlet.keys():
            region = self.inlet[inlet_pid][0]
            if verbose:
                self.logger.log(f"inlet boundary reigon = {region}")

            branch = self.inlet[inlet_pid][1]
            area_i = tree.segment_dict[branch-1].area()
            inlet_pressure = self.pressure_inlet

            model=gf.Model('real') # real or complex space.
            model.add_fem_variable("u", self.v_element_dict[branch])
            model.add_initialized_data("flux_in", [inlet_pressure * area_i])

            cvfid = self.v_element_dict[branch].mesh().region(region)
            pid_in_face = self.v_element_dict[branch].mesh().pid_in_faces(cvfid)
            points_on_cvid = self.v_element_dict[branch].mesh().pid_in_cvids(cvfid[0])
            if verbose:
                self.logger.log(f"Inlet face = {self.v_element_dict[branch].mesh().region(region)[1]}")
                self.logger.log(f"PID on face = {pid_in_face}")
                self.logger.log(f"PID on CVID = {points_on_cvid}")

            condition = points_on_cvid[cvfid[1]] == pid_in_face
            if verbose:
                self.logger.log(f"Conditional = {condition[0]}")

            # Orientate the source term based on the presumed direction
            if self.v_element_dict[branch].mesh().region(region)[1] == 1:
                model.add_source_term_brick(self.mim, "u", "flux_in", region)
            elif self.v_element_dict[branch].mesh().region(region)[1] == 0:
                model.add_source_term_brick(self.mim, "u", "-flux_in", region)
            else:
                raise ValueError("Unexpected face in 1D boundary assignment")


            model.assembly("build_rhs")

            F_inlet_i = model.rhs()
            dof = self.u_dof_ref[branch]
            offset = self._count_offset(branch-1)
            Vec[offset:offset+dof] += F_inlet_i

        # loop through all outlet boundaries
        for outlet_pid in self.outlet.keys():
            region = self.outlet[outlet_pid][0]
            if verbose:
                self.logger.log(f"outlet boundary reigon = {region}")

            branch = self.outlet[outlet_pid][1]
            area_i = tree.segment_dict[branch-1].area()
            outlet_pressure = self.pressure_outlet

            model=gf.Model('real') # real or complex space.
            model.add_fem_variable("u", self.v_element_dict[branch])
            model.add_initialized_data("flux_out", [outlet_pressure * area_i])


            # Orientate the source term based on the presumed direction
            if self.v_element_dict[branch].mesh().region(region)[1] == 1:
                model.add_source_term_brick(self.mim, "u", "flux_out", region)
            elif self.v_element_dict[branch].mesh().region(region)[1] == 0:
                model.add_source_term_brick(self.mim, "u", "-flux_out", region)
            else:
                raise ValueError("Unexpected face in 1D boundary assignment")

                    
            model.assembly("build_rhs")

            F_outlet_i = model.rhs()
            dof = self.u_dof_ref[branch]
            offset = self._count_offset(branch-1)
            Vec[offset:offset+dof] += F_outlet_i

        return Vec

    def make_haematocrit_elements(self, tree:Tree, mesh1D):
        self.h_element_dict = {}
        self.h_dofs_ref = {}
        total = 0
        for keys in tree.segment_dict.keys():
            temp_element = gf.MeshFem(mesh1D)
            convex_IDs = mesh1D.region(keys+1)
            temp_element.set_fem(gf.Fem('FEM_PK(1,1)'), convex_IDs[0])
            self.h_element_dict.update({keys+1:temp_element})
            dofs = len(mesh1D.pid_in_regions(keys+1))
            self.h_dofs_ref.update({keys+1:dofs})
            total += dofs

        self.h_dofs_ref.update({"total":total})

        return self.h_element_dict

    def algorithmic_haematocrit_recursive(self, tree:Tree, mesh1D, u_solution:Union[dict,None]=None, h_solution:Union[dict,None]=None, verbose=False) -> np.ndarray:
        # Check if a u_solution was supplied
        if u_solution == None:
            raise ValueError("You must pass a velocity solution to this function")

        # Check if v_elements exist
        if type(self.v_element_dict) != dict:
            raise ValueError("This function can only be called after build_monolithic_vessel as it requires some internal components")
        
        # Create the h_elements if necessary
        if not hasattr(self, 'h_element_dict'):
            self.make_haematocrit_elements(tree, mesh1D)

        # check if h_solution was supplied and set it to initial if it is not
        if h_solution == None:
            h_solution = {}
            for keys in self.h_element_dict.keys():
                temp_h_sol_vec = np.full(self.h_dofs_ref[keys],self.haematocrit_inlet)
                h_solution.update({keys:temp_h_sol_vec})

        num_segments = tree.count_segments()
        self.fluxes_start = np.zeros(num_segments) #
        self.fluxes_end = np.zeros(num_segments) #
        direction = np.zeros(num_segments)
        self.H_segment_start = np.zeros(num_segments) #
        self.H_segment_end = np.zeros(num_segments) #
        self.H_populated = np.zeros(num_segments) #

        num_nodes = tree.count_nodes()
        self.num_segments_into_node = np.zeros(num_nodes) #
        self.num_segments_outof_node = np.zeros(num_nodes) #
        inlet_marker = np.zeros(num_nodes)

        for keys in tree.segment_dict.keys():
            area = tree.area(keys)
            velocity = np.mean(u_solution[keys+1])

            self.H_segment_start[keys] = h_solution[keys+1][0]
            self.H_segment_end[keys] = h_solution[keys+1][-1]

            node_1_id = tree.segment_dict[keys].node_1_id
            node_2_id = tree.segment_dict[keys].node_2_id

            if velocity > 0:
                direction[keys] += 1
                velocity_start = np.abs(u_solution[keys+1][0])
                velocity_end = np.abs(u_solution[keys+1][-1])

                flux_start = area*velocity_start
                self.fluxes_start[keys] = flux_start
            
                flux_end = area*velocity_end
                self.fluxes_end[keys] = flux_end

                self.num_segments_into_node[node_2_id] += 1
                self.num_segments_outof_node[node_1_id] += 1
            else:
                direction[keys] -= 1
                velocity_start = np.abs(u_solution[keys+1][-1])
                velocity_end = np.abs(u_solution[keys+1][0])

                flux_start = area*velocity_start
                self.fluxes_start[keys] = flux_start
            
                flux_end = area*velocity_end
                self.fluxes_end[keys] = flux_end

                self.num_segments_into_node[node_1_id] += 1
                self.num_segments_outof_node[node_2_id] += 1

        # for keys in tree.node_dict.keys():
        #     if self.num_segments_outof_node[keys] == 2 and self.num_segments_into_node[keys] == 0:
        #         self.logger.log("ERROR IN FLOW DIRECTIONS USING LAST H SOLUTION")
        #         x_output = []
        #         for keys in h_solution.keys():
        #             x_output.extend(np.array(h_solution[keys]))
        #         return x_output, self.h_element_dict
        #     if self.num_segments_outof_node[keys] == 0 and self.num_segments_into_node[keys] == 2:
        #         self.logger.log("ERROR IN FLOW DIRECTIONS USING LAST H SOLUTION")
        #         x_output = []
        #         for keys in h_solution.keys():
        #             x_output.extend(np.array(h_solution[keys]))
        #         return x_output, self.h_element_dict

        self.split_frac = np.ones(num_segments)

        for keys in tree.node_dict.keys():
            if self.num_segments_outof_node[keys] > self.num_segments_into_node[keys]:
                if tree.check_node_internal(keys):
                    in_seg_ids = []
                    out_seg_ids = []
                    seg_ids, node_side = tree.get_segment_ids_on_node(keys)
                    for i in range(len(seg_ids)):
                        if direction[seg_ids[i]] > 0:
                            if node_side[i] == 2:
                                in_seg_ids.append(seg_ids[i])
                            else:
                                out_seg_ids.append(seg_ids[i])
                        else:
                            if node_side[i] == 1:
                                in_seg_ids.append(seg_ids[i])
                            else:
                                out_seg_ids.append(seg_ids[i])

                    if verbose:
                        self.logger.log(f"SEG_IDS = {seg_ids}, NODE_SIDE = {node_side}")
                        self.logger.log(f"IN_SEG_IDS = {in_seg_ids}, OUT_SEG_IDS = {out_seg_ids}")
                        for seg_ids in in_seg_ids:
                            self.logger.log(f"INLET VELOCITY = {self.fluxes_end[seg_ids]}")
                        for seg_ids in out_seg_ids:
                            self.logger.log(f"OUTLET VELOCITY = {self.fluxes_start[seg_ids]}")

                    D_parent = 2*tree.segment_dict[in_seg_ids[0]].radius * 10**6
                    D_1 = 2*tree.segment_dict[out_seg_ids[0]].radius * 10**6
                    D_2 = 2*tree.segment_dict[out_seg_ids[1]].radius * 10**6

                    if direction[in_seg_ids[0]] > 0:
                        parent_flux = self.fluxes_end[in_seg_ids[0]]
                    else:
                        parent_flux = self.fluxes_start[in_seg_ids[0]]
                    if direction[out_seg_ids[0]] > 0:    
                        child_flux_1 = self.fluxes_end[out_seg_ids[0]]
                    else:
                        child_flux_1 = self.fluxes_start[out_seg_ids[0]]
                    if direction[out_seg_ids[1]] > 0:    
                        child_flux_2 = self.fluxes_end[out_seg_ids[1]]
                    else:
                        child_flux_2 = self.fluxes_start[out_seg_ids[1]]

                    FQB = child_flux_1/parent_flux

                    H_parent = self.H_segment_end[in_seg_ids[0]]

                    
                    FQE1 = self._fractional_Erythrocytes(FQB, D_parent, D_1, D_2, H_parent)
                    FQE2 = 1-FQE1

                    self.split_frac[out_seg_ids[0]] = FQE1#*(parent_flux/child_flux_1)
                    self.split_frac[out_seg_ids[1]] = FQE2#*(parent_flux/child_flux_2)
                else:
                    inlet_marker[keys] = 1

        if verbose:
            self.logger.log("SPLIT FRAC FINAL:")
            self.logger.log(self.split_frac)

        for i in range(len(inlet_marker)):
            if inlet_marker[i] == 1:
                seg_id, node_side = tree.get_segment_ids_on_node(i)
                self.H_populated[seg_id[0]] = 1
                if direction[seg_id[0]] > 0:
                    self.H_segment_start[seg_id[0]] = self.haematocrit_inlet
                    self.H_segment_end[seg_id[0]] = self.H_segment_start[seg_id[0]]*self.fluxes_start[seg_id[0]]/self.fluxes_end[seg_id[0]]
                self._populate_haematocrit_recursive(tree,seg_id[0],i,verbose)

        if verbose:
            self.logger.log(f"Automatic Population from inlets:")
            self.logger.log(f"Number of segments: {num_segments}")
            self.logger.log(f"Number populated with H: {len(np.where(self.H_populated == 1)[0])}")

        for i in range(len(self.H_populated)):
            zero_positions = np.where(self.H_populated == 0)[0]
            
            # Check if there are no zeros left
            if len(zero_positions) == 0:
                break  # Exit the loop
            
            # Process the array
            for current_segment in zero_positions:
                if direction[current_segment] > 0:
                    starting_node = tree.segment_dict[current_segment].node_1_id
                else:
                    starting_node = tree.segment_dict[current_segment].node_2_id
                
                if self.num_segments_into_node[starting_node] == 2:
                    local_segments, _ = tree.get_segment_ids_on_node(starting_node)
                    local_segments.remove(current_segment)

                    populated_parents = True
                    for parent in local_segments:
                        if self.H_populated[parent] == 0:
                            populated_parents = False
                    
                    H_in_child = 0
                    if populated_parents:
                        for parent in local_segments:
                            H_in_child += self.H_segment_end[parent] * self.fluxes_end[parent] / self.fluxes_start[current_segment]

                        self.H_populated[current_segment] = 1
                        self.H_segment_start[current_segment] = H_in_child
                        self.H_segment_end[current_segment] = self.H_segment_start[current_segment] * self.fluxes_start[current_segment] / self.fluxes_end[current_segment]
                        if verbose:
                            self.logger.log(f"Segment {current_segment} populated with H_value through Anastomosis: {self.H_segment_start[current_segment]}")
                        self._populate_haematocrit_recursive(tree,current_segment,starting_node,verbose)
            
            # Check if the number of zeros is decreasing
            new_zero_positions = np.where(self.H_populated == 0)[0]
            
            if len(new_zero_positions) >= len(zero_positions):
                raise ValueError("No progress populating Haematocrit Values.")
        
        if verbose:
            self.logger.log("All segments populated with H values")
            self.logger.log(direction)

        x_output = []
        for keys in self.h_element_dict.keys():
            # temp_h_sol_vec = np.full(self.h_element_dict[keys].nbdof(),self.H_segment[keys-1])
            if direction[keys-1] > 0:
                temp_h_sol_vec = np.linspace(self.H_segment_start[keys-1],self.H_segment_end[keys-1],self.h_dofs_ref[keys])
            else:
                temp_h_sol_vec = np.linspace(self.H_segment_end[keys-1],self.H_segment_start[keys-1],self.h_dofs_ref[keys])

            if verbose:
                self.logger.log(f"Segment {keys-1}")
                self.logger.log(temp_h_sol_vec)
            x_output.extend(temp_h_sol_vec)

        x_output = np.array(x_output)

        if verbose:
            self.logger.log("Initial Vector")
            self.logger.log(h_solution)
            self.logger.log("Final Vector")
            self.logger.log(x_output)

        # cleanup
        self.fluxes_start = [] #
        self.fluxes_end = [] #
        self.H_segment_start = [] #
        self.H_segment_end = [] #
        self.H_populated = [] #
        self.num_segments_into_node = [] #
        self.num_segments_outof_node = [] #


        return x_output, self.h_element_dict

    def algorithmic_haematocrit(self, tree:Tree, mesh1D, u_solution:Union[dict,None]=None, h_solution:Union[dict,None]=None, verbose=False) -> np.ndarray:
        # Check if a u_solution was supplied
        if u_solution == None:
            raise ValueError("You must pass a velocity solution to this function")

        # Check if v_elements exist
        if type(self.v_element_dict) != dict:
            raise ValueError("This function can only be called after build_monolithic_vessel as it requires some internal components")
        
        # Create the h_elements if necessary
        if not hasattr(self, 'h_element_dict'):
            self.make_haematocrit_elements(tree, mesh1D)

        # check if h_solution was supplied and set it to initial if it is not
        if h_solution == None:
            h_solution = {}
            for keys in self.h_element_dict.keys():
                temp_h_sol_vec = np.full(self.h_dofs_ref[keys],self.haematocrit_inlet)
                h_solution.update({keys:temp_h_sol_vec})

        num_segments = tree.count_segments()
        self.fluxes_start = np.zeros(num_segments) #
        self.fluxes_end = np.zeros(num_segments) #
        direction = np.zeros(num_segments)
        self.H_segment_start = np.zeros(num_segments) #
        self.H_segment_end = np.zeros(num_segments) #
        self.H_populated = np.zeros(num_segments) #

        num_nodes = tree.count_nodes()
        self.num_segments_into_node = np.zeros(num_nodes) #
        self.num_segments_outof_node = np.zeros(num_nodes) #
        inlet_marker = np.zeros(num_nodes)

        for keys in tree.segment_dict.keys():
            area = tree.area(keys)
            velocity = np.mean(u_solution[keys+1])

            self.H_segment_start[keys] = h_solution[keys+1][0]
            self.H_segment_end[keys] = h_solution[keys+1][-1]

            node_1_id = tree.segment_dict[keys].node_1_id
            node_2_id = tree.segment_dict[keys].node_2_id

            if velocity >= 0:
                direction[keys] += 1
                velocity_start = np.abs(u_solution[keys+1][0])
                velocity_end = np.abs(u_solution[keys+1][-1])

                flux_start = area*velocity_start
                self.fluxes_start[keys] = flux_start
            
                flux_end = area*velocity_end
                self.fluxes_end[keys] = flux_end

                self.num_segments_into_node[node_2_id] += 1
                self.num_segments_outof_node[node_1_id] += 1
            else:
                direction[keys] -= 1
                velocity_start = np.abs(u_solution[keys+1][-1])
                velocity_end = np.abs(u_solution[keys+1][0])

                flux_start = area*velocity_start
                self.fluxes_start[keys] = flux_start
            
                flux_end = area*velocity_end
                self.fluxes_end[keys] = flux_end

                self.num_segments_into_node[node_1_id] += 1
                self.num_segments_outof_node[node_2_id] += 1

        self.split_frac = np.ones(num_segments)

        for keys in tree.node_dict.keys():
            if self.num_segments_outof_node[keys] > self.num_segments_into_node[keys]:
                if tree.check_node_internal(keys):
                    in_seg_ids = []
                    out_seg_ids = []
                    seg_ids, node_side = tree.get_segment_ids_on_node(keys)
                    for i in range(len(seg_ids)):
                        if direction[seg_ids[i]] > 0:
                            if node_side[i] == 2:
                                in_seg_ids.append(seg_ids[i])
                            else:
                                out_seg_ids.append(seg_ids[i])
                        else:
                            if node_side[i] == 1:
                                in_seg_ids.append(seg_ids[i])
                            else:
                                out_seg_ids.append(seg_ids[i])

                    if verbose:
                        self.logger.log(f"SEG_IDS = {seg_ids}, NODE_SIDE = {node_side}")
                        self.logger.log(f"IN_SEG_IDS = {in_seg_ids}, OUT_SEG_IDS = {out_seg_ids}")
                        for seg_ids in in_seg_ids:
                            self.logger.log(f"INLET VELOCITY = {self.fluxes_end[seg_ids]}")
                        for seg_ids in out_seg_ids:
                            self.logger.log(f"OUTLET VELOCITY = {self.fluxes_start[seg_ids]}")

                    if len(in_seg_ids) == 0:
                        raise ValueError(f"No segments flow into internal node {keys}, in_seg_ids={in_seg_ids}, out_seg_ids={out_seg_ids}")
                    if len(out_seg_ids) == 0:
                        raise ValueError(f"No segments flow out of internal node {keys}, in_seg_ids={in_seg_ids}, out_seg_ids={out_seg_ids}")

                    D_parent = 2*tree.segment_dict[in_seg_ids[0]].radius * 10**6
                    D_1 = 2*tree.segment_dict[out_seg_ids[0]].radius * 10**6
                    D_2 = 2*tree.segment_dict[out_seg_ids[1]].radius * 10**6

                    if direction[in_seg_ids[0]] > 0:
                        parent_flux = self.fluxes_end[in_seg_ids[0]]
                    else:
                        parent_flux = self.fluxes_start[in_seg_ids[0]]
                    if direction[out_seg_ids[0]] > 0:    
                        child_flux_1 = self.fluxes_end[out_seg_ids[0]]
                    else:
                        child_flux_1 = self.fluxes_start[out_seg_ids[0]]
                    if direction[out_seg_ids[1]] > 0:    
                        child_flux_2 = self.fluxes_end[out_seg_ids[1]]
                    else:
                        child_flux_2 = self.fluxes_start[out_seg_ids[1]]

                    FQB1 = child_flux_1/parent_flux
                    FQB2 = child_flux_2/parent_flux

                    H_parent = self.H_segment_end[in_seg_ids[0]]

                    
                    FQE1 = self._fractional_Erythrocytes(FQB1, D_parent, D_1, D_2, H_parent)
                    FQE2 = 1-FQE1
                    # FQE2 = self._fractional_Erythrocytes(FQB2, D_parent, D_1, D_2, H_parent)

                    self.split_frac[out_seg_ids[0]] = FQE1#*(parent_flux/child_flux_1)
                    self.split_frac[out_seg_ids[1]] = FQE2#*(parent_flux/child_flux_2)
                else:
                    inlet_marker[keys] = 1

        if verbose:
            self.logger.log("SPLIT FRAC FINAL:")
            self.logger.log(self.split_frac)

        graph = self._make_directed_graph(tree,direction)
        self._populate_haematocrit_directed_dfs(tree,graph,self.haematocrit_inlet)

        x_output = []
        for keys in self.h_element_dict.keys():
            # temp_h_sol_vec = np.full(self.h_element_dict[keys].nbdof(),self.H_segment[keys-1])
            if direction[keys-1] > 0:
                temp_h_sol_vec = np.linspace(self.H_segment_start[keys-1],self.H_segment_end[keys-1],self.h_dofs_ref[keys])
            else:
                temp_h_sol_vec = np.linspace(self.H_segment_end[keys-1],self.H_segment_start[keys-1],self.h_dofs_ref[keys])

            if verbose:
                self.logger.log(f"Segment {keys-1}")
                self.logger.log(temp_h_sol_vec)
            x_output.extend(temp_h_sol_vec)

        x_output = np.array(x_output)

        if verbose:
            self.logger.log("Initial Vector")
            self.logger.log(h_solution)
            self.logger.log("Final Vector")
            self.logger.log(x_output)

        # cleanup
        self.fluxes_start = [] #
        self.fluxes_end = [] #
        self.H_segment_start = [] #
        self.H_segment_end = [] #
        self.H_populated = [] #
        self.num_segments_into_node = [] #
        self.num_segments_outof_node = [] #


        return x_output, self.h_element_dict, self.capped_haematocrit

    def _make_directed_graph(self,tree:Tree,direction:np.ndarray) -> GraphRepresentation:
        graph = GraphRepresentation()
        graph.build_directed(tree,direction)
        return graph

    def _populate_haematocrit_directed_dfs(self, tree:Tree, graph:GraphRepresentation, h_value:float, verbose=False):
        start_nodes = tree.get_node_ids_inlet()
        stack = [(start_node, 0) for start_node in start_nodes]  # List of tuples: [(node, incoming_edges_remaining), ...]
        buffer = []
        visited = set()
        # self.logger.log(graph)
        if verbose:
            self.logger.log(f"#################### START H DFS ####################")
        

        while stack:
            # self.logger.log(f"stack = {stack}")
            current_node, incoming_edges_remaining = stack.pop()

            # Check if all incoming edges are processed
            if incoming_edges_remaining == 0:
                # Your logic to populate values for a specific node goes here
                # Case where current node is an inlet 
                if graph.incoming_edges_count[current_node] == 0:
                    for neighbor in graph.adjacency_list[current_node]:
                        downstream_seg_id = graph.edge_list[current_node][neighbor]
                        self.H_populated[downstream_seg_id] = 1
                        self.H_segment_start[downstream_seg_id] = h_value
                        self.H_segment_end[downstream_seg_id] = h_value
                        if verbose:
                            self.logger.log(f"Segment {downstream_seg_id} populated with H = {self.H_segment_start[downstream_seg_id]:.2g} and {self.H_segment_end[downstream_seg_id]:.2g} via case 1")

                # Case where the current node has 1 upstream vessel
                if graph.incoming_edges_count[current_node] == 1:
                    # Identify the segment ID associated with the upstream vessel
                    local_seg_ids, _ = tree.get_segment_ids_on_node(current_node)
                    for downstream_node in graph.adjacency_list[current_node]:
                        remove_seg_id = graph.edge_list[current_node][downstream_node]
                        local_seg_ids.remove(remove_seg_id)
                    # Case where the current node has 1 or 2 downstream vessels
                    for neighbor in graph.adjacency_list[current_node]:
                        downstream_seg_id = graph.edge_list[current_node][neighbor]
                        # self.logger.log(f"Downstream seg IDs: {downstream_seg_id}")
                        # self.logger.log(f"Local seg IDs: {local_seg_ids}")
                        for upstream_seg_id in local_seg_ids:
                            self.H_populated[downstream_seg_id] = 1
                            """
                            For good velocity solutions H should never exceed 1, but as we are optimising it can happen in the velocity solution is bad
                            Hence we are now setting a 0-1 cap on the H values
                            """
                            self.H_segment_start[downstream_seg_id] = self.H_segment_end[upstream_seg_id] * self.split_frac[downstream_seg_id] * self.fluxes_end[upstream_seg_id] / self.fluxes_start[downstream_seg_id]
                            if len(graph.adjacency_list[current_node]) == 2 and verbose:
                                self.logger.log(f"1-1:{self.H_segment_start[downstream_seg_id]:.4g} = {self.H_segment_end[upstream_seg_id]:.4g} * {self.split_frac[downstream_seg_id]:.4g} * {self.fluxes_end[upstream_seg_id]:.4g} / {self.fluxes_start[downstream_seg_id]:.4g}")
                            self.H_segment_end[downstream_seg_id] = self.H_segment_start[downstream_seg_id] * self.fluxes_start[downstream_seg_id] / self.fluxes_end[downstream_seg_id]
                            # if len(graph.adjacency_list[current_node]) == 2:
                            #     self.logger.log(f"1-2:{self.H_segment_end[downstream_seg_id]:.4g} = {self.H_segment_start[downstream_seg_id]:.4g} * {self.fluxes_start[downstream_seg_id]:.4g} / {self.fluxes_end[downstream_seg_id]:.4g}")

                        if verbose:
                            self.logger.log(f"Segment {downstream_seg_id} populated with H = {self.H_segment_start[downstream_seg_id]:.2g} and {self.H_segment_end[downstream_seg_id]:.2g} via case 2 with split frac {self.split_frac[downstream_seg_id]:.2g}")
                            # self.logger.log(f"Upstream H {self.H_segment_end[upstream_seg_id]}, Split frac {self.split_frac[downstream_seg_id]}, Upstream Flux {self.fluxes_end[upstream_seg_id]}, Local Flux {self.fluxes_start[downstream_seg_id]}")

                # Case where the current node has 2 upstream vessels
                if graph.incoming_edges_count[current_node] == 2:
                    # Case where the current node has 1 downstream vessels
                    for neighbor in graph.adjacency_list[current_node]:
                        downstream_seg_id = graph.edge_list[current_node][neighbor]
                        local_seg_ids, _ = tree.get_segment_ids_on_node(current_node)
                        local_seg_ids.remove(downstream_seg_id)
                        self.H_segment_start[downstream_seg_id] = 0
                        string_1 = f"2-1:"
                        equation_string = []
                        for upstream_seg_id in local_seg_ids:
                            self.H_populated[downstream_seg_id] = 1
                            self.H_segment_start[downstream_seg_id] += self.H_segment_end[upstream_seg_id] * self.split_frac[downstream_seg_id] * self.fluxes_end[upstream_seg_id] / self.fluxes_start[downstream_seg_id]
                            equation_string.append(f"{self.H_segment_end[upstream_seg_id]:.4g} * {self.split_frac[downstream_seg_id]:.4g} * {self.fluxes_end[upstream_seg_id]:.4g} / {self.fluxes_start[downstream_seg_id]:.4g}")
                        string_2 = f"{self.H_segment_start[downstream_seg_id]:.4g} = "
                        final_string = string_1+string_2+" + ".join(equation_string)
                        if verbose:
                            self.logger.log(final_string)
                        self.H_segment_end[downstream_seg_id] = self.H_segment_start[downstream_seg_id] * self.fluxes_start[downstream_seg_id] / self.fluxes_end[downstream_seg_id]
                        # self.logger.log(f"2-2:{self.H_segment_end[downstream_seg_id]:.4g} = {self.H_segment_start[downstream_seg_id]:.4g} * {self.fluxes_start[downstream_seg_id]:.4g} / {self.fluxes_end[downstream_seg_id]:.4g}")
                        if verbose:
                            self.logger.log(f"Segment {downstream_seg_id} populated with H = {self.H_segment_start[downstream_seg_id]} and {self.H_segment_end[downstream_seg_id]} via case 2")
                # Mark the node as visited
                visited.add(current_node)

                # Add unvisited neighbors to the stack with updated incoming edges count
                stack.extend((neighbor, graph.incoming_edges_count[neighbor] - 1) for neighbor in graph.adjacency_list[current_node] if neighbor not in visited)
                for neighbor in graph.adjacency_list[current_node]:
                    if neighbor in visited:
                        # self.logger.log(f"visited neighbor {neighbor}")
                        for i, (node_id, incoming_edges_remaining) in enumerate(buffer):
                            if node_id == neighbor:
                                # Decrement the incoming_edges_remaining by 1
                                # self.logger.log("Process to decrement buffer")
                                buffer[i] = (node_id, incoming_edges_remaining - 1)
                                break  # Stop searching once the ID is found
            else:
                # Push the current node back onto the stack with decreased incoming edges count
                buffer.append((current_node, incoming_edges_remaining))
                # Mark the node as visited
                visited.add(current_node)

            # self.logger.log(f"buffer = {buffer}")

            # Check if any two entries in the stack are the same
            i = 0
            while i < len(stack):
                j = i + 1
                while j < len(stack):
                    if stack[i] == stack[j]:
                        # self.logger.log("Matching stack entries.")
                        a, b = stack.pop(j)
                        stack[i] = (a, b - 1)
                        # After popping, do not increment j, because elements have shifted
                    else:
                        j += 1
                i += 1


            for i, (node_id, incoming_edges_remaining) in enumerate(buffer):
                if incoming_edges_remaining == 0:
                    # pop the id and edges in no more incoming info needed
                    node_id, incoming_edges_remaining = buffer.pop(i)
                    stack.extend([(node_id, incoming_edges_remaining)])

        # Cap the haematocrit values if they are outside the reasonable range of 0-1
        self.capped_haematocrit = False
        for segment_id, _ in enumerate(self.H_segment_start):
            if self.H_segment_start[segment_id] > 1:
                self.capped_haematocrit = True
                # self.logger.log(f"Haematocrit capped from {self.H_segment_start[segment_id]} to 1 at the start of segment {segment_id}")
                self.H_segment_start[segment_id] = 0.99
            if self.H_segment_end[segment_id] > 1:
                self.capped_haematocrit = True
                # self.logger.log(f"Haematocrit capped from {self.H_segment_end[segment_id]} to 1 at the end of segment {segment_id}")
                self.H_segment_end[segment_id] = 0.99

        if verbose:
            self.logger.log(f"##################### END H DFS #####################")
        return

    def _populate_haematocrit_recursive(self,tree:Tree,seg_id:int,starting_node:int,verbose=False):
        node_1_id = tree.segment_dict[seg_id].node_1_id
        node_2_id = tree.segment_dict[seg_id].node_2_id

        if node_1_id == starting_node:
            next_node = node_2_id
        elif node_2_id == starting_node:
            next_node = node_1_id
        else:
            raise ValueError(f"Starting Node for recursion ({starting_node}) does not match available node ({node_1_id},{node_2_id})")

        if self.num_segments_outof_node[next_node] >= self.num_segments_into_node[next_node]:
            next_segments, _ = tree.get_segment_ids_on_node(next_node)
            next_segments.remove(seg_id)
            for next_segment in next_segments:
                self.H_populated[next_segment] = 1
                self.H_segment_start[next_segment] = self.H_segment_end[seg_id] * self.split_frac[next_segment] * self.fluxes_end[seg_id] / self.fluxes_start[next_segment]
                self.H_segment_end[next_segment] = self.H_segment_start[next_segment] * self.fluxes_start[next_segment] / self.fluxes_end[next_segment]
                if verbose:
                    self.logger.log(f"Segment {next_segment} populated with H_value through Recursion: {self.H_segment_start[next_segment]}")
                self._populate_haematocrit_recursive(tree,next_segment,next_node,verbose)

        return
    
    # @staticmethod
    def _fractional_Erythrocytes(self, FQB, D_f, D_1, D_2, H_f):
        scale = (1-H_f)/D_f
        A = -13.29*((((D_1)**2 / (D_2)**2) - 1) / (((D_1)**2 / (D_2)**2) + 1) ) * scale
        B = 1 + 6.98 * scale
        X0 = 0.964 * scale

        if FQB <= X0:
            FQE = 0
        elif X0 < FQB and FQB < (1-X0):
            step_1 = (FQB - X0) / (1 - 2*X0)
            step_2 = A + B * logit(step_1)
            step_3 = expit(step_2)
            FQE = step_3
        elif FQB >= (1-X0):
            FQE = 1
        else:
            FQE = 0.5
            self.logger.log(f"Error in haematocrit distribution function, relative flow not numeric.")

        # self.logger.log(f"FQE={FQE:.4g}, FQB={FQB:.4g}, A={A:.4g}, B={B:.4g}, X0={X0:.4g}")
        
        return FQE


    def make_oxygen_elements(self, mesh1D):
        self.o_element = gf.MeshFem(mesh1D)
        self.o_element.set_fem(gf.Fem('FEM_PK(1,1)'))
        self.o_dofs = mesh1D.nbpts()

        return self.o_element

    def build_monolithic_oxygen(self,tree:Tree,mesh1D,u_solution:Dict,h_solution:Dict,o_solution:np.ndarray=None, output="Spmat",verbose=False) -> Tuple[gf.Spmat, np.ndarray, gf.MeshFem, np.ndarray]:
        if not (output == "Spmat"):
            raise ValueError("Supported output strings are only 'Spmat'")

        # Check if a u_solution was supplied
        if u_solution == None:
            raise ValueError("You must pass a velocity solution to this function")

        # Check if a h_solution was supplied
        if h_solution == None:
            raise ValueError("You must pass a haematocrit solution to this function")

        # Check if v_elements exist
        if type(self.v_element_dict) != dict:
            raise ValueError("This function can only be called after build_monolithic_vessel as it requires some internal components")

        # Check if h_elements exist
        if type(self.h_element_dict) != dict:
            raise ValueError("This function can only be called after build_monolithic_haematocrit as it requires some internal components")
        
        # Create the o_elements if necessary
        if not hasattr(self, 'o_elements'):
            self.make_oxygen_elements(mesh1D)

        # check if h_solution was supplied and set it to initial is not
        if o_solution is None:
            o_solution = np.full(self.o_dofs,self.solubility_O2 * self.P_O2_in)
        
        # Construct the monolithic matrix system
        o_dofs = self.o_dofs
        O_MAT = gf.Spmat("empty", o_dofs, o_dofs)
        O_VEC = np.zeros(o_dofs)
        if self.recalculate_O2_matrix is True:
            self.Static_O2_Mat = gf.Spmat("empty", o_dofs, o_dofs)
            self.Static_O2_VEC = np.zeros(o_dofs)
        x = np.zeros(o_dofs)

        # Iterate through each segment to obtain its local self contribution
        for keys in tree.segment_dict.keys():
            uv_i_sol = u_solution[keys+1]
            h_i_sol = h_solution[keys+1]
            if self.recalculate_O2_matrix is True:
                Dovi = self._build_Dovi(tree, keys)
                self.Static_O2_Mat.add(range(0,o_dofs),range(0,o_dofs),Dovi)

            
            Aovi = self._build_Aovi(tree, keys, uv_i_sol, h_i_sol, o_solution)
            O_MAT.add(range(0,o_dofs),range(0,o_dofs),Aovi)
            # Add the Dovi and Aovi component to the monolithic matrix
        # This is the EXPERIMENTAL DECATING DIFFUSIVITY
        DDovi = self._build_DecayingDovi()
        O_MAT.add(range(0,o_dofs),range(0,o_dofs),DDovi) 
           
        if self.recalculate_O2_matrix is True:
            Fo_vector = np.zeros(o_dofs)
            Oo_empty = gf.Spmat("empty", o_dofs, o_dofs)
            Oo, Fo_vector, dirichlet_rows = self._build_o_boundary(tree,Oo_empty,Fo_vector,u_solution)

            self.Static_O2_Mat.add(range(0,o_dofs),range(0,o_dofs), Oo)
            self.Static_O2_VEC += Fo_vector

            # Apply dirichlet rows
            for rows in dirichlet_rows:
                self.Static_O2_Mat.clear(rows,range(0,o_dofs))
                self.Static_O2_Mat.add(rows,rows,1)
                if verbose:
                    self.logger.log(f"Applied dirichlet condition to row: {rows}")

        O_MAT.add(range(0,o_dofs),range(0,o_dofs),self.Static_O2_Mat)
        O_VEC = self.Static_O2_VEC
        ox = np.empty(o_dofs)

        self.recalculate_O2_matrix = False
        
        return O_MAT, O_VEC, self.o_element, x
    
    def set_decaying_diffusivity(self,error_value,iteration):
        """
        This function updates the value of the error dependent decaying diffusivity term for the stabilisation of the oxygen problem

        :param error_value: A float indicating the current error value of the oxygen problem.
        :param error_value: A int indicating the current iteration of the oxygen problem.
        """
        D0 = 1e-8
        # alpha controls the iteration dependent decay rate
        alpha = 3e-2
        # beta controls the error dependent decay rate, at 6.9e-7: Dk ~= 1/1000 D0 at error = 1e-7
        beta = 6.9e-5
        self.decaying_diffusivity = D0 * np.exp(-(beta/error_value+alpha*iteration))
        return

    def _build_Dovi(self, tree:Tree, segment_id:int) -> gf.Spmat:
        """         # Establish parameters specific to the oxygen system
        diffusivity_O2_v = self.diffusivity_O2

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("o", self.o_element)

        model.add_initialized_data("diffusivity_O2_v", diffusivity_O2_v)
        model.add_initialized_data("area",tree.segment_dict[segment_id].area())

        # expression = f"area*diffusivity_O2_v*Grad_o.Grad_Test_o"
        Dovi = model.add_linear_term(self.mim, "area*diffusivity_O2_v*Grad_o.Grad_Test_o", segment_id+1)

        model.assembly("build_matrix")
        Dov_mat = model.tangent_matrix() """

        area = tree.segment_dict[segment_id].area()
        scale_term = area*self.diffusivity_O2
        dof_total = self.o_dofs
        scale = np.ones(dof_total)
        scale = scale*scale_term
        Dovi = gf.asm_laplacian(self.mim, self.o_element, self.o_element, scale, segment_id+1)

        return Dovi
    
    def _build_DecayingDovi(self) -> gf.Spmat:
        """         # Establish parameters specific to the oxygen system
        diffusivity_O2_v = self.diffusivity_O2

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("o", self.o_element)

        model.add_initialized_data("diffusivity_O2_v", diffusivity_O2_v)
        model.add_initialized_data("area",tree.segment_dict[segment_id].area())

        # expression = f"area*diffusivity_O2_v*Grad_o.Grad_Test_o"
        Dovi = model.add_linear_term(self.mim, "area*diffusivity_O2_v*Grad_o.Grad_Test_o", segment_id+1)

        model.assembly("build_matrix")
        Dov_mat = model.tangent_matrix() """

        area = np.pi*(6e-6)**2
        scale_term = area*self.decaying_diffusivity
        dof_total = self.o_dofs
        scale = np.ones(dof_total)
        scale = scale*scale_term
        Dovi = gf.asm_laplacian(self.mim, self.o_element, self.o_element, scale)

        return Dovi

    def _build_Aovi(self, tree:Tree, segment_id:int, u_solution:Dict, h_solution:Dict, o_solution:np.ndarray=None) -> gf.Spmat:
        # Establish parameters specific to the oxygen system
        kappa1 = self.kappa1
        solubility_O2 = self.solubility_O2
        hill_exponent = self.hill_exponent
        kappa2 = self.kappa2
        lx, ly, lz = tree.get_tangent_versor(segment_id)

        # Establish the model and assign variables and data to the model
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("o", self.o_element)
        model.add_fem_variable("u", self.v_element_dict[segment_id+1])
        model.add_fem_variable("h", self.h_element_dict[segment_id+1])
        model.add_fem_variable("O_k", self.o_element)

        model.add_initialized_data("radius", tree.segment_dict[segment_id].radius)
        model.add_initialized_data("area", tree.segment_dict[segment_id].area())
        model.add_initialized_data("kappa1", [kappa1])
        model.add_initialized_data("solubility_O2", [solubility_O2])
        model.add_initialized_data("hill_exponent", [hill_exponent])
        model.add_initialized_data("hill_minus", [hill_exponent-1])
        model.add_initialized_data("kappa2", [kappa2])
        model.add_initialized_data("one", [1])
        model.add_initialized_data("lx", [lx])
        model.add_initialized_data("ly", [ly])
        model.add_initialized_data("lz", [lz])

        model.set_variable("u", u_solution)
        model.set_variable("h", h_solution)
        model.set_variable("O_k", o_solution)

        mod_vel_string = "kappa1*h*((pow(O_k,hill_minus))/(pow(O_k,hill_exponent)+kappa2))"
        do_ds = "(Grad_o(1).[lx,ly,lz](1)+Grad_o(2).[lx,ly,lz](2)+Grad_o(3).[lx,ly,lz](3))"
        du_ds = "(Grad_u(1).[lx,ly,lz](1)+Grad_u(2).[lx,ly,lz](2)+Grad_u(3).[lx,ly,lz](3))"
        Aov1_string = f"area*u*(one+{mod_vel_string})*{do_ds}.Test_o"
        Aov2_string = f"area*{du_ds}*(one+{mod_vel_string})*o.Test_o"

        Aov1 = model.add_nonlinear_term(self.mim, Aov1_string,segment_id+1)
        Aov2 = model.add_nonlinear_term(self.mim, Aov2_string,segment_id+1)

        model.disable_variable("u")
        model.disable_variable("h")
        model.disable_variable("O_k")

        model.assembly("build_matrix")
        Aovi_mat = model.tangent_matrix()
        return Aovi_mat

    def _build_o_boundary(self, tree:Tree, Mat:gf.Spmat, Vec:np.ndarray, u_solution:Dict, verbose=False) -> Tuple[gf.Spmat, np.ndarray,List[int]]:
        solubility_O2 = self.solubility_O2
        P_O2_in = self.P_O2_in
        C_O2_in = solubility_O2 * P_O2_in
        beta_ov = self.beta_ov
        P_O2_out = self.P_O2_out
        C_O2_out = solubility_O2 * P_O2_out
        dirichlet_rows = []
        
        # loop through all inlet boundaries
        for inlet_pid in self.inlet.keys():
            region = self.inlet[inlet_pid][0]
            if verbose:
                self.logger.log(f"inlet boundary reigon = {region}")

            branch = self.inlet[inlet_pid][1]
            area_i = tree.segment_dict[branch-1].area()

            model=gf.Model('real') # real or complex space.
            model.add_fem_variable("o", self.o_element)
            model.add_initialized_data("C_O2_in", [C_O2_in])
            model.add_initialized_data("C_O2_in2", [-C_O2_in])

            cvfid = self.v_element_dict[branch].mesh().region(region)
            pid_in_face = self.v_element_dict[branch].mesh().pid_in_faces(cvfid)
            points_on_cvid = self.v_element_dict[branch].mesh().pid_in_cvids(cvfid[0])
            if verbose:
                self.logger.log(f"Inlet face = {self.v_element_dict[branch].mesh().region(region)[1]}")
                self.logger.log(f"PID on face = {pid_in_face}")
                self.logger.log(f"PID on CVID = {points_on_cvid}")

            condition = points_on_cvid[cvfid[1]] == pid_in_face
            if verbose:
                self.logger.log(f"Conditional = {condition[0]}")

            # Orientate the source term based on the presumed direction
            if self.v_element_dict[branch].mesh().region(region)[1] == 1:
                model.add_Dirichlet_condition_with_simplification("o",region,"C_O2_in")
            elif self.v_element_dict[branch].mesh().region(region)[1] == 0:
                model.add_Dirichlet_condition_with_simplification("o",region,"C_O2_in2")
            else:
                raise ValueError("Unexpected face in 1D boundary assignment")

            model.assembly()

            F_inlet_i = model.rhs()
            Vec += F_inlet_i
            dirichlet_rows.append(np.nonzero(F_inlet_i))

        # loop through all outlet boundaries
        for outlet_pid in self.outlet.keys():
            region = self.outlet[outlet_pid][0]
            if verbose:
                self.logger.log(f"outlet boundary reigon = {region}")

            branch = self.outlet[outlet_pid][1]
            area_i = tree.segment_dict[branch-1].area()

            model=gf.Model('real') # real or complex space.
            model.add_fem_variable("o", self.o_element)
            model.add_initialized_data("area", [area_i])
            model.add_initialized_data("betav", [beta_ov])
            model.add_initialized_data("C_O2_out", [C_O2_out])

            # Orientate the source term based on the presumed direction
            if self.v_element_dict[branch].mesh().region(region)[1] == 1:
                Aov_beta = model.add_linear_term(self.mim, "-area*betav*o.Test_o", region)
                Fov_beta = model.add_source_term_brick(self.mim, "o", "-area*betav*C_O2_out", region)
            elif self.v_element_dict[branch].mesh().region(region)[1] == 0:
                Aov_beta = model.add_linear_term(self.mim, "area*betav*o.Test_o", region)
                Fov_beta = model.add_source_term_brick(self.mim, "o", "area*betav*C_O2_out", region)
            else:
                raise ValueError("Unexpected face in 1D boundary assignment")
       
            model.assembly()

            Aov_beta_i = model.tangent_matrix()
            Fov_beta_i = model.rhs()

            dof = self.o_dofs
            Mat.add(range(0,dof),range(0,dof), Aov_beta_i)
            Vec += Fov_beta_i

        return Mat, Vec, dirichlet_rows

    def build_Reynolds_elements(self, mesh): 
        re_element = gf.MeshFem(mesh)
        re_element.set_fem(gf.Fem('FEM_PK(1,1)'))
        self.re_dofs = mesh.nbpts()
        return re_element

    def post_process_Reynolds(self, tree:Tree, mesh1D, u_solution:Dict):
        self.re_element = self.build_Reynolds_elements(mesh1D)
        re_dofs = self.re_dofs

        Re_MAT = gf.Spmat("empty", re_dofs, re_dofs)
        Re_VEC = np.zeros(re_dofs)

        for keys in tree.segment_dict.keys():
            Re_partial_MAT, Re_partial_vec = self._build_Reynolds_solution_partial(tree, u_solution, keys)
            # Add the Mvvi component to the monolithic matrix
            Re_MAT.add(range(0,re_dofs),range(0,re_dofs),Re_partial_MAT)
            Re_VEC += Re_partial_vec

        return Re_MAT, Re_VEC, self.re_element

    def _build_Reynolds_solution_partial(self, tree:Tree, u_solution:Dict, segment_id:int):
        L = tree.length(segment_id)
        u_sol = u_solution[segment_id+1]
        viscosity_sol = self.fluid_viscosity[segment_id+1]
        fluid_density = 1000 # kg/m3 = 1 g/mL (blood density ~= water density)

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("re", self.re_element)

        model.add_fem_variable("u", self.v_element_dict[segment_id+1])
        model.set_variable("u", u_sol)
        model.disable_variable("u")

        model.add_fem_variable("mu_v", self.h_element_dict[segment_id+1])
        model.set_variable("mu_v", viscosity_sol)
        model.disable_variable("mu_v")

        model.add_initialized_data("L", [L])
        model.add_initialized_data("rho", [fluid_density])
        
        model.add_linear_term(self.mim, "re.Test_re", segment_id+1)
        model.add_source_term(self.mim, "((rho*u*L)/(mu_v))", segment_id+1)

        model.assembly()
        RE_MAT = model.tangent_matrix()
        RE_VEC = model.rhs()

        return RE_MAT, RE_VEC

    def build_wss_elements(self, mesh): 
        wss_element = gf.MeshFem(mesh)
        wss_element.set_fem(gf.Fem('FEM_PK(1,1)'))
        self.wss_dofs = mesh.nbpts()
        return wss_element

    def post_process_wss(self, tree:Tree, mesh1D, u_solution:Dict):
        self.wss_element = self.build_wss_elements(mesh1D)
        wss_dofs = self.wss_dofs

        wss_MAT = gf.Spmat("empty", wss_dofs, wss_dofs)
        wss_VEC = np.zeros(wss_dofs)

        for keys in tree.segment_dict.keys():
            wss_partial_MAT, wss_partial_vec = self._build_wss_matrix(tree, u_solution, keys)
            # Add the Mvvi component to the monolithic matrix
            wss_MAT.add(range(0,wss_dofs),range(0,wss_dofs),wss_partial_MAT)
            wss_VEC += wss_partial_vec

        return wss_MAT, wss_VEC, self.wss_element

    def _build_wss_matrix(self, tree:Tree, u_solution:Dict, segment_id:int):
        R = tree.segment_dict[segment_id].radius

        u_sol = u_solution[segment_id+1]
        viscosity_sol = self.fluid_viscosity[segment_id+1]

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("wss", self.wss_element)

        model.add_fem_variable("u", self.v_element_dict[segment_id+1])
        model.set_variable("u", u_sol)
        model.disable_variable("u")

        model.add_fem_variable("mu_v", self.h_element_dict[segment_id+1])
        model.set_variable("mu_v", viscosity_sol)
        model.disable_variable("mu_v")

        model.add_initialized_data("R", [R])
        model.add_initialized_data("two", [2])

        model.add_linear_term(self.mim, "wss.Test_wss", segment_id+1)
        model.add_source_term(self.mim, "((two*mu_v*u)/(R))", segment_id+1)

        model.assembly()
        WSS_MAT = model.tangent_matrix()
        WSS_VEC = model.rhs()

        return WSS_MAT, WSS_VEC

    def save_post_process_vtk(self,save_string:str,run_string:str,wssfem:gf.MeshFem,wssx:np.ndarray,refem:gf.MeshFem,rex:np.ndarray):
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("wss", wssfem)
        model.add_fem_variable("re", refem)

        model.set_variable("wss",wssx)
        model.set_variable("re",rex)

        wssfem.export_to_vtk(save_string+"/Post_Process_Quantities_"+run_string+".vtk",\
            wssfem,model.variable("wss"),"Wall Shear Stress", refem,model.variable("re"),"Reynolds Number")

        return
    
    def save_haemodynamic_1D(self,save_string:str,run_string:str,uvfem_dict:Dict,uvx:Dict,pvfem:gf.MeshFem,pvx:np.ndarray,hvfem_dict:Dict,hx:Dict,iteration:Union[int,None]=None):
        if not iteration is None:
            run_string = f"{run_string}_haemodynamic_{iteration}"

        model=gf.Model('real') # real or complex space.

        for keys in uvfem_dict:
            uvfem = uvfem_dict[keys]
            string = f"uv{keys}"
            model.add_fem_variable(string, uvfem)
            model.set_variable(string,uvx[keys])

        model.add_fem_variable("pv", pvfem)
        model.set_variable("pv",pvx)

        for keys in hvfem_dict:
            hvfem = hvfem_dict[keys]
            string = f"hv{keys}"
            model.add_fem_variable(string, hvfem)
            model.set_variable(string,hx[keys])

        for keys in uvfem_dict:
            uvfem = uvfem_dict[keys]
            hvfem = hvfem_dict[keys]
            string = f"{keys}"
            string1 = f"uv{keys}"
            string2 = f"hv{keys}"
            uvfem.export_to_vtk(save_string+"/segment_results/velocity_and_haematocrit_branch_"+string+"_"+run_string+".vtk",uvfem, model.variable(string1), "Velocity Vessel",hvfem, model.variable(string2), "Haematocrit Vessel")
        
        pvfem.export_to_vtk(save_string+"/tree_results/full_vessel_pressure_and_oxygen_"+run_string+".vtk",pvfem, model.variable("pv"), "Pressure Vessel")


    def __str__(self):
        return f"""
        1D Settings:
          Inlet Pressure: {self.pressure_inlet}
          Outlet Pressure: {self.pressure_outlet}
          Inlet Haematocrit: {self.haematocrit_inlet}
          Inlet Haematocrit: {self.wall_hydraulic_conductivity}
          Cells per segment: {self.num_cells}
        """ 

class GetFEMHandler3D(Tissue,Tree):
    """
    A class used to handle the Fenics implementation for the generation of segment specific submatricies.

    ----------
    Class Attributes
    ----------
    None

    ----------
    Instance Attributes
    ----------
    corner_1 : [int,int,int]
        An array of ints specifying the location of base corner of the 3D tissue mesh

    corner_2 : [int,int,int]
        An array of ints specifying the location of the opposite corner of the 3D tissue mesh

    num_cells : [int,int,int]
        An array of ints specifying the number of cells in each of the XYZ directions in the mesh

    pressure_boundary : int
        An integer stating the pressure boundary condition at the inlet  

    ----------
    Class Methods
    ----------  
    load_config(filename)
        Returns a class instance from file

    ----------
    Instance Methods
    ----------  
    haemodynamic_submatrix(segment, node_list[segment.node_1_id], node_list[segment.node_2_id], h_solution_partial)
        Returns a segment specific submatrix and subvector for the haemodynamics (Pressure and Velocity)

    haematocrit_submatrix(segment, node_list[segment.node_1_id], node_list[segment.node_2_id], u_solution_partial)
        Returns a segment specific submatrix and subvector for the Haematocrit dynamics

    oxygen_submatrix(segment, node_list[segment.node_1_id], node_list[segment.node_2_id], h_solution_partial, u_solution_partial)
        Returns a segment specific submatrix and subvector for the Oxygen dymanics

    """

    def __init__(self, distant_pressure:float, boundary_conductivity:float, tissue_hydraulic_conductivity:float, interstitial_viscosity:float, wall_hydraulic_conductivity:float, oncotic_gradient:float):
        self.distant_pressure = distant_pressure
        self.boundary_conductivity = boundary_conductivity
        self.tissue_hydraulic_conductivity = tissue_hydraulic_conductivity
        self.interstitial_viscosity = interstitial_viscosity
        self.wall_hydraulic_conductivity = wall_hydraulic_conductivity
        self.oncotic_gradient = oncotic_gradient
        self.recalculate_O2_matrix = True
        

    @classmethod
    def load_config(cls, config:Config):
        data = config.config_access["3D_CONDITIONS"]
        object = cls(data["distant_pressure"],data["boundary_conductivity"],data["tissue_hydraulic_conductivity"],data["interstitial_viscosity"],data["wall_hydraulic_conductivity"],data["oncotic_gradient"])
        object.tissue_diffusion_coefficient_O2 = data["tissue_diffusion_coefficient_O2"] # type: ignore
        object.consumption_rate_O2 = data["consumption_rate_O2"] # type: ignore
        object.michaelis_menten_constant = data["michaelis_menten_constant"] # type: ignore
        object.tissue_solubility_O2 = data["tissue_solubility_O2"] # type: ignore
        object.beta_ot = data["beta_ot"] # type: ignore
        object.far_field_O2 = data["far_field_O2"] # type: ignore
        object.wall_O2_permeability = data["wall_O2_permeability"] # type: ignore
        object.O2_reflection = data["O2_reflection"] # type: ignore
        object.logger = config.logger # type: ignore
        return object

    @staticmethod
    def _make_mesh(tissue:Tissue, cylinder=False, mesh_name="tissue.msh",rotation=False) -> gf.Mesh:
        if cylinder:
            mo = gf.MesherObject('cylinder', [5,5,0], [0,0,1], [10], [5])
            h = 10/tissue.num_cells[2] # type: ignore
            mesh3D = gf.Mesh('generate', mo, h, 2)

            # Define the scaling factors for each coordinate axis
            scale_x = 1e-5
            scale_y = 1e-5
            scale_z = 1e-5

            # Get the mesh vertices
            vertices = mesh3D.pts()

            # Scale the vertices
            vertices[0,:] *= scale_x
            vertices[1,:] *= scale_y
            vertices[2,:] *= scale_z

            # Update the mesh vertices
            mesh3D.set_pts(vertices)

            mesh3D.save(mesh_name)
        else:
            bottom_corner = tissue.bottom_corner()
            top_corner = tissue.top_corner()

            x0,y0,z0 = bottom_corner
            x1,y1,z1 = top_corner

            #position test
            # x0 = x0+3
            # x1 = x1+3

            nx,ny,nz = tissue.num_cells # type: ignore

            mesh3D = gf.Mesh("regular simplices", np.arange(x0,x1+(x1-x0)/nx,(x1-x0)/nx), np.arange(y0,y1+(y1-y0)/ny,(y1-y0)/ny), np.arange(z0,z1+(z1-z0)/nz,(z1-z0)/nz))
            mesh3D.save(mesh_name)

            if rotation:
                rotation_angle = 3* np.pi / 2# 90 degrees
                c = np.cos(rotation_angle)
                s = np.sin(rotation_angle)
                rotation_matrix = np.array([[c, 0, s],
                                            [0, 1, 0],
                                            [-s, 0, c]])

                translation_vector = np.array([x1/2,y1/2,z1/2])

                # translate the matrix to be centred on the origin
                mesh3D.translate(-translation_vector)
                # rotate the matrix around the y axis
                mesh3D.transform(rotation_matrix)
                # move the matrix back off the origin
                mesh3D.translate(translation_vector)
                mesh3D.save(mesh_name)

        return mesh3D

    # @staticmethod
    # def make_cylinder(cell_number:int):
    #     # Initialize Gmsh
    #     gmsh.initialize()
    #     gmsh.option.setNumber("General.Verbosity", 0)

    #     # Create a new model
    #     gmsh.model.add("cylinder")

    #     # Define parameters
    #     radius = 5e-5
    #     height = 10e-5
    #     center = 5e-5
    #     side_number = 40
    #     cell_size = height / cell_number

    #     # Generate circle points
    #     angles = np.linspace(0, 2 * np.pi, side_number, endpoint=False)
    #     x = radius * np.cos(angles) + center
    #     y = radius * np.sin(angles) + center

    #     # Add points to the model
    #     points = [gmsh.model.geo.addPoint(x[i], y[i], 0) for i in range(side_number)]

    #     # Add lines to form the circular loop
    #     lines = [gmsh.model.geo.addLine(points[i], points[(i + 1) % side_number]) for i in range(side_number)]

    #     # Create the curve loop and surface
    #     curve_loop = gmsh.model.geo.addCurveLoop(lines)
    #     surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    #     gmsh.model.geo.synchronize()

    #     # Make Spoke like structure
    #     p0 = gmsh.model.geo.addPoint(center, center, 0)
    #     lines2 = [gmsh.model.geo.addLine(p0, points[i]) for i in range(side_number)]
    #     gmsh.model.geo.synchronize()
    #     gmsh.model.mesh.embed(1, lines2, 2, surface)

    #     # Extrude the surface to create the 3D cylinder
    #     gmsh.model.geo.extrude([(2, surface)], 0, 0, height, numElements=[50], recombine=False)

    #     # Set mesh size on the curved surface
    #     gmsh.model.geo.synchronize()

    #     field_radius = gmsh.model.mesh.field.add("MathEval")
    #     gmsh.model.mesh.field.setString(field_radius, "F", "sqrt((x-5e-5)^2 + (y-5e-5)^2)")

    #     # Set the mesh field as a background mesh
    #     resolution = cell_size
    #     threshold = gmsh.model.mesh.field.add("Threshold")
    #     gmsh.model.mesh.field.setNumber(threshold, "IField", field_radius)
    #     gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
    #     gmsh.model.mesh.field.setNumber(threshold, "LcMax", 25*resolution)
    #     gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.1*radius)
    #     gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.5*radius)
    #     gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

    #     gmsh.model.geo.synchronize()
    #     gmsh.option.setNumber("Mesh.RecombineAll", 0)

    #     # Generate the mesh
    #     gmsh.model.mesh.generate(3)

    #     # Save the mesh to a file
    #     gmsh.write("structured.msh")

    #     # Finalize Gmsh
    #     gmsh.finalize()

    #     # Suppress getfem text outputs
    #     gf.util_trace_level(level=0)
    #     gf.util_warning_level(0)
    #     mesh = gf.Mesh('import', "gmsh", "structured.msh")
    #     mesh.save("structured2.msh")

    def make_haemodynamic_elements(self,mesh:gf.Mesh=None,mesh3D:gf.Mesh=None) -> Tuple[gf.MeshFem, gf.MeshFem, gf.MeshIm, gf.Mesh]:
        if mesh3D is None:
            mesh3D = gf.Mesh('load', mesh)

        # Establish FEM variables
        self.v_element = gf.MeshFem(mesh3D,3)
        self.p_element = gf.MeshFem(mesh3D,1)

        self.v_element.set_fem(gf.Fem('FEM_RT0(3)'))
        self.p_element.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))

        # Establish mesh integration measure      
        self.mim = gf.MeshIm(mesh3D, gf.Integ('IM_TETRAHEDRON(8)'))  

        self.ut_dofs = self.v_element.nbdof()
        self.pt_dofs = mesh3D.nbcvs()

        return self.p_element,self.v_element, self.mim, mesh3D

    def make_mesh(self, new_mesh=False, mesh="tissue.msh", tissue:Union[Tissue,None]=None, cylinder_test=False, verbose=False) -> gf.Mesh:
        # Make/Import mesh
        if new_mesh:
            if type(tissue) is Tissue:
                if verbose:
                    self.logger.log("Building new mesh") # type: ignore
                mesh3D = self._make_mesh(tissue,cylinder_test,mesh)  
            else: 
                raise ValueError(f"If generating new mesh you must supply a tissue object that indicates the bounds of the mesh.")  
        else:
            if verbose:
                self.logger.log("Loading Mesh...") # type: ignore
            mesh3D = gf.Mesh('load', mesh)
        
        return mesh3D

    def build_monolithic_tissue(self, tree:Union[Tree,None]=None, tissue:Union[Tissue,None]=None, mesh3D:gf.Mesh=None, boundary="ROBIN", p_data=None, cylinder_test=False,verbose=False,mem_handler=None) -> Tuple[gf.Spmat, np.ndarray, gf.MeshIm, gf.MeshFem, gf.MeshFem]:
        # self.memory_tracker = mem_handler
        # self.memory_tracker.print_mem("Starting 3D Haemodynamics")
        if not (boundary == "ROBIN" or boundary == "DIR"):
            raise ValueError("Supported boundary strings are only 'ROBIN' and 'DIR'")
        
        # Suppress getfem text outputs
        gf.util_trace_level(level=1)

        if mesh3D is None:
            self.make_mesh(True,tissue=tissue)

        if cylinder_test:
            tol = 1e-8
            boundary_region_bottom = mesh3D.outer_faces_with_direction([0, 0, -1], tol)
            boundary_region_top = mesh3D.outer_faces_with_direction([0, 0, 1], tol)
            boundary_region_side = mesh3D.outer_faces_with_direction([1, 0, 0], tol) + \
                                mesh3D.outer_faces_with_direction([-1, 0, 0], tol)

            mesh3D.set_region(1,boundary_region_side)
            mesh3D.set_region(2,boundary_region_top)
            mesh3D.set_region(3,boundary_region_bottom)
            tissue_boundaries = [1,2,3]
        else:
            # get face boundaries of the tissue
            cvfid1 = mesh3D.outer_faces_with_direction([0,0,-1],1)
            cvfid2 = mesh3D.outer_faces_with_direction([0,0,1],1)
            cvfid3 = mesh3D.outer_faces_with_direction([0,-1,0],1)
            cvfid4 = mesh3D.outer_faces_with_direction([0,1,0],1)
            cvfid5 = mesh3D.outer_faces_with_direction([-1,0,0],1)
            cvfid6 = mesh3D.outer_faces_with_direction([1,0,0],1)

            BOTTOM = mesh3D.set_region(1,cvfid1)
            TOP = mesh3D.set_region(2,cvfid2)
            LEFT = mesh3D.set_region(3,cvfid3)
            RIGHT = mesh3D.set_region(4,cvfid4)
            BACK = mesh3D.set_region(5,cvfid5)
            FRONT = mesh3D.set_region(6,cvfid6)
            
            tissue_boundaries = [1,2,3,4,5,6]
        # self.logger.log(mesh3D)
        # self.memory_tracker.print_mem("Post Boundary Regions")

        self.make_haemodynamic_elements(mesh3D=mesh3D)
        # self.memory_tracker.print_mem("Post FE creation")

        ut_dofs = self.ut_dofs
        pt_dofs = self.pt_dofs
        dof_total = ut_dofs + pt_dofs

        


        Full_MAT = gf.Spmat("empty", dof_total, dof_total)
        Full_VEC = np.zeros(dof_total)

        # self.memory_tracker.print_mem("Post MAT Initialization")

        Mtt = self._build_Mtt()
        # self.memory_tracker.print_mem("Post Mtt")
        Dtt = self._build_Dtt()
        # self.memory_tracker.print_mem("Post Dtt")

        Full_MAT.add(range(0,ut_dofs),range(0,ut_dofs), Mtt)

        Full_MAT.add(range(0,ut_dofs),range(ut_dofs,ut_dofs+pt_dofs), -Dtt)
        Dtt.transpose()
        Full_MAT.add(range(ut_dofs,ut_dofs+pt_dofs),range(0,ut_dofs),Dtt)
        # self.memory_tracker.print_mem("Post Full MAT")

        if boundary == "ROBIN":
            Mtt_aug, RHS = self._build_tissue_bc_robin(p_data, tissue_boundaries, cylinder_test)
            Full_MAT.add(range(0,ut_dofs),range(0,ut_dofs), Mtt_aug)
            Full_VEC[:ut_dofs] = RHS
        if boundary == "DIR":
            RHS = self._build_tissue_bc_dir(tissue_boundaries,cylinder_test)
            Full_VEC[:ut_dofs] = RHS

        # if cylinder_test:
        #     identity = gf.Spmat("identity", 1)

        #     mask = np.abs(Full_VEC[:ut_dofs]) > 1e-6
        #     pos = np.where(mask)[0]

        #     Full_MAT.clear(pos, range(0, ut_dofs))
        #     Full_MAT.add(pos, pos, identity)
        #     Full_VEC[pos] = 0

        return Full_MAT, Full_VEC, self.mim, self.p_element, self.v_element#A, b, ux, px, domain, x, offset

    def _build_Mtt(self) -> gf.Spmat:
        scale_term = self.interstitial_viscosity/self.tissue_hydraulic_conductivity
        Mtt = gf.asm_mass_matrix(self.mim, self.v_element, self.v_element)
        Mtt.scale(scale_term)
        #Mtt.display()
        
        return Mtt

    def _build_Dtt(self) -> gf.Spmat:
        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("p", self.p_element)
        model.add_fem_variable("u", self.v_element)

        pv_expression = "p.Div_Test_u"
        Dtt = gf.asm_generic(self.mim, 2, pv_expression, -1, model, 'select_output', 'u','p')
        #Dtt.display()

        return Dtt

    def _build_tissue_bc_dir(self, tissue_boundaries:List[int], cylinder_test=False) -> np.ndarray:
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("u", self.v_element)

        for boundary in tissue_boundaries:
            val = self.distant_pressure*133.32

            p_string = f"p_{boundary}"
            model.add_initialized_data(p_string, [val])
            source_string = f"-{p_string}.(Test_u.Normal)"
            model.add_source_term(self.mim, source_string, boundary)

        model.assembly("build_rhs")
        RHS = model.rhs()
        #self.logger.log(RHS.size)
        return RHS

    def _build_tissue_bc_robin(self, p_data:np.ndarray, tissue_boundaries:List[int],cylinder_test=False) -> Tuple[gf.Spmat,np.ndarray]:
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("u", self.v_element)
        model.add_fem_variable("p_t", self.p_element)

        # assign the p_data to the p_t variable
        if p_data is None:
            p_shape = model.variable("p_t")
            p_data = np.full_like(p_shape,1e-6)

        model.set_variable("p_t", p_data)

        #beta = 2.22e-6
        beta = self.boundary_conductivity
        model.add_initialized_data("beta", [beta])

        if cylinder_test:
            model.add_initialized_data("beta2", [beta*1e-6])

        for boundary in tissue_boundaries:
            val = self.distant_pressure*133.32

            p_string = f"p_{boundary}"
            model.add_initialized_data(p_string, [val])

            if cylinder_test and boundary != 1:
                source_string = f"-{p_string}.(Test_u.Normal)"
                model.add_source_term(self.mim, source_string, boundary)
                model.add_source_term(self.mim, "p_t.(Test_u.Normal)", boundary)
                model.add_nonlinear_term(self.mim, "(1/beta2)*(u.Normal).(Test_u.Normal)", boundary)
            else:
                source_string = f"-{p_string}.(Test_u.Normal)"
                model.add_source_term(self.mim, source_string, boundary)
                model.add_source_term(self.mim, "p_t.(Test_u.Normal)", boundary)
                model.add_nonlinear_term(self.mim, "(1/beta)*(u.Normal).(Test_u.Normal)", boundary)

        model.disable_variable("p_t")
        model.assembly()
        Mtt_aug = model.tangent_matrix()
        RHS = model.rhs()
        #self.logger.log(RHS.size)
        return Mtt_aug, RHS

    def make_oxygen_elements(self,mesh3D:gf.Mesh) -> gf.MeshFem:
        self.o_element = gf.MeshFem(mesh3D,1)
        self.o_element.set_fem(gf.Fem('FEM_PK(3,1)'))
        self.ot_dofs = mesh3D.nbpts()

        return self.o_element

    def build_monolithic_oxygen_tissue(self, mesh3D:gf.Mesh, u_solution:np.ndarray=None, o_solution:np.ndarray=None, cylinder_test=False, verbose=False) -> Tuple[gf.Spmat, np.ndarray, gf.MeshIm, gf.MeshFem]:
        # Check if a u_solution was supplied
        if u_solution is None:
            raise ValueError("You must pass a velocity solution to this function")

        # Check if a velocity elements have been established
        if not hasattr(self, 'v_element'):
            raise ValueError("You must run GetFEMHandler3D.build_monolithic_tissue() before this function will operate properly")
        
        # Suppress getfem text outputs
        gf.util_trace_level(level=1)

        

        if cylinder_test:
            tol = 1e-8
            boundary_region_bottom = mesh3D.outer_faces_with_direction([0, 0, -1], tol)
            boundary_region_top = mesh3D.outer_faces_with_direction([0, 0, 1], tol)
            boundary_region_all = mesh3D.outer_faces()

            mesh3D.set_region(1,boundary_region_all)
            mesh3D.set_region(2,boundary_region_top)
            mesh3D.set_region(3,boundary_region_bottom)

            # remove the top and bottom from region 1 to get the sides
            mesh3D.region_subtract(1,2)
            mesh3D.region_subtract(1,3)

            tissue_boundaries = [1,2,3]
        else:
            # get face boundaries of the tissue
            cvfid1 = mesh3D.outer_faces_with_direction([0,0,-1],1)
            cvfid2 = mesh3D.outer_faces_with_direction([0,0,1],1)
            cvfid3 = mesh3D.outer_faces_with_direction([0,-1,0],1)
            cvfid4 = mesh3D.outer_faces_with_direction([0,1,0],1)
            cvfid5 = mesh3D.outer_faces_with_direction([-1,0,0],1)
            cvfid6 = mesh3D.outer_faces_with_direction([1,0,0],1)

            BOTTOM = mesh3D.set_region(1,cvfid1)
            TOP = mesh3D.set_region(2,cvfid2)
            LEFT = mesh3D.set_region(3,cvfid3)
            RIGHT = mesh3D.set_region(4,cvfid4)
            BACK = mesh3D.set_region(5,cvfid5)
            FRONT = mesh3D.set_region(6,cvfid6)
            
            tissue_boundaries = [1,2,3,4,5,6]

        self.make_oxygen_elements(mesh3D)
        dof_total = self.ot_dofs

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("o", self.o_element)
        # Create placeholder for o_solution in the tissue
        if o_solution is None:
            o_shape = model.variable("o")
            o_solution = np.zeros_like(o_shape)

        Full_MAT = gf.Spmat("empty", dof_total, dof_total)
        Full_VEC = np.zeros(dof_total)
        if self.recalculate_O2_matrix is True:
            self.Static_O2_MAT = gf.Spmat("empty", dof_total, dof_total)

            Dot = self._build_Dot()
            Aot = self._build_Aot(u_solution)
            self.Static_O2_MAT.add(range(0,dof_total),range(0,dof_total), Dot)
            # self.Static_O2_MAT.add(range(0,dof_total),range(0,dof_total), Aot)


        Dot_aug, RHS = self._build_oxygen_bc_robin(tissue_boundaries)
        Full_MAT.add(range(0,dof_total),range(0,dof_total), Dot_aug)
        Full_VEC += RHS

        Rot = self._build_Rot(o_solution)
        Full_MAT.add(range(0,dof_total),range(0,dof_total), Rot)

        Full_MAT.add(range(0,dof_total),range(0,dof_total), self.Static_O2_MAT)
        self.recalculate_O2_matrix = False
        # if cylinder_test:
        #     identity = gf.Spmat("identity", 1)

        #     mask = np.abs(Full_VEC[:ut_dofs]) > 1e-6
        #     pos = np.where(mask)[0]

        #     Full_MAT.clear(pos, range(0, ut_dofs))
        #     Full_MAT.add(pos, pos, identity)
        #     Full_VEC[pos] = 0

        return Full_MAT, Full_VEC, self.mim, self.o_element

    def _build_Dot(self) -> gf.Spmat:
        scale_term = self.tissue_diffusion_coefficient_O2 # type: ignore
        dof_total = self.ot_dofs
        scale = np.ones(dof_total)
        scale = scale*scale_term
        Dot = gf.asm_laplacian(self.mim, self.o_element, self.o_element, scale)
        
        # Dot = gf.asm_mass_matrix(self.mim, self.o_element, self.o_element)
        # Dot.scale(scale_term)

        return Dot

    def _build_Aot(self, u_solution:np.ndarray) -> gf.Spmat:
        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("o", self.o_element)
        model.add_fem_variable("u", self.v_element)

        model.set_variable("u", u_solution)

        Aot1 = model.add_nonlinear_term(self.mim, "u.Grad_o.Test_o")
        Aot2 = model.add_nonlinear_term(self.mim, "Div_u*o.Test_o")

        model.disable_variable("u")
        model.assembly("build_matrix")
        Aot = model.tangent_matrix()

        return Aot

    def _build_Rot(self, o_solution:np.ndarray) -> gf.Spmat:
        KM = self.tissue_solubility_O2*self.michaelis_menten_constant # type: ignore

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("o", self.o_element)
        model.add_fem_variable("O_k", self.o_element)

        #o_solution[o_solution < 0] = 0
        model.set_variable("O_k", o_solution)

        model.add_initialized_data("M0", [self.consumption_rate_O2]) # type: ignore
        model.add_initialized_data("KM", [KM])

        consumption_string = "((M0)/(O_k+KM))"
        Rot = model.add_nonlinear_term(self.mim, f"{consumption_string}*o.Test_o")

        model.disable_variable("O_k")
        model.assembly("build_matrix")
        Rot = model.tangent_matrix()

        return Rot

    def _build_oxygen_bc_robin(self, tissue_boundaries:List[int], test=False) -> Tuple[gf.Spmat,np.ndarray]:
        beta_ot = self.beta_ot # type: ignore
        beta_ot = 1e-6
        cot = self.far_field_O2 # type: ignore

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("o", self.o_element)

        model.add_initialized_data("beta_ot", [beta_ot])
        model.add_initialized_data("cot", [cot])

        model.add_initialized_data("override_beta", [1e-6])
        model.add_initialized_data("override_cot", [-100])
        model.add_initialized_data("override_cot2", [100])

        for boundary in tissue_boundaries:
            if boundary == 3 and test:
                model.add_source_term(self.mim, "-override_beta*override_cot.Test_o", boundary)
                model.add_linear_term(self.mim, "override_beta*o.Test_o", boundary)
            elif boundary == 4 and test:
                model.add_source_term(self.mim, "-override_beta*override_cot2.Test_o", boundary)
                model.add_linear_term(self.mim, "override_beta*o.Test_o", boundary)
            else:
                model.add_source_term(self.mim, "-beta_ot*cot.Test_o", boundary)
                model.add_linear_term(self.mim, "beta_ot*o.Test_o", boundary)

        model.assembly()
        LHS = model.tangent_matrix()
        RHS = model.rhs()

        return LHS, RHS

    def _build_auxillary_matricies(self, mesh1D:gf.Mesh, pv:gf.MeshFem, pt:gf.MeshFem, tree:Tree, tissue:Tissue, verbose=False) -> Tuple[gf.Spmat, gf.Spmat]:
        pv_dofs = mesh1D.nbpts()
        pt_dofs = self.pt_dofs
        gf.util_warning_level(level= 1)

        # build the interpolation matrix projecting:  pt -> pv
        Mlin = gf.asm_interpolation_matrix(pt, pv)
        Mlin.transpose()

        # preallocate the interpolation matrix projecting:  pt -> cylinder projection of pv
        Mbar = gf.Spmat("empty", pv_dofs, pt_dofs)
        """ 
        #define the number of discretisation point for 3D-1D interpolation
        N_int = 50

        # iterate through all points associated with all convexes
        point_vectors = {}
        for pid in mesh1D.pid():
            cvids_for_pid = mesh1D.cvid_from_pid(pid, share=True)
            if verbose:
                self.logger.log(f"################Iterating on pid: {pid:2}#################")
                self.logger.log(f"    pid on convexes: {cvids_for_pid}")
            normal_vectors = []
            radii = []
            for cvid in cvids_for_pid:
                pid0 = mesh1D.pid_from_cvid(cvid)[0][0]
                pid1 = mesh1D.pid_from_cvid(cvid)[0][1]

                nodes1 = mesh1D.pts(pid1)
                nodes0 = mesh1D.pts(pid0)

                if verbose:
                    self.logger.log(f"    Iterating convex: {cvid}")
                    self.logger.log(f"        pid0 = {pid0}")
                    self.logger.log(f"                {nodes0[0]}{nodes0[1]}{nodes0[2]}")
                    self.logger.log(f"        pid1 = {pid1}")
                    self.logger.log(f"                {nodes1[0]}{nodes1[1]}{nodes1[2]}")

                convex_vector = nodes1 - nodes0
                convex_vector = convex_vector / np.linalg.norm(convex_vector)
                # self.logger.log(convex_vector)
                normal_vectors.append(convex_vector)
                
                for region in mesh1D.regions():
                    if pid0 in mesh1D.pid_in_regions(region) and pid1 in mesh1D.pid_in_regions(region):
                        radius = tree.segment_dict[region-1].radius
                        radii.append(radius)
                        break
            
            # self.logger.log(f"radii = {radii}")
            final_vector = [[0],[0],[0]]
            for vectors in normal_vectors:
                final_vector += vectors
                
                
            point_vectors.update({pid:{"vec":final_vector,"radius":radius}})
            if verbose:
                self.logger.log(f"#####################################################")
        
        if verbose:
            self.logger.log(mesh1D)
    
       
        # iterate through all 1D dofs to get the projected cylinder points in the 3D
        counter = 0
        percent = 0
        point_data = []
        for i in range(pv_dofs):
            
            # construct orthonormal system v0, v1, v2
            # retrieve v0 and radius from previous step
            mesh_pid = mesh1D.pid_from_coords(pv.basic_dof_nodes(i))
            if verbose:
                self.logger.log(f"DOF id: {i:2}, Mesh id: {mesh_pid[0]:2}")
            v0 = point_vectors[mesh_pid[0]]["vec"]
            radius = point_vectors[mesh_pid[0]]["radius"]

            # generate orthogonal vectors of v0
            # Can use cross product to get third component
            v1 = [[0.0], -v0[2], v0[1]]
            v2 = [v0[1]*v1[2]-v0[2]*v1[1], v0[2]*v1[0]-v0[0]*v1[2],v0[0]*v1[1]-v0[1]*v1[0]]
            #[v0[1]*v0[1] + v0[2]*v0[2], -v0[0]*v0[1], -v0[0]*v0[2]]

            # correct for edge case where v1 and v2 are colinear 
            if np.linalg.norm(v2) < 1.0e-8 * np.linalg.norm(v0):
                v1 = np.array([-v0[1], v0[0], [0.0]])
                v2 = np.array([-v0[0]*v0[2], -v0[1]*v0[2], v0[0]*v0[0]+v0[1]*v0[1]])

            # normalize v1 and v2
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # build the circle orthogonal to the vector associated with the dof
            # plot this to check
            circle_points = np.empty((N_int,3))
            for j in range(N_int):
                pts = pv.basic_dof_nodes(i) + radius*(np.cos(2*np.pi*j/N_int)*v1 + np.sin(2*np.pi*j/N_int)*v2)
                #self.logger.log(pts)
                circle_points[j,:] = pts.squeeze()
                point_data.append(pts.squeeze())

                

            # It is possible for points on this cricle to go outside the borders of the mesh, 
            # here we restrict this by capping values to the max or min allowed by the geometry
            # Possibly the issue, plot to check, Throw the values away instead, only applicable for box domains.
            # xyz_min = tissue.bottom_corner()
            # xyz_max = tissue.top_corner()
            # circle_points[:] = np.clip(circle_points,xyz_min,xyz_max)

            # THE PROBLEM IS IN THIS CONSTRUCTION
            # Create the interpolation matrix associated with the circle
            Mbari = gf.asm_interpolation_matrix(pt, circle_points)
            Mbari = Mbari.full()
            # self.logger.log(f"Mbari shape = {Mbari.shape}")
            # self.logger.log(f"Mbari sum = {np.sum(Mbari)}")

            # Apply the interpolation matrix associated with the circle to the dofs of the 1D as an average.
            # when removing points from boundaries n_points will decrease
            for j in range(N_int):
                row = Mbari[j,:]
                sum_row = np.sum(row)
                for col_index, value in np.ndenumerate(row):
                    if value != 0:
                        # self.logger.log(f"Adding: {value/sum_row} to row: {i}, col: {col_index}")
                        Mbar.add(i, col_index, value/sum_row)
        
        # mesh = pyvista.PolyData(point_data)
        # plot = pyvista.Plotter()
        # plot.add_points(mesh)
        # plot.show()
        # #plot.export_html("projected_circles.html")
        # mesh.save('projected_circles.vtk')
        points = np.array(point_data)
        with open("projected_circles.csv", "w") as f:
            f.write("X,Y,Z\n")
            np.savetxt(f, points, delimiter=",", fmt="%.6f")
        """
        if verbose:
            self.logger.log("") # type: ignore


        return Mbar, Mlin

    def _build_exchange_matricies(self, mimv:gf.MeshIm, p_element:gf.MeshFem, Mbar:gf.Spmat, Mlin:gf.Spmat, mesh1D:gf.Mesh, tree:Tree, alternate=True, verbose=False) -> Tuple[gf.Spmat, gf.Spmat, gf.Spmat, gf.Spmat]:
        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("p", p_element)

        data = gf.MeshFem(mesh1D)
        data.set_fem(gf.Fem('FEM_PK(1,1)'))
        model.add_fem_data("data", data)

        model.add_initialized_data("Lp", [self.wall_hydraulic_conductivity])

        # iterate through all fem dofs associated with all segments
        pv_dofs = mesh1D.nbpts()
        pv_surface = np.empty(pv_dofs)
        for keys in tree.segment_dict.keys():
            region = keys+1
            pids_in_region = p_element.dof_on_region(region)
            surface = tree.segment_dict[keys].circumference()
            pv_surface[pids_in_region] = surface

        # Check if the array is fully populated
        if verbose:
            fully_populated = np.all(pv_surface != 0)

            if fully_populated:
                self.logger.log("The array is fully populated.") # type: ignore
            else:
                self.logger.log("The array is not fully populated.") # type: ignore

        """ # iterate through all points associated with all convexes
        i = 0
        pv_dofs = p_element.nbdof()
        pv_surface = np.empty(pv_dofs)
        while i < pv_dofs:
            for keys in tree.segment_dict.keys():
                convexes_in_branch = mesh1D.region(keys+1)[0]
                for cvid in convexes_in_branch:
                    # get the points associated with the convex
                    pid0 = mesh1D.pid_from_cvid(cvid)[0][0]
                    pid1 = mesh1D.pid_from_cvid(cvid)[0][1]

                    # get the area associated with the convex
                    surface = tree.segment_dict[keys].circumference()

                    # apply the area to the appropriate points
                    if i == 0:
                        pv_surface[pid0] = surface
                        i += 1
                    
                    pv_surface[pid1] = surface
                    i += 1 """

        # Assemble Bvv
        model.set_variable("data", pv_surface)
        model.add_nonlinear_term(mimv, "data*Lp*p.Test_p")
        model.assembly("build_matrix")
        Bvv = model.tangent_matrix()
        model.clear()

        # Build Bvt using the auxilary matricies
        Bvt = gf.Spmat('mult', Bvv, Mbar)

        if alternate:
            Btv = gf.Spmat('mult', Mlin, Bvv)
            Bvt = gf.Spmat('copy', Btv)
            Bvt.transpose()
            # Mlin.transpose()
            Btt = gf.Spmat('mult',Mlin,Bvt)
            # Mlin.transpose()

            """ 
            Btv = gf.Spmat('copy', Bvt)
            Btv.transpose()
            Mbar.transpose()
            Btt = gf.Spmat('mult',Mbar,Bvt)
            Mbar.transpose()
            """
        else:
            # Mlin.transpose()
            # Build Btv using the auxialry matricies
            Btv = gf.Spmat('mult', Mlin, Bvv)

            # Build Btt using the auxialry matricies
            Btt = gf.Spmat('mult', Mlin, Bvt)

        

        return Bvv, Bvt, Btv, Btt

    def _build_oxygen_exchange(self, mimv:gf.MeshIm, o_element:gf.MeshFem, Mbar:gf.Spmat, Mlin:gf.Spmat, mesh1D:gf.Mesh, tree:Tree, alternate=True, verbose=False) -> Tuple[gf.Spmat, gf.Spmat, gf.Spmat, gf.Spmat]:
        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("o", o_element)

        data = gf.MeshFem(mesh1D)
        data.set_fem(gf.Fem('FEM_PK(1,1)'))
        model.add_fem_data("data", data)

        # iterate through all fem dofs associated with all segments
        ov_dofs = o_element.nbdof()
        ov_surface = np.empty(ov_dofs)
        for keys in tree.segment_dict.keys():
            region = keys+1
            pids_in_region = o_element.dof_on_region(region)
            surface = tree.segment_dict[keys].circumference()
            ov_surface[pids_in_region] = surface

        # Check if the array is fully populated
        if verbose:
            fully_populated = np.all(ov_surface != 0)

            if fully_populated:
                self.logger.log("The array is fully populated.") # type: ignore
            else:
                self.logger.log("The array is not fully populated.") # type: ignore

        """     
        # iterate through all points associated with all convexes
        i = 0
        ov_dofs = o_element.nbdof()
        ov_surface = np.empty(ov_dofs)
        while i < ov_dofs:
            for keys in tree.segment_dict.keys():
                convexes_in_branch = mesh1D.region(keys+1)[0]
                for cvid in convexes_in_branch:
                    # get the points associated with the convex
                    pid0 = mesh1D.pid_from_cvid(cvid)[0][0]
                    pid1 = mesh1D.pid_from_cvid(cvid)[0][1]

                    # get the area associated with the convex
                    surface = tree.segment_dict[keys].circumference()

                    # apply the area to the appropriate points
                    if i == 0:
                        ov_surface[pid0] = surface
                        i += 1
                    
                    ov_surface[pid1] = surface
                    i += 1
        """
        # Assign area quantities to each point
        model.set_variable("data", ov_surface)
        
        model.add_initialized_data("Po", [self.wall_O2_permeability]) # type: ignore
        model.add_initialized_data("Phiv", [0])
        model.add_initialized_data("one_minus_sigma", [1-self.O2_reflection]) # type: ignore
        model.add_initialized_data("half",[0.5])
        
        # Assemble Bvv and auxiliaries for the advective component
        advective_flux = model.add_nonlinear_term(mimv, "data*(one_minus_sigma)*half*Phiv*o.Test_o")
        
        # Assemble Bvv and auxiliaries for the diffusive component
        diffusive_flux = model.add_nonlinear_term(mimv, "data*Po*o.Test_o")
        model.assembly("build_matrix")
        Bvv = model.tangent_matrix()
        model.clear()

        # Build Bvt using the auxilary matricies
        Bvt = gf.Spmat('mult', Bvv, Mbar)

        if alternate:
            Btv = gf.Spmat('mult', Mlin, Bvv)
            Bvt = gf.Spmat('copy', Btv)
            Bvt.transpose()
            # Mlin.transpose()
            Btt = gf.Spmat('mult',Mlin,Bvt)
            # Mlin.transpose()
            """ 
            Btv = gf.Spmat('copy', Bvt)
            Btv.transpose()
            Mbar.transpose()
            Btt = gf.Spmat('mult',Mbar,Bvt)
            Mbar.transpose()
            """
        else:
            # Mlin.transpose()
            # Build Btv using the auxilary matricies
            Btv = gf.Spmat('mult', Mlin, Bvv)

            # Build Btt using the auxilary matricies
            Btt = gf.Spmat('mult', Mlin, Bvt)

       
        return Bvv, Bvt, Btv, Btt

    def make_vegf_elements(self,mesh3D:gf.Mesh) -> gf.MeshFem:
        self.vegf_element = gf.MeshFem(mesh3D,1)
        self.vegf_element.set_fem(gf.Fem('FEM_PK(3,1)'))

        return self.vegf_element

    def build_monolithic_vegf_tissue(self, mesh3D:gf.Mesh, u_solution:np.ndarray=None, o_solution:np.ndarray=None, cylinder_test=False, verbose=False) -> Tuple[gf.Spmat, np.ndarray, gf.MeshIm, gf.MeshFem]:
        # Check if a u_solution was supplied
        if u_solution is None:
            raise ValueError("You must pass a velocity solution to this function")

        if o_solution is None:
            raise ValueError("You must pass an oxygen solution to this function")

        # Check if a velocity elements have been established
        if not hasattr(self, 'v_element'):
            raise ValueError("You must run GetFEMHandler3D.build_monolithic_tissue() before this function will operate properly")

        # Check if oxygen elements have been established
        if not hasattr(self, 'o_element'):
            raise ValueError("You must run GetFEMHandler3D.build_monolithic_oxygen_tissue() before this function will operate properly")
        
        # Suppress getfem text outputs
        gf.util_trace_level(level=1)

        if cylinder_test:
            tol = 1e-8
            boundary_region_bottom = mesh3D.outer_faces_with_direction([0, 0, -1], tol)
            boundary_region_top = mesh3D.outer_faces_with_direction([0, 0, 1], tol)
            boundary_region_side = mesh3D.outer_faces_with_direction([1, 0, 0], tol) + \
                                mesh3D.outer_faces_with_direction([-1, 0, 0], tol)

            mesh3D.set_region(1,boundary_region_side)
            mesh3D.set_region(2,boundary_region_top)
            mesh3D.set_region(3,boundary_region_bottom)
            tissue_boundaries = [1,2,3]
        else:
            # get face boundaries of the tissue
            cvfid1 = mesh3D.outer_faces_with_direction([0,0,-1],1)
            cvfid2 = mesh3D.outer_faces_with_direction([0,0,1],1)
            cvfid3 = mesh3D.outer_faces_with_direction([0,-1,0],1)
            cvfid4 = mesh3D.outer_faces_with_direction([0,1,0],1)
            cvfid5 = mesh3D.outer_faces_with_direction([-1,0,0],1)
            cvfid6 = mesh3D.outer_faces_with_direction([1,0,0],1)

            BOTTOM = mesh3D.set_region(1,cvfid1)
            TOP = mesh3D.set_region(2,cvfid2)
            LEFT = mesh3D.set_region(3,cvfid3)
            RIGHT = mesh3D.set_region(4,cvfid4)
            BACK = mesh3D.set_region(5,cvfid5)
            FRONT = mesh3D.set_region(6,cvfid6)
            
            tissue_boundaries = [1,2,3,4,5,6]

        self.make_vegf_elements(mesh3D)

        dof_total = self.vegf_element.nbdof()

        Full_MAT = gf.Spmat("empty", dof_total, dof_total)
        Full_VEC = np.zeros(dof_total)

        Dvt = self._build_Dvt()
        Kvt = self._build_Kvt()
        # Avt = self._build_Avt(u_solution)

        Ovt, Ovt_RHS = self._build_Ovt(o_solution)

        Full_MAT.add(range(0,dof_total),range(0,dof_total), -Dvt)
        Full_MAT.add(range(0,dof_total),range(0,dof_total), -Kvt)
        Full_MAT.add(range(0,dof_total),range(0,dof_total), Ovt)
        # Full_MAT.add(range(0,dof_total),range(0,dof_total), Avt)

        Dvt_aug, RHS = self._build_vegf_bc_robin(tissue_boundaries, cylinder_test)
        Full_MAT.add(range(0,dof_total),range(0,dof_total), Dvt_aug)
        Full_VEC += RHS
        Full_VEC -= Ovt_RHS

        return Full_MAT, Full_VEC, self.mim, self.vegf_element

    def _build_Dvt(self) -> gf.Spmat:
        # In vitro 3D collective sprouting angiogenesis under orchestrated ANG-1 and VEGF gradients
        # 1.2E-12 m2s-1 in cells 30um thick
        # Conversion to 3D diffusion: D3 = D2/(4h)
        # Therefore: 1E-8
        vegf_diffusion_coef = 1e-12
        scale_term = vegf_diffusion_coef
        dof_total = self.vegf_element.nbdof()
        scale = np.ones(dof_total)
        scale = scale*scale_term
        Dvt = gf.asm_laplacian(self.mim, self.vegf_element, self.vegf_element, scale)

        return Dvt

    def _build_Avt(self, u_solution:np.ndarray) -> gf.Spmat:
        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("vegf", self.vegf_element)
        model.add_fem_variable("u", self.v_element)

        model.set_variable("u", u_solution)

        Avt1 = model.add_nonlinear_term(self.mim, "u.Grad_vegf.Test_vegf")
        Avt2 = model.add_nonlinear_term(self.mim, "Div_u*vegf.Test_vegf")

        model.disable_variable("u")
        model.assembly("build_matrix")
        Avt = model.tangent_matrix()

        return Avt

    def _build_Kvt(self) -> gf.Spmat:
        degradation_rate_constant = 2.82e-3 #×10−3 s−1
        Kv = degradation_rate_constant

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("vegf", self.vegf_element)
        model.add_initialized_data("Kv", [Kv])

        model.add_linear_term(self.mim, "Kv*vegf.Test_vegf")

        model.assembly("build_matrix")
        Kvt = model.tangent_matrix()

        return Kvt


    def _build_Ovt(self, o_solution:np.ndarray) -> gf.Spmat:
        basal_rate = 1.97e-3 #×10−3 pM s−1

        o_solution = o_solution/self.tissue_solubility_O2 # type: ignore
        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("vegf", self.vegf_element)
        model.add_fem_variable("o", self.o_element)
        model.add_fem_variable("high", self.o_element)
        model.add_fem_variable("mid", self.o_element)
        model.add_fem_variable("low", self.o_element)

        o_mask_high = (o_solution >= 20).astype(int)
        o_mask_mid = ((o_solution > 1) & (o_solution < 20)).astype(int)
        o_mask_low = (o_solution <= 1).astype(int)

        model.set_variable("o", o_solution)
        model.set_variable("high", o_mask_high)
        model.set_variable("mid", o_mask_mid)
        model.set_variable("low", o_mask_low)

        model.add_initialized_data("Mvegf", [basal_rate])
        model.add_initialized_data("six", [6])
        model.add_initialized_data("five", [5])
        model.add_initialized_data("one", [1])
        model.add_initialized_data("twenty", [20])
        model.add_initialized_data("nineteen", [19])
        model.add_initialized_data("three", [3])

        high_string = "Mvegf"
        Ovt1 = model.add_source_term(self.mim, f"{high_string}*high.Test_vegf")

        mid_string = "Mvegf*(one+five*pow((twenty-o)/nineteen,three))"
        Ovt2 = model.add_source_term(self.mim, f"{mid_string}*mid.Test_vegf")

        low_string = "six*Mvegf"
        Ovt3 = model.add_source_term(self.mim, f"{low_string}*low.Test_vegf")

        model.disable_variable("o")
        model.disable_variable("high")
        model.disable_variable("mid")
        model.disable_variable("low")
        model.assembly()
        Ovt = model.tangent_matrix()
        Ovt_RHS = model.rhs()

        return Ovt, Ovt_RHS

    def _build_vegf_bc_robin(self, tissue_boundaries:List[int], cylinder_test:bool) -> Tuple[gf.Spmat, np.ndarray]:
        beta_vt = 0
        far_field_vegf = 0
        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("vegf", self.vegf_element)

        model.add_initialized_data("beta_vt", [beta_vt])
        model.add_initialized_data("cvt", [far_field_vegf])

        for boundary in tissue_boundaries:
            model.add_source_term(self.mim, "-beta_vt*cvt.Test_vegf", boundary)
            model.add_linear_term(self.mim, "beta_vt*vegf.Test_vegf", boundary)

        model.assembly()
        LHS = model.tangent_matrix()
        RHS = model.rhs()

        return LHS, RHS

    def make_attractor_elements(self, mesh3D:gf.Mesh):
        self.attractor_element = gf.MeshFem(mesh3D,1)
        self.attractor_element.set_fem(gf.Fem('FEM_PK(3,1)'))

        return self.attractor_element

    def build_base_attractor_field(self, mesh3D:gf.Mesh, verbose=False) -> Tuple[gf.Spmat, np.ndarray, gf.MeshIm, gf.MeshFem]:
        # Suppress getfem text outputs
        gf.util_trace_level(level=1)

        cvfid1 = mesh3D.outer_faces_with_direction([0,0,-1],1)
        cvfid2 = mesh3D.outer_faces_with_direction([0,0,1],1)
        cvfid3 = mesh3D.outer_faces_with_direction([0,-1,0],1)
        cvfid4 = mesh3D.outer_faces_with_direction([0,1,0],1)
        cvfid5 = mesh3D.outer_faces_with_direction([-1,0,0],1)
        cvfid6 = mesh3D.outer_faces_with_direction([1,0,0],1)

        BOTTOM = mesh3D.set_region(1,cvfid1)
        TOP = mesh3D.set_region(2,cvfid2)
        LEFT = mesh3D.set_region(3,cvfid3)
        RIGHT = mesh3D.set_region(4,cvfid4)
        BACK = mesh3D.set_region(5,cvfid5)
        FRONT = mesh3D.set_region(6,cvfid6)
        
        tissue_boundaries = [1,2,3,4,5,6]

        self.make_attractor_elements(mesh3D)

        dof_total = self.attractor_element.nbdof()

        Full_MAT = gf.Spmat("empty", dof_total, dof_total)
        Full_VEC = np.zeros(dof_total)

        Dat = self._build_Dat()
        Kat = self._build_Kat()
        Full_MAT.add(range(0,dof_total),range(0,dof_total), -Dat)
        Full_MAT.add(range(0,dof_total),range(0,dof_total), -Kat)

        Dat_aug, RHS = self._build_attractor_bc_robin(tissue_boundaries)
        Full_MAT.add(range(0,dof_total),range(0,dof_total), Dat_aug)
        Full_VEC += RHS

        # RHS = self._build_attractor_signal(mesh1D, tree, tissue)
        # Full_VEC += RHS

        return Full_MAT, Full_VEC, self.mim, self.attractor_element

    def _build_Dat(self)-> gf.Spmat:
        attractor_diffusion_coef = 1e-8
        scale_term = attractor_diffusion_coef
        dof_total = self.attractor_element.nbdof()
        scale = np.ones(dof_total)
        scale = scale*scale_term
        Dat = gf.asm_laplacian(self.mim, self.attractor_element, self.attractor_element, scale)

        return Dat

    def _build_Kat(self) -> gf.Spmat:
        degradation_rate_constant = 6e-2 #×10−3 s−1
        Ka = degradation_rate_constant

        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("a", self.vegf_element)
        model.add_initialized_data("Ka", [Ka])

        model.add_linear_term(self.mim, "Ka*a.Test_a")

        model.assembly("build_matrix")
        Kat = model.tangent_matrix()

        return Kat

    def _build_attractor_bc_robin(self, tissue_boundaries:List[int]) -> Tuple[gf.Spmat, np.ndarray]:
        beta_at = 0
        far_field_attractor = 0
        # Establish the FEM model, and assign variables to it
        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("a", self.attractor_element)

        model.add_initialized_data("beta_at", [beta_at])
        model.add_initialized_data("cat", [far_field_attractor])

        for boundary in tissue_boundaries:
            model.add_source_term(self.mim, "-beta_at*cat.Test_a", boundary)
            model.add_linear_term(self.mim, "beta_at*a.Test_a", boundary)

        model.assembly()
        LHS = model.tangent_matrix()
        RHS = model.rhs()

        return LHS, RHS
        
    def _build_vessel_inhibition_signal(self, mesh1D:gf.Mesh, tree:Tree, tissue:Tissue, signal_intensity:float=1) -> np.ndarray:
        attractor_element_1d = gf.MeshFem(mesh1D,1)
        attractor_element_1d.set_fem(gf.Fem('FEM_PK(1,1)'))
        mimv = gf.MeshIm(mesh1D, gf.Integ('IM_GAUSS1D(6)')) 

        Mbar, Mlin = self._build_auxillary_matricies(mesh1D,attractor_element_1d,self.attractor_element,tree,tissue)

        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("a", attractor_element_1d)
        model.add_linear_term(mimv, "a.Test_a")
        model.assembly("build_matrix")
        Bvv = model.tangent_matrix()
        model.clear()

        Btv = gf.Spmat('mult', Mlin, Bvv)

        signal_points = None
        used_dofs = []
        for keys in tree.segment_dict.keys():
            dof_ids = attractor_element_1d.basic_dof_on_region(keys+1)
            # Iterate through the subset and check if an ID has been used
            filtered_subset = [id for id in dof_ids if id not in used_dofs]
            # Record the retrieved IDs in the used_ids list
            used_dofs.extend(filtered_subset)
            signal_points_on_region = attractor_element_1d.basic_dof_nodes(filtered_subset)
            # Create a row filled with the segment_id
            new_row1 = np.zeros_like(signal_points_on_region[0, :])
            new_row2 = np.full_like(signal_points_on_region[0, :], keys)
            # Stack the original array with the new row
            signal_points_on_region = np.vstack([signal_points_on_region, new_row1, new_row2])
            if signal_points is None:
                signal_points = signal_points_on_region
            else:
                signal_points = np.hstack([signal_points,signal_points_on_region])

        signal_points = np.transpose(signal_points)
        signal_vector = tree.measure_inhibition_signal(signal_points)
        #Inhibited regions have a binary signal where true is inhibited. We want non-inhibited regions to attract.
        signal_vector = np.ones_like(signal_vector)-signal_vector
        signal_vector *= signal_intensity
        signal_vector = signal_vector*1e-10
        

        RHS_segment_signal = Btv.mult(signal_vector)
        return RHS_segment_signal

    def _build_tip_cell_signal(self, sprout:Sprout, signal_intensity:float=2) -> np.ndarray:
        temp_point_mesh = gf.Mesh("empty", 3)
        tip_loc = sprout.get_tip_loc()
        temp_point_mesh.add_point(np.transpose(tip_loc))
        
        pts = temp_point_mesh.pts()
        Mlin = gf.asm_interpolation_matrix(self.attractor_element,pts)
        signal_vector = np.full(1,signal_intensity)*1e-14

        Mlin.transpose()
        RHS_tip_signal = Mlin.mult(signal_vector)

        return RHS_tip_signal


    def _build_final_MAT(self, Mat3D:gf.Spmat, Mat1D:gf.Spmat, Bvv:gf.Spmat, Bvt:gf.Spmat, Btv:gf.Spmat, Btt:gf.Spmat) -> gf.Spmat:
        # Get sizes necessary for matrix arrangement
        tissue_size = Mat3D.size()
        Btt_size = Btt.size()

        vessel_size = Mat1D.size()
        Bvv_size = Bvv.size()
        
        Tt = tissue_size[0]
        Ut = tissue_size[0] - Btt_size[0]
        #Pt = Btt_size[0]

        Tv = vessel_size[0]
        Uv = vessel_size[0] - Bvv_size[0]
        #Pv = Bvv_size[0]

        FinalMat = gf.Spmat('empty',Tt+Tv, Tt+Tv)
        # Assign 3D contributions to the matrix
        FinalMat.add(range(0,Tt),range(0,Tt),Mat3D)
        # Assign 1D-3D exchange contributions to the matrix
        FinalMat.add(range(Tt+Uv,Tt+Tv),range(Tt+Uv,Tt+Tv),Bvv)
        FinalMat.add(range(Ut,Tt),range(Ut,Tt),Btt)
        FinalMat.add(range(Tt+Uv,Tt+Tv),range(Ut,Tt),-Bvt)
        FinalMat.add(range(Ut,Tt),range(Tt+Uv,Tt+Tv),-Btv)

        FinalMat.add(range(Tt,Tt+Tv),range(Tt,Tt+Tv),Mat1D)

        return FinalMat

    def _build_final_VEC(self, Vec3D:np.ndarray, Vec1D:np.ndarray, Bvv:gf.Spmat, Btv:gf.Spmat, verbose=False) -> np.ndarray:
        # construct the final vector from the velocity boundary conditions and
        # the oncotic contributions to the 1D-3D exchange

        # Get sizes necessary for matrix arrangement
        Tt = Vec3D.size
        Ut = Vec3D.size - Btv.size()[0]
        Pt = Btv.size()[0]

        Tv = Vec1D.size
        Uv = Vec1D.size - Bvv.size()[0]
        Pv = Bvv.size()[0]

        # define constants
        delta_pi = self.oncotic_gradient * 133.32
        sigma  = 0.95

        pi_coef = sigma*delta_pi

        pi_vec = np.full(Pv,pi_coef)

        aux_tissue_vec = Btv.mult(pi_vec)
        aux_vessel_vec = Bvv.mult(pi_vec)

        FinalVec = np.empty(Tt+Tv)
        FinalVec[:Tt] = Vec3D
        FinalVec[Ut:Tt] += -aux_tissue_vec
        FinalVec[Tt:Tt+Tv] = Vec1D
        FinalVec[Tt+Uv:Tt+Tv] += aux_vessel_vec

        if verbose:
            self.logger.log("########################################################") # type: ignore
            self.logger.log("1D Exchange Vector:") # type: ignore
            self.logger.log(aux_vessel_vec) # type: ignore
            self.logger.log("########################################################") # type: ignore

        return FinalVec

    def _build_final_oxygen_MAT(self, Mat3D:gf.Spmat, Mat1D:gf.Spmat, Bvv:gf.Spmat, Bvt:gf.Spmat, Btv:gf.Spmat, Btt:gf.Spmat, verbose=False) -> gf.Spmat:
        # Get sizes necessary for matrix arrangement
        tissue_size = Mat3D.size()
        vessel_size = Mat1D.size()
        
        Tt = tissue_size[0]
        Tv = vessel_size[0]

        FinalMat = gf.Spmat('empty',Tt+Tv, Tt+Tv)

        # Assign 3D contributions to the matrix
        FinalMat.add(range(0,Tt),range(0,Tt),Mat3D)
        # Assign 1D contributions to the matrix
        FinalMat.add(range(Tt,Tt+Tv),range(Tt,Tt+Tv),Mat1D)
        # Assign 1D-3D exchange contributions to the matrix
        FinalMat.add(range(Tt,Tt+Tv),range(Tt,Tt+Tv),Bvv)
        FinalMat.add(range(0,Tt),range(0,Tt),Btt)
        FinalMat.add(range(Tt,Tt+Tv),range(0,Tt),-Bvt)
        FinalMat.add(range(0,Tt),range(Tt,Tt+Tv),-Btv)

        if verbose:
            self.logger.log("##################  O MATRIX!!  #####################") # type: ignore
            self.logger.log(Mat1D.full()) # type: ignore
            np.savetxt("O_MAT_1D.txt",Mat1D.full())
            self.logger.log("##################  O MATRIX!!  #####################") # type: ignore
            self.logger.log("##################  O BVV MAT!  #####################") # type: ignore
            self.logger.log(Bvv.full()) # type: ignore
            np.savetxt("O_EXCHANGE.txt",Bvv.full())
            self.logger.log("##################  O BVV MAT!  #####################") # type: ignore

        return FinalMat

    def _build_final_oxygen_VEC(self, Vec3D:np.ndarray, Vec1D:np.ndarray ,verbose=False) -> np.ndarray:
        # construct the final vector from the velocity boundary conditions and
        # the oncotic contributions to the 1D-3D exchange

        # Get sizes necessary for matrix arrangement
        Tt = Vec3D.size
        Tv = Vec1D.size

        FinalVec = np.empty(Tt+Tv)
        FinalVec[:Tt] = Vec3D
        FinalVec[Tt:Tt+Tv] = Vec1D

        if verbose:
            self.logger.log("##################  O VECTOR!!  #####################") # type: ignore
            self.logger.log(Vec1D) # type: ignore
            np.savetxt("O_VEC_1D.txt",Vec1D)
            self.logger.log("##################  O VECTOR!!  #####################") # type: ignore

        return FinalVec
        

    def save_haemodynamic_3D(self,save_string:str,run_string:str,utfem:gf.MeshFem,utx:np.ndarray,ptfem:gf.MeshFem,ptx:np.ndarray,iteration:Union[int,None]=None):
        if not iteration is None:
            run_string = f"{run_string}_{iteration}"
            
        print_string = save_string+"/getfem_full_tissue_"+run_string+".vtk"

        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("ut", utfem)
        model.add_fem_variable("pt", ptfem)

        model.set_variable("ut",utx)
        model.set_variable("pt",ptx)

        utfem.export_to_vtk(save_string+"/getfem_full_tissue_"+run_string+".vtk",utfem, model.variable("ut"), "Velocity Tissue",ptfem, model.variable("pt"), "Pressure Tissue")
        self.logger.log(f"Saving 3D mesh (Haemodynamic) to {print_string}") # type: ignore
        return
    
    def save_oxygen(self,save_string:str,run_string:str,utfem:gf.MeshFem,utx:np.ndarray,ptfem:gf.MeshFem,ptx:np.ndarray,uvfem_dict:Dict,uvx:Dict,pvfem:gf.MeshFem,pvx:np.ndarray,hvfem_dict:Dict,hx:Dict,otfem:gf.MeshFem,otx:np.ndarray,ovfem:gf.MeshFem,ovx:np.ndarray,iteration:Union[int,None]=None):
        if not iteration is None:
            run_string = f"{run_string}_{iteration}"
            
        print_string = save_string+"/getfem_full_tissue_"+run_string+".vtk"

        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("ut", utfem)
        model.add_fem_variable("pt", ptfem)

        model.set_variable("ut",utx)
        model.set_variable("pt",ptx)

        for keys in uvfem_dict:
            uvfem = uvfem_dict[keys]
            string = f"uv{keys}"
            model.add_fem_variable(string, uvfem)
            model.set_variable(string,uvx[keys])

        model.add_fem_variable("pv", pvfem)
        model.set_variable("pv",pvx)

        for keys in hvfem_dict:
            hvfem = hvfem_dict[keys]
            string = f"hv{keys}"
            model.add_fem_variable(string, hvfem)
            model.set_variable(string,hx[keys])

        model.add_fem_variable("ot", otfem)
        model.add_fem_variable("ov", ovfem)

        model.set_variable("ot",otx)
        model.set_variable("ov",ovx)

        utfem.export_to_vtk(save_string+"/getfem_full_tissue_"+run_string+".vtk",utfem, model.variable("ut"), "Velocity Tissue",ptfem, model.variable("pt"), "Pressure Tissue",otfem, model.variable("ot"), "Oxygen Tissue")
        
        self.logger.log(f"Saving 3D mesh (Oxygen) to {print_string}") # type: ignore
        return

    def save_vegf(self,save_string:str,run_string:str,utfem:gf.MeshFem,utx:np.ndarray,ptfem:gf.MeshFem,ptx:np.ndarray,uvfem_dict:Dict,uvx:Dict,pvfem:gf.MeshFem,pvx:np.ndarray,hvfem_dict:Dict,hx:Dict,otfem:gf.MeshFem,otx:np.ndarray,ovfem:gf.MeshFem,ovx:np.ndarray,vtfem:gf.MeshFem,vtx:np.ndarray,tree:Tree,iteration=None):
        if not iteration is None:
            run_string = f"{run_string}_{iteration}"
            
        print_string = save_string+"/getfem_full_tissue_"+run_string+".vtk"

        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("ut", utfem)
        model.add_fem_variable("pt", ptfem)

        model.set_variable("ut",utx)
        model.set_variable("pt",ptx)

        for keys in uvfem_dict:
            uvfem = uvfem_dict[keys]
            string = f"uv{keys}"
            model.add_fem_variable(string, uvfem)
            model.set_variable(string,uvx[keys])
            qvfem = uvfem_dict[keys]
            string = f"qv{keys}"
            model.add_fem_variable(string, qvfem)
            area = tree.area(keys-1)
            flux = area*uvx[keys]
            model.set_variable(string,flux)

        model.add_fem_variable("pv", pvfem)
        model.set_variable("pv",pvx)

        for keys in hvfem_dict:
            hvfem = hvfem_dict[keys]
            string = f"hv{keys}"
            model.add_fem_variable(string, hvfem)
            model.set_variable(string,hx[keys])

        model.add_fem_variable("ot", otfem)
        model.add_fem_variable("ov", ovfem)

        model.set_variable("ot",otx)
        model.set_variable("ov",ovx)

        model.add_fem_variable("vt", vtfem)
        model.set_variable("vt",vtx)

        utfem.export_to_vtk(save_string+"/getfem_full_tissue_"+run_string+".vtk",\
            utfem,model.variable("ut"),"Velocity Tissue", ptfem,model.variable("pt"),"Pressure Tissue",\
                otfem,model.variable("ot"),"Oxygen Tissue", vtfem,model.variable("vt"),"VEGF Tissue")
        
        self.logger.log(f"Saving 3D mesh (VEGF) to {print_string}") # type: ignore
        return

    def save_attraction_field(self,save_string:str,run_string:str,afem:gf.MeshFem,x_tree:np.ndarray,x_sprouts:dict,iteration:Union[int,None]=None):
        if not iteration is None:
            run_string = f"{run_string}_{iteration}"

        model=gf.Model('real') # real or complex space.
        model.add_fem_variable("a_vessel", afem)

        model.set_variable("a_vessel",x_tree)

        afem.export_to_vtk(save_string+"/attractor_results/attractor_field_vessel_"+run_string+".vtk",\
            afem,model.variable("a_vessel"),"Vessel Attractor Field")

        for key in x_sprouts.keys():
            x_sprout = x_sprouts[key]
            string = f"a_sprout_{key}"
            model.add_fem_variable(string, afem)
            model.set_variable(string,x_sprout)

            afem.export_to_vtk(save_string+f"/attractor_results/attractor_field_sprout_{key}_"+run_string+".vtk",\
                afem,model.variable(string),f"Sprout {key} Attractor Field")
        



    def __str__(self):
      return f"""
        3D Settings:
          Distant Pressure (mmHg): {self.distant_pressure}
          Boundary Conductivity: {self.boundary_conductivity}
          Tissue Hydraulic Conductivity: {self.tissue_hydraulic_conductivity}
          Interstitial Viscosity: {self.interstitial_viscosity}
          Wall Hydraulic Conductivity: {self.wall_hydraulic_conductivity}
          Oncotic Gradient: {self.oncotic_gradient}
      """  





        






