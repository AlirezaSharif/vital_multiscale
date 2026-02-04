import json
import csv
from os.path import join as os_path_join
import copy
import numpy as np # type: ignore
import pyvista as pv # type: ignore
import matplotlib as plt # type: ignore
import networkx as nx # type: ignore
from scipy.integrate import simpson # type: ignore
from scipy.spatial import distance # type: ignore
from operator import length_hint
from typing import List, Dict, Tuple, Union
from classes.ConfigClass import Config
from collections import defaultdict, deque
import concurrent.futures
import heapq


class Node:
    """
    A class used to handle the information needed to define a single node in the vascular tree.

    ----------
    Class Attributes
    ----------
    __NODE_COUNT : int
        An integer stating the total number of nodes in the system
    
    __INLET_COUNT : int
        An integer stating the total number of nodes in the system that are inlets
    
    __OUTLET_COUNT : int
        An integer stating the total number of nodes in the system that are outlets
    
    __X_LIM : [int, int]
        A range of valid X coordinate values

    __Y_LIM : [int, int]
        A range of valid Y coordinate values

    __Z_LIM : [int, int]
        A range of valid Z coordinate values

    ----------
    Instance Attributes
    ----------
    node_id : int
        A unique id number representing the node

    spatial_location : [int, int, int]
        An XYZ coordinate representing the location of the node in 3D space.

    _node_type : int
        An int indicating whether the node is an inlet, an outlet, or a junction
    
    junction_id : int
        A unique id number representing the junction associated with the node, if it is a junction

    ----------
    Class Methods
    ----------  
    increment_count(amount=0)
        Adds or subtracts from the __NODE_COUNT Attribute

    reset_count()
        Resets the __NODE_COUNT Attribute
    
    set_limits([x_min,x_max], [y_min,y_max], [z_min,z_max])
        Sets the values of the limits on valid spatial location coordinates

    ----------
    Instance Methods
    ----------  
    node_type()
        Returns the type of node indicated by _node_type as a string

    x_coord()
        Returns the x component of the spatial location

    y_coord()
        Returns the y component of the spatial location

    z_coord()
        Returns the z component of the spatial location

    location()
        Returns the [x,y,z] spatial location

    move_node([x,y,z])
        Moves the node from its current location by the input vector
    
    """

    __NODE_COUNT = 0
    __INLET_COUNT = 0
    __OUTLET_COUNT = 0
    __X_LIM = [0, 100]
    __Y_LIM = [0, 100]
    __Z_LIM = [0, 100]

    def __init__(self, node_id:int, spatial_location:List[float], _node_type:int, junction_id:int=None): # type: ignore
        """
        Initializes a Node object with a unique ID, spatial location, node type, and optional junction ID.

        :param node_id: A unique id number representing the node.
        :param spatial_location: An XYZ coordinate representing the location of the node in 3D space.
        :param _node_type: An int indicating whether the node is an inlet, an outlet, or a junction.
        :param junction_id: A unique id number representing the junction associated with the node, if it is a junction.
        """
        if node_id is None:
            self.node_id = self.__NODE_COUNT
        else:
            self.node_id = node_id

        if spatial_location[0] < Node.__X_LIM[0] or spatial_location[0] > Node.__X_LIM[1]:
            raise ValueError(f"The x coordinate of this node ({spatial_location[0]}) is not within the spatial limits ({Node.__X_LIM})")
        elif spatial_location[1] < Node.__Y_LIM[0] or spatial_location[1] > Node.__Y_LIM[1]:
            raise ValueError(f"The y coordinate of this node ({spatial_location[1]}) is not within the spatial limits ({Node.__Y_LIM})")
        elif spatial_location[2] < Node.__Z_LIM[0] or spatial_location[2] > Node.__Z_LIM[1]:
            raise ValueError(f"The z coordinate of this node ({spatial_location[2]}) is not within the spatial limits ({Node.__Z_LIM})")
        else:
            self.spatial_location = spatial_location

        self._node_type = _node_type
        self.junction_id = junction_id

    @classmethod
    def increment_count(cls, amount=1):
        """
        Adds or subtracts from the __NODE_COUNT attribute.

        :param amount: The amount to add to the __NODE_COUNT attribute. Defaults to 1.
        """
        cls.__NODE_COUNT += amount

    @classmethod
    def reset_count(cls):
        """
        Resets the __NODE_COUNT, __INLET_COUNT, and __OUTLET_COUNT attributes to 0.
        """
        cls.__NODE_COUNT = 0
        cls.__INLET_COUNT = 0
        cls.__OUTLET_COUNT = 0

    @classmethod
    def increment_inlet(cls, amount=1):
        """
        Adds or subtracts from the __INLET_COUNT attribute.

        :param amount: The amount to add to the __INLET_COUNT attribute. Defaults to 1.
        """
        cls.__INLET_COUNT += amount

    @classmethod
    def increment_outlet(cls, amount=1):
        """
        Adds or subtracts from the __OUTLET_COUNT attribute.

        :param amount: The amount to add to the __OUTLET_COUNT attribute. Defaults to 1.
        """
        cls.__OUTLET_COUNT += amount

    @classmethod
    def set_limits(cls, xlim, ylim, zlim):
        """
        Sets the values of the limits on valid spatial location coordinates.

        :param xlim: A range of valid X coordinate values.
        :param ylim: A range of valid Y coordinate values.
        :param zlim: A range of valid Z coordinate values.
        """
        cls.__X_LIM = xlim
        cls.__Y_LIM = ylim
        cls.__Z_LIM = zlim
    
    def node_type(self) -> str:
        """
        Returns the type of node indicated by _node_type as a string.

        :return: A string representing the type of node ('Inlet', 'Outlet', 'Internal').
        """
        if self._node_type == 0:
            return "Inlet"
        elif self._node_type == 1:
            return "Outlet"
        elif self._node_type == 2:
            return "Internal"
        else:
            raise ValueError("The node_type is not correctly set, valid values are: 0, 1, 2")

    def x_coord(self) -> float:
        """
        Returns the x component of the spatial location.

        :return: The x coordinate of the node.
        """
        return self.spatial_location[0]

    def y_coord(self) -> float:
        """
        Returns the y component of the spatial location.

        :return: The y coordinate of the node.
        """
        return self.spatial_location[1]

    def z_coord(self) -> float:
        """
        Returns the z component of the spatial location.

        :return: The z coordinate of the node.
        """
        return self.spatial_location[2]

    def location(self) -> np.ndarray:
        """
        Returns the [x, y, z] spatial location.

        :return: A numpy array representing the XYZ coordinates of the node.
        """
        return np.array([self.spatial_location[0], self.spatial_location[1], self.spatial_location[2]])

    def move_node(self,movement_vector:list) -> np.ndarray:
        """
        Changes the location of the node by the input vector [x, y, z] and returns the new [x, y, z] spatial location.
        Bounded such that node position will not move out of the range defined by the set_limits() method

        :param movement_vector: A list representing the movement vector.
        :return: A numpy array representing the new XYZ coordinates of the node.
        """
        new_location = self.spatial_location + movement_vector
        if new_location[0] > Node.__X_LIM[0] and new_location[0] < Node.__X_LIM[1]:
            self.spatial_location[0] = new_location[0]
        elif new_location[1] > Node.__Y_LIM[0] and new_location[1] < Node.__Y_LIM[1]:
            self.spatial_location[0] = new_location[1]
        elif new_location[2] > Node.__Z_LIM[0] and new_location[2] < Node.__Z_LIM[1]:
            self.spatial_location[0] = new_location[2]

        return np.array([self.spatial_location[0], self.spatial_location[1], self.spatial_location[2]])

    def __str__(self):
        """
        Returns a string representation of the node information.

        :return: A string representing the node's ID, coordinates, type, and junction ID.
        """
        return f"""
        Node Information:
          Node ID: {self.node_id}
          XYZ coordinate: {self.spatial_location}
          Node Type: {self.node_type()}
          Junction ID: {self.junction_id}
      """
 

class Segment:
    """
    A class used to handle the information needed to define a single segment in the vascular tree.

    ----------
    Class Attributes
    ----------
    __SEGMENT_COUNT : int
        An integer stating the total number of segments in the system
    
    __DEFAULT_RADIUS : int
        An integer stating the default radius of newly created segments

    __RADIUS_RANGE : [int, int]
        A pair of integers setting lower and upper bounds of the valid radius values

    ----------
    Instance Attributes
    ----------
    segment_id : int
        A unique id number representing the segment

    radius : int
        A value for the radius of the segment in meters

    node_1_id : int
        The id number associated with the first node of the segment

    node_2_id : int
        The id number associated with the second node of the segment

    ----------
    Class Methods
    ----------  
    increment_count(amount=1)
        Adds or subtracts from the __SEGMENT_COUNT attribute

    reset_count()
        Resets the __SEGMENT_COUNT attribute

    set_default_radius(default_radius)
        Sets the default radius of a newly created segment

    ----------
    Instance Methods
    ----------  
    radius
        Returns the radius of the segment
    
    set_radius(radius)
        Sets the radius of the segment
    
    area()
        Returns the area of the segment based on its radius

    circumference()
        Returns the circumference of the segment based on its radius

    """

    __SEGMENT_COUNT = 0
    __DEFAULT_RADIUS = 3e-6
    __RADIUS_RANGE = [0, 30e-6]

    def __init__(self, segment_id:int, radius:float, node_1_id:int, node_2_id:int):
        """
        Initializes a Segment object with a unique ID, radius, and node IDs.

        :param segment_id: A unique id number representing the segment.
        :param radius: A value for the radius of the segment in meters.
        :param node_1_id: The id number associated with the first node of the segment.
        :param node_2_id: The id number associated with the second node of the segment.
        """
        if segment_id is None:
            self.segment_id = self.__SEGMENT_COUNT
        else:
            self.segment_id = segment_id

        if radius is None:
            radius = self.__DEFAULT_RADIUS

        if radius < Segment.__RADIUS_RANGE[0]:
            raise ValueError("The provided radius was too small.")
        elif radius > Segment.__RADIUS_RANGE[1]:
            raise ValueError("The provided radius was too large.")
        else:
            self._radius = radius

        self.node_1_id = node_1_id
        self.node_2_id = node_2_id

    @classmethod
    def increment_count(cls, amount:int=1):
        """
        Adds or subtracts from the __SEGMENT_COUNT attribute.

        :param amount: The amount to add to the __SEGMENT_COUNT attribute. Defaults to 1.
        """
        cls.__SEGMENT_COUNT += amount

    @classmethod
    def reset_count(cls):
        """
        Resets the __SEGMENT_COUNT attribute to 0.
        """
        cls.__SEGMENT_COUNT = 0

    @classmethod
    def set_default_radius(cls, default_radius:int):
        """
        Sets the default radius of a newly created segment.

        :param default_radius: The default radius value to be set.
        """
        cls.__DEFAULT_RADIUS = default_radius

    @property
    def radius(self):
        """
        Returns the radius of the segment.

        :return: The radius of the segment.
        """
        return self._radius

    @radius.setter
    def radius(self, radius:float):
        """
        Sets the radius of the segment, ensuring it is within the valid range.

        :param radius: The radius value to be set.
        :raises ValueError: If the radius is outside the valid range.
        """
        if radius < Segment.__RADIUS_RANGE[0]:
            raise ValueError("The provided radius was too small.")
        elif radius > Segment.__RADIUS_RANGE[1]:
            raise ValueError("The provided radius was too large.")
        else:
            self._radius = radius

    def area(self):
        """
        Returns the area of the segment based on its radius.

        :return: The area of the segment.
        """
        return np.pi * self.radius * self.radius

    def circumference(self):
        """
        Returns the circumference of the segment based on its radius.

        :return: The circumference of the segment.
        """
        return 2 * np.pi * self.radius
    
    def reverse_segment(self):
        """
        Swaps the node ids associated with node 1 and node 2.
        """
        self.node_1_id, self.node_2_id = self.node_2_id, self.node_1_id

    def __str__(self):
        """
        Returns a string representation of the segment information.

        :return: A string representing the segment's ID, radius, and associated node IDs.
        """
        return f"""
        Segment Information:
          Segment ID: {self.segment_id}
          Segment Radius: {self.radius}
          Node 1 ID: {self.node_1_id}
          Node 2 ID: {self.node_2_id}
      """
    


class Junction:
    """
    A class used to handle the information needed to define a single junction in the vascular tree.

    ----------
    Class Attributes
    ----------
    __JUNCTION_COUNT : int
        An integer stating the total number of junctions in the system

    ----------
    Instance Attributes
    ----------
    junction_id : int
        A unique id number representing the junction

    node_id : int
        A unique id number representing the node associated with the junction

    segment_1_id : int
        The id number associated with the first segment of the junction

    segment_2_id : int
        The id number associated with the second segment of the junction

    segment_3_id : int, optional
        The id number associated with the third segment of the junction (default is None)

    ----------
    Class Methods
    ----------  
    increment_count(amount=1)
        Adds or subtracts from the __JUNCTION_COUNT attribute

    reset_count()
        Resets the __JUNCTION_COUNT attribute

    ----------
    Instance Methods
    ----------
    is_connection()
        Returns true if the junction connects only two segments

    is_bifurcation()
        Returns true if the junction connects three segments

    segments_in_junction()
        Returns the number of segments connected by the junction
    """

    __JUNCTION_COUNT = 0

    def __init__(self, junction_id:int, node_id:int, segment_1_id:int, segment_2_id:int, segment_3_id:int=None): # type: ignore
        """
        Initializes a Junction object with a unique ID, node ID, and segment IDs.

        :param junction_id: A unique id number representing the junction.
        :param node_id: A unique id number representing the node associated with the junction.
        :param segment_1_id: The id number associated with the first segment of the junction.
        :param segment_2_id: The id number associated with the second segment of the junction.
        :param segment_3_id: The id number associated with the third segment of the junction (optional).
        """
        if junction_id is None:
            self.junction_id = self.__JUNCTION_COUNT
        else:
            self.junction_id = junction_id

        self.node_id = node_id
        self.segment_1_id = segment_1_id
        self.segment_2_id = segment_2_id
        self.segment_3_id = segment_3_id

    @classmethod
    def increment_count(cls, amount:int=1):
        """
        Adds or subtracts from the __JUNCTION_COUNT attribute.

        :param amount: The amount to add to the __JUNCTION_COUNT attribute. Defaults to 1.
        """
        cls.__JUNCTION_COUNT += amount

    @classmethod
    def reset_count(cls):
        """
        Resets the __JUNCTION_COUNT attribute to 0.
        """
        cls.__JUNCTION_COUNT = 0

    def is_connection(self) -> bool:
        """
        Checks if the junction connects only two segments.

        :return: True if the junction connects only two segments, otherwise False.
        """
        return self.segment_3_id is None

    def is_bifurcation(self) -> bool:
        """
        Checks if the junction connects three segments.

        :return: True if the junction connects three segments, otherwise False.
        """
        return self.segment_3_id is not None

    def segments_in_junction(self) -> int:
        """
        Returns the number of segments connected by the junction.

        :return: The number of segments connected by the junction.
        """
        return 2 if self.segment_3_id is None else 3

    def __str__(self):
        """
        Returns a string representation of the junction information.

        :return: A string representing the junction's ID, node ID, and associated segment IDs.
        """
        return f"""
        Junction Information:
          Junction ID: {self.junction_id}
          Node ID: {self.node_id}
          Segment 1 ID: {self.segment_1_id}
          Segment 2 ID: {self.segment_2_id}
          Segment 3 ID: {self.segment_3_id}
      """
class Tissue:
    """
    A class used to handle the information needed to define the tissue space.

    ----------
    Class Attributes
    ----------
    __X_LIM : [int, int]
        Min and Max X coordinate values for the tissue

    __Y_LIM : [int, int]
        Min and Max Y coordinate values for the tissue

    __Z_LIM : [int, int]
        Min and Max Z coordinate values for the tissue

    __TISSUE_COUNT : int
        An integer stating the total number of tissues generated in the system

    ----------
    Instance Attributes
    ----------
    tissue_id : int
        A unique id number representing the tissue

    x_values : [int, int]
        The x coordinates defining the opposite sides of the tissue block

    y_values : [int, int]
        The y coordinates defining the opposite sides of the tissue block

    z_values : [int, int]
        The z coordinates defining the opposite sides of the tissue block

    num_cells : [int, int, int]
        A list of integers specifying the number of cells in the tissue mesh in each of the cardinal directions

    cell_type : int
        An int specifying the type of cells making up the mesh of the tissue, 0 = Tetrahedron

    ----------
    Class Methods
    ----------
    increment_count(amount=1)
        Adds or subtracts from the __TISSUE_COUNT attribute

    reset_count()
        Resets the __TISSUE_COUNT attribute

    set_limits([x_min, x_max], [y_min, y_max], [z_min, z_max])
        Sets the values of the limits on valid spatial location coordinates

    ----------
    Instance Methods
    ----------
    bottom_corner()
        Returns the XYZ coordinates of the bottom corner of the tissue

    top_corner()
        Returns the XYZ coordinates of the top corner of the tissue

    define_box([x_min, x_max], [y_min, y_max], [z_min, z_max])
        Sets the values of the limits on valid spatial location coordinates

    define_cells([x_cells, y_cells, z_cells], cell_type)
        Sets the cell characteristics of the tissue mesh

    tissue_limits()
        Returns the limits of the tissue block in XYZ directions

    check_point_in_bounds(point)
        Checks if a given point is within the bounds of the tissue block
    """

    __X_LIM = [0, 100]
    __Y_LIM = [0, 100]
    __Z_LIM = [0, 100]
    __TISSUE_COUNT = 0

    def __init__(self, x_values: List[float], y_values: List[float], z_values: List[float], num_cells: int, config:Config=None, cell_type: int = 0, tissue_id: int = None): # type: ignore
        """
        Initializes a Tissue object with specified coordinates, number of cells, and cell type.

        :param x_values: The x coordinates defining the opposite sides of the tissue block.
        :param y_values: The y coordinates defining the opposite sides of the tissue block.
        :param z_values: The z coordinates defining the opposite sides of the tissue block.
        :param num_cells: A list of integers specifying the number of cells in the tissue mesh in each of the cardinal directions.
        :param cell_type: An int specifying the type of cells making up the mesh of the tissue (default is 0, Tetrahedron).
        :param tissue_id: A unique id number representing the tissue (default is None).
        """
        if tissue_id is None:
            self.tissue_id = self.__TISSUE_COUNT
        else:
            self.tissue_id = tissue_id

        self.define_box(x_values, y_values, z_values)
        self.num_cells = num_cells
        self._cell_type = cell_type
        if config is None:
            self.config = None
            self.logger = None
        else:
            self.config = config
            self.logger = config.logger

    @classmethod
    def increment_count(cls, amount: int = 1):
        """
        Adds or subtracts from the __TISSUE_COUNT attribute.

        :param amount: The amount to add to the __TISSUE_COUNT attribute (default is 1).
        """
        cls.__TISSUE_COUNT += amount

    @classmethod
    def reset_count(cls):
        """
        Resets the __TISSUE_COUNT attribute to 0.
        """
        cls.__TISSUE_COUNT = 0

    @classmethod
    def set_limits(cls, xlim: List[int], ylim: List[int], zlim: List[int]):
        """
        Sets the values of the limits on valid spatial location coordinates.

        :param xlim: A list of integers specifying the min and max X coordinate values.
        :param ylim: A list of integers specifying the min and max Y coordinate values.
        :param zlim: A list of integers specifying the min and max Z coordinate values.
        """
        cls.__X_LIM = xlim
        cls.__Y_LIM = ylim
        cls.__Z_LIM = zlim

    def define_box(self, x_values: List[float], y_values: List[float], z_values: List[float]):
        """
        Sets the values of the limits on valid spatial location coordinates for the tissue block.

        :param x_values: The x coordinates defining the opposite sides of the tissue block.
        :param y_values: The y coordinates defining the opposite sides of the tissue block.
        :param z_values: The z coordinates defining the opposite sides of the tissue block.
        """
        if x_values[0] < Tissue.__X_LIM[0] or x_values[0] > Tissue.__X_LIM[1] or x_values[1] < Tissue.__X_LIM[0] or x_values[1] > Tissue.__X_LIM[1]:
            raise ValueError("The x coordinate of the tissue is not within the spatial limits")
        if y_values[0] < Tissue.__Y_LIM[0] or y_values[0] > Tissue.__Y_LIM[1] or y_values[1] < Tissue.__Y_LIM[0] or y_values[1] > Tissue.__Y_LIM[1]:
            raise ValueError("The y coordinate of the tissue is not within the spatial limits")
        if z_values[0] < Tissue.__Z_LIM[0] or z_values[0] > Tissue.__Z_LIM[1] or z_values[1] < Tissue.__Z_LIM[0] or z_values[1] > Tissue.__Z_LIM[1]:
            raise ValueError("The z coordinate of the tissue is not within the spatial limits")

        self.x_values = x_values
        self.y_values = y_values
        self.z_values = z_values

    @property
    def cell_type(self):
        """
        Returns the type of cells making up the mesh of the tissue.

        :return: A string representing the cell type.
        """
        if self._cell_type == 0:
            return "Tetrahedron"
        else:
            raise ValueError("The cell_type is not correctly set, valid values are currently: 0")

    def define_cells(self, num_cells: List[int], cell_type: int = 0):
        """
        Sets the cell characteristics of the tissue mesh.

        :param num_cells: A list of integers specifying the number of cells in the tissue mesh in each of the cardinal directions.
        :param cell_type: An int specifying the type of cells making up the mesh of the tissue (default is 0, Tetrahedron).
        """
        self.num_cells = num_cells
        self._cell_type = cell_type

    def bottom_corner(self) -> List[float]:
        """
        Returns the XYZ coordinates of the bottom corner of the tissue.

        :return: A list of floats representing the bottom corner coordinates.
        """
        return [self.x_values[0], self.y_values[0], self.z_values[0]]

    def top_corner(self) -> List[float]:
        """
        Returns the XYZ coordinates of the top corner of the tissue.

        :return: A list of floats representing the top corner coordinates.
        """
        return [self.x_values[1], self.y_values[1], self.z_values[1]]

    def tissue_limits(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Returns the limits of the tissue block in XYZ directions.

        :return: Three lists representing the limits in the X, Y, and Z directions.
        """
        xlims = [self.x_values[0], self.x_values[1]]
        ylims = [self.y_values[0], self.y_values[1]]
        zlims = [self.z_values[0], self.z_values[1]]
        return xlims, ylims, zlims

    def check_point_in_bounds(self, point: List[float]) -> bool:
        """
        Checks if a given point is within the bounds of the tissue block.

        :param point: A list of floats representing the point's coordinates.
        :return: True if the point is within the bounds, otherwise False.
        """
        corner1 = np.array(self.bottom_corner())
        corner2 = np.array(self.top_corner())
        point = np.array(point)
        tol = 1e-10

        # Ensure corners are ordered correctly
        min_corner = np.minimum(corner1, corner2)
        max_corner = np.maximum(corner1, corner2)

        # Check if the point is within the bounding box
        return np.all(np.logical_and(min_corner + tol <= point, point <= max_corner - tol))
    
    def get_tissue_volume(self):
        """
        Returns the volume of the tissue in m3.

        :return: Float representing the tissue volume.
        """
        corner1 = np.array(self.bottom_corner())
        corner2 = np.array(self.top_corner())

        volume = np.abs((corner2[0]-corner1[0])*(corner2[1]-corner1[1])*(corner2[2]-corner1[2]))

        return volume
    
    def generate_subdomain(self,volume_percentage=25., gaussian_bias=False):
        """
        Generates a random subdomain box within a given domain.

        Parameters:
        - volume_percentage: float, the volume percentage of the domain that the subdomain should occupy (0 < volume_percentage <= 100).
        - gaussian_bias: bool, if True, centers of subdomains are biased towards the domain center using a Gaussian distribution.

        Returns:
        - subdomain_corners: tuple of two points (min_corner, max_corner) defining the subdomain.
        """
        min_corner = np.array(self.bottom_corner())
        max_corner = np.array(self.top_corner())

        x_min = min_corner[0]
        y_min = min_corner[1]
        z_min = min_corner[2]

        x_max = max_corner[0]
        y_max = max_corner[1]
        z_max = max_corner[2]

        # Calculate domain size and center
        dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
        domain_center = (x_min + dx / 2, y_min + dy / 2, z_min + dz / 2)

        # Calculate subdomain dimensions
        domain_volume = dx * dy * dz
        subdomain_volume = volume_percentage / 100 * domain_volume
        sub_dx = (subdomain_volume ** (1 / 3)) * (dx / (dx * dy * dz) ** (1 / 3))
        sub_dy = (subdomain_volume ** (1 / 3)) * (dy / (dx * dy * dz) ** (1 / 3))
        sub_dz = (subdomain_volume ** (1 / 3)) * (dz / (dx * dy * dz) ** (1 / 3))

        # Determine subdomain center
        if gaussian_bias:
            # Generate center coordinates using a Gaussian distribution centered at the domain center
            cx = np.clip(np.random.normal(domain_center[0], dx / 6), x_min + sub_dx / 2, x_max - sub_dx / 2)
            cy = np.clip(np.random.normal(domain_center[1], dy / 6), y_min + sub_dy / 2, y_max - sub_dy / 2)
            cz = np.clip(np.random.normal(domain_center[2], dz / 6), z_min + sub_dz / 2, z_max - sub_dz / 2)
        else:
            # Uniform random sampling of the center
            cx = np.random.uniform(x_min + sub_dx / 2, x_max - sub_dx / 2)
            cy = np.random.uniform(y_min + sub_dy / 2, y_max - sub_dy / 2)
            cz = np.random.uniform(z_min + sub_dz / 2, z_max - sub_dz / 2)

        # Calculate subdomain corners
        sx_min, sx_max = cx - sub_dx / 2, cx + sub_dx / 2
        sy_min, sy_max = cy - sub_dy / 2, cy + sub_dy / 2
        sz_min, sz_max = cz - sub_dz / 2, cz + sub_dz / 2

        lower_corner = [sx_min, sy_min, sz_min]
        upper_corner = [sx_max, sy_max, sz_max]

        subdomain = tuple(lower_corner), tuple(upper_corner)

        return subdomain
    
    def sample_subdomain_elements(self,tree,subdomain_corners):
        """
        Samples Nodes and Segments within the given subdomain.

        Parameters:
        - Tree: A tree object containing dictionaries of the Node and Segment information
        - subdomain_corners: tuple of two points (min_corner, max_corner) defining the subdomain.

        Returns:
        - subdomain_nodes: dictionary of Nodes within the subdomain.
        - subdomain_segments: dictionary of Segments fully contained within the subdomain.
        """
        min_corner, max_corner = np.array(subdomain_corners[0]), np.array(subdomain_corners[1])
        nodes = tree.node_dict
        segments = tree.segment_dict

        # Filter nodes within the subdomain
        subdomain_nodes = {
            node_id: node for node_id, node in nodes.items()
            if np.all(min_corner <= node.location()) and np.all(node.location() <= max_corner)
        }

        # Filter segments where both nodes are in the subdomain
        subdomain_segments = {
            segment_id: segment
            for segment_id, segment in segments.items()
            if segment.node_1_id in subdomain_nodes and segment.node_2_id in subdomain_nodes
        }

        return subdomain_nodes, subdomain_segments
    
    def get_domain_network(self,tree):
        """
        This function converts the tree into a graph structure and then retrieves the segment IDs associated with each individual vessel segment
        
        :return: A tuple containing:
                - associated_vessels_dict: A dictionary mapping each vessel ID to its corresponding segment {vessel_id:list of segment ids}.
                - adjacency_dict: A dictionary mapping each vessel ID to a set of adjacent vessel IDs {vessel_id:list of vessel ids}..
        """
        graph = GraphRepresentation()
        graph.build_undirected(tree)
        associated_vessels_dict, adjacency_dict = graph.get_vessels()

        return associated_vessels_dict, adjacency_dict
    
    def extract_subdomain_networks(self, tree, subdomain_segments_dict):

        gobal_vessel_dict, global_adjacency_dict = self.get_domain_network(tree)

        segment_list = list(subdomain_segments_dict.keys())
        # Convert segment_list to a set for fast lookup
        segment_set = set(segment_list)
        
        # Filter the vessel dictionary
        reduced_vessel_dict = {
            vessel_id: [seg for seg in segments if seg in segment_set]
            for vessel_id, segments in gobal_vessel_dict.items()
        }
        
        # Remove empty vessels
        reduced_vessel_dict = {v_id: segs for v_id, segs in reduced_vessel_dict.items() if segs}
        
        # Extract remaining and removed vessel IDs
        remaining_vessels = list(reduced_vessel_dict.keys())
        removed_vessels = [v_id for v_id in gobal_vessel_dict if v_id not in reduced_vessel_dict]

         # Convert remaining_vessels to a set for fast lookup
        remaining_set = set(remaining_vessels)
        
        # Filter the adjacency list to keep only remaining vessels
        filtered_adjacency_dict = {
            vessel_id: [adj for adj in adj_list if adj in remaining_set]
            for vessel_id, adj_list in global_adjacency_dict.items()
            if vessel_id in remaining_set  # Only keep keys that are still valid vessels
        }

        visited = set()
        network_dict = {}
        network_id = 0

        def bfs(start_vessel):
            """Breadth-First Search to find all vessels in the same connected component."""
            queue = deque([start_vessel])
            network_vessels = {}

            while queue:
                vessel = queue.popleft()
                if vessel in visited:
                    continue
                visited.add(vessel)
                network_vessels[vessel] = reduced_vessel_dict[vessel]  # Store vessel and its segments
                
                for neighbor in filtered_adjacency_dict.get(vessel, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            return network_vessels

        # Find all connected components
        for vessel in filtered_adjacency_dict:
            if vessel not in visited:
                network_vessels = bfs(vessel)
                if network_vessels:
                    network_dict[network_id] = network_vessels
                    network_id += 1

        return network_dict


        

    
    # def classify_subdomain_networks(self, subdomain_nodes, subdomain_segments):
    #     """
    #     Classifies Nodes and Segments in a Subdomain into disconnected networks.

    #     Parameters:
    #     - subdomain_nodes: dictionary of Nodes in the Subdomain {node_id: (x, y, z)}.
    #     - subdomain_segments: dictionary of Segments in the Subdomain 
    #                         {segment_id: (node1_id, node2_id, radius)}.

    #     Returns:
    #     - network_dict: dictionary of disconnected networks in the form
    #                     {network_id: [network_nodes, network_segments]}.
    #     """
    #     # Create a graph from the Nodes and Segments
    #     G = nx.Graph()
    #     G.add_nodes_from(subdomain_nodes.keys())  # Add nodes
    #     G.add_edges_from((segment.node_1_id, segment.node_2_id) for _,segment in subdomain_segments.items())  # Add edges

    #     # Find connected components
    #     connected_components = list(nx.connected_components(G))
    #     network_dict = {}

    #     for network_id, component_nodes in enumerate(connected_components):
    #         # Get nodes and segments for the current connected component
    #         component_nodes = set(component_nodes)
    #         component_segments = {
    #             seg_id: seg_data
    #             for seg_id, seg_data in subdomain_segments.items()
    #             if seg_data.node_1_id in component_nodes and seg_data.node_2_id in component_nodes
    #         }

    #         # Store the network in the dictionary
    #         network_dict[network_id] = [
    #             {node_id: subdomain_nodes[node_id] for node_id in component_nodes},  # Network nodes
    #             component_segments,  # Network segments
    #         ]

    #     return network_dict
    
    # def classify_vessels(self, network_dict):
    #     """
    #     Classifies segments in each network into vessels.

    #     Parameters:
    #     - network_dict: dictionary of the form {network_id: [network_nodes, network_segments]}.
    #     - network_nodes: dictionary {node_id: (x, y, z)}.
    #     - network_segments: dictionary {segment_id: (node1_id, node2_id, radius)}.

    #     Returns:
    #     - vesselized_network_dict: dictionary of the form {network_id: {vessel_id: list of segment_ids}}.
    #     """
    #     vesselized_network_dict = {}

    #     for network_id, (network_nodes, network_segments) in network_dict.items():
    #         # Build adjacency information
    #         node_to_segments = defaultdict(list)
    #         for segment_id, segment in network_segments.items():
    #             node_to_segments[segment.node_1_id].append(segment_id)
    #             node_to_segments[segment.node_2_id].append(segment_id)

    #         # Identify terminal and bifurcation nodes
    #         branch_or_terminal_nodes = {node for node, segments in node_to_segments.items() if len(segments) != 2}

    #         # Initialize tracking structures
    #         visited_segments = set()
    #         vessels = {}
    #         vessel_id = 0

    #         def traverse_segment_chain(start_segment):
    #             """
    #             Traverses both directions from the start_segment until reaching a branch or terminal node.
    #             """
    #             chain_segments = set()  # Use a set to avoid duplication
    #             to_visit = [start_segment]

    #             while to_visit:
    #                 current_segment = to_visit.pop()
    #                 if current_segment in visited_segments:
    #                     continue

    #                 visited_segments.add(current_segment)
    #                 chain_segments.add(current_segment)

    #                 # Get both nodes of the current segment
    #                 node1 = network_segments[current_segment].node_1_id
    #                 node2 = network_segments[current_segment].node_2_id

    #                 for next_node in (node1, node2):  # Explore both directions
    #                     if next_node in branch_or_terminal_nodes:
    #                         continue  # Stop traversal at a branch or terminal

    #                     for next_segment in node_to_segments[next_node]:
    #                         if next_segment not in visited_segments:
    #                             to_visit.append(next_segment)

    #             return list(chain_segments)

    #         # Iterate through all segments and classify them into vessels
    #         for segment_id, segment in network_segments.items():
    #             if segment_id not in visited_segments:
    #                 # Start a new vessel traversal
    #                 vessel_segments = traverse_segment_chain(segment_id)
    #                 vessels[vessel_id] = vessel_segments
    #                 vessel_id += 1

    #         # Store the vessels for the current network
    #         vesselized_network_dict[network_id] = vessels

    #     return vesselized_network_dict

    def get_subdomain_vessel_statistics(self, vesselized_network_dict, tree):
        """
        Takes the lists of segments in the vessels of the networks of a subdomain and obtains the vascular statistics.

        Parameters:
        - vesselized_network_dict: dictionary of the form {network_id: {vessel_id: list of segment_ids}}.
        - tree: a Tree object containing the original node and segment data.

        Returns:
        - vascular_statistics_dict: A dictionary containing the information for vascular statistics
        - luminal_volume: A float representing the total volume of the vessels.
        """
        network_stats = {}
        luminal_volume = 0
        # Calculate the statistics for each vessel
        for network_id, vessel_id in vesselized_network_dict.items():
            vessel_radii = {}
            vessel_length = {}
            for vessel_id, segment_ids in vesselized_network_dict.items():
                total_length = 0
                total_radii = 0
                for segment_id in segment_ids:
                    length = tree.length(segment_id)
                    total_length += length
                    total_radii += tree.segment_dict[segment_id].radius * length
                    luminal_volume += length * np.pi * (tree.segment_dict[segment_id].radius ** 2)

                mean_radii = total_radii / total_length
                vessel_radii[vessel_id] = mean_radii

                vessel_length[vessel_id] = total_length

                # self.logger.log(f"total_length = {total_length}") # type: ignore 

                network_stats[network_id] = {vessel_id:[mean_radii,total_length]}

        return network_stats, luminal_volume
    
    def sample_n_subdomains(self,n_iterations:int,subdomain_volume_percent:float,tree,generation:int=None,gaussian_bias=True,verbose=False, growth_handler=None): # type: ignore
        """
        Sample n subdomains obtaining the vessels of the networks of a subdomain and the resulting vascular statistics.

        Parameters:
        - n_iterations: Number of subdomains to generate and sample.
        - subdomain_volume_percent: Percentage of volume of the domain each subdomain will occupy.
        - tree: a Tree object containing the original node and segment data.
        - generation: An optional int that indicates an additional generation marker in the saved files. 
        - gaussian_bias: A Boolean indicating whether subdomains should be generated using a gaussian distribution or a uniform distribution.

        Returns:
        - collated_network_stats: A dictionary containing the information for vascular statistics.
        - collated_luminal_volumes: A dictionary containing the total volume of the vessels in the subdomains.
        - expected_subdomain_volume: A float representing the volume of each subdomain.
        """
        collated_network_stats = {}
        collated_luminal_volumes = {}
        expected_subdomain_volume = (subdomain_volume_percent/100) * self.get_tissue_volume()
        subdomain_boundaries = []
        subdomain_points = {}
        subdomain_tissue_data = {}

        for i in range(n_iterations):
            luminal_volume = 0
            collated_network_stats[i] = {}
            subdomain = self.generate_subdomain(subdomain_volume_percent,True)
            subdomain_boundaries.append([subdomain[0][0], subdomain[1][0], 
                  subdomain[0][1], subdomain[1][1], 
                  subdomain[0][2], subdomain[1][2]])
            if verbose:
                self.logger.log(f"Subdomain {i} has boundaries ({subdomain})") # type: ignore
            nodes, segments = self.sample_subdomain_elements(tree,subdomain)
            # networks = self.classify_subdomain_networks(nodes, segments)
            # vesselized_networks = self.classify_vessels(networks)
            vesselized_networks = self.extract_subdomain_networks(tree,segments)
            
            for network_id, vessels in vesselized_networks.items():
                if verbose:
                    self.logger.log(f"Network {network_id}:") # type: ignore
                collated_network_stats[i][network_id] = {}
                for vessel_id, vessel_segments in vessels.items():
                    if verbose:
                        self.logger.log(f"  Vessel {vessel_id}: {vessel_segments}") # type: ignore
                    length_weighted_radii_sum = 0
                    vessel_length = 0
                    for segment_id in vessel_segments:
                        length = tree.length(segment_id)
                        vessel_length += length
                        length_weighted_radii_sum += tree.segment_dict[segment_id].radius * length
                        luminal_volume += length * np.pi * (tree.segment_dict[segment_id].radius ** 2)
                
                    vessel_radii = length_weighted_radii_sum / vessel_length
                    if verbose:
                        self.logger.log(f"  Radii {vessel_radii} | Length {vessel_length}") # type: ignore
                    collated_network_stats[i][network_id][vessel_id] = [vessel_radii,vessel_length]

            if not growth_handler is None:
                lower_corner, upper_corner = subdomain  
                x = np.linspace(lower_corner[0], upper_corner[0], 50)
                y = np.linspace(lower_corner[1], upper_corner[1], 50)
                z = np.linspace(lower_corner[2], upper_corner[2], 50)
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                grid_col = np.hsplit(grid_points, 3)
                growth_handler._get_growth_info()
                vegf_values = growth_handler.sample_vegf_values(grid_col)
                oxygen_values = growth_handler.sample_oxygen_values(grid_col)
                subdomain_tissue_data.update({i:{"VEGF_Values": vegf_values,"Oxygen_Values": oxygen_values}})
            
            collated_luminal_volumes[i] = luminal_volume
   
        # Specify the output CSV filename
        _, _, _, output_path, test_name = self.config.parse_run() # type: ignore
        filepath = os_path_join(output_path,test_name)
        case = self.config.growth_case # type: ignore
        file_path = "./"+filepath+"/statistic_results/"+f"case_{case}/"
        if gaussian_bias:
            csv_filename = file_path+"collated_vessel_data_gauss"
        else:
            csv_filename = file_path+"collated_vessel_data_uniform"

        if not generation is None:
            csv_filename += f"_{generation}"
        csv_filename += ".csv"

        # Writing to CSV
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(["Subdomain", "Network_ID", "Vessel_ID", "Vessel_Radii", "Vessel_Length"])
            
            # Write data
            if verbose:
                self.logger.log(collated_network_stats)  # type: ignore
            for subdomain, data in collated_network_stats.items():
                if isinstance(data, dict):  # Check if it's another dictionary
                    if verbose:
                        self.logger.log(f"Subdomain {subdomain}:")# type: ignore
                    for network_id, vessels in data.items():
                        if isinstance(vessels, dict):  # Check if it's another dictionary
                            if verbose:
                                self.logger.log(f"  Network {network_id}:") # type: ignore
                            for vessel_id, (vessel_radius, vessel_length) in vessels.items():
                                if verbose:
                                    self.logger.log(f"      Vessel {vessel_id}| {vessel_radius}, {vessel_length}") # type: ignore
                                writer.writerow([subdomain, network_id, vessel_id, vessel_radius, vessel_length])

        if gaussian_bias:
            subdomain_filename = file_path+"subdomain_data_gauss"
        else:
            subdomain_filename = file_path+"subdomain_data_uniform"
        if not generation is None:
            subdomain_filename += f"_{generation}"
        subdomain_filename += ".csv"
         # Writing to CSV
        with open(subdomain_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(["Subdomain", "X_min", "X_max", "Y_min", "Y_max", "Z_min", "Z_max", "Luminal_Volume", "Subdomain_volume"])
            for i,subdomain in enumerate(subdomain_boundaries):
                writer.writerow([i, subdomain[0], subdomain[1], subdomain[2], subdomain[3], subdomain[4], subdomain[5],\
                                  collated_luminal_volumes[i], expected_subdomain_volume])
        
        if not growth_handler is None:
            if gaussian_bias:
                vegf_filename = file_path+"tissue_data_gauss"
            else:
                vegf_filename = file_path+"tissue_data_uniform"
            if not generation is None:
                vegf_filename += f"_{generation}"
            vegf_filename += ".csv"
            # Writing to CSV
            with open(vegf_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                
                # Write header
                writer.writerow(["Subdomain", "VEGF_Values", "Oxygen_Values"])
                
                for subdomain, data in subdomain_tissue_data.items():
                    vegf_values = data["VEGF_Values"]
                    oxygen_values = data["Oxygen_Values"]
                    
                    for vegf, oxygen in zip(vegf_values, oxygen_values):
                        writer.writerow([subdomain, vegf, oxygen])
        

        return collated_network_stats, collated_luminal_volumes, expected_subdomain_volume


    def __str__(self):
        """
        Returns a string representation of the tissue information.

        :return: A string representing the tissue's ID, coordinates, number of cells, and cell type.
        """
        return f"""
        Tissue Information:
          Tissue ID: {self.tissue_id}
          Bottom coordinate: {self.bottom_corner()}
          Top coordinate: {self.top_corner()}
          Mesh elements: {self.num_cells}
          Cell Type: {self.cell_type()} """ # type: ignore

class TissueHandler(Tissue):
    """
    A class used to handle the information needed to define the tissue space.

    ----------
    Instance Attributes
    ----------
    tissue_dict : Dict[int, Tissue]
        A dictionary containing Tissue class instances

    tissue_age : int
        An integer recording the number of generations the Tissue has gone through

    ----------
    Class Methods
    ----------
    load_file(filepath: str, age: int)
        Returns class instance from file

    ----------
    Instance Methods
    ----------
    save_file(filepath: str)
        Saves class instance to file

    add_tissue(x_values: List[int], y_values: List[int], z_values: List[int], num_cells: int, cell_type: int = 0, tissue_id: int = None)
        Adds a new Tissue instance to the handler

    grow_older()
        Increment the tissue age

    volume(tissue_id: int, verbose: bool = False, units: str = "um3")
        Returns the volume of a tissue instance

    count_tissues()
        Returns the length of the tissue dictionary

    current_tissue()
        Returns the tissue which is currently most recent
    """

    def __init__(self, tissues: Dict[int, Tissue] = None, age: int = None): # type: ignore
        """
        Initializes a TissueHandler object with specified tissues and age.

        :param tissues: A dictionary containing Tissue class instances (default is None).
        :param age: An integer recording the number of generations the Tissue has gone through (default is None).
        """
        Tissue.reset_count()
        self.tissue_dict = {}
        if tissues is not None:
            for key, obj in tissues.items():
                self.tissue_dict.update({int(key): Tissue(obj["x_values"], obj["y_values"], obj["z_values"], obj["_num_cells"], obj["_num_cells"], obj["tissue_id"])}) # type: ignore

        if age is not None:
            self._tissue_age = age
        else:
            self._tissue_age = 0

        self.config = None

        Tissue.increment_count(self.count_tissues())


    def set_config(self,config:Config):
        self.config = config
        self.logger = config.logger
        return

    @staticmethod
    def to_jsons(dict: Dict[int, Tissue]) -> str:
        """
        Converts a dictionary of Tissue objects to a JSON string.

        :param dict: A dictionary containing Tissue class instances.
        :return: A JSON string representation of the dictionary.
        """
        json_string = ''
        for key, obj in dict.items():
            json_string += json.dumps({key: obj.__dict__}) 
            json_string += ";\;" # type: ignore
        return json_string

    @classmethod
    def load_file(cls, tissue_path: str, age: int):
        """
        Returns a class instance from a file.

        :param tissue_path: The file path to load the tissue data from.
        :param age: An integer recording the number of generations the Tissue has gone through.
        :return: An instance of the TissueHandler class.
        """
        with open(tissue_path, "r") as f:
            data = json.load(f)
            tissues = {}
            for partial in data.split(";\;")[:-1]: # type: ignore
                tissues.update(json.loads(partial))
        
            return cls(tissues, age)

    def save_file(self, tissue_path: str):
        """
        Saves the class instance to a file.

        :param tissue_path: The file path to save the tissue data to.
        """
        with open(tissue_path, "w") as f:
            save_string = self.to_jsons(self.tissue_dict)
            json.dump(save_string, f)

    def add_tissue(self, x_values: List[float], y_values: List[float], z_values: List[float], num_cells: int, cell_type: int = 0, tissue_id: int = None) -> int:  # type: ignore
        """
        Adds a new Tissue instance to the handler.

        :param x_values: The x coordinates defining the opposite sides of the tissue block.
        :param y_values: The y coordinates defining the opposite sides of the tissue block.
        :param z_values: The z coordinates defining the opposite sides of the tissue block.
        :param num_cells: A list of integers specifying the number of cells in the tissue mesh in each of the cardinal directions.
        :param cell_type: An int specifying the type of cells making up the mesh of the tissue (default is 0, Tetrahedron).
        :param tissue_id: A unique id number representing the tissue (default is None).
        :return: The tissue_id of the newly added tissue.
        """
        temp_tissue = Tissue(x_values, y_values, z_values, num_cells, self.config, cell_type, tissue_id) # type: ignore
        self.tissue_dict.update({temp_tissue.tissue_id: temp_tissue})
        Tissue.increment_count()

        return temp_tissue.tissue_id

    def volume(self, tissue_id: int, verbose: bool = False, units: str = "um3") -> float:
        """
        Returns the volume of a tissue instance.

        :param tissue_id: The unique id number representing the tissue.
        :param verbose: If True, prints the volume of the tissue (default is False).
        :param units: The units of the volume ("um3" or "m3", default is "um3").
        :return: The volume of the tissue instance.
        """
        if tissue_id in self.tissue_dict:
            bottom_corner = self.tissue_dict[tissue_id].bottom_corner()
            top_corner = self.tissue_dict[tissue_id].top_corner()

            volume = (top_corner[0] - bottom_corner[0]) * (top_corner[1] - bottom_corner[1]) * (top_corner[2] - bottom_corner[2])
        else:
            raise ValueError("The provided tissue_id does not correspond to an existing tissue")
        
        if units == "um3":
            volume *= 1e18
        elif units == "m3":
            volume = volume
        else: 
            raise ValueError("The provided units string is not valid, use 'um3' or 'm3'")

        if verbose:
            print("The Volume of the tissue is: ", volume, units)
        
        return volume

    @property
    def tissue_age(self) -> int:
        """
        Returns the age of the tissue.

        :return: The age of the tissue.
        """
        return self._tissue_age
        
    @tissue_age.setter
    def tissue_age(self, new_age: int):
        """
        Sets the age of the tissue.

        :param new_age: The new age of the tissue.
        """
        if new_age < 0:
            raise ValueError("The age of the tissue must be >= 0")
        else:
            self._tissue_age = new_age

    def count_tissues(self) -> int:
        """
        Returns the number of tissues in the tissue dictionary.

        :return: The length of the tissue dictionary.
        """
        return int(length_hint(self.tissue_dict))

    def current_tissue(self) -> Tissue:
        """
        Returns the most recent tissue.

        :return: The tissue which is currently most recent.
        """
        return self.tissue_dict[self.count_tissues() - 1]

    def __str__(self):
        """
        Returns a string representation of the tissue handler information.

        :return: A string representing the number of tissues and the tissue handler age.
        """
        return f"""
        Tissue Handler Information:
          Number of Tissues: {self.count_tissues()}
          Tissue Handler Age: {self.tissue_age}
      """

class Tree(Node,Segment,Junction):
    """
    A class used to handle the information needed to define the entire vascular tree.

    ----------
    Class Attributes
    ----------
    None
    ----------
    Instance Attributes
    ----------
    segment_dict : Dictionary of Segment instances
        A dictionary containing Segment class instances

    node_dict : Dictionary of Node instances
        A dictionary containing Node class instances

    junction_dict : Dictionary of Junction instances
        A dictionary containing Junction class instances

    tree_age : int
        An integer recording the number of generations the tree has gone through

    ----------
    Class Methods
    ----------  
    load_file(filepath)
        Returns class instance from file

    ----------
    Instance Methods
    ----------  
    save_file(filepath)
        Saves class instance to file
    
    add_node(spatial_location)
        Adds a new node to the vascular tree

    add_inlet(spatial_location)
        Adds a new inlet node to the vascular tree

    add_outlet(spatial_location)
        Adds a new outlet node to the vascular tree

    add_segment(segment_id, radius, node_1_id, node_2_id)
        Adds a new segment to the vascular tree
    
    add_junction(junction_id, node_id, segment_1_id, segment_2_id, segment_3_id = None)
        Adds a new junction to the vascular tree

    break_segment(segment_id, new_node_location)
        Breaks a segment into two segments by adding a new node in the middle that is associated with two segments and serves as a junction.

    grow_older()
        Increment the tree age

    length(segment_id, verbose = False)
        Returns the distance between node 1 and node 2 in the segment

    volume(segment_id, verbose = False)
        Returns the volume, length, and area of a segment

    count_nodes()
        Returns the length of the node list

    count_segments()
        Returns the length of the segment list

    count_junctions()
        Returns the length of the junction list

    get_tangent_versor(segment_id)
        Returns the tangent versors in the x, y, and z direction for a specific segment

    get_segment_node_locations(segment_id)
        Returns the x,y,z locations of the two nodes associated with a segment

    push_nodes(Tissue)
        Pushes node locations so that they are internal to the bounding box of the tissue

    apply_maximum_segment_length(cell_size)
        Breaks every segment such that no segment has length > cell_size
    """

    def __init__(self, nodes:Dict[int,Node]=None, segments:Dict[int,Segment]=None, junctions:Dict[int,Junction]=None, age:int=None): # type: ignore
        Node.reset_count()
        Segment.reset_count()
        Junction.reset_count()

        self.node_dict = {}
        if nodes != None:
            for key, object in nodes.items():
                self.node_dict.update({int(key): Node(object["node_id"],object["spatial_location"],object["_node_type"],object["junction_id"])}) # type: ignore
        
        self.segment_dict = {}
        if segments != None:
            for key, object in segments.items():
                self.segment_dict.update({int(key): Segment(object["segment_id"],object["_radius"],object["node_1_id"],object["node_2_id"])}) # type: ignore
        
        self.junction_dict = {}
        if junctions != None:
            for key, object in junctions.items():
                self.junction_dict.update({int(key): Junction(object["junction_id"],object["node_id"],object["segment_1_id"],object["segment_2_id"],object["segment_3_id"])}) # type: ignore
        
        if age != None:
            self._tree_age = age
        else:
            self._tree_age = 0

        Node.increment_count(self.count_nodes())
        Segment.increment_count(self.count_segments())
        Junction.increment_count(self.count_junctions())

    @staticmethod
    def to_jsons(dict:Dict):
        """
        Converts a dictionary object to a json string for file saving purposes

        :param dict: The dict to be converted to json
        :return: The json string to be used for saving to file.
        """
        def convert(obj):
            # If the object is a numpy ndarray, convert it to a list
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # If the object has a __dict__, return that; otherwise, use the object itself
            return obj.__dict__ if hasattr(obj, '__dict__') else obj

        json_string = ''
        for key, object in dict.items():
            # Serialize the object with the custom convert function
            json_string += json.dumps({key: convert(object)}, default=convert)
            json_string += ";\;"
        return json_string
    
    @classmethod
    def load_from_config(cls, config:Config, aged:bool=False, age=0):
        """
        Loads an instance of the tree from file.

        :param config: A config object which contains the filepath to the location the tree data is saved
        :return: A Tree class object loaded from file.
        """
        if aged is True:
            config.set_age_to_load(age)
            node_path = config.return_filepath("node",True)
            segment_path = config.return_filepath("node",True)
            junction_path = config.return_filepath("node",True)
        elif aged is False:
            age = config.age_to_load
            node_path = config.return_filepath("node")
            segment_path = config.return_filepath("segment")
            junction_path = config.return_filepath("junction")
        
        
        with open(node_path, "r") as f:
            data = json.load(f)
            nodes = {}
            for partial in data.split(";\;")[:-1]:
                nodes.update(json.loads(partial))
        
        with open(segment_path, "r") as f:
            data = json.load(f)
            segments = {}
            for partial in data.split(";\;")[:-1]:
                segments.update(json.loads(partial))
        
        with open(junction_path, "r") as f:
            data = json.load(f)
            junctions = {}
            for partial in data.split(";\;")[:-1]:
                junctions.update(json.loads(partial))
        
        class_object = cls(nodes, segments, junctions, age)
        class_object.logger = config.logger # type: ignore

        return class_object

    def save_from_config(self, config:Config):
        """
        Saves the current Tree class object to file

        :param config: A config object which contains the filepath to the location the tree data is saved.
        """
        node_path = config.return_filepath("node",True)
        segment_path = config.return_filepath("segment",True)
        junction_path = config.return_filepath("junction",True)

        with open(node_path, "w") as f:
            save_string = self.to_jsons(self.node_dict)
            json.dump(save_string, f)
            config.logger.log(f"Nodes Saved to {node_path}")
        with open(segment_path, "w") as f:
            save_string = self.to_jsons(self.segment_dict)
            json.dump(save_string, f)
            config.logger.log(f"Segments Saved to {segment_path}")
        with open(junction_path, "w") as f:
            save_string = self.to_jsons(self.junction_dict)
            json.dump(save_string, f)
            config.logger.log(f"Junctions Saved to {junction_path}")
        

    def add_node(self, spatial_location:List[float], node_id:int = None): # type: ignore
        """
        Adds a node to the tree

        :param spatial_location: A list detailing the x,y,z coordinates of the node.
        :param node_id: An optional parameter specifying the desired node ID.
        :return: The ID of the created node.
        """
        for other_node_id, node in self.node_dict.items():
            if all(np.isclose(node.location(),spatial_location,atol=1e-10)):
                return other_node_id
        temp_node = Node(node_id, spatial_location, 2, None) # type: ignore
        self.node_dict.update({temp_node.node_id: temp_node})
        Node.increment_count()

        
        return temp_node.node_id
        
    
    def add_inlet(self, spatial_location:List[float], node_id:int = None): # type: ignore
        """
        Adds an inlet node to the tree

        :param spatial_location: A list detailing the x,y,z coordinates of the node.
        :param node_id: An optional parameter specifying the desired node ID.
        :return: The ID of the created inlet node.
        """
        for other_node_id, node in self.node_dict.items():
            if all(np.isclose(node.location(),spatial_location,atol=1e-10)):
                if node._node_type != 0:
                    Node.increment_inlet()
                    node._node_type = 0
                return other_node_id
        temp_node = Node(node_id, spatial_location, 0, None) # type: ignore
        self.node_dict.update({temp_node.node_id: temp_node})
        Node.increment_count()
        Node.increment_inlet()

        return temp_node.node_id

    def add_outlet(self, spatial_location:List[float], node_id:int = None): # type: ignore
        """
        Adds an outlet node to the tree

        :param spatial_location: A list detailing the x,y,z coordinates of the node.
        :param node_id: An optional parameter specifying the desired node ID.
        :return: The ID of the created outlet node.
        """
        for other_node_id, node in self.node_dict.items():
            if all(np.isclose(node.location(),spatial_location,atol=1e-10)):
                if node._node_type != 1:
                    Node.increment_outlet()
                    node._node_type = 1
                return other_node_id
        temp_node = Node(node_id, spatial_location, 1, None) # type: ignore
        self.node_dict.update({temp_node.node_id: temp_node})
        Node.increment_count()
        Node.increment_outlet()

        return temp_node.node_id

    def add_segment(self, node_1_id:int, node_2_id:int, segment_id:int = None, radius:float = None): # type: ignore
        """
        Adds a segment to the tree

        :param node_1_id: The node ID of one end of the segment.
        :param node_2_id: The node ID of the other end of the segment.
        :param segment_id: An optional parameter specifying the desired segment ID.
        :param radius: An optional parameter specifying the radius value of the segment.
        :return: The ID of the created segment.
        """
        # NEED TO CHECK IF THE NODES THAT ARE BEING ADDED EXIST
        if not (node_1_id in self.node_dict and node_2_id in self.node_dict):
            raise ValueError("Neither provided node id corresponded to an existing node")
        if not node_1_id in self.node_dict:
            raise ValueError("The provided node_1_id does not correspond to an existing node")
        if not node_2_id in self.node_dict:
            raise ValueError("The provided node_2_id does not correspond to an existing node")


        new = True
        # NEED TO CHECK IF THE NODES ALREADY HAVE A CONNECTING SEGMENT
        node_1_segments, _ = self.get_segment_ids_on_node(node_1_id)
        node_2_segments, _ = self.get_segment_ids_on_node(node_2_id)

        if len(node_1_segments) > 0:
            for seg_id in node_1_segments:
                if seg_id in node_2_segments:
                    new = False
                    segment_id = seg_id
                    break

        if new:
            temp_segment = Segment(segment_id, radius, node_1_id, node_2_id)
            self.segment_dict.update({temp_segment.segment_id: temp_segment})
            Segment.increment_count()
            segment_id = temp_segment.segment_id

        return segment_id

    def add_junction(self, node_id:int, segment_1_id:int, segment_2_id:int, segment_3_id:int = None, junction_id:int = None): # type: ignore
        """
        Adds a junction to the tree

        :param node_id: The node ID upon which the junction is located.
        :param segment_1_id: The segment ID of the first segment connected to the junction.
        :param segment_2_id: The segment ID of the second segment connected to the junction.
        :param segment_3_id: An optional segment ID of the third segment connected to the junction if it is a bifurcation.
        :param junction_id: An optional parameter specifying the desired junction ID.
        :return: The ID of the created Junction.
        """
        # NEED TO CHECK IF THE NODES AND SEGMENTS THAT ARE BEING ADDED EXIST
        if not node_id in self.node_dict:
            raise ValueError("The provided node_id does not correspond to an existing node")

        segment_1_real = False
        segment_2_real = False
        segment_3_real = False

        if segment_1_id in self.segment_dict:
            segment_1_real = True
        if segment_2_id in self.segment_dict:
            segment_2_real = True
        if segment_3_id in self.segment_dict:
            segment_3_real = True

        if segment_3_id != None and segment_1_real == False and segment_2_real == False and segment_3_real == False:
            raise ValueError("None of the provided segment_id's correspond to existing segments")
        if segment_1_real == False and segment_2_real == False:
            raise ValueError("The provided segment_1_id and the segment_2_id do not correspond to existing segments")
        if segment_3_id != None and segment_1_real == False and segment_3_real == False:
            raise ValueError("The provided segment_1_id and the segment_3_id do not correspond to existing segments")
        if segment_3_id != None and segment_3_real == False and segment_3_real == False:
            raise ValueError("The provided segment_2_id and the segment_3_id do not correspond to existing segments")
        if segment_1_real == False:
            raise ValueError("The provided segment_1_id does not correspond to an existing segment")
        if segment_2_real == False:
            raise ValueError("The provided segment_2_id does not correspond to an existing segment")
        if segment_3_id != None and segment_3_real == False:
            raise ValueError("The provided segment_3_id does not correspond to an existing segment")

        temp_junction = Junction(junction_id, node_id, segment_1_id, segment_2_id, segment_3_id)
        self.junction_dict.update({temp_junction.junction_id: temp_junction})
        Junction.increment_count()

        # ADD JUNCTION INFORMATION TO THE NODE
        self.node_dict[node_id].junction_id = temp_junction.junction_id

        return temp_junction.junction_id
    
    def remove_node(self, node_id:int):
        """
        Removes a node from the tree

        :param node_id: The node ID of the node to be removed.
        """
        # NEED TO CHECK THAT THE NODE BEING REMOVED EXISTS
        if not node_id in self.node_dict:
            raise ValueError(f"Attempted to remove nonexistant node id: {node_id}")
        
        self.node_dict.pop(node_id)
    
    def remove_segment(self, segment_id:int):
        """
        Removes a segment from the tree

        :param segment_id: The segment ID of the segment to be removed.
        """
        # NEED TO CHECK THAT THE SEGMENT BEING REMOVED EXISTS
        if not segment_id in self.segment_dict:
            raise ValueError(f"Attempted to remove nonexistant segment id: {segment_id}")
        
        popped_seg:Segment = self.segment_dict.pop(segment_id)
        node_1_id = popped_seg.node_1_id
        node_2_id = popped_seg.node_2_id

        segs_on_node, _ = self.get_segment_ids_on_node(node_1_id)
        if len(segs_on_node) == 0:
            self.remove_node(node_1_id)

        segs_on_node, _ = self.get_segment_ids_on_node(node_2_id)
        if len(segs_on_node) == 0:
            self.remove_node(node_2_id)
            
        return

    @property
    def tree_age(self):
        """
        Returns the age parameter of the tree

        :return: The age of the tree.
        """
        return self._tree_age
        
    @tree_age.setter
    def tree_age(self, new_age:int):
        """
        Sets the age of the tree

        :param new_age: The age the tree is to be set to
        """
        if new_age < 0:
            raise ValueError("The age of the tree must be >= 0")
        else:
            self._tree_age = new_age

    def grow_older(self, amount = 1, verbose=False):
        """
        Changes the age of the tree

        :param amount: The amount to change the age of the tree
        :param verbose: Boolean determining if the resulting age should be printed
        """
        self.tree_age += amount
        if verbose:
            print("The tree has grown to be", self.tree_age, "generations old")

    def length(self, segment_id:int, verbose=False) -> float:
        """
        Returns the length of the specified segment

        :param segment_id: The ID of the segment to get the length of.
        :param verbose: Boolean determining if the length should be printed
        :return: The length of the segment
        """
        
        if segment_id in self.segment_dict:
            node_1_id = self.segment_dict[segment_id].node_1_id
            node_2_id = self.segment_dict[segment_id].node_2_id
        else:
            raise ValueError("The provided segment_id does not correspond to an existing segment")
        
        if verbose:
            print("Segment ", segment_id, " is assoicated with node ", node_1_id, " and node ", node_2_id)

        if node_1_id in self.node_dict:
                node_1_pos = self.node_dict[node_1_id].spatial_location
        else:
            raise ValueError("Node id 1 associated with the segment does not correspond to an existing node")
        if node_2_id in self.node_dict:
                node_2_pos = self.node_dict[node_2_id].spatial_location
        else:
            raise ValueError("Node id 2 associated with the segment does not correspond to an existing node")
        
        length = np.linalg.norm(np.array(node_1_pos)-np.array(node_2_pos))

        if verbose:
            print(f"The length of segment {segment_id} is: {length}")
            print(f"Position 1: {node_1_pos}")
            print(f"Position 2: {node_2_pos}")
        
        return length


    def area(self, segment_id:int, verbose=False) -> float: # type: ignore
        """
        Returns the cross sectional area of the specified segment

        :param segment_id: The ID of the segment to get the cross sectional area of.
        :param verbose: Boolean determining if the area should be printed
        :return: The cross sectional area of the segment
        """
        if segment_id in self.segment_dict:
            area = self.segment_dict[segment_id].area()
        else:
            raise ValueError("The provided segment_id does not correspond to an existing segment")
        
        if verbose:
            print("The Area of the segment is: ", area)

        return area

    def volume(self, segment_id:int, verbose=False) -> float:
        """
        Returns the volume of the specified segment

        :param segment_id: The ID of the segment to get the cross sectional area of.
        :param verbose: Boolean determining if the volume should be printed
        :return: The volume of the segment
        """
        if segment_id in self.segment_dict:
            area = self.segment_dict[segment_id].area()
        else:
            raise ValueError("The provided segment_id does not correspond to an existing segment")
        
        length = self.length(segment_id, verbose)
        volume = length * area

        if verbose:
            print("The Volume of the segment is: ", volume)
        
        return volume
    
    def get_network_volume(self, verbose=False) -> float:
        """
        Returns the total luminal volume of the tree

        :param verbose: Boolean determining if the total luminal volume should be printed.
        :return: The total luminal volume of the tree.
        """
        total_volume = 0
        for seg_id in self.segment_dict.keys():
            local_volume = self.volume(seg_id)
            total_volume += local_volume

        if verbose:
            print(f"The total luminal volume of the network is: {total_volume}")

        return total_volume

    def get_network_surface(self, verbose=False) -> float:
        """
        Returns the total surface area of the tree

        :param verbose: Boolean determining if the total surface area should be printed.
        :return: The total surface area of the tree.
        """
        total_surface = 0
        for seg_id in self.segment_dict.keys():
            local_circumference = self.segment_dict[seg_id].circumference()
            local_length = self.length(seg_id)
            total_surface += local_length * local_circumference

        if verbose:
            print(f"The total surface area of the network is: {total_surface}")

        return total_surface

    def break_segment(self, segment_id:int, new_node_location:List[float]) -> Tuple[int,List]:
        """
        Create a new node on the tree and then break a segment such that both ends of the segment now connect to the new node rather than eachother.

        :param segment_id: The ID of the segment to be broken.
        :param new_node_location: a list detailing the x,y,z coordiantes of the new node in space
        :return: The ID of the new node and a list of the resulting segment IDs.
        """
        if segment_id in self.segment_dict:
            node_1_id = self.segment_dict[segment_id].node_1_id
            node_2_id = self.segment_dict[segment_id].node_2_id

            node_1_loc = self.node_dict[node_1_id].location()
            node_2_loc = self.node_dict[node_2_id].location()

            if np.allclose(new_node_location, node_1_loc, 1e-9):
                if self.get_junction_on_node(node_1_id) is False:
                    seg_ids = self.get_segment_ids_on_node(node_1_id)
                    self.add_junction(node_1_id, seg_ids[0][0], seg_ids[0][1])
                return node_1_id, [segment_id]
            if np.allclose(new_node_location, node_2_loc, 1e-9):
                if self.get_junction_on_node(node_2_id) is False:
                    seg_ids = self.get_segment_ids_on_node(node_2_id)
                    self.add_junction(node_2_id, seg_ids[0][0], seg_ids[0][1])
                return node_2_id, [segment_id]
        else:
            raise ValueError("Segment ID is not valid")

        new_node_id = self.add_node(new_node_location)
        self.segment_dict[segment_id].node_2_id = new_node_id

        new_segment_id = self.add_segment(new_node_id,node_2_id, radius=self.segment_dict[segment_id].radius)
        self.add_junction(new_node_id, int(segment_id), new_segment_id)

        if self.node_dict[node_2_id].node_type == 2:
            if self.junction_dict[self.node_dict[node_2_id].junction_id].segment_1_id == segment_id:
                self.junction_dict[self.node_dict[node_2_id].junction_id].segment_1_id = new_segment_id
            if self.junction_dict[self.node_dict[node_2_id].junction_id].segment_2_id == segment_id:
                self.junction_dict[self.node_dict[node_2_id].junction_id].segment_2_id = new_segment_id
            if self.junction_dict[self.node_dict[node_2_id].junction_id].segment_3_id == segment_id:
                self.junction_dict[self.node_dict[node_2_id].junction_id].segment_3_id = new_segment_id

        return new_node_id, [segment_id,new_segment_id]

    

    def get_fractional_segment_position(self, segment_id:int, fraction:float) -> List[float]:
        """
        Returns the x,y,z coordinates of a specified point on the line made by a segment between node 1 and node 2.

        :param segment_id: The ID of the segment in question.
        :param fraction: The fraction between [0,1] that determines how far along the segments line to return
        :return: x,y,z coordinates
        """
        if not (fraction >= 0 and fraction <= 1):
            raise ValueError(f"Attempted to access fraction {fraction}, fraction must be in the range [0,1]")
        node_first_id = self.segment_dict[segment_id].node_1_id
        node_last_id = self.segment_dict[segment_id].node_2_id

        node_first_pos = self.node_dict[node_first_id].location()
        node_last_pos = self.node_dict[node_last_id].location()

        fractional_pos = (1-fraction)*node_first_pos + fraction*node_last_pos
        return fractional_pos

    def check_node_in_junction(self,node_id:int) -> bool:
        """
        Returns a boolean describing if the node ID exists at a bifurcation of the network

        :param node_id: The ID of the node in question.
        :return: True/False, Is the node at a bifurcation?
        """
        for keys in self.junction_dict.keys():
            if self.junction_dict[keys].node_id == node_id:
                return self.junction_dict[keys].is_bifurcation()
        return False

    def get_junction_on_node(self,node_id:int) -> Union[int,bool]:
        """
        Returns the junction ID associated with a node

        :param node_id: The ID of the node in question.
        :return: Junction ID/False
        """
        for keys in self.junction_dict.keys():
            if self.junction_dict[keys].node_id == node_id:
                return self.junction_dict[keys].junction_id
        return False

    def check_node_internal(self,node_id:int) -> bool:
        """
        Checks if the node is marked as 'Internal' in the tree

        :param node_id: The ID of the node in question.
        :return: True/False Internal or Not
        """
        return self.node_dict[node_id].node_type() == "Internal"

    def get_node_ids_inlet(self) -> List:
        """
        Returns the list of nodes marked as inlets

        :return: List of node IDs asscoiated with nodes marked as inlets
        """
        inlet_list = []
        for id, node in self.node_dict.items():
            if node.node_type() == "Inlet":
                inlet_list.append(id)
        return inlet_list

    def get_node_ids_outlet(self) -> List:
        """
        Returns the list of nodes marked as outlets

        :return: List of node IDs asscoiated with nodes marked as outlets
        """
        outlet_list = []
        for id, node in self.node_dict.items():
            if node.node_type() == "Outlet":
                outlet_list.append(id)
        return outlet_list

    def get_node_ids_internal(self) -> List:
        """
        Returns the list of nodes marked as internal

        :return: List of node IDs asscoiated with nodes marked as internal
        """
        internal_list = []
        for id, node in self.node_dict.items():
            if node.node_type() == "Internal":
                internal_list.append(id)
        return internal_list

    def get_node_ids_by_type(self) -> Tuple[List,List,List]:
        """
        Returns the tuple of the lists of nodes that are inlets, outlets, and internal

        :return: Tuple of the Lists of nodes that are inlets, outlets, and internal
        """
        inlets = self.get_node_ids_inlet()
        outlets = self.get_node_ids_outlet()
        internals = self.get_node_ids_internal()
        return inlets,outlets,internals
    
    def get_node_ids_on_segment(self,segment_id:int) -> Tuple[int,int]:
        """
        Finds the Node IDs associated with the prescribed segment

        :param segment_id: The segment ID in question
        :return: Node 1 ID, Node 2 ID
        """
        
        node_1 = self.segment_dict[segment_id].node_1_id
        node_2 = self.segment_dict[segment_id].node_2_id
        return node_1, node_2

    def get_segment_ids_on_node(self,node_id:int) -> Tuple[List,List]:
        """
        Finds the segment IDs and presumed side of the segment associated with the node ID

        :param node_id: The node ID in question
        :return: List of segment ID's, List of segment sides.
        """
        
        segment_ids = []
        node_side = []
        for keys in self.segment_dict.keys():
            if self.segment_dict[keys].node_1_id == node_id:
                segment_ids.append(keys)
                node_side.append(1)
            if self.segment_dict[keys].node_2_id == node_id:
                segment_ids.append(keys)
                node_side.append(2)
        return segment_ids, node_side

    def count_segments_connected_to_node(self, node_id:int) -> int:
        """
        Counts the segments associated with the node ID

        :param node_id: The node ID in question.
        :return: the number of segments containing that node.
        """
        count = 0
        for keys in self.segment_dict.keys():
            if self.segment_dict[keys].node_1_id == node_id:
                count += 1
            if self.segment_dict[keys].node_2_id == node_id:
                count += 1
        return count

    def count_nodes(self) -> int:
        """
        Returns the number of nodes currently in the tree

        :return: The number of nodes currently in the tree
        """
        return int(length_hint(self.node_dict))
    
    def count_segments(self) -> int:
        """
        Returns the number of segments currently in the tree

        :return: The number of segments currently in the tree
        """
        return int(length_hint(self.segment_dict))
    
    def count_junctions(self) -> int:
        """
        Returns the number of junctions currently in the tree

        :return: The number of junctions currently in the tree
        """
        return int(length_hint(self.junction_dict))

    def specify_radius(self, segment_id:int, radius:float):
        """
        Changes the radius of a segment on the tree

        :param segment_id: The segment you want to change the radius of
        :param radius: The radius you want to change the segment to
        """
        self.segment_dict[segment_id].radius = radius

    def reset_node_and_segment_numbering(self):
        """
        Resets the ID numbers associated with each node and segment.
        After this method is used, it is recommended to run .populate_junctions() to correct the junction information for the new numbering.
        """
        # Create a mapping from old node IDs to new continuous node IDs
        old_node_ids = sorted(self.node_dict.keys())
        new_node_id_map = {old_id: new_id for new_id, old_id in enumerate(old_node_ids)}

        # Create the new node dictionary with continuous IDs
        new_node_dict = {new_id: self.node_dict[old_id] for old_id, new_id in new_node_id_map.items()}
        # Update the nodes' local IDs
        for new_id, node in new_node_dict.items():
            node.node_id = new_id

        # Update the node dictionary
        self.node_dict = new_node_dict

        # Update the node IDs in the segments
        for segment in self.segment_dict.values():
            segment.node_1_id = new_node_id_map[segment.node_1_id]
            segment.node_2_id = new_node_id_map[segment.node_2_id]

        # Create a mapping from old segment IDs to new continuous segment IDs
        old_seg_ids = sorted(self.segment_dict.keys())
        new_seg_id_map = {old_id: new_id for new_id, old_id in enumerate(old_seg_ids)}

        # Create the new segment dictionary with continuous IDs
        new_segment_dict = {new_id: self.segment_dict[old_id] for new_id, old_id in new_seg_id_map.items()}
        # Update the segments' local IDs
        for new_id, segment in new_segment_dict.items():
            segment.segment_id = new_id

        # Update the segment dictionary
        self.segment_dict = new_segment_dict

        Node.reset_count()
        Node.increment_count(len(self.node_dict.keys()))
        Segment.reset_count()
        Segment.increment_count(len(self.segment_dict.keys()))

        return
    
    def _set_node_and_segement_numbers_by_bfs(self):
        """
        Resets the ID numbers associated with each node and segment.
        This method starts from an inlet node and then explores the entire tree using bfs renumbering the nodes and segments.
        After this method is used, it is recommended to run .populate_junctions() to correct the junction information for the new numbering.
        """
        inlet_id_numbers = self.get_node_ids_inlet()
        if len(inlet_id_numbers) == 0:
            raise ValueError(f"Cannot renumber from inlets using dfs since there are no inlets in the tree")
        first_inlet_id = inlet_id_numbers[0]
        node_num = 0
        segment_num = 0

        node_dict = {first_inlet_id:node_num}
        node_num += 1
        segment_dict = {}

        seen_node_ids = [first_inlet_id]
        seen_segment_ids = []
        segment_id_queue = []
        seg_ids,_ = self.get_segment_ids_on_node(first_inlet_id)
        for seg_id in seg_ids:
            if not seg_id in seen_segment_ids:
                segment_id_queue.append(seg_id)
                seen_segment_ids.append(seg_id)
                segment_dict.update({seg_id:segment_num})
                segment_num += 1

        while len(segment_id_queue) > 0:
            current_seg_id = segment_id_queue.pop(0)
            node_1, node_2 = self.get_node_ids_on_segment(current_seg_id)
            nodes = [node_1,node_2]
            for node_id in nodes:
                if not node_id in seen_node_ids:
                    seen_node_ids.append(node_id)
                    node_dict.update({node_id:node_num})
                    node_num += 1
                    seg_ids,_ = self.get_segment_ids_on_node(node_id)
                    for seg_id in seg_ids:
                        if not seg_id in seen_segment_ids:
                            segment_id_queue.append(seg_id)
                            seen_segment_ids.append(seg_id)
                            segment_dict.update({seg_id:segment_num})
                            segment_num += 1

        # Create the new node dictionary with continuous IDs
        new_node_dict = {new_id: self.node_dict[old_id] for old_id, new_id in node_dict.items()}
        # Update the nodes' local IDs
        for new_id, node in new_node_dict.items():
            node.node_id = new_id

        # Update the node dictionary
        self.node_dict = new_node_dict

        # Update the node IDs in the segments
        for segment in self.segment_dict.values():
            segment.node_1_id = node_dict[segment.node_1_id]
            segment.node_2_id = node_dict[segment.node_2_id]

        # Create the new segment dictionary with continuous IDs
        new_segment_dict = {new_id: self.segment_dict[old_id] for old_id, new_id in segment_dict.items()}
        # Update the segments' local IDs
        for new_id, segment in new_segment_dict.items():
            segment.segment_id = new_id

        # Update the segment dictionary
        self.segment_dict = new_segment_dict

        Node.reset_count()
        Node.increment_count(len(self.node_dict.keys()))
        Segment.reset_count()
        Segment.increment_count(len(self.segment_dict.keys()))

        return

    def reset_junctions(self):
        """
        Resets the Junction dicitonary to nothing.
        After this Method is used it is recommended to run .populate_junctions().
        """

        self.junction_dict = {}
        Junction.reset_count()
        return

    def populate_junctions(self):
        """
        Iterate through the tree to populate the junction information based on the node-segment connectivity
        """
        self.reset_junctions()
        for node_keys in self.node_dict.keys():
            node_id = self.node_dict[node_keys].node_id
            found = False
            if self.check_node_internal(node_id):
                for junction_keys in self.junction_dict.keys():
                    if self.junction_dict[junction_keys].node_id == node_id:
                        found = True
                        break
                if found:
                    segments_on_node,_ = self.get_segment_ids_on_node(node_id)
                    if len(segments_on_node) > self.junction_dict[junction_keys].segments_in_junction(): # type: ignore
                        self.junction_dict[junction_keys].segment_1_id = segments_on_node[0] # type: ignore
                        self.junction_dict[junction_keys].segment_2_id = segments_on_node[1] # type: ignore
                        self.junction_dict[junction_keys].segment_3_id = segments_on_node[2] # type: ignore
                else:
                    segments_on_node,_ = self.get_segment_ids_on_node(node_id)
                    if len(segments_on_node) == 2:
                        self.add_junction(node_id,segments_on_node[0],segments_on_node[1])
                    elif len(segments_on_node) == 3:
                        self.add_junction(node_id,segments_on_node[0],segments_on_node[1],segments_on_node[2])
                    else:
                        # self.visualize_geometry()
                        raise ValueError(f"Unusual number of segments ({len(segments_on_node)}) with IDs: ({segments_on_node}) connecting to internal node {node_id} found")
        return

    def __str__(self):
        """
        Returns a string representation of the tree information.

        :return: A string representing the number of nodes, segments, junctions, and the age of the tree
        """
        return f"""
        Tree Information:
          Number of Nodes: {self.count_nodes()}
          Number of Segments: {self.count_segments()}
          Number of Junctions: {self.count_junctions()}
          Tree Age: {self.tree_age}
        """ 

    def visualize_geometry(self):
        """
        Create and display a basic 3D plot of the nodes and segments of the tree.
        """
        point_list = []
        node_ids = []  # List to store node IDs
        for keys in self.node_dict.keys():
            point = self.node_dict[keys].spatial_location
            point_list.append(point)
            node_ids.append(keys)  # Append node ID to the list
        
        line_connectivity = []
        for keys in self.segment_dict.keys():
            line_connectivity.extend([2, self.segment_dict[keys].node_1_id, self.segment_dict[keys].node_2_id])
        
        number_of_lines = self.count_segments()

        mesh = pv.PolyData(point_list, lines=line_connectivity, n_lines=number_of_lines)

        # Add point data for node IDs
        mesh.point_data["Node IDs"] = node_ids

        # Plot the mesh with annotations
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, line_width=5, render_points_as_spheres=True)
        
        # Show the plot
        plotter.show()

        return
        

    def get_tangent_versor(self, segment_id:int) -> Tuple[float,float,float]:
        """
        Returns the normalized direction vector of a segment from node 1 to node 2

        :param segment_id: The segment ID in question.
        :return: The x,y,z components of the characteristic vector of the segment
        """
        node_1_pos, node_2_pos = self.get_segment_node_locations(segment_id)

        lx = node_2_pos[0]-node_1_pos[0]
        ly = node_2_pos[1]-node_1_pos[1]
        lz = node_2_pos[2]-node_1_pos[2]

        l_norm = np.sqrt(lx*lx+ly*ly+lz*lz)

        x = lx / l_norm
        y = ly / l_norm
        z = lz / l_norm

        return x,y,z

    def get_segment_node_locations(self, segment_id:int) -> Tuple[List,List]:
        """
        Returns the spatial positions of the nodes associated with a segment

        :param segment_id: The segment ID in question.
        :return node_1_pos: The x,y,z positions of node 1
        :return node_2_pos: The x,y,z positions of node 2
        """
        if not segment_id in self.segment_dict:
            raise ValueError(f"Segment ID {segment_id} is not valid")

        node_1_id = self.segment_dict[segment_id].node_1_id
        node_2_id = self.segment_dict[segment_id].node_2_id

        node_1_pos = self.node_dict[node_1_id].location()
        node_2_pos = self.node_dict[node_2_id].location()

        return node_1_pos, node_2_pos

    def get_midpoint(self, segment_id:int) -> List:
        """
        Returns the midpoint of the line of a segment from node 1 to node 2

        :param segment_id: The segment ID in question.
        :return: The x,y,z location of the midpoint of the segment
        """
        pos1, pos2 = self.get_segment_node_locations(segment_id)
        mid_pos = [(x + y) / 2 for x, y in zip(pos1, pos2)]

        return mid_pos

    def push_nodes(self, tissue:Tissue):
        """
        Moves all nodes that exist outside fo the spatial constraints of the provided tissue into the tissue.

        :param tissue: A Tissue object defining the spatial constraints of the tree.
        """
        x0,y0,z0 = tissue.bottom_corner()
        x1,y1,z1 = tissue.top_corner()
        for keys in self.node_dict.keys():
            x,y,z = self.node_dict[keys].location()
            if x <= x0:
                self.node_dict[keys].spatial_location[0] = x0 + 1e-7
            elif x >= x1:
                self.node_dict[keys].spatial_location[0] = x1 - 1e-7

            if y <= y0:
                self.node_dict[keys].spatial_location[1] = y0 + 1e-7
            elif y >= y1:
                self.node_dict[keys].spatial_location[1] = y1 - 1e-7

            if z <= z0:
                self.node_dict[keys].spatial_location[2] = z0 + 1e-7
            elif z >= z1:
                self.node_dict[keys].spatial_location[2] = z1 - 1e-7

    def apply_maximum_segment_length(self, cell_size:float, verbose=False):
        """
        Modifies the tree to enforce a maximum segment length by subdiving segments that are too long.
        THERE IS AN ERROR IN HERE OR IN subdivide_segment() THAT CAUSES INCORRECT CONNECTIVITY

        :param cell_size: The maximum allowed segment length.
        :param verbose: Boolean determining if information about the subdivision is needed.
        """
        # Create a copy of the dictionary keys before iterating
        
        key_list = list(self.segment_dict.keys())
        any_changes = False

        for keys in key_list:
            current_length = self.length(keys)
            subdivisions = int(current_length // cell_size)
            if subdivisions > 0:
                if verbose:
                    print(f"Segment {keys}, has length: {current_length}")
                    print(f"this results in {subdivisions} subdivisions given cell size: {cell_size}")
                self.subdivide_segment(keys,subdivisions)
                any_changes = True
                # node_1_pos, node_2_pos = self.get_segment_node_locations(keys)
                # positions = np.linspace(node_1_pos,node_2_pos,2+subdivisions)
                # positions = positions[1:-1]
                # for pos in positions[::-1]:
                #     self.break_segment(keys,pos)

        if any_changes:
            self._set_node_and_segement_numbers_by_bfs()
            # self.populate_junctions()

        return

    def subdivide_segment(self, segment_id:int, new_num_segments:int):
        """
        Take an existing segment and seperate it into n smaller segments
        It is recommended that you call populate_junctions after using this function. 
        THERE IS AN ERROR IN HERE OR IN apply_maximum_segment_length() THAT CAUSES INCORRECT CONNECTIVITY

        :param segment_id: The ID of the segment to be subdivided.
        :param new_num_segments: The number of new segments that should result from the subdivision
        """
        if segment_id in self.segment_dict:
            node_first_id = self.segment_dict[segment_id].node_1_id
            node_last_id = self.segment_dict[segment_id].node_2_id
        else:
            raise ValueError("Segment ID is not valid")
        
        node_first_pos = self.node_dict[node_first_id].location()
        node_last_pos = self.node_dict[node_last_id].location()

        pos_list = np.linspace(node_first_pos,node_last_pos, new_num_segments+2)
        
        node_id_list = []
        segment_id_list = []
        radius = self.segment_dict[segment_id].radius

        for position in range(1,len(pos_list)-1):
            new_node_id = self.add_node(pos_list[position])
            node_id_list.append(new_node_id)

        new_segment_id = self.add_segment(node_first_id,node_id_list[0],radius=radius)
        segment_id_list.append(new_segment_id) 

        for i,node_id in enumerate(node_id_list[:-1]):
            new_segment_id = self.add_segment(node_id_list[i],node_id_list[i+1],radius=radius)
            segment_id_list.append(new_segment_id)

        new_segment_id = self.add_segment(node_id_list[-1],node_last_id,radius=radius)
        segment_id_list.append(new_segment_id)

        self.segment_dict.pop(segment_id)

        return segment_id_list
        # if segment_id in self.segment_dict:
        #     node_first_id = self.segment_dict[segment_id].node_1_id
        #     node_last_id = self.segment_dict[segment_id].node_2_id
        # else:
        #     raise ValueError("Segment ID is not valid")
        
        # node_first_pos = self.node_dict[node_first_id].location()
        # node_last_pos = self.node_dict[node_last_id].location()

        # pos_list = np.linspace(node_first_pos,node_last_pos, new_num_segments+2)
        
        # node_id_list = [node_first_id]
        # segment_id_list = [segment_id]
        # radius = self.segment_dict[segment_id].radius

        # for position in range(1,len(pos_list)-1):
        #     new_node_id = self.add_node(pos_list[position])
        #     node_id_list.append(new_node_id)
        #     if position == 1:
        #         self.segment_dict[segment_id].node_2_id = node_id_list[-1]
        #     else:
        #         new_segment_id = self.add_segment(node_id_list[-2],node_id_list[-1],radius=radius)
        #         segment_id_list.append(new_segment_id)
                
        # node_id_list.append(node_last_id)
        # new_segment_id = self.add_segment(node_id_list[-2],node_id_list[-1],radius=radius)
        # segment_id_list.append(new_segment_id)

        # return segment_id_list
    
    @staticmethod
    def _distance_point_to_line(query_point: List[float], line_start: List[float], line_end: List[float]) -> Tuple[float, np.ndarray]:
        """
        Calculates the distance from the querry point to the a line specified by two points

        :param query_point: x,y,z coordiantes specifying the location of the querry point
        :param line_start: x,y,z coordiantes specifying the location of the start of the line
        :param line_end: x,y,z coordiantes specifying the location of the end fo the line
        :return: The distance between the point and the line and the location of the closest point on the line
        """
        # Convert input lists to NumPy arrays for vectorized operations
        query_point = np.array(query_point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)
        
        # Calculate the vector along the line segment
        line_vec = line_end - line_start # type: ignore

        # Calculate the vector from line_start to the query_point
        point_vec = query_point - line_start # type: ignore

        # Calculate the length of the line segment
        line_len = np.linalg.norm(line_vec)

        # Calculate the unit vector along the line
        line_unitvec = line_vec / line_len

        # Calculate the projection of point_vec onto the line
        proj = np.dot(point_vec, line_unitvec)
        
        if proj <= 0:
            # Closest point is at the start of the line segment
            closest_point = line_start
            distance = np.linalg.norm(query_point - line_start) # type: ignore
        elif proj >= line_len:
            # Closest point is at the end of the line segment
            closest_point = line_end
            distance = np.linalg.norm(query_point - line_end) # type: ignore
        else:
            # Closest point is somewhere on the line segment
            closest_point = line_start + proj * line_unitvec
            distance = np.linalg.norm(query_point - closest_point)

        return distance, closest_point

    def in_search_radius(self, l1:float, l2:float, query_point:List[float], segment_id:int) -> Tuple[Union[None,List[float]],float]:
        """
        Calculates if a point is within the sum of two distances of a segment.

        :param l1: The first characteristic distance
        :param l2: The second characteristic distance
        :param query_point: x,y,z coordiantes specifying the location of the querry point
        :param segment_id: The segment ID in question
        :return: The distance between the point and the line and the location of the closest point on the line
        """
        p1, p2 = self.get_segment_node_locations(segment_id)
        closest_distance, closest_point = self._distance_point_to_line(query_point, p1, p2)

        # Check if the closest distance is greater than l (outside the delineated space)
        if closest_distance > l1+l2:
            return None, 0
        else:
            return closest_point, closest_distance
            
    def attach_sprout(self, sprout:"Sprout",use_base_raidus=True, reverse=False) -> Tuple[int,List]:
        """
        Converts a Sprout object into a series of nodes, segments, and junctions on the tree. 
        Sprout is attached from base to tip unless reverse is set to true. In which case the sprout is attached from tip to base.
        Includes a safety check to ensure that the base node has not become a bifurcating junction.

        :param sprout: The Sprout object to be attached
        :param use_base_raidus: Boolean to determine if you use the radius of the segment the sprout connects to for the sprout size
        :param reverse: Boolean to determine if the sprout should be attached base to tip or tip to base.
        :return: The ID of the node at the tip of the sprout and a list of the segment ID's generated
        """
        # get information associated with the base of the sprout
        node_1_id = sprout.base_node_id
        base_segments = []
        connected_nodes = []
        seg_ids = []
        base_segments, sides = self.get_segment_ids_on_node(node_1_id)

        # Check if the base of the sprout has become a bifurcating junciton.
        if len(base_segments) > 2:
            # Get the adjacent node ids
            for i, segment_id in enumerate(base_segments):
                if sides[i] == 1:
                    connected_nodes.append(self.segment_dict[segment_id].node_2_id)
                if sides[i] == 2:
                    connected_nodes.append(self.segment_dict[segment_id].node_1_id)

            # Find the distance between the sprout connection and the adjacent nodes:
            distances = {
            node: np.linalg.norm(self.node_dict[node].location() - sprout.snail_trail_dict[1])
            for node in connected_nodes
            }

            # Sort connected nodes by distance
            node_segment_pairs = list(zip(connected_nodes, base_segments))  # Pair nodes with their respective segments
            sorted_pairs = sorted(node_segment_pairs, key=lambda pair: distances[pair[0]])  # Sort by node distance
            sorted_nodes, sorted_segments = zip(*sorted_pairs)  # Unzip into sorted nodes and segments

            # Check for an adjacent node with fewer than 3 segments
            for node in sorted_nodes:
                if not self.check_node_in_junction(node):
                    node_1_id = node
                    break

            # If no adjacent noeds are valid break one of the segments to make a new node.
            for segment in sorted_segments:
                # Use appropriate segment ID and break that segment in the middle
                node_1_loc, node_2_loc = self.get_segment_node_locations(segment)
                node_1_id, _ = self.break_segment(segment, [(c1 + c2) / 2 for c1, c2 in zip(node_1_loc, node_2_loc)])
                break



        # turn the sprout into segments
        first = True
        for key, value in sprout.snail_trail_dict.items():
            if key != 0:
                # Create the next node and the connecting segment. 
                node_2_id = self.add_node(sprout.snail_trail_dict[key])
                if use_base_raidus:
                    R1 = self.segment_dict[base_segments[0]].radius
                    R2 = self.segment_dict[base_segments[1]].radius
                    seg_id = self.add_segment(node_1_id,node_2_id,radius=(R1+R2)/2)
                else:
                    seg_id = self.add_segment(node_1_id,node_2_id)

                # add the junction associated with the base of the sprout
                if first:
                    junction_id = self.get_junction_on_node(node_1_id)
                    if junction_id is False:
                        segment_ids,_ = self.get_segment_ids_on_node(node_1_id)
                        self.add_junction(node_1_id,segment_ids[0],segment_ids[1],segment_ids[2])
                    else:
                        self.junction_dict[junction_id].segment_3_id = seg_id
                    first = False
                node_1_id = node_2_id
                seg_ids.append(seg_id)

        if reverse:
            for seg_id in seg_ids:
                self.segment_dict[seg_id].reverse_segment()

        # return the id of the node associated with the tip of the sprout for anastomosis purposes.
        if first:
            return node_1_id, seg_ids
        else:
            tip_node_id = node_2_id # type: ignore
            return tip_node_id, seg_ids

    def set_and_propagate_inhibition(self,inhibited_nodes:np.ndarray,inhibition_distance:float=120e-6) -> dict:
        """
        Takes a set of points on the tree and creates an inhbition map based on travel distance along the tree.

        :param inhibited_nodes: Array containing x,y,z location, seg id for sources of inhibition.
        :param inhibition_distance: Distance along the tree the inhibition should travel from the source points
        :return: A dictionary containing the inhibition score of each node on the tree.
        """
        # This function is to approximate the effect of delta-notch inhibition signalling 
        inhibitor_signal_dict = {}
        for keys in self.node_dict.keys():
            inhibitor_signal_dict.update({keys:-1})

        to_propagate_from = []

        # Inhibit Inlets and Outlets
        for inlet_id in self.get_node_ids_inlet():
            inhibitor_signal_dict[inlet_id] = inhibition_distance/2
            to_propagate_from.extend([inlet_id])
        for outlet_id in self.get_node_ids_outlet():
            inhibitor_signal_dict[outlet_id] = inhibition_distance/2
            to_propagate_from.extend([outlet_id])
        
        for node_id in self.get_node_ids_internal():
            if self.check_node_in_junction(node_id):
                inhibitor_signal_dict[node_id] = inhibition_distance/6
                to_propagate_from.extend([node_id])
        
        #Propogate from inhibited points
        for i in range(len(inhibited_nodes[:,0])):
            seg_id = inhibited_nodes[i,4]
            inhibition_point_xyz = inhibited_nodes[i,:3]
            node_1_pos, node_2_pos = self.get_segment_node_locations(seg_id)

            node_1_id = self.segment_dict[seg_id].node_1_id
            node_2_id = self.segment_dict[seg_id].node_2_id

            node_1_id = self.segment_dict[seg_id].node_1_id
            node_2_id = self.segment_dict[seg_id].node_2_id

            distance = np.linalg.norm(node_1_pos - inhibition_point_xyz)
            inhibitor_signal_dict[node_1_id] = max(inhibitor_signal_dict[node_1_id], inhibition_distance - distance)

            distance = np.linalg.norm(node_2_pos - inhibition_point_xyz)
            inhibitor_signal_dict[node_2_id] = max(inhibitor_signal_dict[node_2_id], inhibition_distance - distance)

            to_propagate_from.extend([node_1_id, node_2_id])

        for node_ids in to_propagate_from:
            inhibitor_signal_dict = self._propagate_inhibition(node_ids, inhibitor_signal_dict)

        self.inhibited_nodes = inhibited_nodes
        self.inhibitor_signal_dict = inhibitor_signal_dict

        return inhibitor_signal_dict

    def _propagate_inhibition(self, node_id:int, inhibitor_signal_dict:dict) -> dict:
        """
        Recursively Propogates the inhibition signal throughout the tree

        :param node_id: The node ID to propagate from.
        :param inhibitor_signal_dict: A dictionary containing the inhibition score of each node on the tree.
        :return: A dictionary containing the inhibition score of each node on the tree.
        """
        node_signal = inhibitor_signal_dict[node_id]
        segment_ids, node_side = self.get_segment_ids_on_node(node_id)
        for i, seg_id in enumerate(segment_ids):
            segment_length = self.length(seg_id)
            other_node_side = 3-node_side[i]
            if other_node_side == 1:
                other_node_id:int = self.segment_dict[seg_id].node_1_id
            else: # other_node_side == 2:
                other_node_id:int = self.segment_dict[seg_id].node_2_id
            

            other_node_signal = node_signal - segment_length
            if other_node_signal > inhibitor_signal_dict[other_node_id]:
                inhibitor_signal_dict.update({other_node_id:other_node_signal})
                if other_node_signal > 0:
                    inhibitor_signal_dict = self._propagate_inhibition(other_node_id,inhibitor_signal_dict)

        return inhibitor_signal_dict

    def measure_inhibition_signal(self,sorted_array:list,passed_array:list=None,inhibitor_signal_dict:dict=None,inhibition_range:float=80e-6) -> np.ndarray: # type: ignore
        """
        Measures the inhibition score at locations on the tree.

        :param sorted_array: An array containing the information about the points to evaluate the inhibition score of
        :param passed_array: An array containing the information about the points causing inhibition
        :param inhibitor_signal_dict: A dictionary containing the inhibition score of each node on the tree.
        :param inhibition_range: The effective range of inhibition.
        :return: A binary inhibited/non-inhibited array for the points evaluated.
        """
        if passed_array is None:
            passed_array = self.inhibited_nodes
        if inhibitor_signal_dict is None:
            inhibitor_signal_dict = self.inhibitor_signal_dict
        
        row = 0
        signal = []
        
        while row < sorted_array.shape[0]: # type: ignore
            same_seg_signal = -1
            seg_id = sorted_array[row,4] # type: ignore
            P = sorted_array[row,:3] # type: ignore
            P1, P2 = self.get_segment_node_locations(seg_id)

            indices = [index for index, value in enumerate(passed_array[:,4]) if value == seg_id] # type: ignore
            for index in indices:
                P3 = passed_array [index,:3] # type: ignore
                same_seg_dist = np.linalg.norm(np.array(P)-np.array(P3))
                same_seg_signal = max(same_seg_signal,inhibition_range-same_seg_dist)
                

            node_1_id = self.segment_dict[seg_id].node_1_id
            node_2_id = self.segment_dict[seg_id].node_2_id

            # Calculate alpha
            alpha = np.linalg.norm(np.array(P) - np.array(P1)) / np.linalg.norm(np.array(P2) - np.array(P1))
            
            # Ensure alpha is between 0 and 1
            alpha = max(0.0, min(1.0, alpha))

            local_signal = (1-alpha)*inhibitor_signal_dict[node_1_id] + alpha*inhibitor_signal_dict[node_2_id]
            final_signal = max(same_seg_signal,local_signal)
            if final_signal >= 0:
                signal.append(1)
            else:
                signal.append(0)
            row += 1


        return np.array(signal)
    
    def add_segment_with_max_length(self, node_1_id:int, node_2_id:int, max_segment_length:float, segment_id:int=None, radius:float=None) -> list: # type: ignore
        """
        Adds a segment to the tree with a maximum length constraint. 
        If the length of the added segment exceeds the constraint the segment will be broken into multiple segments satisfying the constraint

        :param node_1_id: The node ID of one end of the segment.
        :param node_2_id: The node ID of the other end of the segment.
        :param max_segment_length: The maximum length the generated segments can be.
        :param segment_id: An optional parameter specifying the desired segment ID.
        :param radius: An optional parameter specifying the radius value of the segment.
        :return: A list containing the ID's of the created segments.
        """
        segment_id = self.add_segment(node_1_id,node_2_id,segment_id,radius)
        segment_id_list = [segment_id]

        current_length = self.length(segment_id)
        subdivisions = int(current_length // max_segment_length)
        if subdivisions > 0:
            segment_id_list = self.subdivide_segment(segment_id,subdivisions)
        
        return segment_id_list
    
    def clean_up_overlapping_nodes(self,point_dist_tolerance:float=2e-6) -> np.ndarray:
        """
        Calculates the distance from each node on the tree to each other node on the tree in real space,
        then merges nodes that are under the specified tolerance into a single node. 
        Then corrects segment connectivity based on the new nodes.
        THIS MAY RESULT IN ILLEGAL TRIFURCATIONS.
        FUNCTION IS UNFINISHED AND UNUSED

        :param point_dist_tolerance: The minimum distance any nodes can be from one another before being merged
        """

        # Get the list of node ids and their positions
        node_ids = list(self.node_dict.keys())
        positions = np.array([self.node_dict[node_id].location() for node_id in node_ids])
        
        # Calculate pairwise distances between nodes
        dist_matrix = distance.cdist(positions, positions, 'euclidean')
        
        # Get the indices of the upper triangle of the distance matrix (excluding the diagonal)
        upper_triangle_indices = np.triu_indices_from(dist_matrix, k=1)
        
        # Sort the distances in ascending order and get the sorted indices
        sorted_dist_indices = np.argsort(dist_matrix[upper_triangle_indices])
        
        # List to store merged nodes and their new positions
        merged_nodes_info = []
        merged_set = set()  # To track nodes that have already been merged
        
        for idx in sorted_dist_indices:
            i, j = upper_triangle_indices[0][idx], upper_triangle_indices[1][idx]
            
            if dist_matrix[i, j] > point_dist_tolerance:
                break  # Stop if the distance is greater than the tolerance
            
            node_id_i = node_ids[i]
            node_id_j = node_ids[j]
            
            if node_id_i in merged_set or node_id_j in merged_set:
                continue  # Skip if either node has already been merged
            
            # Calculate the merged position as the average of the two nodes' positions
            merged_position = (self.node_dict[node_id_i].location() + self.node_dict[node_id_j].location()) / 2
            
            # Record the merged nodes and their new position
            merged_nodes_info.append(((node_id_i, node_id_j), merged_position))
            
            # Mark these nodes as merged
            merged_set.add(node_id_i)
            merged_set.add(node_id_j)
        

        for (node_id_i, node_id_j), merged_position in merged_nodes_info:
            node_id_new = self.add_node(merged_position)
            for key in self.segment_dict.keys():
                segment = self.segment_dict[key]
                node_1_replaced = False
                node_2_replaced = False
                if segment.node_1_id == node_id_i or segment.node_1_id == node_id_j:
                    segment.node_1_id = node_id_new
                    node_1_replaced = True
                if segment.node_2_id == node_id_i or segment.node_2_id == node_id_j:
                    segment.node_2_id = node_id_new
                    node_2_replaced = True
                if node_1_replaced and node_2_replaced:
                    self.segment_dict.pop(key)


        return
    
    def export_vessel_array_to_csv(self, output_file:str):
        """
        Exports the custom vessel array data structure to a CSV file for use with circulatory autogen.
        Format takes the form of 5 collumns  as follows:
            name: Name of the vessel
            BC_type: some combination of p's and v's indicating pressure or flow boundaries
            vessel_type: string referencing the local behaviour template
            inp_vessels: List of input vessel names.
            out_vessels: List of output vessel names.

        :param output_file: str. Path to the output CSV file.
        """

        # Build vessel_array info from the segment dictionary.
        vessel_array = []
        for segment_key in self.segment_dict.keys():
            vessel = {}
            vessel["name"] = f"segment_{self.segment_dict[segment_key].segment_id}"
            vessel["BC_type"] = "pv"
            vessel["vessel_type"] = "capillary"
            inp_vessels = []
            for segment_id in self.get_segment_ids_on_node(self.segment_dict[segment_key].node_1_id):
                if segment_id != self.segment_dict[segment_key].segment_id:
                    inp_vessels.append(f"segment_{segment_id}")
            vessel["inp_vessels"] = inp_vessels
            out_vessels = []
            for segment_id in self.get_segment_ids_on_node(self.segment_dict[segment_key].node_1_id):
                if segment_id != self.segment_dict[segment_key].segment_id:
                    out_vessels.append(f"segment_{segment_id}")
            vessel["out_vessels"] = out_vessels


        # Write to CSV file
        with open(output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(["name", "BC_type", "vessel_type", "inp_vessels", "out_vessels"])
            
            # Write vessel data
            for vessel in vessel_array:
                writer.writerow([
                    vessel["name"],
                    vessel["BC_type"],
                    vessel["vessel_type"],
                    " ".join(vessel["inp_vessels"]),  # Space-separated string
                    " ".join(vessel["out_vessels"])  # Space-separated string
                ])

        print(f"Vessel array successfully exported to {output_file}")

    def homogenous_fit_to_tissue(self, old_tissue:Tissue, new_tissue:Tissue):
        """
        This function scales the network homogenously with the transformation from an old tissue to a new tissue.
        First a transformation from the old_tissue bounding box to the new_tissue bounding box will be calculated
        Then that transformation will be applied to all nodal locations.

        :param old_tissue: A Tissue object for the previous tissue state.
        :param new_tissue: A Tissue object for the new tissue state.
        """

        old_corner_1 = old_tissue.bottom_corner()
        old_corner_2 = old_tissue.top_corner()

        new_corner_1 = new_tissue.bottom_corner()
        new_corner_2 = new_tissue.top_corner()

        for node in self.node_dict.values():
            node_pos = node.location()

            node_x_relative_old = (node_pos[0]-old_corner_1[0]) / (old_corner_2[0]-old_corner_1[0])
            node_y_relative_old = (node_pos[1]-old_corner_1[1]) / (old_corner_2[1]-old_corner_1[1])
            node_z_relative_old = (node_pos[2]-old_corner_1[2]) / (old_corner_2[2]-old_corner_1[2])

            node_relative_pos = [node_x_relative_old,node_y_relative_old,node_z_relative_old]
            mask = [pos>1 or pos<0 for pos in node_relative_pos]
            if any(mask):
                raise ValueError(f"Node found outside bounds of supplied old tissue. This suggests the tissue and tree do not match.")

            node_x_new = (new_corner_2[0]-new_corner_1[0]) * node_x_relative_old + new_corner_1[0]
            node_y_new = (new_corner_2[1]-new_corner_1[1]) * node_y_relative_old + new_corner_1[1]
            node_z_new = (new_corner_2[2]-new_corner_1[2]) * node_z_relative_old + new_corner_1[2]

            node.spatial_location[0] = node_x_new
            node.spatial_location[1] = node_y_new
            node.spatial_location[2] = node_z_new

        

class Sprout():
    """
    A class used to handle the information needed to define a sprout in the process of growing.

    ----------
    Class Attributes
    ----------
    __SPROUT_COUNT : int
        An integer stating the total number of sprouts in the system

    ----------
    Instance Attributes
    ----------
    sprout_id : int
        A unique id number representing the sprout

    snail_trail_dict : dict
        A dictionary of the form {shadow_node_id:[x,y,z]}

    starting_node : int
        An int specifying at which node the sprout begins

    tip_node : int
        An int specifying at which node the sprout currently ends

    base_segment_vector : xyz vector
        A vector in 3D space of the segment the sprout is growing from

    ----------
    Class Methods
    ----------  
    increment_count(amount=0)
        Adds or subtracts from the __SPROUT_COUNT Attribute

    reset_count()
        Resets the __SEGMENT_COUNT Attribute

    ----------
    Instance Methods
    ----------  
    add_step(x,y,z)
        Adds a new node to the system

    in_search_radius(sprout_signal_dist,search_dist,query_xyz)
        Function that checks if the sprout is within the radius of a searching tip node, 
        returns either: -1 for False or the id of the closest shadow node for True.

    sever_sprout(shadow_node_id)
        Severs the sprout removing nodes after the supplied shadow node id

    """

    __SPROUT_COUNT = 0

    def __init__(self, sprout_id:Union[int,None], xyz:List[float], segment_id:int, tree:Tree, base_node_id:int, oxygen_tension:float):
        if sprout_id == None:
            self.sprout_id = self.__SPROUT_COUNT
            self.increment_count()
        else:
            self.sprout_id = sprout_id

        self.snail_trail_dict = {}
        self.snail_trail_dict.update({0:xyz})
        self.base_node_id = base_node_id
        self.starting_node = 0
        self.tip_node = 0
        self.oxygen_tension_score = oxygen_tension

        self.base_segment_vector = tree.get_tangent_versor(segment_id)
        self.macrophage_connection = None
        self.num_macrophages = 0
        self.base_radius = tree.segment_dict[segment_id]._Segment__DEFAULT_RADIUS

    @classmethod
    def increment_count(cls, amount=1):
        cls.__SPROUT_COUNT += amount

    @classmethod
    def reset_count(cls):
        cls.__SPROUT_COUNT = 0

    def add_step(self,xyz:List[float]):
        self.snail_trail_dict.update({self.tip_node+1:xyz})
        self.tip_node += 1

    def calculate_angle(self,vector:List[float]) -> Tuple[float, List[float], bool]:
        if self.tip_node == 0:
            vector2 = self.base_segment_vector
            base_case = True
        else:
            vector2 = self.snail_trail_dict[self.tip_node] - self.snail_trail_dict[self.tip_node-1]
            base_case = False
        
        # Calculate the cross product to find the normal vector of the plane
        normal_vector = np.cross(vector, vector2)
        
        # Normalize the normal vector to get a unit vector
        normal_unit = normal_vector / np.linalg.norm(normal_vector)
        
        # Calculate the dot product of the unit normal vector and v1
        dot_product = np.dot(vector, normal_unit)
        
        # Take the absolute value of the dot product to ensure the angle is acute
        dot_product = abs(dot_product)
        
        # Calculate the angle in radians using arccosine
        angle_rad = np.arccos(dot_product)
        
        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg, vector2, base_case # type: ignore

    @staticmethod
    def _distance_point_to_line(point:List[float], line_start:List[float], line_end:List[float]) -> float:
        line_vec = line_end - line_start # type: ignore
        point_vec = point - line_start # type: ignore
        line_len = np.linalg.norm(line_vec)

        # Check if line_len is zero (i.e., line_start and line_end are the same point)
        if line_len == 0:
            return np.linalg.norm(point - line_start) # type: ignore

        line_unitvec = line_vec / line_len
        proj = np.dot(point_vec, line_unitvec)
        
        if proj <= 0:
            return np.linalg.norm(point - line_start) # type: ignore
        if proj >= line_len:
            return np.linalg.norm(point - line_end) # type: ignore

        closest_point = line_start + proj * line_unitvec
        return np.linalg.norm(point - closest_point)

    def in_search_radius(self, filopodia_length:float, macrophage_size:float, query_point:List[float]) -> Tuple[Union[None,int],float]:
        num_points = len(self.snail_trail_dict.keys())

        closest_distance = float('inf')
        closest_point = None

        # Check points not including the base
        for i in range(num_points)[1:]:
            p1 = self.snail_trail_dict[i]
            p2 = self.snail_trail_dict[min((i + 1),num_points-1)]
            
            distance_to_line = self._distance_point_to_line(query_point, p1, p2)
            # If point is the tip adjust for filopodia length
            if i == self.tip_node:
                distance_to_line -= filopodia_length

            if distance_to_line < closest_distance:
                closest_distance = distance_to_line
                closest_point = i

        # Check if the closest distance is greater than l (outside the delineated space)
        if closest_distance > filopodia_length+macrophage_size:
            return None, 0
        else:
            return closest_point, closest_distance

    def sever_sprout(self, cutoff:int) -> list:
        # Extract values that are cut from the dictionary
        removed_values = [value for key, value in self.snail_trail_dict.items() if key > cutoff]
        # Use dictionary comprehension to filter out entries with keys greater than cutoff
        self.snail_trail_dict = {key: value for key, value in self.snail_trail_dict.items() if key <= cutoff}
        self.tip_node = cutoff

        return removed_values

    def get_tip_loc(self) -> List[float]:
        return self.snail_trail_dict[self.tip_node]

    def get_base_loc(self) -> List[float]:
        return self.snail_trail_dict[0]

    def add_macrophage_connection(self, target_type:str, target_id:int, target_pos:Union[int,float]):
        if self.check_has_macrophage():
            raise ValueError(f"Sprout {self.sprout_id} already has a macrophage connection defined as: {self.macrophage_connection}")
        self.macrophage_connection = {}
        self.macrophage_connection.update({"type":target_type})
        self.macrophage_connection.update({"id":target_id})
        self.macrophage_connection.update({"position":target_pos})
        self.num_macrophages +=1
        return

    def remove_macrophage_connection(self):
        self.macrophage_connection = None
        self.num_macrophages = 0
        return

    def check_has_macrophage(self):
        if self.num_macrophages == 0:
            return False
        else:
            return True

    def get_sprout_age(self):
        return len(self.snail_trail_dict.keys())

    def calculate_bending(self):
        # Check if sprout is long enough to make a second derviative
        if self.get_sprout_age() < 3:
            return 0
        
        second_derivatives = []
        for i in range(1,self.get_sprout_age()-1):
            p1 = self.snail_trail_dict[i-1]
            p2 = self.snail_trail_dict[i]
            p3 = self.snail_trail_dict[i+1]
            p1_p2_grad = (p2-p1)/np.linalg.norm(p2-p1)
            p2_p3_grad = (p3-p2)/np.linalg.norm(p3-p2)
            second_derivative = (p2_p3_grad - p1_p2_grad) / (np.linalg.norm(p3-p1)/2)
            second_derivatives.append(np.linalg.norm(second_derivative))

        # Integrate the norm of second derivatives over the length of the line
        line_length = 0
        for i in range(0, self.get_sprout_age()-1):
            line_length += np.linalg.norm(self.snail_trail_dict[i + 1] - self.snail_trail_dict[i])

        bending = simpson(second_derivatives, dx=line_length / len(second_derivatives))

        return bending

    def __str__(self):
        return f"""
        Sprout Information:
          Sprout ID: {self.sprout_id}
          Phantom Nodes: {self.snail_trail_dict}
          Base Node ID: {self.base_node_id}
          Tip ID: {self.tip_node}
          Tip Node Loc: {self.get_tip_loc()}
      """  


class GraphRepresentation():
    """
    A class used to handle the formation of directed graphs to navigate the vascular tree.

    ----------
    Class Attributes
    ----------
    ----------
    Instance Attributes
    ----------
    adjacency_list : dict
        A dictionary of the form {node_id:[adjacent_downstream_node1,adjacent_downstream_node2...]}

    incoming_edges_count : dict
        A dictionary of the form {node_id:number of incoming segments}

    edge_list : dict
        A dictionary of the form {node_id_1:{node_id_2:edge_id}}

    ----------
    Class Methods
    ----------  
    ----------
    Instance Methods
    ----------  
    add_directed_edge(u,v,edge_id)
        Adds a directed edge to the graph

    reset_graph()
        Resets the graph

    build_directed(tree, direction)
        Takes information about the geometry of the tree and the directionality to add all of the directed edges to the graph

    build_undirected(tree)
        Takes information about the geometry of the tree to add all of the edges to the graph

    add_undirected_edge(u,v,edge_id)
        Adds an undirected edge to the graph

    _dfs(self, u, parent, time, discovery_time, low, visited, bridges)
        A recursive depth first search algorithm used to determine the location of critical bridges in the graph

    get_critical_connections()
        Finds critical bridges in the graph using _dfs and returns a dictionary of the form {edge_id:bool_is_critical}

    traverse_segment(start_vertex,visited_edges)
        Variant of dfs that explores to the next junciton or endpoint on the graph.

    get_segments()
        Finds the groups of edges in the graph that constitute continuous segments between junctions.
    """

    def __init__(self):
        self.reset_graph()

    def add_directed_edge(self, u:int, v:int, edge_id:int):
        """
        Function used to add an undirected edge to the graph using node adjacency and edge information

        :param u: A graph vertex ID associated with the starting point of the edge.
        :param v: A graph vertex ID associated with the ending point of the edge.
        :param edge_id: A graph edge ID associated with the pair of vertex ID's u and v.
        """
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
            self.incoming_edges_count[u] = 0
            self.edge_list[u] = {}

        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
            self.incoming_edges_count[v] = 0
            self.edge_list[v] = {}

        self.adjacency_list[u].append(v)
        self.incoming_edges_count[v] += 1
        self.edge_list[u][v] = edge_id
        return

    def shortest_path(self, start: int, end: int, edge_lengths: dict) -> float:
        """
        Computes the shortest path distance between two vertices based on edge lengths.

        :param start: The starting vertex.
        :param end: The destination vertex.
        :param edge_lengths: A dictionary mapping edge IDs to their lengths.
        :return: The shortest path distance or float('inf') if no path exists.
        """
        pq = [(0, start)]  # Priority queue storing (distance, vertex)
        distances = {vertex: float('inf') for vertex in self.adjacency_list}
        distances[start] = 0

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)

            # If we reached the target vertex, return the distance
            if current_vertex == end:
                return current_distance

            # If a shorter path to current_vertex has already been found, skip processing
            if current_distance > distances[current_vertex]:
                continue

            # Process all neighbors
            for neighbor in self.adjacency_list[current_vertex]:
                edge_id = self.edge_list[current_vertex][neighbor]
                edge_length = edge_lengths.get(edge_id, float('inf'))  # Default to infinity if edge length is missing

                new_distance = current_distance + edge_length

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor))

        return float('inf')  # Return infinity if no path exists

    def shortest_distances_from_source(self, start: int, edge_lengths: dict) -> dict:
        """
        Computes the shortest travel distances from a source vertex to all other vertices.

        :param start: The starting vertex.
        :param edge_lengths: A dictionary mapping edge IDs to their lengths.
        :return: A dictionary mapping each vertex to its shortest distance from the source.
        """
        pq = [(0, start)]  # Min-heap storing (distance, vertex)
        distances = {vertex: float('inf') for vertex in self.adjacency_list}
        distances[start] = 0

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)

            # If a shorter path to current_vertex has already been found, skip processing
            if current_distance > distances[current_vertex]:
                continue

            # Process all neighbors
            for neighbor in self.adjacency_list[current_vertex]:
                edge_id = self.edge_list[current_vertex][neighbor]
                edge_length = edge_lengths.get(edge_id, float('inf'))  # Default to infinity if missing

                new_distance = current_distance + edge_length

                if new_distance < distances[neighbor]:  # Relaxation step
                    distances[neighbor] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor))

        return distances  # Dictionary of shortest distances from the source to every vertex

    def shortest_distances_from_sources(self, sources: list, edge_lengths: dict) -> dict:
        """
        Computes the shortest travel distance from any of multiple source vertices to every other vertex.

        :param sources: A list of starting vertices.
        :param edge_lengths: A dictionary mapping edge IDs to their lengths.
        :return: A dictionary mapping each vertex to its shortest distance from any source.
        """
        pq = []  # Min-heap storing (distance, vertex)
        distances = {vertex: float('inf') for vertex in self.adjacency_list}

        # Initialize all source vertices with distance 0
        for source in sources:
            distances[source] = 0
            heapq.heappush(pq, (0, source))

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)

            # If a shorter path to current_vertex has already been found, skip processing
            if current_distance > distances[current_vertex]:
                continue

            # Process all neighbors
            for neighbor in self.adjacency_list[current_vertex]:
                edge_id = self.edge_list[current_vertex][neighbor]
                edge_length = edge_lengths.get(edge_id, float('inf'))  # Default to infinity if missing

                new_distance = current_distance + edge_length

                if new_distance < distances[neighbor]:  # Relaxation step
                    distances[neighbor] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor))

        return distances  # Dictionary of shortest distances from any source to every vertex


    def reset_graph(self):
        """
        Resets the dictionaries containing the graph information.
        """
        self.adjacency_list = {}
        self.incoming_edges_count = {}
        self.edge_list = {}
        return

    def build_directed(self,tree:Tree,direction:np.ndarray):
        """
        Takes a tree object and a numpy array indicating direction associated with the segments of the tree to build a directed graph

        :param tree: A Tree object.
        :param direction: A numpy array indicating flow direction derived from the haemodynamic solution associated with the tree object.
        """
        for seg_id in tree.segment_dict.keys():
            node_1_id = tree.segment_dict[seg_id].node_1_id
            node_2_id = tree.segment_dict[seg_id].node_2_id
            if direction[seg_id] > 0:
                self.add_directed_edge(node_1_id,node_2_id,seg_id)
            elif direction[seg_id] < 0:
                self.add_directed_edge(node_2_id,node_1_id,seg_id)
            else:
                raise ValueError("Undetermined vessel flow direction in directional graph building.")

    def build_undirected(self, tree:Tree):
        """
        Takes a tree object to build an undirected graph

        :param tree: A Tree object.
        """
        self.connectivity_graph = []
        for segment_id in tree.segment_dict.keys():
            node_1_id = tree.segment_dict[segment_id].node_1_id
            node_2_id = tree.segment_dict[segment_id].node_2_id
            self.add_undirected_edge(node_1_id,node_2_id,segment_id) #(u, v, id) where u and v are the vertices and id is the edge ID.
        
        return
    
    def build_undirected_sub(self, segment_dict:dict):
        """
        Takes a dictionary of segment objects to build an undirected graph.

        :param segment_dict: A Dictionary containing Segment objects.
        """
        self.connectivity_graph = []
        for segment_id in segment_dict.keys():
            node_1_id = segment_dict[segment_id].node_1_id
            node_2_id = segment_dict[segment_id].node_2_id
            self.add_undirected_edge(node_1_id,node_2_id,segment_id) #(u, v, id) where u and v are the vertices and id is the edge ID.
        
        return

    def add_undirected_edge(self, u:int, v:int, edge_id:int):
        """
        Function to add an undirected edge to the graph.

        :param u: A graph vertex ID associated with the starting point of the edge.
        :param v: A graph vertex ID associated with the ending point of the edge.
        :param edge_id: A graph edge ID associated with the pair of vertex ID's u and v.
        """
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
            self.edge_list[u] = {}

        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
            self.edge_list[v] = {}

        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)
        self.edge_list[u][v] = edge_id
        self.edge_list[v][u] = edge_id

        return

    

    def _dfs(self, u:int, parent:int, time:int, discovery_time:list, low:list, visited:list, bridges:list) -> int:
        """
        Performs a Depth-First Search (DFS) to find bridges in the graph.

        :param u: The current vertex.
        :param parent: The parent vertex in the DFS tree.
        :param time: The current discovery time.
        :param discovery_time: List storing discovery times of visited vertices.
        :param low: List storing the lowest discovery times reachable from each vertex.
        :param visited: List indicating whether a vertex has been visited.
        :param bridges: List to store the found bridges.
        :return: Updated discovery time.
        """
        visited[u] = True
        discovery_time[u] = low[u] = time
        time += 1

        for v in self.adjacency_list[u]:
            if not visited[v]:
                time = self._dfs(v, u, time, discovery_time, low, visited, bridges)
                low[u] = min(low[u], low[v])
                
                if low[v] > discovery_time[u]:
                    bridges.append((u, v, self.edge_list[u][v]))
            elif v != parent:
                low[u] = min(low[u], discovery_time[v])
        
        return time
    
    def _dfs_iterative(self, u:int, parent:int, time:int, discovery_time:dict, low:dict, visited:dict, bridges:list) -> int:
        """
        Performs an iterative Depth-First Search (DFS) to find bridges in the graph.

        :param u: The current vertex.
        :param parent: The parent vertex in the DFS tree.
        :param time: The current discovery time.
        :param discovery_time: List storing discovery times of visited vertices.
        :param low: List storing the lowest discovery times reachable from each vertex.
        :param visited: List indicating whether a vertex has been visited.
        :param bridges: List to store the found bridges.
        :return: Updated discovery time.
        """
        # Stack to simulate the recursion stack in an iterative manner.
        stack = [(u, parent, time, 0)]  # (current vertex, parent, discovery time, child index)
        
        while stack:
            u, parent, discovery, child_index = stack.pop()
            
            if not visited[u]:
                # Mark the current vertex as visited and set its discovery and low times.
                visited[u] = True
                discovery_time[u] = low[u] = discovery
                time = discovery + 1

            # Process the children of the current vertex.
            for i in range(child_index, len(self.adjacency_list[u])):
                v = self.adjacency_list[u][i]

                if not visited[v]:
                    # Push the current state to the stack before moving to the child.
                    stack.append((u, parent, discovery, i + 1))
                    # Push the child to the stack.
                    stack.append((v, u, time, 0))
                    break
                elif v != parent:
                    # Update the low value of the current vertex.
                    low[u] = min(low[u], discovery_time[v])
            
            else:
                # After processing all children, update the parent's low value.
                if parent != -1:
                    low[parent] = min(low[parent], low[u])
                    # Check if the current edge is a bridge.
                    if low[u] > discovery_time[parent]:
                        bridges.append((parent, u, self.edge_list[parent][u]))

        return time

    def get_critical_connections(self) -> dict:
        """
        Finds all bridges (critical connections) in the undirected graph.

        :return: A dictionary with edge IDs as keys and boolean values indicating whether the edge is critical.
        """
        # Initialize discovery_time, low, and visited as dictionaries to handle arbitrary node IDs
        discovery_time = {i: -1 for i in self.adjacency_list.keys()}
        low = {i: -1 for i in self.adjacency_list.keys()}
        visited = {i: False for i in self.adjacency_list.keys()}
        bridges = []

        time = 0
        for i in self.adjacency_list.keys():
            if not visited[i]:
                time = self._dfs_iterative(i, -1, time, discovery_time, low, visited, bridges)

        # Create a dictionary to map edge IDs to whether they are critical or not
        edge_id_map = {self.edge_list[u][v]: False for u in self.edge_list for v in self.edge_list[u]}

        # Mark critical edges
        for u, v, edge_id in bridges:
            edge_id_map[edge_id] = True

        return edge_id_map

    def traverse_segment(self, start_vertex: int, visited_edges: set) -> list:
        """
        Traverses through the graph starting from a given vertex and collects all edges 
        until a non-2-edge vertex is encountered.

        :param start_vertex: The starting vertex for the traversal.
        :param visited_edges: A set to track visited edges to avoid revisiting.
        :return: A list of edge IDs that form the segment.
        """
        current_vertex = start_vertex
        segment_edges = []

        while current_vertex is not None:
            next_vertex = None
            for neighbor in self.adjacency_list[current_vertex]:
                edge_id = self.edge_list[current_vertex][neighbor]
                if (current_vertex, neighbor) not in visited_edges and (neighbor, current_vertex) not in visited_edges:
                    visited_edges.add((current_vertex, neighbor))
                    visited_edges.add((neighbor, current_vertex))
                    segment_edges.append(edge_id)
                    if len(self.adjacency_list[neighbor]) == 2:
                        next_vertex = neighbor
                        break
                    else:
                        next_vertex = None
                        break
            current_vertex = next_vertex

        return segment_edges

    def get_vessels(self) -> tuple[dict,dict]:
        """
        Finds all segments of edges between non-2-edge vertices in the graph, determines adjacency between vessels,
        and creates a mapping from each segment to its vessel.

        :return: A tuple containing:
                - associated_vessels_dict: A dictionary mapping each vessel ID to its corresponding segment {vessel_id:list of segment ids}.
                - adjacency_dict: A dictionary mapping each vessel ID to a set of adjacent vessel IDs {vessel_id:list of vessel ids}.
        """
        visited_edges = set()
        segments = []
        vessel_nodes = {}               # Dictionary to store which nodes belong to which vessel
        adjacency_dict = {}             # Dictionary to track adjacency between vessels
        associated_vessels_dict = {}    # Dictionary to map vessel IDs to their segments

        for vertex in self.adjacency_list:
            if len(self.adjacency_list[vertex]) != 2:
                for neighbor in self.adjacency_list[vertex]:
                    edge_id = self.edge_list[vertex][neighbor]
                    if (vertex, neighbor) not in visited_edges and (neighbor, vertex) not in visited_edges:
                        visited_edges.add((vertex, neighbor))
                        visited_edges.add((neighbor, vertex))
                        segment_edges = [edge_id]
                        if len(self.adjacency_list[neighbor]) == 2:
                            segment_edges.extend(self.traverse_segment(neighbor, visited_edges))
                        segments.append(segment_edges)

                        # Track nodes in the current vessel and assign it an ID
                        vessel_id = len(segments) - 1
                        vessel_nodes[vessel_id] = set([vertex])

                        # Add all nodes from each edge in segment_edges
                        for edge_id in segment_edges:
                            # Assuming self.edge_list has an entry like {vertex: {neighbor: edge_id}}
                            # Find the vertices for each edge_id and add them to the set
                            for node1, neighbors in self.edge_list.items():
                                for node2, edge in neighbors.items():
                                    if edge == edge_id:
                                        vessel_nodes[vessel_id].update([node1, node2])

                        # Map vessel_id to its segment
                        associated_vessels_dict[vessel_id] = segment_edges


        # Build adjacency dictionary based on shared nodes
        for vessel_id, nodes in vessel_nodes.items():
            for other_vessel_id, other_nodes in vessel_nodes.items():
                if vessel_id != other_vessel_id and not nodes.isdisjoint(other_nodes):
                    if vessel_id not in adjacency_dict:
                        adjacency_dict[vessel_id] = set()
                    adjacency_dict[vessel_id].add(other_vessel_id)

        return associated_vessels_dict, adjacency_dict


    def __str__(self):
        return f"""
        Graph Information:
          Adjaceny list: {self.adjacency_list}
          Incoming edge count: {self.incoming_edges_count}
          Edge list: {self.edge_list}
      """  

class Visualizer():
    """
    A class used to handle the visualization of the geometry and the results.

    ----------
    Class Attributes
    ----------
    ----------
    Instance Attributes
    ----------
    vtk_mesh_dict : dict
        A dictionary of the form {mesh_layer_id:{}}

    save_path : str
        A string indicating where the file should be saved

    file_name : str
        A string containing the name of the file to be saved.

    ----------
    Class Methods
    ----------  
    ----------
    Instance Methods
    ----------  
    add_mesh_to_layer(mesh_layer_id,mesh)
        Adds a mesh to the specfied mesh layer

    create_mesh_object(point_1,point_2)
        Creates a vtk mesh consisting of a line between two points

    add_cell_data_to_mesh(mesh,data_name,data)
        Adds the specified cell data to the specified mesh
    
    add_point_data_to_mesh(mesh,data_name,data)
        Adds the specified point data to the specified mesh

    set_save_path(save_path)
        Sets the save path of the class object

    augment_save_path(save_path_augmentation)
        Augments the existing save_path such that save_path = save_path+save_path_augmentation

    set_file_name(file_name)
        Sets the filename of the objects saved by the class object

    load_settings_from_config(config)
        Takes information from the Config object to set the save details

    save_to_file([list of mesh_layer_ids])
        Saves to file the specified mesh layers

    display_mesh([list of mesh_layer_ids])
        Displays the specified mesh layers.

    clear_mesh_layer(mesh_layer_id)
        Removes a mesh layer from the vtk_mesh_dict

    reset_visualizer()
        Resets the attributes of the class object to their base state

    """

    def __init__(self,config:Config):
        self.reset_visualizer()
        if config != None:
            self.load_settings_from_config(config)
        return

    def reset_visualizer(self):
        """
        Function to reset the attributes of the class object to their base state
        """
        
        self.reset_mesh_dict()
        self.save_path = ""
        self.file_name = ""
        return
    
    def reset_mesh_dict(self):
        """
        Function to reset the mesh dictionary of the class object to its base state
        """

        self.vtk_mesh_dict = {}
        return

    def load_settings_from_config(self,config:Config):
        """
        Function to take information from the Config object to set the save details

        :param config: A Config object containing the save file details.
        """
        _,_,_, output_path, test_name = config.parse_run()
        save_path = os_path_join(output_path, test_name)
        self.set_save_path(save_path)
        self.logger = config.logger
        return

    def set_save_path(self,save_path:str):
        """
        Function to set the save path of the class object

        :param save_path: A string indicating the path where the files will be saved.
        """
        self.save_path = save_path
        return
    
    def augment_save_path(self,save_path_augmentation:str):
        """
        Function to make augmentations to the default save path

        :param save_path_augmentation: A string indicating the augmentation to the path where the files will be saved.
        """
        self.save_path += "/"
        self.save_path += save_path_augmentation
        return

    def set_file_name(self,file_name):
        """
        Function to set the name of the file to be saved.

        :param save_path: A string indicating the name of the file to be saved.
        """
        self.file_name = file_name
        return
    
    def add_mesh_to_layer(self,mesh_layer_id:int,mesh:pv.PolyData,mesh_id:int):
        """
        Function to add a mesh to the specfied mesh layer. Mesh layers are used to differentiate the source of the mesh.

        :param mesh_layer_id: An integer indicating the relevant mesh layer
        :param mesh: A Pyvista Polydata object containing the mesh part. 
        :param mesh_id: An integer indicating the specific mesh being added. 
        """
        if not mesh_layer_id in self.vtk_mesh_dict.keys():
            self.vtk_mesh_dict.update({mesh_layer_id:{}})
            # self.logger.log(f"Creating Mesh Layer for layer id: {mesh_layer_id}, existing layers are: {self.vtk_mesh_dict.keys()}")
        self.vtk_mesh_dict[mesh_layer_id].update({mesh_id:mesh})
        return

    def clear_mesh_layer(self, mesh_layer_id:int, verbose:bool=False):
        """
        Function to remove a mesh layer from the vtk_mesh_dict

        :param mesh_layer_id: An integer indicating the relevant mesh layer being removed.
        """
        if mesh_layer_id in self.vtk_mesh_dict.keys():
            self.vtk_mesh_dict.pop(mesh_layer_id)
        elif verbose:
            self.logger.log(f"No meshes layer found using id: {mesh_layer_id}, existing layers are: {self.vtk_mesh_dict.keys()}")
        return
    
    def clear_mesh_object(self, mesh_layer_id:int, mesh_id:int, verbose:bool=False):
        """
        Function to remove a mesh layer from the vtk_mesh_dict

        :param mesh_layer_id: An integer indicating the relevant mesh layer being removed from.
        :param mesh_id: An integer indicating the specific mesh being added. 
        """
        if mesh_layer_id in self.vtk_mesh_dict.keys():
            if mesh_id in self.vtk_mesh_dict[mesh_layer_id].keys():
                self.vtk_mesh_dict[mesh_layer_id].pop(mesh_id)
            elif verbose:
                self.logger.log(f"No meshes found using id: {mesh_layer_id}, existing meshes are: {self.vtk_mesh_dict[mesh_layer_id].keys()}")
        elif verbose:
            self.logger.log(f"No meshes layer found using id: {mesh_layer_id}, existing layers are: {self.vtk_mesh_dict.keys()}")
        return

    def create_mesh_object(self,point1:np.ndarray,point2:np.ndarray) -> pv.PolyData:
        """
        Function to create a vtk mesh consisting of a line between two points

        :param point1: An [x,y,z] numpy array indicating the location of point 1.
        :param point2: An [x,y,z] numpy array indicating the location of point 2.
        :return: A Pyvista Polydata object containing the mesh part. 
        """
        point_list = [point1, point2]
        line_connectivity = [2, 0, 1]
        number_of_lines = 1
        
        # Create PolyData
        mesh = pv.PolyData(point_list, lines=line_connectivity, n_lines=number_of_lines)

        # Batch addition of cell and point data
        cell_data = {
            "Mesh Type": np.nan,
            "Segment_id": np.nan,
            "WSS": np.nan,
            "Flow": np.nan,
            "Haematocrit": np.nan,
            "Viscosity": np.nan,
            "Length": np.nan,
            "Pressure Drop": np.nan,
            "Pressure Drop by Length": np.nan,
            "Vessel Number": np.nan,
            "Total Luminal Volume": np.nan,
            "Sprout_id": np.nan
        }

        point_data = {
            "Radius": [np.nan, np.nan],
            "Pressure": [np.nan, np.nan],
            "Oxygen": [np.nan, np.nan],
            "Oxygen Proportion": [np.nan, np.nan],
            "Node_id": [np.nan, np.nan]
        }

        for name, data in cell_data.items():
            mesh.cell_data[name] = data
        for name, data in point_data.items():
            mesh.point_data[name] = data

        return mesh
    
    def create_full_mesh_for_adaptation(self,tree:Tree,haemodynamic_solution:dict,haematocrit_solution:dict,node_point_ref:dict):
        """
        Function to create a vtk mesh for the entire Tree at once

        :param tree: A tree object representing the geometry to be constructed.
        :param haemodynamic_solution: A dictionary containing the haemodynamic solution
        :param haemodynamic_solution: A dictionary containing the haematocrit solution
        :return: A Pyvista Polydata object containing the full mesh. 
        """

        node_locs = np.array([tree.node_dict[node_key].location() for node_key in tree.node_dict])
        
        # Vectorized retrieval of node pairs for all segment_ids
        node_pairs = np.array([tree.get_node_ids_on_segment(seg_id) for seg_id in tree.segment_dict])
        # Prepare an array of 2s for each line segment
        twos = np.full((node_pairs.shape[0], 1), 2)
        # Concatenate the 2s and the node pairs along the columns
        connectivity_array = np.column_stack((twos, node_pairs))
        connectivity_array.flatten()
        
        # Create a combined PolyData object
        combined_mesh = pv.PolyData(node_locs, lines=connectivity_array)

        # Batch addition of cell and point data
        number_of_cells = combined_mesh.n_cells
        number_of_points = combined_mesh.n_points
        volume = tree.get_network_volume()

        # Create Baseline
        self.cell_data = {
            "Mesh Type": np.full(number_of_cells, 0),
            "Segment_id": np.full(number_of_cells, np.nan),
            "WSS": np.full(number_of_cells, np.nan),
            "Flow": np.full(number_of_cells, np.nan),
            "Haematocrit": np.full(number_of_cells, np.nan),
            "Viscosity": np.full(number_of_cells, np.nan),
            "Length": np.full(number_of_cells, np.nan),
            "Pressure Drop": np.full(number_of_cells, np.nan),
            "Pressure Drop by Length": np.full(number_of_cells, np.nan),
            "Vessel Number": np.full(number_of_cells, np.nan),
            "Total Luminal Volume": np.full(number_of_cells, volume),
            "Sprout_id": np.full(number_of_cells, np.nan)
        }

        self.point_data = {
            "Radius": np.full((number_of_points,), np.nan),
            "Pressure": np.full((number_of_points,), np.nan),
            "Oxygen": np.full((number_of_points,), np.nan),
            "Oxygen Proportion": np.full((number_of_points,), np.nan),
            "Node_id": np.full((number_of_points,), np.nan)
        }

        self.populate_full_data(tree,haemodynamic_solution,haematocrit_solution,node_point_ref=node_point_ref,haemodynamic=True)

        # Assigning all cell data at once
        for name, data in self.cell_data.items():
            combined_mesh.cell_data[name] = data
        
        # Assigning all point data at once
        for name, data in self.point_data.items():
            combined_mesh.point_data[name] = data
            
        self.add_mesh_to_layer(0,combined_mesh,0)

        return combined_mesh
    
    def create_full_geometry_mesh(self,tree:Tree):
        """
        Function to create a vtk mesh for the entire Tree at once for the Growth purpose

        :param tree: A tree object representing the geometry to be constructed.
        :param haemodynamic_solution: A dictionary containing the haemodynamic solution
        :param haematocrit_solution: A dictionary containing the haematocrit solution
        :param oxygen_solution: A dictionary containing the oxygen solution
        :param node_point_ref: A dictionary containing the node_point_ref
        :return: A Pyvista Polydata object containing the full mesh. 
        """

        node_locs = np.array([tree.node_dict[node_key].location() for node_key in tree.node_dict])
        # Vectorized retrieval of node pairs for all segment_ids
        node_pairs = np.array([tree.get_node_ids_on_segment(seg_id) for seg_id in tree.segment_dict])
        # Prepare an array of 2s for each line segment
        twos = np.full((node_pairs.shape[0], 1), 2)
        # Concatenate the 2s and the node pairs along the columns
        connectivity_array = np.column_stack((twos, node_pairs))
        connectivity_array.flatten()
        
        # Create a combined PolyData object
        self.original_tree_locs = node_locs
        self.original_connectivity = connectivity_array
        combined_mesh = pv.PolyData(node_locs, lines=connectivity_array)

        # Batch addition of cell and point data
        number_of_cells = combined_mesh.n_cells
        number_of_points = combined_mesh.n_points
        volume = tree.get_network_volume()
        self.num_cells = number_of_cells
        self.num_points = number_of_points

        # Create Baseline
        self.cell_data = {
            "Mesh Type": np.full(number_of_cells, 0),
            "Segment_id": np.full(number_of_cells, np.nan),
            "WSS": np.full(number_of_cells, np.nan),
            "Velocity": np.full(number_of_cells, np.nan),
            "Flow": np.full(number_of_cells, np.nan),
            "Haematocrit": np.full(number_of_cells, np.nan),
            "Viscosity": np.full(number_of_cells, np.nan),
            "Length": np.full(number_of_cells, np.nan),
            "Pressure Drop": np.full(number_of_cells, np.nan),
            "Pressure Drop by Length": np.full(number_of_cells, np.nan),
            "Vessel Number": np.full(number_of_cells, np.nan),
            "Total Luminal Volume": np.full(number_of_cells, volume),
            "Sprout_id": np.full(number_of_cells, np.nan)
        }

        self.point_data = {
            "Radius": np.full((number_of_points,), np.nan),
            "Pressure": np.full((number_of_points,), np.nan),
            "Oxygen": np.full((number_of_points,), np.nan),
            "Oxygen Proportion": np.full((number_of_points,), np.nan),
            "Node_id": np.full((number_of_points,), np.nan)
        }

        self.populate_full_data(tree)

        # Assigning all cell data at once
        for name, data in self.cell_data.items():
            combined_mesh.cell_data[name] = data
        
        # Assigning all point data at once
        for name, data in self.point_data.items():
            combined_mesh.point_data[name] = data
            
        self.add_mesh_to_layer(0,combined_mesh,0)

        return combined_mesh
    
    def create_full_mesh_for_growth(self,tree:Tree,haemodynamic_solution:dict,haematocrit_solution:dict,oxygen_solution:dict,node_point_ref:dict):
        """
        Function to create a vtk mesh for the entire Tree at once for the Growth purpose

        :param tree: A tree object representing the geometry to be constructed.
        :param haemodynamic_solution: A dictionary containing the haemodynamic solution
        :param haematocrit_solution: A dictionary containing the haematocrit solution
        :param oxygen_solution: A dictionary containing the oxygen solution
        :param node_point_ref: A dictionary containing the node_point_ref
        :return: A Pyvista Polydata object containing the full mesh. 
        """

        node_locs = np.array([tree.node_dict[node_key].location() for node_key in tree.node_dict])
        # Vectorized retrieval of node pairs for all segment_ids
        node_pairs = np.array([tree.get_node_ids_on_segment(seg_id) for seg_id in tree.segment_dict])
        # Prepare an array of 2s for each line segment
        twos = np.full((node_pairs.shape[0], 1), 2)
        # Concatenate the 2s and the node pairs along the columns
        connectivity_array = np.column_stack((twos, node_pairs))
        connectivity_array.flatten()
        
        # Create a combined PolyData object
        self.original_tree_locs = node_locs
        self.original_connectivity = connectivity_array
        combined_mesh = pv.PolyData(node_locs, lines=connectivity_array)

        # Batch addition of cell and point data
        number_of_cells = combined_mesh.n_cells
        number_of_points = combined_mesh.n_points
        volume = tree.get_network_volume()
        self.num_cells = number_of_cells
        self.num_points = number_of_points

        # Create Baseline
        self.cell_data = {
            "Mesh Type": np.full(number_of_cells, 0),
            "Segment_id": np.full(number_of_cells, np.nan),
            "WSS": np.full(number_of_cells, np.nan),
            "Velocity": np.full(number_of_cells, np.nan),
            "Flow": np.full(number_of_cells, np.nan),
            "Haematocrit": np.full(number_of_cells, np.nan),
            "Viscosity": np.full(number_of_cells, np.nan),
            "Length": np.full(number_of_cells, np.nan),
            "Pressure Drop": np.full(number_of_cells, np.nan),
            "Pressure Drop by Length": np.full(number_of_cells, np.nan),
            "Vessel Number": np.full(number_of_cells, np.nan),
            "Total Luminal Volume": np.full(number_of_cells, volume),
            "Sprout_id": np.full(number_of_cells, np.nan)
        }

        self.point_data = {
            "Radius": np.full((number_of_points,), np.nan),
            "Pressure": np.full((number_of_points,), np.nan),
            "Oxygen": np.full((number_of_points,), np.nan),
            "Oxygen Proportion": np.full((number_of_points,), np.nan),
            "Node_id": np.full((number_of_points,), np.nan)
        }

        self.populate_full_data(tree,haemodynamic_solution,haematocrit_solution,oxygen_solution,node_point_ref,True,True,True)

        # Assigning all cell data at once
        for name, data in self.cell_data.items():
            combined_mesh.cell_data[name] = data
        
        # Assigning all point data at once
        for name, data in self.point_data.items():
            combined_mesh.point_data[name] = data
            
        self.add_mesh_to_layer(0,combined_mesh,0)

        return combined_mesh
    
    def update_mesh_for_growth(self,tree:Tree,sprout_dict:dict,sprout_to_segment_id_list:list,growth_vector_dict:dict):
        """
        Function to populate growth on top of an established baseline for growth (should be run after: create_full_mesh_for_growth())
        :param tree: A tree object representing the geometry to be constructed.
        :param sprout_dict: A dictionary containing all of the growing sprouts
        :param sprout_to_segment_id_list: A list containing all of the segments of the tree that are new.
        :param growth_vector_dict: A dictionary containing all of the growth vectors at the tips of the sprouts
        :return:
        """

        master_point_locs = copy.copy(self.original_tree_locs)
        master_connectivity = copy.copy(self.original_connectivity)
        master_cell_data = copy.copy(self.cell_data)
        master_point_data = copy.copy(self.point_data)

        original_len = len(master_point_locs)
        points_count = len(master_point_locs)
        
        # SPROUT TO SEGMENT CONTRIBUTION
        if len(sprout_to_segment_id_list) > 0:
            # Vectorized retrieval of node pairs for all segment_ids
            node_pairs = np.array([tree.get_node_ids_on_segment(seg_id) for seg_id in sprout_to_segment_id_list])
            lengths = np.array([tree.length(seg_id) for seg_id in sprout_to_segment_id_list])
            # Vectorized retrieval of all node locs for specified segments
            node_locs = np.array([tree.node_dict[node_key].location() for node_key in np.unique(node_pairs)])
            # Create New Mesh Points for the nodes on New Segments
            node_id_to_mesh_point_mapping = {node_id:i+original_len for i, node_id in enumerate(np.unique(node_pairs))}
            # Vectorize the dictionary lookup
            vectorized_lookup = np.vectorize(lambda node_id: node_id_to_mesh_point_mapping[node_id])

            # Apply the vectorized lookup to each element in node_pairs
            updated_node_pairs = vectorized_lookup(node_pairs)


            # Prepare an array of 2s for each line segment
            twos = np.full((updated_node_pairs.shape[0], 1), 2)
            # Concatenate the 2s and the node pairs along the columns
            connectivity_array = np.column_stack((twos, updated_node_pairs))
            connectivity_array.flatten()
            offset_points = len(np.unique(node_pairs))
            cells_count = len(sprout_to_segment_id_list)
            node_ids = np.array([tree.node_dict[node_key].node_id for node_key in np.unique(node_pairs)])
            base_radius = tree.segment_dict[sprout_to_segment_id_list[0]]._Segment__DEFAULT_RADIUS

            # Create Baseline
            ss_extend_cell_data = {
                "Mesh Type": np.full(cells_count, 1),
                "Segment_id": np.array(sprout_to_segment_id_list),
                "WSS": np.full(cells_count, np.nan),
                "Velocity": np.full(cells_count, np.nan),
                "Flow": np.full(cells_count, np.nan),
                "Haematocrit": np.full(cells_count, np.nan),
                "Viscosity": np.full(cells_count, np.nan),
                "Length": np.array(lengths),
                "Pressure Drop": np.full(cells_count, np.nan),
                "Pressure Drop by Length": np.full(cells_count, np.nan),
                "Vessel Number": np.full(cells_count, np.nan),
                "Total Luminal Volume": np.full(cells_count, np.nan),
                "Sprout_id": np.full(cells_count, np.nan)
            }

            ss_extend_point_data = {
                "Radius": np.full((offset_points,), base_radius),
                "Pressure": np.full((offset_points,), np.nan),
                "Oxygen": np.full((offset_points,), np.nan),
                "Oxygen Proportion": np.full((offset_points,), np.nan),
                "Node_id": np.array(node_ids)
            }

            # self.logger.log(f"Segments: {sprout_to_segment_id_list}")
            # self.logger.log(f"Connectivity: {connectivity_array}")
            # self.logger.log(f"Node_pairs: {node_pairs}")

            master_point_locs = np.append(master_point_locs,node_locs)
            master_connectivity = np.append(master_connectivity,connectivity_array)
            master_cell_data = {key: np.hstack((master_cell_data[key], ss_extend_cell_data[key])) for key in master_cell_data}
            master_point_data = {key: np.hstack((master_point_data[key], ss_extend_point_data[key])) for key in master_point_data}

            # node_ids = np.array([tree.node_dict[node_key].node_id for node_key in tree.node_dict])
            # master_point_data["Node_id"] = node_ids
            points_count = original_len+len(node_ids)


        
        # SPROUT CONTRIBUTION
        # Step 1: Vectorized stacking of locations (remove the first entry)
        all_locations = {sprout_id: np.vstack(list(sprout_obj.snail_trail_dict.values())) 
                        for sprout_id, sprout_obj in sprout_dict.items()}

        # Step 2: Vectorized creation of connectivity maps and replacements
        all_connectivity_maps = {}
        # self.logger.log(f"len(master_points before sprout) = {len(master_point_locs)}")
        total_offset = 0

        for sprout_id, sprout_obj in sprout_dict.items():
            locations = all_locations[sprout_id]
            n = len(locations)  # Number of points before removing the first

            # Step 3: Apply the custom offset and update the offset
            new_offset = n
            
            if len(locations) > 1:
                
                # Create connectivity map for each sprout
                connectivity_map = np.column_stack((
                    np.full(n-1, 2),
                    np.arange(points_count, points_count + n-1),  # Target points (from the 2nd onward)
                    np.arange(points_count+1, points_count + n)   # Source points
                )).ravel()

                points_count += new_offset
                total_offset += new_offset

                # Replace the first source point with the sprout's base_node_id
                # connectivity_map[1] = sprout_obj.base_node_id

                # Store the result
                all_connectivity_maps[sprout_id] = connectivity_map

                sprout_extend_cell_data = {
                    "Mesh Type": np.full(new_offset-1, 2),
                    "Segment_id": np.full(new_offset-1, np.nan),
                    "WSS": np.full(new_offset-1, np.nan),
                    "Velocity": np.full(new_offset-1, np.nan),
                    "Flow": np.full(new_offset-1, np.nan),
                    "Haematocrit": np.full(new_offset-1, np.nan),
                    "Viscosity": np.full(new_offset-1, np.nan),
                    "Length": np.full(new_offset-1, np.nan),
                    "Pressure Drop": np.full(new_offset-1, np.nan),
                    "Pressure Drop by Length": np.full(new_offset-1, np.nan),
                    "Vessel Number": np.full(new_offset-1, np.nan),
                    "Total Luminal Volume": np.full(new_offset-1, np.nan),
                    "Sprout_id": np.full(new_offset-1, sprout_id)
                }

                sprout_extend_point_data = {
                    "Radius": np.full((new_offset,), sprout_obj.base_radius),
                    "Pressure": np.full((new_offset,), np.nan),
                    "Oxygen": np.full((new_offset,), np.nan),
                    "Oxygen Proportion": np.full((new_offset,), np.nan),
                    "Node_id": np.full((new_offset,), np.nan)
                }

                # self.logger.log(f"len(locations of sprout {sprout_id}) = {len(locations)}")
                master_point_locs = np.append(master_point_locs,locations)
                # self.logger.log(f"len(master_points after sprout {sprout_id}) = {len(master_point_locs)}")
                # self.logger.log(master_point_locs)
                master_connectivity = np.append(master_connectivity,connectivity_map)
                master_cell_data = {key: np.hstack((master_cell_data[key], sprout_extend_cell_data[key])) for key in master_cell_data}
                master_point_data = {key: np.hstack((master_point_data[key], sprout_extend_point_data[key])) for key in master_point_data}

        growth_locs = []
        growth_connectivity = []
        mesh_type = []
        radii = []
        growth_cells = 0
        growth_points = 0
        for keys in growth_vector_dict.keys():
            if "alpha" in  growth_vector_dict[keys].keys():
                tip = sprout_dict[keys].tip_node
                xyz1 = sprout_dict[keys].snail_trail_dict[tip]
                alpha_vector = growth_vector_dict[keys]["alpha"]
                alpha_vector = alpha_vector * 25e-6
                xyz2 = xyz1+alpha_vector
                growth_locs.extend([xyz1,xyz2])
                growth_connectivity.extend([2,points_count,points_count+1])
                mesh_type.extend([3])
                radii.extend([6e-6,1e-6])
                points_count += 2
                growth_points += 2
                growth_cells += 1
            if "beta" in  growth_vector_dict[keys].keys():
                tip = sprout_dict[keys].tip_node
                xyz1 = sprout_dict[keys].snail_trail_dict[tip]
                beta_vector = growth_vector_dict[keys]["beta"]
                beta_vector = beta_vector * 25e-6
                xyz2 = xyz1+beta_vector
                growth_locs.extend([xyz1,xyz2])
                growth_connectivity.extend([2,points_count,points_count+1])
                mesh_type.extend([4])
                radii.extend([6e-6,1e-6])
                points_count += 2
                growth_points += 2
                growth_cells += 1
            if "gamma" in  growth_vector_dict[keys].keys():
                tip = sprout_dict[keys].tip_node
                xyz1 = sprout_dict[keys].snail_trail_dict[tip]
                gamma_vector = growth_vector_dict[keys]["gamma"]
                gamma_vector = gamma_vector * 25e-6
                xyz2 = xyz1+gamma_vector
                growth_locs.extend([xyz1,xyz2])
                growth_connectivity.extend([2,points_count,points_count+1])
                mesh_type.extend([5])
                radii.extend([6e-6,1e-6])
                points_count += 2
                growth_points += 2
                growth_cells += 1

        # Create Baseline
        gv_extend_cell_data = {
            "Mesh Type": np.array(mesh_type),
            "Segment_id": np.full(growth_cells, np.nan),
            "WSS": np.full(growth_cells, np.nan),
            "Velocity": np.full(growth_cells, np.nan),
            "Flow": np.full(growth_cells, np.nan),
            "Haematocrit": np.full(growth_cells, np.nan),
            "Viscosity": np.full(growth_cells, np.nan),
            "Length": np.full(growth_cells, np.nan),
            "Pressure Drop": np.full(growth_cells, np.nan),
            "Pressure Drop by Length": np.full(growth_cells, np.nan),
            "Vessel Number": np.full(growth_cells, np.nan),
            "Total Luminal Volume": np.full(growth_cells, np.nan),
            "Sprout_id": np.full(growth_cells, np.nan)
        }

        gv_extend_point_data = {
            "Radius": np.array(radii),
            "Pressure": np.full((growth_points,), np.nan),
            "Oxygen": np.full((growth_points,), np.nan),
            "Oxygen Proportion": np.full((growth_points,), np.nan),
            "Node_id": np.full((growth_points,), np.nan)
        }

        if growth_cells > 0:
            master_point_locs = np.append(master_point_locs,growth_locs)
            master_connectivity =  np.append(master_connectivity,growth_connectivity)
            master_cell_data = {key: np.hstack((master_cell_data[key], gv_extend_cell_data[key])) for key in master_cell_data}
            master_point_data = {key: np.hstack((master_point_data[key], gv_extend_point_data[key])) for key in master_point_data}

        # Create the mesh
        combined_mesh = pv.PolyData(master_point_locs, lines=master_connectivity)


        # self.logger.log(f"len(original_points) = {original_len}")
        # self.logger.log(f"len(master_points) = {len(master_point_locs)}")
        # self.logger.log(f"len(sprout_offset) = {total_offset}")
        # self.logger.log(f"len(growth_points) = {growth_points}")

        # Assigning all cell data at once
        for name, data in master_cell_data.items():
            # self.logger.log(f"Assinging {name}")
            combined_mesh.cell_data[name] = data
        
        # Assigning all point data at once
        for name, data in master_point_data.items():
            # self.logger.log(f"Assinging {name}")
            combined_mesh.point_data[name] = data
            
        self.add_mesh_to_layer(0,combined_mesh,0)

        return combined_mesh


    
    def populate_full_data(self,tree:Tree,haemodynamic_solution=None,haematocrit_solution=None,oxygen_solution=None,node_point_ref:dict=None,geometry=True,haemodynamic=False,oxygen=False): # type: ignore
        """
        Function to populate the cell data and point data objects with information for plotting with a single full mesh

        :param tree: A tree object representing the geometry to be constructed.
        :param haemodynamic_solution: A dictionary containing the haemodynamic solution
        :param haemodynamic_solution: A dictionary containing the haematocrit solution
        :param haemodynamic_solution: A dictionary containing the oxygen solution
        :param node_point_ref: A dictionary the reference dictionary to map form nodes to points on the mesh
        :param geometry: A boolean indicating if the geometry should be updated
        :param hameodynamic: A boolean indicating if the hameodynamic solution should be updated
        :param oxygen: A boolean indicating if the oxygen solution should be updated
        :return:
        """
        # Create Pressure Info
        # Get all node IDs and associated pressures at once
        
        seg_ids = np.array([tree.segment_dict[seg_id].segment_id for seg_id in tree.segment_dict])
        radii = np.array([tree.segment_dict[seg_id].radius for seg_id in seg_ids])
        lengths = np.array([tree.length(seg_id) for seg_id in seg_ids])
        node_ids = np.array([tree.node_dict[node_key].node_id for node_key in tree.node_dict])
            
            
        if geometry:
            node_pairs = np.array([tree.get_node_ids_on_segment(seg_id) for seg_id in seg_ids])
            # Flatten the node pairs so we can map radii to both nodes
            node_ids_flat = node_pairs.flatten()

            # Repeat each radius twice (once for node_1, once for node_2)
            repeated_radii = np.repeat(radii, 2)

            # Create an empty array to store the maximum radii for each node
            max_node_id = np.max(node_ids_flat) + 1
            max_radii = np.zeros(max_node_id)  # This will store the maximum radius for each node

            # Use np.maximum.at to get the maximum radius for each node
            np.maximum.at(max_radii, node_ids_flat, repeated_radii)

            self.point_data['Radius'] = max_radii
            self.point_data["Node_id"] = node_ids
            self.cell_data['Length'] = lengths
            self.cell_data['Segment_id'] = seg_ids
            
        if haemodynamic:
            if haemodynamic_solution is None:
                raise ValueError(f"Updating the Haemodynamic Solution requires a haemodynamic solution to be provided")
            if haematocrit_solution is None:
                raise ValueError(f"Updating the Haemodynamic Solution requires a haematocrit solution to be provided") 
            if node_point_ref is None:
                raise ValueError(f"Updating the Haemodynamic Solution requires a node point reference to be provided") 
            pressures = np.array([haemodynamic_solution["pvx"][node_point_ref[node_id]] for node_id in node_ids])
            # Batch calculate WSS (tw = D dP / (4L)) for all segments
            node_1_ids = np.array([tree.segment_dict[seg_id].node_1_id for seg_id in tree.segment_dict])
            node_2_ids = np.array([tree.segment_dict[seg_id].node_2_id for seg_id in tree.segment_dict])
            pressure_diffs = np.abs(pressures[node_1_ids] - pressures[node_2_ids])
            wss = radii * pressure_diffs / (2 * lengths)

        
            self.point_data['Pressure'] = pressures
            self.cell_data['WSS'] = wss
            self.cell_data['Pressure Drop'] = pressure_diffs
            self.cell_data['Pressure Drop by Length'] = pressure_diffs / lengths

            # velocity and flow
            areas = np.array([tree.area(seg_id) for seg_id in seg_ids])
            start_velocities = np.array([haemodynamic_solution["uvx"][segment_id+1][0] for segment_id in seg_ids])
            end_velocities = np.array([haemodynamic_solution["uvx"][segment_id+1][-1]  for segment_id in seg_ids])
            
            start_flows = areas*start_velocities
            end_flows = areas*end_velocities

            # Compute mean velocities and mean flows
            mean_velocities = np.mean([start_velocities, end_velocities], axis=0)
            mean_flows = np.mean([start_flows, end_flows], axis=0)

            self.cell_data["Velocity"] = mean_velocities
            self.cell_data["Flow"] = mean_flows
    
            # Get haematocrit values at the start and end for each segment
            H_starts = np.array([haematocrit_solution["hx"][segment_id + 1][0] for segment_id in seg_ids])
            H_ends = np.array([haematocrit_solution["hx"][segment_id + 1][-1] for segment_id in seg_ids])
            # Compute mean haematocrit
            H_means = np.mean([H_starts, H_ends], axis=0)
            temperature = 37
            diameters = 2 * radii * 1e-6

            # Viscosity calculations (vectorized)
            viscosity_water_0 = 1.808  # Centipoise viscosity at 0 degrees Celsius
            viscosity_ref = 1.8 * viscosity_water_0 / (1 + 0.0337 * temperature + 0.00022 * temperature**2)

            viscosity_nominal_ref = (6 * np.exp(-0.085 * diameters) + 3.2 - 2.44 * np.exp(-0.06 * np.power(diameters, 0.645)))
            C11 = (0.8 + np.exp(-0.075 * diameters))
            C12 = (-1 + 1 / (1 + np.power(diameters, 12) * 1e-11))
            C1 = C11 * C12
            C2 = 1 / (1 + np.power(diameters, 12) * 1e-11)
            C = C1 + C2

            V1 = viscosity_nominal_ref - 1
            V2_top = np.power(1 - H_means, C) - 1
            V2_bot = np.power(1 - 0.45, C) - 1
            V2 = V2_top / V2_bot
            V3 = np.power((diameters / (diameters - 1.1)), 2)

            # Compute relative viscosity
            relative_viscosity = (1 + V1 * V2 * V3) * V3
            viscosity_vessel = viscosity_ref * relative_viscosity * 1e-3

            self.cell_data['Haematocrit'] = H_means
            self.cell_data['Viscosity'] = viscosity_vessel
            
        if oxygen:
            if oxygen_solution is None:
                raise ValueError(f"Updating the Oxygen Solution requires a oxygen solution to be provided")
            if node_point_ref is None:
                raise ValueError(f"Updating the Oxygen Solution requires a node point reference to be provided") 
            # oxygen
            # Get the Inlet Values:
            inlet_ids = tree.get_node_ids_inlet()
            num_of_inlets = 0
            total_inlet_O2 = 0
            for inlet_id in inlet_ids:
                inlet_pid = node_point_ref[inlet_id]
                O2_value = oxygen_solution["ovx"][inlet_pid]
                num_of_inlets += 1
                total_inlet_O2 += O2_value
            
            mean_inlet_O2 = total_inlet_O2/num_of_inlets

            # Get node ids associated with the segment in the tree representation
            oxygens = np.array([oxygen_solution["ovx"][node_point_ref[node_id]] for node_id in node_ids])
            oxygen_proportions = oxygens / mean_inlet_O2
            self.point_data['Oxygen'] = oxygens
            self.point_data['Oxygen Proportion'] = oxygen_proportions
            
        return

    def add_cell_data_to_mesh(self,mesh:pv.PolyData,data_name:str,data) -> pv.PolyData:
        """
        Function to add the specified cell data to the specified mesh.
        Since the mesh pieces we are working with are a single line connecting two points, data consists of a single value

        :param mesh: A Pyvista Polydata object containing the mesh part. 
        :param data_name: A string indicating the type of data being added eg: radius, wss, velocity
        :param data: A piece of data attached to that cell/segment. Should be a number or string
        :return: A Pyvista Polydata object containing the mesh part now with cell data. 
        """
        mesh.cell_data[data_name] = data
        return mesh
    
    def add_point_data_to_mesh(self,mesh:pv.PolyData,data_name:str,data) -> pv.PolyData:
        """
        Function to add the specified point data to the specified mesh.
        Since the mesh pieces we are working with are a single line connecting two points, data consists of a list with two data values

        :param mesh: A Pyvista Polydata object containing the mesh part. 
        :param data_name: A string indicating the type of data being added eg: radius, wss, velocity
        :param data: A piece of data attached to that cell/segment. Should be a list of numbers
        :return: A Pyvista Polydata object containing the mesh part now with point data. 
        """
        mesh.point_data[data_name] = data
        return mesh
    
    def save_to_file(self,mesh_layer_ids:list):
        """
        Function to save to file the specified mesh layers, mesh layer ids are to be passed in a list [id1,id2]
        The file will be saved at the location indicated by self.save_path with name indicated by self.file_name

        :param mesh_layer_ids: A list of integers indicating the relevant mesh layers
        """
        final_vtk_mesh = None
        mesh_layer_id = None
        for mesh_layer_id in mesh_layer_ids:
            if mesh_layer_id in self.vtk_mesh_dict.keys():
                meshes_to_add = self.vtk_mesh_dict[mesh_layer_id].values()
                for mesh in meshes_to_add:
                    if final_vtk_mesh == None:
                        final_vtk_mesh = copy.deepcopy(mesh)
                    else:
                        final_vtk_mesh += mesh

        if final_vtk_mesh == None:
            self.logger.log(f"No meshes were added to final mesh using layers: {mesh_layer_id}, existing layers are: {self.vtk_mesh_dict.keys()}")
        else:
            save_string = self.save_path+"/"+self.file_name
            final_vtk_mesh.save(save_string)
            self.logger.log(f"Saved file to {save_string}")
        return
    
    def save_specific_mesh_to_file(self,mesh):
        """
        Function to save to file the specified mesh
        The file will be saved at the location indicated by self.save_path with name indicated by self.file_name

        :param mesh: A singular vtk mesh
        """
        
        save_string = self.save_path+"/"+self.file_name
        mesh.save(save_string)
        self.logger.log(f"Saved file to {save_string}")
        return
    
    def display_mesh(self,mesh_layer_ids:list,property_to_color:str):
        """
        Function to display the specified mesh layers, mesh layer ids are to be passed in a list [id1,id2]
        The file will be saved at the location indicated by self.save_path with name indicated by self.file_name

        :param mesh_layer_ids: A list of integers indicating the relevant mesh layers
        """
        final_vtk_mesh = None
        mesh_layer_id = None
        for mesh_layer_id in mesh_layer_ids:
            if mesh_layer_id in self.vtk_mesh_dict.keys():
                meshes_to_add = self.vtk_mesh_dict[mesh_layer_id].values()
                for mesh in meshes_to_add:
                    if final_vtk_mesh == None:
                        final_vtk_mesh = copy.deepcopy(mesh)
                    else:
                        final_vtk_mesh.merge(mesh,merge_points=False)

        if final_vtk_mesh == None:
            self.logger.log(f"No meshes were added to final mesh using layers: {mesh_layer_id}, existing layers are: {self.vtk_mesh_dict.keys()}")
        else:
            plotter = pv.Plotter()
            if property_to_color in final_vtk_mesh.array_names:
                scalar = property_to_color
                plotter.add_mesh(final_vtk_mesh, scalars=scalar,cmap="viridis")
            else:
                raise ValueError(f"{property_to_color} is not in scalars: {final_vtk_mesh.array_names}")

            plotter.view_vector([4,-2,-1.5])
            plotter.camera.roll = -180
            # plotter.show_axes()
            plotter.show()
            
        return
    
    def process_segment(self, tree:Tree, segment_id:int, volume:float):
        # Extract data for this segment
        node1, node2 = tree.get_node_ids_on_segment(segment_id)
        point1, point2 = tree.get_segment_node_locations(segment_id)
        
        # Create the mesh
        mesh = self.create_mesh_object(point1, point2)
        
        # Add relevant data to the mesh
        radius = tree.segment_dict[segment_id].radius
        length = tree.length(segment_id)
        mesh = self.add_point_data_to_mesh(mesh, "Radius", [radius, radius])
        mesh = self.add_cell_data_to_mesh(mesh, "Mesh Type", 0)
        mesh = self.add_cell_data_to_mesh(mesh, "Segment_id", segment_id)
        mesh = self.add_cell_data_to_mesh(mesh, "Total Luminal Volume", volume)
        mesh = self.add_cell_data_to_mesh(mesh, "Length", length)
        mesh = self.add_point_data_to_mesh(mesh, "Node_id", [node1, node2])
        
        return mesh, segment_id

    def add_tree_to_visualizer(self, tree: Tree):
        volume = tree.get_network_volume()

        # Parallelize the processing of each segment
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each segment
            future_to_segment = {
                executor.submit(self.process_segment, tree, key, volume): key
                for key in tree.segment_dict.keys()
            }
            
            # As tasks complete, add meshes to the visualizer
            for future in concurrent.futures.as_completed(future_to_segment):
                segment_id = future_to_segment[future]
                try:
                    mesh, segment_id = future.result()
                    
                    # Access to shared resources needs to be synchronized
                    self.add_mesh_to_layer(0, mesh, segment_id)
                except Exception as exc:
                    print(f"Segment {segment_id} generated an exception: {exc}")
        
        # Continue with the graph representation part (can also be parallelized if needed)
        graph = GraphRepresentation()
        graph.build_undirected(tree)
        associated_vessels_dict,_ = graph.get_vessels()

        for vessel_num, segment_list in associated_vessels_dict.items():
            for segment_id in segment_list:
                # Add cell data to each segment for vessel number
                mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id], "Vessel Number", vessel_num)

        return
    
    def add_specific_segments_to_visualizer(self,tree:Tree,segment_ids:list):
        for segment_id in segment_ids:
            node1, node2 = tree.get_node_ids_on_segment(segment_id)
            point1, point2 = tree.get_segment_node_locations(segment_id)
            mesh = self.create_mesh_object(point1,point2)
            length = tree.length(segment_id)
            radius = tree.segment_dict[segment_id].radius
            mesh = self.add_point_data_to_mesh(mesh,"Radius", [radius,radius])
            mesh = self.add_cell_data_to_mesh(mesh,"Mesh Type", 1)
            mesh = self.add_cell_data_to_mesh(mesh,"Segment_id", segment_id)
            mesh = self.add_cell_data_to_mesh(mesh,"Length", length)
            mesh = self.add_point_data_to_mesh(mesh,"Node_id", [node1, node2])
            self.add_mesh_to_layer(1, mesh, segment_id)
        return
    
    def add_sprout_dict_to_visualizer(self,sprout_dict:dict):
        for key, sprout in sprout_dict.items():
            self.add_sprout_to_visualizer(key,sprout)
        return

    def add_sprout_to_visualizer(self,key:int,sprout:Sprout):
        if sprout.tip_node > 0:
            final_sprout_mesh = None
            for i in range(sprout.tip_node):
                if i == 0:
                    xyz1 = sprout.snail_trail_dict[i]
                xyz2 = sprout.snail_trail_dict[i+1]
                mesh = self.create_mesh_object(xyz1,xyz2) # type: ignore
                mesh = self.add_point_data_to_mesh(mesh,"Radius", [6e-6,6e-6])
                mesh = self.add_cell_data_to_mesh(mesh,"Mesh Type", 2)
                mesh = self.add_cell_data_to_mesh(mesh,"Sprout_id", key)

                if final_sprout_mesh == None:
                    final_sprout_mesh = copy.deepcopy(mesh)
                else:
                    final_sprout_mesh += mesh
                
                xyz1 = xyz2

            self.add_mesh_to_layer(2, final_sprout_mesh, key)
        return
    
    def update_sprout_dict_to_visualizer(self,sprout_dict:dict):
        for key, sprout in sprout_dict.items():
            self.update_sprout_to_visualizer(key,sprout)
        return


    def update_sprout_to_visualizer(self,key:int,sprout:Sprout):
        if sprout.tip_node < 1:
            return
        elif sprout.tip_node == 1:
            xyz1 = sprout.snail_trail_dict[0]
            xyz2 = sprout.snail_trail_dict[1]
            mesh = self.create_mesh_object(xyz1,xyz2)
            mesh = self.add_point_data_to_mesh(mesh,"Radius", [6e-6,6e-6])
            mesh = self.add_cell_data_to_mesh(mesh,"Mesh Type", 2)
            mesh = self.add_cell_data_to_mesh(mesh,"Sprout_id", key)
            self.add_mesh_to_layer(2, mesh, key)
            return
        else:
            if not key in self.vtk_mesh_dict[2]:
                self.logger.log(f"Sprouts in mesh are: {self.vtk_mesh_dict[2].keys()}")
            starting_mesh = self.vtk_mesh_dict[2][key]
            xyz1 = sprout.snail_trail_dict[sprout.tip_node-1]
            xyz2 = sprout.snail_trail_dict[sprout.tip_node]
            mesh = self.create_mesh_object(xyz1,xyz2)
            mesh = self.add_point_data_to_mesh(mesh,"Radius", [6e-6,6e-6])
            mesh = self.add_cell_data_to_mesh(mesh,"Mesh Type", 2)
            mesh = self.add_cell_data_to_mesh(mesh,"Sprout_id", key)
            final_sprout_mesh = starting_mesh + mesh
            self.add_mesh_to_layer(2, final_sprout_mesh, key)
            return
            
    
    def add_growth_directions_to_visualizer(self,sprout_dict:dict,growth_vector_dict:dict):
        # self.logger.log(sprout_dict.keys())
        # self.logger.log(growth_vector_dict.keys())
        for keys in growth_vector_dict.keys():
            if "alpha" in  growth_vector_dict[keys].keys():
                tip = sprout_dict[keys].tip_node
                xyz1 = sprout_dict[keys].snail_trail_dict[tip-1]
                alpha_vector = growth_vector_dict[keys]["alpha"]
                alpha_vector = alpha_vector * 25e-6
                xyz2 = xyz1+alpha_vector
                mesh = self.create_mesh_object(xyz1,xyz2)
                mesh = self.add_point_data_to_mesh(mesh,"Radius", [2e-6,2e-6])
                mesh = self.add_cell_data_to_mesh(mesh,"Mesh Type", 3)
                self.add_mesh_to_layer(3, mesh, sprout_dict[keys].sprout_id)
            if "beta" in  growth_vector_dict[keys].keys():
                tip = sprout_dict[keys].tip_node
                xyz1 = sprout_dict[keys].snail_trail_dict[tip-1]
                beta_vector = growth_vector_dict[keys]["beta"]
                beta_vector = beta_vector * 25e-6
                xyz2 = xyz1+beta_vector
                mesh = self.create_mesh_object(xyz1,xyz2)
                mesh = self.add_point_data_to_mesh(mesh,"Radius", [2e-6,2e-6])
                mesh = self.add_cell_data_to_mesh(mesh,"Mesh Type", 4)
                self.add_mesh_to_layer(4, mesh, sprout_dict[keys].sprout_id)
            if "gamma" in  growth_vector_dict[keys].keys():
                tip = sprout_dict[keys].tip_node
                xyz1 = sprout_dict[keys].snail_trail_dict[tip-1]
                gamma_vector = growth_vector_dict[keys]["gamma"]
                gamma_vector = gamma_vector * 25e-6
                xyz2 = xyz1+gamma_vector
                mesh = self.create_mesh_object(xyz1,xyz2)
                mesh = self.add_point_data_to_mesh(mesh,"Radius", [2e-6,2e-6])
                mesh = self.add_cell_data_to_mesh(mesh,"Mesh Type", 5)
                self.add_mesh_to_layer(5, mesh, sprout_dict[keys].sprout_id)

    def apply_numbering_to_tree_mesh(self,segment_id:int,tree:Tree):
        node_1_id = tree.segment_dict[segment_id].node_1_id
        node_2_id = tree.segment_dict[segment_id].node_2_id
        
        mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Segment_id", segment_id)
        mesh = self.add_point_data_to_mesh(self.vtk_mesh_dict[0][segment_id], "Node_id", [node_1_id, node_2_id])
        
        return mesh

    def apply_pressure_info_to_tree_mesh(self,haemodynamic_solution:dict,segment_id:int,tree:Tree,node_point_ref:dict):
        # pressure and wss
        # Formula for wss: tw = D dP / (4L)
        radii = tree.segment_dict[segment_id].radius
        length = tree.length(segment_id)

        # Get node ids associated with the segment in the tree representation
        node_1_id = tree.segment_dict[segment_id].node_1_id
        node_2_id = tree.segment_dict[segment_id].node_2_id

        # Get the PID associated with the node IDs
        pid1 = node_point_ref[node_1_id]
        pid2 = node_point_ref[node_2_id]

        # Access the value of the solution at the points on the mesh
        # INTERPOLATION MAY BE NECESSARY HERE
        pressure1 = haemodynamic_solution["pvx"][pid1]
        pressure2 = haemodynamic_solution["pvx"][pid2]
        wss = abs(radii * (pressure1-pressure2) / (2*length))

        mesh = self.add_point_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Pressure", [pressure1,pressure2])
        mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"WSS", wss)
        mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Pressure Drop", abs(pressure1-pressure2))
        mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Pressure Drop by Length", abs(pressure1-pressure2)/length)

        return mesh
    
    def apply_velocity_info_to_tree_mesh(self,haemodynamic_solution:dict,segment_id:int,tree:Tree):
        # velocity and flow
        area = tree.area(segment_id)
        start_velocity = haemodynamic_solution["uvx"][segment_id+1][0]
        end_velocity = haemodynamic_solution["uvx"][segment_id+1][-1]

        start_flow = area*start_velocity
        end_flow = area*end_velocity

        mean_velocity = np.mean([start_velocity,end_velocity])
        mean_flow = np.mean([start_flow,end_flow])

        mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Velocity", mean_velocity)
        mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Flow", mean_flow)
        return mesh
    
    def apply_haematocrit_info_to_tree_mesh(self,haematocrit_solution:dict,segment_id:int,tree:Tree):
        # haematocrit and viscosisty
        H_start = haematocrit_solution["hx"][segment_id+1][0]
        H_end = haematocrit_solution["hx"][segment_id+1][-1]
        radius = tree.segment_dict[segment_id].radius * 1e6
        temperature = 37

        # VISCOSITY FORMULA
        #basic establishing coefficients
        H = np.mean([H_start,H_end])
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
        viscosity_vessel *= 1e-3

        mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Haematocrit", [H])
        mesh = self.add_cell_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Viscosity", viscosity_vessel)

        return mesh
    
    def apply_oxygen_info_to_tree_mesh(self,oxygen_solution:dict,segment_id:int,tree:Tree,node_point_ref:dict):
        # oxygen
        # Get the Inlet Values:
        inlet_ids = tree.get_node_ids_inlet()
        num_of_inlets = 0
        total_inlet_O2 = 0
        for inlet_id in inlet_ids:
            inlet_pid = node_point_ref[inlet_id]
            O2_value = oxygen_solution["ovx"][inlet_pid]
            num_of_inlets += 1
            total_inlet_O2 += O2_value
        
        mean_inlet_O2 = total_inlet_O2/num_of_inlets

        # Get node ids associated with the segment in the tree representation
        node_1_id = tree.segment_dict[segment_id].node_1_id
        node_2_id = tree.segment_dict[segment_id].node_2_id

        # Get the PID associated with the node IDs
        pid1 = node_point_ref[node_1_id]
        pid2 = node_point_ref[node_2_id]

        # Access the value of the solution at the points on the mesh
        # INTERPOLATION MAY BE NECESSARY HERE
        oxygen1 = oxygen_solution["ovx"][pid1]
        oxygen2 = oxygen_solution["ovx"][pid2]

        mesh = self.add_point_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Oxygen", [oxygen1,oxygen2])
        mesh = self.add_point_data_to_mesh(self.vtk_mesh_dict[0][segment_id],"Oxygen Proportion", [oxygen1/mean_inlet_O2,oxygen2/mean_inlet_O2])

        return mesh
    
    def update_tree_radii(self,tree:Tree):
        for segment in tree.segment_dict.keys():
            segment_id = tree.segment_dict[segment].segment_id
            mesh = self.vtk_mesh_dict[0][segment_id]
            mesh = self.add_cell_data_to_mesh(mesh,"Radius", tree.segment_dict[segment_id].radius)

        return
    
    def remove_pruned_segments(self,pruned_seg_ids:list):
        for segment_id in pruned_seg_ids:
            self.vtk_mesh_dict[0].pop(segment_id)
        
        return

    

    


    

