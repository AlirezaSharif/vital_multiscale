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


   

  


def vtk_to_custom_tree(vtk_filename, inlet_coord, radius_array_name="radius", debug=False):
    """
    Parses your specific VTP file and constructs the vascular tree.
    """
    
    # 1. Load the VTK file
    print(f"Loading {vtk_filename}...")
    raw_mesh = pv.read(vtk_filename)
    
    # 2. CLEAN THE MESH (CRITICAL STEP)
    # This merges duplicate points at junctions so they are detected as bifurcations.
    # Given your file scale (e-06), we need a very small tolerance.
    mesh = raw_mesh.clean(tolerance=1e-7)
    
    # Check for radius array
    if radius_array_name not in mesh.point_data:
        raise ValueError(f"Array '{radius_array_name}' not found. Available: {mesh.point_data.keys()}")
    
    radii = mesh.point_data[radius_array_name]
    points = mesh.points
    
    # 3. Identify the Inlet ID
    # Finds the closest node to your input coordinate
    inlet_id = mesh.find_closest_point(inlet_coord)
    
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
    breakpoint()
    # --- VISUAL DEBUGGER ---
    if debug:
        print("Opening debug window... Check if Green Spheres appear at junctions.")
        p = pv.Plotter()
        p.add_mesh(mesh, color="white", opacity=0.3, line_width=2, label="Vessels")
        
        # Categorize points
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
    for i in range(mesh.n_points):
        coord = points[i]
        degree = node_degrees.get(i, 0)
        
        if i == inlet_id:
            node_obj = my_tree.add_inlet(coord, i)
        elif degree == 1:
            node_obj = my_tree.add_outlet(coord, i)
        else:
            # Degrees 2 and 3+ (Bifurcations) handled here
            node_obj = my_tree.add_node(coord, i)
            
        vtk_to_lib_map[i] = node_obj

    # 7. Create Segments
    segment_counter = 0
    for u, v in G.edges():
        node_u = vtk_to_lib_map[u]
        node_v = vtk_to_lib_map[v]
        
        # Use average radius
        seg_radius = (radii[u] + radii[v]) / 2.0
        
        my_tree.add_segment(node_u, node_v, segment_counter, radius=seg_radius)
        segment_counter += 1

    # 8. Finalize
    my_tree.populate_junctions()
    
    print(f"Success! Tree created with {mesh.n_points} nodes and {segment_counter} segments.")
    return my_tree

# --- EXECUTION ---
# Usage with your file:
# inlet_location = [0.1e-4, 1.25e-4, 2.5e-4] # Update this to your actual inlet coordinates
# tree = vtk_to_custom_tree("ex1_simple3D.vtp", inlet_location)



# Initialize tree structure and apply a standard maximum segment length refinement
# my_tree = make_random_tree(num_points=10, cube_size=1e-4)
my_tree = vtk_to_custom_tree("/hpc/ash252/CCO/example_1/ex1_simple3D.vtp", [0, 0.00005, 0.00005])

