from pathlib import Path
import json
import sys
import os
import re
import logging
import shutil
from datetime import datetime

class Config:
    """
    A class used to handle the the configuration setup and conglomorate filepaths.

    ----------
    Class Attributes
    ----------
    None
    
    ----------
    Instance Attributes
    ----------
    node_list_path : string
        A string indicating the filepath to the list containing all node information

    segment_list_path : string
        A string indicating the filepath to the list containing all segment information

    junction_list_path : string
        A string indicating the filepath to the list containing all junction information 

    matrix_A_path : string
        A string indicating the filepath to the submatricies A
    
    vector_b_path : string
        A string indicating the filepath to the subvectors b

    vector_x_path : string
        A string indicating the filepath to the subvectors x

    age_to_load : int
        An integer indicating the desired generation of vascular tree to load.

    ----------
    Class Methods
    ----------  
    None

    ----------
    Instance Methods
    ----------  
    load_config(filename)
        Returns a class instance from file

    set_age_to_load(int)
        Sets age_to_load to the desired value

    parse_run()
        Returns the run parameters associated with the config file

    return_filepath(string, aged = False)
        Returns the filepath associated with the input string, if aged = True, returns a numbered filepath.
        Valid strings are: 'node', 'segment', 'junction', 'mat_A', 'vec_b', 'vec_x'

    """

    def __init__(self, config):
        self.config_access = config
        self.node_list_path = config["PATH"]["NODE_LIST_PATH"]
        self.segment_list_path = config["PATH"]["SEGMENT_LIST_PATH"]
        self.junction_list_path = config["PATH"]["JUNCTION_LIST_PATH"]
        self.matrix_A_path = config["PATH"]["MATRIX_A_PATH"]
        self.vector_b_path = config["PATH"]["VECTOR_B_PATH"]
        self.vector_x_path = config["PATH"]["VECTOR_X_PATH"]
        self.mesh_path = config["PATH"]["MESH_PATH"]
        self.age_to_load = config["AGE"]["LAST_AGE"]
        self.test_name = config["RUN_PARAMETERS"]["test_name"]

        self.setup_output_folder()
        self.lock_file = "instance_lock.txt"
        self.add_to_lock_file()

        filepath = self.return_filepath("output")
        run_name = Path(self.config_access["RUN_PARAMETERS"]["test_name"])
        log_str = "log.log"
        filepath = Path.joinpath(filepath,run_name)
        filepath = Path.joinpath(filepath,log_str)

        log_dir = os.path.dirname(filepath)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.test_case_set = False
        
        self.logger = Logger(filepath) # type: ignore

        self.initial_geo = True
        

    @classmethod
    def load_config(cls, filepath):
        path = Path(filepath)
        with open(path, "r") as f:
            settings = json.load(f)
            return cls(settings)

    def set_age_to_load(self, age:int):
        self.age_to_load = age
        return

    def set_growth_case(self, case:int):
        self.growth_case = case
        self.test_case_set = True
        path = Path(f"./config/Weights/config_{case}.json")
        with open(path, "r") as f:
            self.growth_params = json.load(f)

        self.logger.log(self.growth_params)
        
        return

    def toggle_initial_geo(self):
        self.initial_geo = not self.initial_geo

    def parse_run(self):
        if self.config_access["RUN_PARAMETERS"]["cylinder_test"] == 1:
            cylinder_test = True
        else:
            cylinder_test = False

        if self.config_access["RUN_PARAMETERS"]["predefined_mesh"] == 1:
            new_mesh = False
            mesh_path = self.config_access["RUN_PARAMETERS"]["mesh_path"] 
        else:
            new_mesh = True
            mesh_path = self.return_filepath("mesh")

        output_path = self.return_filepath("output")
        test_name = self.config_access["RUN_PARAMETERS"]["test_name"]

        return cylinder_test, new_mesh, str(mesh_path), str(output_path), test_name

    def return_filepath(self, type, aged=False):
        if aged:
            iteration_string = "_" + str(self.age_to_load)
        else:
            iteration_string = ''

        if type == "node_init":
            filepath = Path(self.node_list_path)
            if self.initial_geo == True:
                run_name = Path(self.config_access["RUN_PARAMETERS"]["initial_geometry"])
            else:
                run_name = Path(self.config_access["RUN_PARAMETERS"]["test_name"])

            node_name = f"Nodes" + iteration_string + f".json"
            file = Path(node_name)
            filepath = Path.joinpath(filepath,run_name,file)
            log_dir = os.path.dirname(filepath)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            return filepath
        elif type == "segment_init":
            filepath = Path(self.segment_list_path)
            if self.initial_geo == True:
                run_name = Path(self.config_access["RUN_PARAMETERS"]["initial_geometry"])
            else:
                run_name = Path(self.config_access["RUN_PARAMETERS"]["test_name"])
            segment_name = f"Segments" + iteration_string + f".json"
            file = Path(segment_name)
            filepath = Path.joinpath(filepath,run_name,file)
            log_dir = os.path.dirname(filepath)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            return filepath
        elif type == "junction_init":
            filepath = Path(self.junction_list_path)
            if self.initial_geo == True:
                run_name = Path(self.config_access["RUN_PARAMETERS"]["initial_geometry"])
            else:
                run_name = Path(self.config_access["RUN_PARAMETERS"]["test_name"])
            junction_name = f"Junctions" + iteration_string + f".json"
            file = Path(junction_name)
            filepath = Path.joinpath(filepath,run_name,file)
            log_dir = os.path.dirname(filepath)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            return filepath
        elif type == "node":
            output_path = self.return_filepath("output")
            test_name = Path(self.config_access["RUN_PARAMETERS"]["test_name"])
            run_name = Path.joinpath(output_path,test_name)

            folder_name = Path("statistic_results")
                
            node_name = f"Nodes" + iteration_string + f".json"
            file = Path(node_name)
            filepath = Path.joinpath(run_name,folder_name,file)
            log_dir = os.path.dirname(filepath)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            return filepath
        elif type == "segment":
            output_path = self.return_filepath("output")
            test_name = Path(self.config_access["RUN_PARAMETERS"]["test_name"])
            run_name = Path.joinpath(output_path,test_name)

            folder_name = Path("statistic_results")
                
            node_name = f"Segments" + iteration_string + f".json"
            file = Path(node_name)
            filepath = Path.joinpath(run_name,folder_name,file)
            log_dir = os.path.dirname(filepath)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            return filepath
        elif type == "junction":
            output_path = self.return_filepath("output")
            test_name = Path(self.config_access["RUN_PARAMETERS"]["test_name"])
            run_name = Path.joinpath(output_path,test_name)

            folder_name = Path("statistic_results")
                
            node_name = f"Junctions" + iteration_string + f".json"
            file = Path(node_name)
            filepath = Path.joinpath(run_name,folder_name,file)
            log_dir = os.path.dirname(filepath)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            return filepath
        elif type == "mat_A":
            filepath = self.matrix_A_path + iteration_string + ".dat"
            return Path(filepath)
        elif type == "vec_b":
            filepath = self.vector_b_path + iteration_string + ".dat"
            return Path(filepath)
        elif type == "vec_x":
            filepath = self.vector_x_path + iteration_string + ".dat"
            return Path(filepath)
        elif type == "output":
            filepath = self.config_access["RUN_PARAMETERS"]["output_path"]
            return Path(filepath)
        elif type == "mesh":
            filepath = self.mesh_path + "/" + self.config_access["RUN_PARAMETERS"]["test_name"] + ".msh"
            return Path(filepath)
        else:
            raise ValueError("Input 'type' must be one of the following strings: 'node', 'segment', 'junction', 'mat_A', 'vec_b', 'vec_x', 'output'.")

    def setup_output_folder(self):
        # Create top level output folder if not present
        filepath = self.return_filepath("output")
        run_name = str(Path(self.config_access["RUN_PARAMETERS"]["test_name"]))  # Convert Path object to string
        filepath = Path.joinpath(filepath, run_name)
        log_str = "log.log"
        log_path = Path.joinpath(filepath, log_str)
        path = os.path.dirname(log_path)
        if not os.path.exists(path):
            os.makedirs(path)
        # Create sub folders if not present
        sub_folders = ["tissue_results", "tree_results", "statistic_results", "test_metrics"]
        for sub_folder_string in sub_folders:
            sub_folder_path = Path.joinpath(filepath, sub_folder_string)
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)

        return
    
    def setup_case_folder(self):
        if self.test_case_set is False:
            raise ValueError(f"No test case has been set as of yet")
        
        cases = [self.growth_case]
        filepath = self.return_filepath("output")
        run_name = str(Path(self.config_access["RUN_PARAMETERS"]["test_name"]))  # Convert Path object to string
        filepath = Path.joinpath(filepath, run_name)
        filepath = Path.joinpath(filepath, "test_metrics")
        for case in cases:
            case_path = Path.joinpath(filepath, f"case_{case}")
            self.logger.log(case_path)
            if not os.path.exists(case_path):
                os.makedirs(case_path)
            else:
                # Delete only files meeting naming criteria
                for item in case_path.iterdir():
                    if item.is_file() and not (self.meets_naming_criteria(item.name,self.age_to_load)):  # Define your naming criteria check
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)  # Optionally, clear subdirectories
                        os.makedirs(case_path)
                
        filepath = self.return_filepath("output")
        run_name = str(Path(self.config_access["RUN_PARAMETERS"]["test_name"]))  # Convert Path object to string
        filepath = Path.joinpath(filepath, run_name)
        filepath = Path.joinpath(filepath, "tissue_results")
        for case in cases:
            case_path = Path.joinpath(filepath, f"case_{case}")
            if not os.path.exists(case_path):
                os.makedirs(case_path)
            else:
                # Delete current content of the folder
                shutil.rmtree(case_path)
                # Recreate the folder
                os.makedirs(case_path)
        
        filepath = self.return_filepath("output")
        run_name = str(Path(self.config_access["RUN_PARAMETERS"]["test_name"]))  # Convert Path object to string
        filepath = Path.joinpath(filepath, run_name)
        filepath = Path.joinpath(filepath, "statistic_results")
        for case in cases:
            case_path = Path.joinpath(filepath, f"case_{case}")
            if not os.path.exists(case_path):
                os.makedirs(case_path)
            else:
                # Delete current content of the folder
                shutil.rmtree(case_path)
                # Recreate the folder
                os.makedirs(case_path)
        return
    
    def meets_naming_criteria(self, filename, iteration_number):
        # Define the pattern to capture a trailing number in the filename
        pattern = re.compile(r"^growth_(\d+)\.vtk$")
        match = pattern.search(filename)
        
        if match:
            # Extract the trailing number from the filename
            trailing_number = int(match.group(1))
            # Check if the trailing number is greater than the specified iteration number
            # if trailing_number > iteration_number:
            #     self.logger.log(f"Removed Trailing Number: {trailing_number}")
            # else:
            #     self.logger.log(f"Kept Trailing Number: {trailing_number}")
            return trailing_number <= iteration_number
        # else:
        #     self.logger.log(f"No Match Found")
        return False    
    
    def reset_output_folder_by_index(self,list_of_subfolder_idx_to_reset:list):
        """
        Function to remove the contents of specific output folders by index
        This function takes a list of index numbers and resets those folders.
        The indexes are:
            0: tissue_results
            1: segment_results
            2: tree_results
            3: attractor_results
            4: test_metrics

        :param list_of_subfolder_idx_to_reset: A list of index numbers indicating folders to reset
        """
        filepath = self.return_filepath("output")
        run_name = str(Path(self.config_access["RUN_PARAMETERS"]["test_name"]))  # Convert Path object to string
        filepath = Path.joinpath(filepath, run_name)
        sub_folders = ["tissue_results", "segment_results", "tree_results", "attractor_results", "test_metrics"]
        for index in list_of_subfolder_idx_to_reset:
            if index < 0 or index >= len(sub_folders):
                self.logger.log(f"Index {index} is out of range when resetting output folders.")
                continue
            sub_folder_string = sub_folders[index]
            sub_folder_path = Path.joinpath(filepath, sub_folder_string)
            self.logger.log(f"Resetting ouput folder: {sub_folder_path}")
            if  os.path.exists(sub_folder_path):
                # Delete current content of the folder
                shutil.rmtree(sub_folder_path)

            # Recreate the folder
            sub_folder_path.mkdir(parents=True,exist_ok=True)

        return
    
    def add_to_lock_file(self):
        # Add a unique string to the lock file
        with open(self.lock_file, "a") as f:
            f.write(str(os.getpid()) + "\n")

    def remove_from_lock_file(self):
        # Remove the process ID from the lock file
        pid = str(os.getpid())
        with open(self.lock_file, "r+") as f:
            lines = f.readlines()
            f.seek(0)
            for line in lines:
                if line.strip() != pid:
                    f.write(line)
            f.truncate()
    

    def __str__(self):
      return f"""
        Paths:
          Node: {self.return_filepath('node')}
          Segment: {self.return_filepath('segment')}
          Junction: {self.return_filepath('junction')}
          Matrix A: {self.return_filepath('mat_A')}
          Vector b: {self.return_filepath('vec_b')}
          Vector x: {self.return_filepath('vec_x')}
        Age: {self.age_to_load}
      """  

class Logger:
    def __init__(self, log_file_path='my_log_file.log'):
        self.log_file_path = log_file_path
        self.logger = self.setup_logger()
        self.last_update_line = False  # Track if the last message was using update_line=True
        
    def setup_logger(self):
        # Create a logger with a unique name
        logger = logging.getLogger('log')
        logger.setLevel(logging.DEBUG)

        # Create a file handler if the log file doesn't exist
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w'):
                pass

        # Create a formatter
        formatter = logging.Formatter('%(message)s')  # Removed %(asctime)s - %(levelname)s

        # Create a file handler
        file_handler = logging.FileHandler(self.log_file_path, 'w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        # Create a console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)

        return logger

    def log(self, message, update_line=False):
        # Get the current system time
        current_time = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
        formatted_message = f"{current_time} - {message}"

        if update_line:
            # Use '\r' to overwrite the current line in the terminal
            sys.stdout.write(f"\r{formatted_message}")
            sys.stdout.flush()  # Make sure to flush to update immediately
            self.last_update_line = True # Mark that update_line was used
        else:
            # Log the message normally (both to file and console)
            if self.last_update_line:
                sys.stdout.write("\n")  # Move to the next line
                sys.stdout.flush()
                
            self.logger.info(formatted_message)
            self.last_update_line = False  # Reset flag

import psutil
from psutil._common import bytes2human   
import inspect
class MemoryHandler():
    def __init__(self):
        self._initialize_memory_value()
        self.loop_count = 0

    def _initialize_memory_value(self):
        self.base_used = psutil.virtual_memory().used

    def print_mem(self,string=None,human=True):
        mem_used = psutil.virtual_memory().used
        if string is None:
            string = f"Memory used"
        if not type(string) is str:
            raise ValueError(f"string must be a string not {type(string)}")
        if human:
            print(string + f": {bytes2human(mem_used-self.base_used)}")
        else:
            print(string + f": {mem_used-self.base_used}")
        return

    def full_summary(self):
        print(psutil.virtual_memory())

    def count_loop(self):
        print(f"Loop count is: {self.loop_count}")
        self.loop_count += 1


    def mem_by_lines_wrapper(self,func):
        def wrapper(*args, **kwargs):
            # Get the source code of the wrapped function
            source_lines, _ = inspect.getsourcelines(func) # type: ignore
            
            print(f"Source code of {func.__name__}:") # type: ignore
            for line in source_lines:
                print(line.strip())


            result = func(*args, **kwargs) # type: ignore
            return result
        return wrapper