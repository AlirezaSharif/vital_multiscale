from classes.GeometryClasses import Node, Tree, TissueHandler, Visualizer, GraphRepresentation
from classes.ConfigClass import Config
from classes.MatrixHandlerClasses import MatrixHandler, MatrixSolver
from classes.GetFEMClasses import GetFEMHandler1D, GetFEMHandler3D
from classes.GrowthClasses import GrowthHandler
import sys
import csv
import numpy as np # type: ignore
import matplotlib.pyplot as plt
import os
from os.path import join as os_path_join


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
    config.test_name = f"ratio_{ratio}"

    output_strings_single = [f"./outputs/VEGF_{i}_GEN" for i in range(1,7)]

    data_dict_single = {}
    for i,output_single in enumerate(output_strings_single):

        config.config_access["RUN_PARAMETERS"]["output_path"] = output_single
        _, _, _, output_path, test_name = config.parse_run() # type: ignore
        filepath = os_path_join(output_path,test_name)
        case = config.growth_case # type: ignore
        file_path = "./"+filepath+"/statistic_results/"+f"case_{case}/"
        csv_filename = file_path+"percieved_vegf_values.csv"
        config.logger.log(f"Unpacking {csv_filename}")
        with open(csv_filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Read the header
            header = next(reader)  # ["X Coordinate", "Y Coordinate", "Z Coordinate", "Distance from Inlet", "Distance from Outlet", "VEGF Values"]
            # Read the rows
            data = [row for row in reader]

        # Ensure `i` key exists before populating
        data_dict_single[i] = {col_name: [] for col_name in header}

        # Populate dictionary with column data
        for j, col_name in enumerate(header):
            data_dict_single[i][col_name] = [float(row[j]) for row in data]

    output_strings_double = [f"./outputs/VEGF_{i}_DOUBLE" for i in range(1,7)]

    data_dict_double = {}
    for i,output_double in enumerate(output_strings_double):
        
        config.config_access["RUN_PARAMETERS"]["output_path"] = output_double
        _, _, _, output_path, test_name = config.parse_run() # type: ignore
        filepath = os_path_join(output_path,test_name)
        case = config.growth_case # type: ignore
        file_path = "./"+filepath+"/statistic_results/"+f"case_{case}/"
        csv_filename = file_path+"percieved_vegf_values.csv"
        config.logger.log(f"Unpacking {csv_filename}")
        with open(csv_filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Read the header
            header = next(reader)  # ["X Coordinate", "Y Coordinate", "Z Coordinate", "Distance from Inlet", "Distance from Outlet", "VEGF Values"]
            # Read the rows
            data = [row for row in reader]

        # Ensure `i` key exists before populating
        data_dict_double[i] = {col_name: [] for col_name in header}

        # Populate dictionary with column data
        for j, col_name in enumerate(header):
            data_dict_double[i][col_name] = [float(row[j]) for row in data]

    # Create an output directory if it doesn't exist
    output_dir = "./diagrams/thresholds"
    os.makedirs(output_dir, exist_ok=True)

    comparison_columns = ["X Coordinate", "Y Coordinate", "Z Coordinate", "Distance from Inlet", "Distance from Outlet"]

    # Create a figure for each comparison column
    for col in comparison_columns:
        # === First plot for data_dict_single ===
        fig, ax = plt.subplots(figsize=(16, 12))

        # Iterate over datasets and plot each on the same figure
        for i, (dataset_name, data) in enumerate(data_dict_single.items()):
            vegf_values = data["VEGF Values"]  # Get VEGF Values for this dataset
            if col == "Distance from Inlet":
                col_values = np.array(data[col])
                min_val = col_values.min()
                max_val = col_values.max()
                # Avoid division by zero
                if max_val > min_val:
                    standardized_col = (col_values - min_val) / (max_val - min_val)
                else:
                    standardized_col = col_values  # or standardized_col = np.zeros_like(col_values)
                x_values = standardized_col
            else:
                x_values = data[col]

            ax.scatter(x_values, vegf_values, s=0.5, alpha=0.5, label=f"Dataset {i+1}")

        ax.set_xlabel(col)
        ax.set_ylabel("VEGF Values")
        ax.set_title(f"{col} vs VEGF for Bifurcation Counts")
        ax.set_ylim(0, 4.5)  # Set y-axis limits
        ax.legend()  # Add legend to differentiate datasets

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(output_dir, f"scatter_plot_{col.replace(' ', '_')}_single.png")
        plt.savefig(save_path, dpi=300)  # Save as high-resolution PNG
        plt.close(fig)  # Close figure to free memory

        config.logger.log(f"Saved plot: {save_path}")  # Print confirmation

         # === Second plot for data_dict_double ===
        fig, ax = plt.subplots(figsize=(16, 12))  # Create a new figure for the second dataset
        
        # Iterate over datasets and plot each on the same figure
        for i, (dataset_name, data) in enumerate(data_dict_double.items()):
            vegf_values = data["VEGF Values"]  # Get VEGF Values for this dataset
            if col == "Distance from Inlet":
                col_values = np.array(data[col])
                min_val = col_values.min()
                max_val = col_values.max()
                # Avoid division by zero
                if max_val > min_val:
                    standardized_col = (col_values - min_val) / (max_val - min_val)
                else:
                    standardized_col = col_values  # or standardized_col = np.zeros_like(col_values)
                x_values = standardized_col
            else:
                x_values = data[col]

            ax.scatter(x_values, vegf_values, s=0.5, alpha=0.5, label=f"Dataset {i+1}")

        ax.set_xlabel(col)
        ax.set_ylabel("VEGF Values")
        ax.set_title(f"{col} vs VEGF for Bifurcation Counts")
        ax.set_ylim(0, 4.5)  # Set y-axis limits
        ax.legend()  # Add legend to differentiate datasets

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(output_dir, f"scatter_plot_{col.replace(' ', '_')}_double.png")
        plt.savefig(save_path, dpi=300)  # Save as high-resolution PNG
        plt.close(fig)  # Close figure to free memory

        config.logger.log(f"Saved plot: {save_path}")  # Print confirmation

    
    def compute_statistics(values):
        return {
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "lower_quartile": np.percentile(values, 25),
            "upper_quartile": np.percentile(values, 75),
            "std_dev": np.std(values, ddof=1),
            "variance": np.var(values, ddof=1),
            "skewness": float(np.mean(((values - np.mean(values)) / np.std(values, ddof=1))**3)),
            "kurtosis": float(np.mean(((values - np.mean(values)) / np.std(values, ddof=1))**4) - 3)
        }

    def save_statistics_to_csv(dict, output_file):
        fieldnames = ["generation", "min", "max", "mean", "median", "lower_quartile", "upper_quartile", "std_dev", "variance", "skewness", "kurtosis"]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, (dataset_name, data) in enumerate(dict.items()):
                values = data["VEGF Values"]
                stats = compute_statistics(values)
                stats["generation"] = i + 1
                writer.writerow(stats)
        config.logger.log(f"Saved table: {output_file}")  # Print confirmation


    save_path = os.path.join(output_dir, f"table_data_single.csv")
    save_statistics_to_csv(data_dict_single,save_path)
    save_path = os.path.join(output_dir, f"table_data_double.csv")
    save_statistics_to_csv(data_dict_double,save_path)


if __name__ == "__main__":
    main()       