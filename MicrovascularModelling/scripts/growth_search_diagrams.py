from classes.GeometryClasses import Node, Tree, TissueHandler, Visualizer, GraphRepresentation
from classes.ConfigClass import Config
from classes.MatrixHandlerClasses import MatrixHandler, MatrixSolver
from classes.GetFEMClasses import GetFEMHandler1D, GetFEMHandler3D
from classes.GrowthClasses import GrowthHandler
import math
import sys
import csv
import ast
import json
import scipy
from pathlib import Path
from copy import deepcopy
import numpy as np  # type: ignore
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from os.path import join as os_path_join
from scipy.special import kl_div 
from scipy.stats import wasserstein_distance 
import scipy.stats as stats
from scipy.stats import describe
from scipy.spatial.distance import jensenshannon

def main():
    config_file = "single_growth"
    growth_case = 5
    age = 0
    ratio = 1.4

    print("Starting Experiment")

    config_string = "./config/Cases/" + config_file + ".json"
    config = Config.load_config(config_string)
    config.set_age_to_load(age)
    config.set_growth_case(growth_case)
    config.test_name = f"ratio_{ratio}"

    
    # Define maximum generation number (adjust as needed)
    max_gen = 9
    # Create an output directory if it doesn't exist
    output_dir = "./diagrams/GROWTH_SEARCH"
    os.makedirs(output_dir, exist_ok=True)

    # # Build a dictionary keyed by generation.
    # # Each key holds a list of dataset dictionaries from the different output strategys.
    # data_dict = {gen: [] for gen in range(max_gen + 1)}

    # for strategy_index, output in enumerate(output_strings):
    #     config.config_access["RUN_PARAMETERS"]["output_path"] = output
    #     _, _, _, output_path, test_name = config.parse_run()  # type: ignore
    #     filepath = os_path_join(output_path, test_name)
    #     case = config.growth_case  # type: ignore
    #     file_path = "./" + filepath + "/statistic_results/" + f"case_{case}/"
        
    #     for gen in range(max_gen + 1):
    #         csv_filename = os.path.join(file_path, f"percieved_vegf_values_{gen}.csv")
    #         config.logger.log(f"Unpacking {csv_filename}")
    #         try:
    #             with open(csv_filename, newline='') as csvfile:
    #                 reader = csv.reader(csvfile)
    #                 # Read the header, e.g. ["X Coordinate", "Y Coordinate", "Z Coordinate", "Distance from Inlet", "Distance from Outlet", "VEGF Values"]
    #                 header = next(reader)
    #                 data = [row for row in reader]
    #         except FileNotFoundError:
    #             config.logger.log(f"File not found: {csv_filename}")
    #             continue

    #         dataset = {col_name: [] for col_name in header}
    #         for j, col_name in enumerate(header):
    #             dataset[col_name] = [float(row[j]) for row in data]
    #         # Record generation and strategy index in the dataset
    #         dataset["generation"] = gen
    #         dataset["strategy"] = strategy_index
    #         data_dict[gen].append(dataset)

   

    # comparison_columns = ["X Coordinate", "Y Coordinate", "Z Coordinate", "Distance from Inlet", "Distance from Outlet"]

    # # ------------------------------
    # # Plot 1: For each generation, plot all strategies.
    # # ------------------------------
    # for gen, datasets in data_dict.items():
    #     for col in comparison_columns:
    #         fig, ax = plt.subplots(figsize=(16, 12))
    #         for dataset in datasets:
    #             vegf_values = dataset["VEGF Values"]
    #             ax.scatter(dataset[col], vegf_values, s=0.5, alpha=0.5,
    #                        label=f"Strategy {dataset['strategy']+1}")
    #         ax.set_xlabel(col)
    #         ax.set_ylabel("VEGF Values")
    #         ax.set_title(f"{col} vs VEGF, Generation {gen}")
    #         ax.set_ylim(0, 4.5)
    #         ax.legend()
    #         plt.tight_layout()

    #         # Save the generation-based plot
    #         save_path = os.path.join(output_dir, f"generation_{gen}_{col.replace(' ', '_')}.png")
    #         plt.savefig(save_path, dpi=300)
    #         plt.close(fig)
    #         config.logger.log(f"Saved plot: {save_path}")

    # # ------------------------------
    # # Plot 2: For each strategy (output strategy), plot all generations.
    # # ------------------------------
    # # First, flatten the data for easier grouping by strategy
    # flat_datasets = []
    # for gen, datasets in data_dict.items():
    #     flat_datasets.extend(datasets)

    # # Group datasets by strategy
    # strategy_data = {}
    # for dataset in flat_datasets:
    #     strategy = dataset["strategy"]
    #     if strategy not in strategy_data:
    #         strategy_data[strategy] = []
    #     strategy_data[strategy].append(dataset)

    # # Now create a plot for each strategy and each comparison column
    # for strategy, datasets in strategy_data.items():
    #     for col in comparison_columns:
    #         fig, ax = plt.subplots(figsize=(16, 12))
    #         for dataset in sorted(datasets, key=lambda d: d["generation"]):
    #             vegf_values = dataset["VEGF Values"]
    #             ax.scatter(dataset[col], vegf_values, s=0.5, alpha=0.5,
    #                        label=f"Generation {dataset['generation']}")
    #         ax.set_xlabel(col)
    #         ax.set_ylabel("VEGF Values")
    #         ax.set_title(f"{col} vs VEGF, Strategy {strategy+1} across Generations")
    #         ax.set_ylim(0, 4.5)
    #         ax.legend()
    #         plt.tight_layout()

    #         # Save the strategy-based plot
    #         save_path = os.path.join(output_dir, f"strategy_{strategy+1}_{col.replace(' ', '_')}.png")
    #         plt.savefig(save_path, dpi=300)
    #         plt.close(fig)
    #         config.logger.log(f"Saved plot: {save_path}")

    # # ------------------------------
    # # Compute and save statistics for each dataset
    # # ------------------------------
    # def compute_statistics(values):
    #     return {
    #         "min": np.min(values),
    #         "max": np.max(values),
    #         "mean": np.mean(values),
    #         "median": np.median(values),
    #         "lower_quartile": np.percentile(values, 25),
    #         "upper_quartile": np.percentile(values, 75),
    #         "std_dev": np.std(values, ddof=1),
    #         "variance": np.var(values, ddof=1),
    #         "skewness": float(np.mean(((values - np.mean(values)) / np.std(values, ddof=1)) ** 3)),
    #         "kurtosis": float(np.mean(((values - np.mean(values)) / np.std(values, ddof=1)) ** 4) - 3)
    #     }

    # def save_statistics_to_csv(datasets, output_file):
    #     fieldnames = ["generation", "strategy", "min", "max", "mean", "median", "lower_quartile",
    #                   "upper_quartile", "std_dev", "variance", "skewness", "kurtosis"]
    #     with open(output_file, 'w', newline='') as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for dataset in datasets:
    #             values = dataset["VEGF Values"]
    #             stats = compute_statistics(np.array(values))
    #             stats["generation"] = dataset["generation"]
    #             stats["strategy"] = dataset["strategy"] + 1
    #             writer.writerow(stats)
    #     config.logger.log(f"Saved table: {output_file}")

    # stats_save_path = os.path.join(output_dir, "table_data_sprout_strat.csv")
    # save_statistics_to_csv(flat_datasets, stats_save_path)

    string = f"./outputs/GROWTH_SEARCH"
    config.config_access["RUN_PARAMETERS"]["output_path"] = string

    parent_folder = Path("./outputs/GROWTH_SEARCH/ratio_1.4/statistic_results/")
    folders = sorted(
    [item for item in parent_folder.iterdir() if item.is_dir() and item.name.startswith("case_")],
    key=lambda f: float(f.name.split("_")[1]))
    
    # If you only want the names, not the full Path objects:
    case_strings = [folder.name for folder in folders]
    config.logger.log(case_strings)
    
    gen_data_by_strat = {}
    tissue_data = {}
    subdomains = []
    subdomain_to_oxygen_values = defaultdict(list)

    for strategy_index, string in enumerate(case_strings):
        vasc_data_by_gen = {}
        # _, _, _, output_path, test_name = config.parse_run()  # type: ignore
        # filepath = os_path_join(output_path, test_name)
        case = config.growth_case  # type: ignore
        filepath = os_path_join(parent_folder, string)

        tissue_data[strategy_index] = {}
        for gen in range(max_gen + 1):
            vasc_filename = os.path.join(filepath, f"vascular_statistics_{gen}.json")
            try:
                with open(vasc_filename, "r") as jsonfile:
                    reader = json.load(jsonfile)
                    vasc_data_by_gen.update({gen:reader})
            except FileNotFoundError:
                config.logger.log(f"File not found: {vasc_filename}")
                continue
            
            # tissue_filename = os.path.join(filepath, f"tissue_data_uniform_{gen}.csv")
            # try:
            #     with open(tissue_filename, "r") as csvfile:
            #         reader = csv.reader(csvfile)
            #         header = next(reader)
            #         for row in reader:
            #             subdomain = int(row[0])
            #             if subdomain not in subdomains:
            #                 subdomains.append(subdomain)
            #             oxygen_str = row[2]
                        
            #             # Convert string like '[-1.50532292e-09]' to float using ast.literal_eval
            #             oxygen_val = ast.literal_eval(oxygen_str)[0]
                        
            #             subdomain_to_oxygen_values[subdomain].append(oxygen_val)
            #         subdomain_to_mean_oxygen = {subdomain: np.mean(values) for subdomain, values in subdomain_to_oxygen_values.items()}
            #         tissue_data[strategy_index].update({gen:subdomain_to_mean_oxygen})
            # except FileNotFoundError:
            #     config.logger.log(f"File not found: {tissue_filename}")
            #     continue
        
        gen_data_by_strat.update({strategy_index:deepcopy(vasc_data_by_gen)})

    # def get_grid_shape(n):
    #     """Return (rows, cols) for at least n subplots in a grid."""
    #     cols = math.ceil(math.sqrt(n))
    #     rows = math.ceil(n / cols)
    #     return rows, cols

    # def save_vasc_to_csv(data_by_strat, output_file):
    #     columns = ["strategy","generation"]
    #     attributes = ["num_vessels","num_bifurcations","inlet_p","incoming_o2","outgoing_o2","global_oxygen_mean","global_VEGF_mean",\
    #                   "Pressure Drop", "Oxygen Discharged from Vessels", "Mean vs Discharged", "Upstream Force", "O2 eff vs Force"]
    #     columns.extend(attributes)
    #     with open(output_file, 'w', newline='') as f:
    #         writer = csv.DictWriter(f, fieldnames=columns)
    #         writer.writeheader()
    #         for strategy_i,gen_dataset in data_by_strat.items():
    #             for gen, data in gen_dataset.items():
    #                 stats = [strategy_i*20+5,gen,data["num_vessels"],data["num_bifurcations"],data["inlet_pressure"],\
    #                         data["incoming_oxygen"],data["outgoing_oxygen"],data["global_oxygen_mean"],data["global_VEGF_mean"],\
    #                         data["inlet_pressure"]-data["outlet_pressure"],data["incoming_oxygen"]-data["outgoing_oxygen"],\
    #                         data["global_oxygen_mean"] / (data["incoming_oxygen"]-data["outgoing_oxygen"]),\
    #                         data["inlet_pressure"]*np.pi*data["radii"][0]**2,\
    #                         (data["global_oxygen_mean"] / (data["incoming_oxygen"]-data["outgoing_oxygen"])) / (data["inlet_pressure"]*np.pi*data["radii"][0]**2)]
    #                 for subdomain in subdomains:
    #                     stats.append(data[subdomain])
    #                 writer.writerow(dict(zip(columns, stats)))
    #     config.logger.log(f"Saved table: {output_file}")
        
    # o2_save_path = os.path.join(output_dir, "vascular_data_table.csv")
    # save_vasc_to_csv(gen_data_by_strat, o2_save_path)

    def plot_attributes_over_generations(data_by_strat, output_dir):
        # Attributes to plot
        attributes = [
            "num_vessels", "num_bifurcations", "inlet_pressure",
            "incoming_oxygen", "outgoing_oxygen",
            "global_oxygen_mean", "global_VEGF_mean"
        ]

        # Get a colormap with enough colors
        num_strategies = len(data_by_strat)
        cmap = plt.get_cmap('viridis')  # Or 'plasma', 'cividis', 'tab20', etc.
        colors = cmap(np.linspace(0, 1, num_strategies))

        # One plot per attribute
        for attribute in attributes:
            plt.figure(figsize=(8, 6))

            for (strategy_i, (strat_key, gen_dataset)) in enumerate(data_by_strat.items()):
                generations = sorted(gen_dataset.keys())
                values = [gen_dataset[gen][attribute] for gen in generations]
                gen_x = [gen+1 for gen in generations]
                plt.plot(gen_x, values, label=f"x_nom {(strat_key*0.2)}", marker='o', color=colors[strategy_i])

            # plt.title(f"{attribute.replace('_', ' ').title()} Over Generations")
            plt.xlabel("Generation", fontsize=14)
            if attribute == "num_vessels":
                plt.ylabel("Number of Vessels", fontsize=14)
            elif attribute == "global_oxygen_mean":
                plt.ylabel("Mean Tissue Oxygen (mLO\u2082/cm\u00b2)", fontsize=14)
                 # Custom formatter: x * 1e6 with 'E-6' suffix
                from matplotlib.ticker import FuncFormatter
                formatter = FuncFormatter(lambda x, _: f"{x * 1e6:.0f}E-6")
                plt.gca().yaxis.set_major_formatter(formatter)
            else:
                plt.ylabel(attribute.replace('_', ' ').title())
            plt.xlim(1,10)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            file_name = f"{attribute}_plot.png"
            save_path = os.path.join(output_dir, file_name)
            plt.savefig(save_path)
            config.logger.log(f"Saved Graph: {save_path}")
        # # One plot per attribute
        # for attribute in attributes:
        #     plt.figure(figsize=(8, 6))

        #     for (strategy_i, (strat_key, gen_dataset)) in enumerate(data_by_strat.items()):
        #         generations = sorted(gen_dataset.keys())
        #         values = [gen_dataset[gen][attribute] for gen in generations]
        #         plt.plot(
        #             generations, values, label=f"x_nom {(strat_key*0.2)}",
        #             marker='o', color=colors[strategy_i]
        #         )

        #     plt.title(f"{attribute.replace('_', ' ').title()} over Generations")
        #     plt.xlabel("Generation")
        #     plt.ylabel(attribute.replace('_', ' ').title())
        #     plt.legend()
        #     plt.grid(True)
        #     plt.tight_layout()
        #     file_name = f"{attribute}_plot.png"
        #     save_path = os.path.join(output_dir, file_name)
        #     plt.savefig(save_path)
        #     config.logger.log(f"Saved Graph: {save_path}")
        #     plt.close()
            
    plot_attributes_over_generations(gen_data_by_strat,output_dir)

    def plot_attributes_surface(data_by_strat, output_dir):
        attributes = [
            "num_vessels", "num_bifurcations", "inlet_pressure",
            "incoming_oxygen", "outgoing_oxygen",
            "global_oxygen_mean", "global_VEGF_mean"
        ]

        # Sort strategy keys and compute ratios
        strategy_keys   = sorted(data_by_strat.keys())
        strategy_ratios = np.array([(key * 0.2) for key in strategy_keys])

        # Collect the union of all generation‐keys
        all_gens = set()
        for inner in data_by_strat.values():
            all_gens.update(inner.keys())
        generations = sorted(all_gens)
        generation_array = np.array(generations)+1

        # Now build meshgrid over every strategy × every generation
        Strat, Gen = np.meshgrid(strategy_ratios, generation_array)

        for attribute in attributes:
           # create a Z full of NaNs, same shape as your meshgrid
            Z = np.full_like(Strat, np.nan, dtype=float)

            # fill only the (strategy, generation) pairs that exist
            for si, strat_key in enumerate(strategy_keys):
                inner = data_by_strat[strat_key]
                for gi, gen in enumerate(generations):
                    if gen in inner and attribute in inner[gen]:
                        Z[gi, si] = inner[gen][attribute]
                    # else: leave Z[gi,si] as NaN

            # Create 3D plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            surf = ax.plot_surface(Strat, Gen, Z, cmap='viridis', edgecolor='k', alpha=0.85)

            # ax.set_title(f"{attribute.replace('_', ' ').title()} Surface")
            ax.set_xlabel("x_nom")
            ax.set_ylabel("Generation")
            if attribute == "num_vessels":
                ax.set_zlabel("Number of Vessels")
            elif attribute == "global_oxygen_mean":
                ax.set_zlabel("Mean Tissue Oxygen (mLO\u2082/cm\u00b2)", labelpad=12)
                 # Custom formatter: x * 1e6 with 'E-6' suffix
                from matplotlib.ticker import FuncFormatter
                formatter = FuncFormatter(lambda x, _: f"{x * 1e6:.0f}E-6")
                plt.gca().zaxis.set_major_formatter(formatter)
            else:
                ax.set_zlabel(attribute.replace('_', ' ').title())
            ax.set_xlim(0, strategy_ratios.max())
            ax.set_ylim(generation_array.min(), generation_array.max())
            num_ticks = 7  # or any number you prefer
            xticks = np.linspace(0, strategy_ratios.max(), num_ticks)
            ax.set_xticks(xticks)
            ax.invert_xaxis()  # Reverse strategy axis (right to left)
            

            # Adjust and manually place the colorbar
            fig.subplots_adjust(left=0.02, right=0.88, bottom=0.15, top=0.95)
            cbar_ax = fig.add_axes([0.89, 0.2, 0.02, 0.6])
            cbar = fig.colorbar(surf, cax=cbar_ax)

            if attribute == "global_oxygen_mean":
                # Apply the same formatter to colorbar ticks
                cbar.ax.yaxis.set_major_formatter(formatter)

            # plt.tight_layout()
            file_name = f"{attribute}_surface_plot.png"
            save_path = os.path.join(output_dir, file_name)
            plt.savefig(save_path)
            config.logger.log(f"Saved 3D Surface: {save_path}")
            plt.close()

    plot_attributes_surface(gen_data_by_strat,output_dir)


    # def plot_list_attribute_violin_subplots(data_by_strat, list_attributes, output_dir):
    #     num_strategies = len(data_by_strat)
    #     all_generations = sorted({gen for strat in data_by_strat.values() for gen in strat})
    #     num_generations = len(all_generations)

    #     for attribute in list_attributes:
    #         # ======== Per-Generation View ========
    #         gen_rows, gen_cols = get_grid_shape(num_generations)
    #         fig_gen, axes_gen = plt.subplots(nrows=gen_rows, ncols=gen_cols, figsize=(gen_cols * 4, gen_rows * 4), sharey=True)
    #         axes_gen = axes_gen.flatten()

    #         for i, gen in enumerate(all_generations):
    #             ax = axes_gen[i]
    #             data_to_plot = []
    #             labels = []
    #             for strategy_i in sorted(data_by_strat.keys()):
    #                 gen_data = data_by_strat[strategy_i].get(gen)
    #                 if gen_data and attribute in gen_data and isinstance(gen_data[attribute], list):
    #                     abs_vals = [abs(v) for v in gen_data[attribute]]
    #                     data_to_plot.append(abs_vals)
    #                     labels.append(f"S{strategy_i+1}")
    #             if data_to_plot:
    #                 ax.violinplot(data_to_plot, showmeans=True)
    #                 ax.set_title(f"Generation {gen}")
    #                 ax.set_xticks(range(1, len(labels) + 1))
    #                 ax.set_xticklabels(labels, rotation=45)
    #                 ax.grid(True)

    #         fig_gen.suptitle(f"{attribute.replace('_', ' ').title()} by Generation", fontsize=16)
    #         fig_gen.tight_layout(rect=[0, 0.03, 1, 0.95])
    #         save_path = os.path.join(output_dir, f"{attribute}_by_generation.png")
    #         fig_gen.savefig(save_path)
    #         plt.close(fig_gen)
    #         config.logger.log(f"Saved Graph: {save_path}")

    #         # ======== Per-Strategy View ========
    #         strat_rows, strat_cols = get_grid_shape(num_strategies)
    #         fig_strat, axes_strat = plt.subplots(nrows=strat_rows, ncols=strat_cols, figsize=(strat_cols * 4, strat_rows * 4), sharey=True)
    #         axes_strat = axes_strat.flatten()

    #         for i, strategy_i in enumerate(sorted(data_by_strat.keys())):
    #             ax = axes_strat[i]
    #             data_to_plot = []
    #             labels = []
    #             for gen in all_generations:
    #                 gen_data = data_by_strat[strategy_i].get(gen)
    #                 if gen_data and attribute in gen_data and isinstance(gen_data[attribute], list):
    #                     abs_vals = [abs(v) for v in gen_data[attribute]]
    #                     data_to_plot.append(abs_vals)
    #                     labels.append(f"G{gen}")
    #             if data_to_plot:
    #                 ax.violinplot(data_to_plot, showmeans=True)
    #                 ax.set_title(f"Strategy {strategy_i+1}")
    #                 ax.set_xticks(range(1, len(labels) + 1))
    #                 ax.set_xticklabels(labels, rotation=45)
    #                 ax.grid(True)

    #         fig_strat.suptitle(f"{attribute.replace('_', ' ').title()} by Strategy", fontsize=16)
    #         fig_strat.tight_layout(rect=[0, 0.03, 1, 0.95])
    #         save_path = os.path.join(output_dir, f"{attribute}_by_strategy.png")
    #         fig_strat.savefig(save_path)
    #         plt.close(fig_strat)
    #         config.logger.log(f"Saved Graph: {save_path}")

    # def draw_blocky_violin(ax, data, center, bin_edges, max_bin_count, width=0.8, color='skyblue'):
    #     hist, _ = np.histogram(data, bins=bin_edges)
    #     # Normalize histogram to global max so all violins scale equally
    #     hist_scaled = hist / max_bin_count * (width / 2)
    #     bin_height = bin_edges[1] - bin_edges[0]
    #     for h, y in zip(hist_scaled, bin_edges[:-1]):
    #         ax.barh(y, h, height=bin_height * 0.9, left=center - h, color=color, edgecolor='black')
    #         ax.barh(y, h, height=bin_height * 0.9, left=center, color=color, edgecolor='black')


    # def plot_blocky_violin_subplots(data_by_strat, list_attributes, output_dir, bins=20):
    #     os.makedirs(output_dir, exist_ok=True)

    #     num_strategies = len(data_by_strat)
    #     all_generations = sorted({gen for strat in data_by_strat.values() for gen in strat})
    #     num_generations = len(all_generations)

    #     for attribute in list_attributes:
    #         # Gather all absolute values for global binning
    #         all_abs_values = []
    #         for strat_data in data_by_strat.values():
    #             for gen_data in strat_data.values():
    #                 values = gen_data.get(attribute)
    #                 if values and isinstance(values, list):
    #                     all_abs_values.extend(abs(v) for v in values)

    #         # Compute shared histogram bin edges and max count
    #         bin_edges = np.histogram_bin_edges(all_abs_values, bins=20)

    #         # Compute the maximum bin count for consistent width scaling
    #         global_max_bin_count = 0
    #         for strat_data in data_by_strat.values():
    #             for gen_data in strat_data.values():
    #                 values = gen_data.get(attribute)
    #                 if values and isinstance(values, list):
    #                     hist, _ = np.histogram([abs(v) for v in values], bins=bin_edges)
    #                     global_max_bin_count = max(global_max_bin_count, hist.max())
    #         # === Per-Generation View ===
    #         gen_rows, gen_cols = get_grid_shape(num_generations)
    #         fig_gen, axes_gen = plt.subplots(nrows=gen_rows, ncols=gen_cols, figsize=(gen_cols * 4, gen_rows * 4), sharey=True)
    #         axes_gen = axes_gen.flatten()

    #         for i, gen in enumerate(all_generations):
    #             ax = axes_gen[i]
    #             centers = []
    #             for j, strategy_i in enumerate(sorted(data_by_strat.keys())):
    #                 gen_data = data_by_strat[strategy_i].get(gen)
    #                 if gen_data and attribute in gen_data and isinstance(gen_data[attribute], list):
    #                     abs_vals = [abs(v) for v in gen_data[attribute]]
    #                     draw_blocky_violin(ax,abs_vals,center=j + 1,bin_edges=bin_edges,max_bin_count=global_max_bin_count)
    #                     centers.append(j + 1)
    #             ax.set_xticks(centers)
    #             ax.set_xticklabels([f"S{i+1}" for i in range(len(centers))])
    #             ax.set_title(f"Generation {gen}")
    #             ax.set_ylabel(attribute.replace('_', ' ').title())
    #             ax.grid(True)

    #         fig_gen.suptitle(f"{attribute.replace('_', ' ').title()} by Generation", fontsize=16)
    #         fig_gen.tight_layout(rect=[0, 0.03, 1, 0.95])
    #         save_path = os.path.join(output_dir, f"{attribute}_by_generation_blocky.png")
    #         fig_gen.savefig(save_path)
    #         plt.close(fig_gen)
    #         config.logger.log(f"Saved Graph: {save_path}")

    #         # === Per-Strategy View ===
    #         strat_rows, strat_cols = get_grid_shape(num_strategies)
    #         fig_strat, axes_strat = plt.subplots(nrows=strat_rows, ncols=strat_cols, figsize=(strat_cols * 4, strat_rows * 4), sharey=True)
    #         axes_strat = axes_strat.flatten()

    #         for i, strategy_i in enumerate(sorted(data_by_strat.keys())):
    #             ax = axes_strat[i]
    #             centers = []
    #             for j, gen in enumerate(all_generations):
    #                 gen_data = data_by_strat[strategy_i].get(gen)
    #                 if gen_data and attribute in gen_data and isinstance(gen_data[attribute], list):
    #                     abs_vals = [abs(v) for v in gen_data[attribute]]
    #                     draw_blocky_violin(ax,abs_vals,center=j + 1,bin_edges=bin_edges,max_bin_count=global_max_bin_count)
    #                     centers.append(j + 1)
    #             labelled_gens = [gen for j, gen in enumerate(all_generations) if data_by_strat[strategy_i].get(gen) and attribute in data_by_strat[strategy_i][gen]]
    #             ax.set_xticks(centers)
    #             ax.set_xticklabels([f"G{gen}" for gen in labelled_gens], rotation=45)
    #             ax.set_title(f"Strategy {strategy_i+1}")
    #             ax.set_ylabel(attribute.replace('_', ' ').title())
    #             ax.grid(True)

    #         fig_strat.suptitle(f"{attribute.replace('_', ' ').title()} by Strategy", fontsize=16)
    #         fig_strat.tight_layout(rect=[0, 0.03, 1, 0.95])
    #         save_path = os.path.join(output_dir, f"{attribute}_by_strategy_blocky.png")
    #         fig_strat.savefig(save_path)
    #         plt.close(fig_strat)
    #         config.logger.log(f"Saved Graph: {save_path}")

    # list_attributes = ["radii","lengths","oxygens","flows"]
    # plot_list_attribute_violin_subplots(gen_data_by_strat,list_attributes,output_dir)
    # plot_blocky_violin_subplots(gen_data_by_strat,list_attributes,output_dir)


    def read_collated_network_stats(csv_filename):
        # Create a nested defaultdict structure
        collated_network_stats = defaultdict(lambda: defaultdict(dict))

        with open(csv_filename, mode="r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                # config.logger.log(f"Row: {row}")
                # Convert keys to int
                subdomain = int(row["Subdomain"])
                network_id = int(row["Network_ID"])
                vessel_id = int(row["Vessel_ID"])

                # Convert values to float
                vessel_radius = float(row["Vessel_Radii"])
                vessel_length = float(row["Vessel_Length"])

                # Assign to nested dictionary
                collated_network_stats[subdomain][network_id][vessel_id] = (vessel_radius, vessel_length)

        # config.logger.log(f"Read File: {csv_filename}")
        return collated_network_stats
    
    def read_in_vivo_csv(csv_filename):
        in_vivo_data = defaultdict(lambda: defaultdict(dict))
        with open(csv_filename, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            last_stack = 0
            vessel_id = -1
            for row in reader:
                # Convert keys to int
                stack = int(row["Stack"])
                network_id = int(row["sub_domain"])

                # Convert values to float
                vessel_radius = float(row["Radius"])*1e-6
                vessel_length = float(row["Length"])*1e-6
                    
                if stack == last_stack:
                    vessel_id += 1
                else:
                    vessel_id = 0
                    last_stack = deepcopy(stack)

                # Assign to nested dictionary
                in_vivo_data[stack][network_id][vessel_id] = (vessel_radius, vessel_length)
        # config.logger.log(f"Read File: {csv_filename}")
                
                
        return in_vivo_data

    dist_types = ["gauss","uniform"]
    val_types = ["vessel_radii","vessel_length"]
    
    in_vivo_filepath = Path(f"./Cortical_ordered.csv")
    in_vivo_data = read_in_vivo_csv(in_vivo_filepath)

    
    for index, val_type in enumerate(val_types):
        results = []
        for sample_num1, subdomain in enumerate(in_vivo_data):  # sorting to ensure order if needed
            set_A = []
            set_B = []
            for sample_num2, subdomain in enumerate(in_vivo_data):  # sorting to ensure order if needed
                subdomain_vals = []
                for network in in_vivo_data[subdomain].values():
                    for values in network.values():
                        subdomain_vals.append(values[index])
                if sample_num1 == sample_num2:
                    set_B.extend(deepcopy(subdomain_vals))
                else:
                    set_A.extend(deepcopy(subdomain_vals))


            # t-test
            t_stat, t_p = scipy.stats.ttest_ind(set_A, set_B,
                                    equal_var=True, alternative='two-sided')

            # Mann-Whitney U
            u_stat, u_p = scipy.stats.mannwhitneyu(set_A, set_B,
                                    alternative='two-sided')
            NaNb = len(set_A)*len(set_B)

            # Wasserstein set_A
            wd = wasserstein_distance(set_A, set_B)

            # Jensen-Shannon distance
            edges = np.histogram_bin_edges(np.concatenate((set_A, set_B)), bins='fd')
            p_hist, _ = np.histogram(set_A,  bins=edges, density=True)
            q_hist, _ = np.histogram(set_B,  bins=edges, density=True)
            p = p_hist / p_hist.sum()
            q = q_hist / q_hist.sum()
            jsd = jensenshannon(p, q, base=2)


            results.append({
                'left_out': sample_num1,
                'n_X': len(set_A), 'n_Y': len(set_B),
                't_stat': abs(t_stat), 't_p': t_p,
                'u_stat': u_stat, 'NaNb': NaNb, "u_normalized":u_stat/NaNb, 'u_p': u_p,
                'wasserstein': wd, 'jensen_shannon': jsd
            })

            
            Xi = set_B
            Yi = set_A

            plt.figure()
            plt.boxplot([Xi, Yi], labels=[f'Sample {sample_num1}', 'Pooled'])
            plt.ylabel('R values')
            plt.title(f'Boxplot: Sample {sample_num1} vs Pooled Rest')
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"drop_one_boxes_sample_{sample_num1}_{val_type}.png")
            plt.savefig(save_path)
            plt.close()
            config.logger.log(f"Saved table: {save_path}")
            
        # Save to CSV (no index column)
        output_path= os.path.join(output_dir, f"in_vivo_drop_one_{val_type}.csv")
        # Get column names from keys in the first dictionary
        fieldnames = results[0].keys()

        with open(output_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  # Write column names
            writer.writerows(results)  # Write all rows

        config.logger.log(f"Saved table: {output_path}")
        
            
            

    for dist_type in dist_types:
        for index, val_type in enumerate(val_types):
            subdomain_info_per_gen_per_case = {}
            for strategy_index, string in enumerate(case_strings):
                subdomain_info_per_gen = {}
                for gen in range(10):
                    filepath = os_path_join(parent_folder, string)
                    subdomain_filename = os.path.join(filepath, f"collated_vessel_data_{dist_type}_{gen}.csv")
                    subdomain_info = read_collated_network_stats(subdomain_filename)
                    subdomain_info_per_gen[gen] = deepcopy(subdomain_info)
                subdomain_info_per_gen_per_case[string] = deepcopy(subdomain_info_per_gen)

            

            set_B = []
            for subdomain in in_vivo_data:  # sorting to ensure order if needed
                subdomain_vals = []
                for network in in_vivo_data[subdomain].values():
                    for values in network.values():
                        subdomain_vals.append(values[index])
                set_B.extend(deepcopy(subdomain_vals))


            results = []
            
            for case, subdomain_info_per_gen in subdomain_info_per_gen_per_case.items():
                subdomain_info = subdomain_info_per_gen[9]
                set_A = []
                for subdomain in subdomain_info:  # sorting to ensure order if needed
                    subdomain_vals = []
                    for network in subdomain_info[subdomain].values():
                        for values in network.values():
                            subdomain_vals.append(values[index])
                    set_A.extend(deepcopy(subdomain_vals))
                

                # t-test
                t_stat, t_p = scipy.stats.ttest_ind(set_B, set_A,
                                        equal_var=True, alternative='two-sided')

                # Mann-Whitney U
                u_stat, u_p = scipy.stats.mannwhitneyu(set_B, set_A,
                                        alternative='two-sided')
                NaNb = len(set_A)*len(set_B)

                # Wasserstein distance
                wd = wasserstein_distance(set_B, set_A)

                # Jensen-Shannon distance
                edges = np.histogram_bin_edges(np.concatenate((set_B, set_A)), bins='fd')
                p_hist, _ = np.histogram(set_B,  bins=edges, density=True)
                q_hist, _ = np.histogram(set_A,  bins=edges, density=True)
                p = p_hist / p_hist.sum()
                q = q_hist / q_hist.sum()
                jsd = jensenshannon(p, q, base=2)

                results.append({
                    'case': case,
                    'n_X': len(set_A), 'n_Y': len(set_B),
                    't_stat': abs(t_stat), 't_p': t_p,
                    'u_stat': u_stat, 'NaNb': NaNb, "u_normalized":u_stat/NaNb, 'u_p': u_p,
                    'wasserstein': wd, 'jensen_shannon': jsd
                })
            
            # Save to CSV (no index column)
            output_path= os.path.join(output_dir, f"in_vivo_compare_stats_{dist_type}_{val_type}.csv")
            # Get column names from keys in the first dictionary
            fieldnames = results[0].keys()

            with open(output_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()  # Write column names
                writer.writerows(results)  # Write all rows

            config.logger.log(f"Saved table: {output_path}")

    for index, val_type in enumerate(val_types):
        # 1) Read the CSV back into a list of dicts
        results = []
        config.logger.log(f"VAL TYPE = {val_type}")
        output_path= os.path.join(output_dir, f"in_vivo_drop_one_{val_type}.csv")
        with open(output_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # convert numeric fields
                parsed = {
                    'left_out':    int(row['left_out']),
                    'n_X':         int(row['n_X']),
                    'n_Y':         int(row['n_Y']),
                    't_stat':      float(row['t_stat']),
                    't_p':         float(row['t_p']),
                    'U':           float(row['u_stat']),
                    'U_p':         float(row['u_p']),
                    'Wasserstein': float(row['wasserstein']),
                    'JSD':         float(row['jensen_shannon'])
                }
                results.append(parsed)

        # 2) Print a tabular summary
        config.logger.log(f"{'LOO':>3} {'n_X':>4} {'n_Y':>4} {'t_stat':>8} {'t_p':>6} {'U':>8} {'U_p':>6} {'W':>8} {'JSD':>6}")
        for r in results:
            config.logger.log(f"{r['left_out']:3d} {r['n_X']:4d} {r['n_Y']:4d} "
                f"{r['t_stat']:8.3f} {r['t_p']:6.3f} {r['U']:8.1f} {r['U_p']:6.3f} "
                f"{r['Wasserstein']:8.3f} {r['JSD']:6.3f}")

        left_out = [r['left_out'] for r in results]

        for metric, label in [('Wasserstein','Wasserstein Distance'),
                            ('JSD','Jensen–Shannon Distance')]:
            vals = [r[metric] for r in results]
            plt.figure()
            plt.bar(left_out, vals)
            plt.xlabel('Left-out sample index')
            plt.ylabel(label)
            plt.title(f'{label} by Leave-One-Out')
            plt.xticks(left_out)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"drop_one_visualized_{val_type}.png")
            plt.savefig(save_path)
            plt.close()
            config.logger.log(f"Saved table: {save_path}")
            
        for metric in ['Wasserstein', 'JSD']:
            vals = [r[metric] for r in results]
            desc = describe(vals)
            mean = desc.mean
            sd   = np.sqrt(desc.variance)  # note: describe returns the unbiased sample variance
            config.logger.log(f"{metric}: mean = {mean:.3f}, SD = {sd:.3f}")





if __name__ == "__main__":
    main()
