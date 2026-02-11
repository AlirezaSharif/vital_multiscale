# vital_multiscale
## Installation
After installing Docker on your machine, enter the following commands to create an image containing all the dependencies you need to run the tutorial:
```shell
sudo docker pull alirezasharifzadeh/vital_multiscale:v1
```
Clone this repository:
```shell
git clone https://github.com/AlirezaSharif/vital_multiscale.git
```
After cloning the current repository on your local machine, run the following command:
```shell
sudo docker run --name vital --rm -v path/to/the/cloned/repo:/code -it alirezasharifzadeh/vital_multiscale:v1
```
**Danger!!!** replace `path/to/the/cloned/repo` with the path to the cloned repository!

**In the case you have problems finding the path:** after cloning the repository:
```shell
cd vital_multiscale
```
For Linux and macOS users:
```shell
sudo docker run --name vital --rm -v $(pwd):/code -it alirezasharifzadeh/vital_multiscale:v1
```
Windows users (cmd):
```shell
sudo docker run --name vital --rm -v %cd%:/code -it alirezasharifzadeh/vital_multiscale:v1
```
**Giving permission to execute the codes:**
```shell
chmod +x ./CCO/vein ./CCO/artery run.sh
```
Now to see the magic, run this command:
```shell
./run.sh
```

## What did just happen?
By running `./run.sh` command in your Docker environment, arterial (```./artery```) and venous (```./vein```) trees inside a specified cube (***ex_3D.vtk***) will be generated with random seeds.
| ```./artery``` | ```./vein``` |
| :---: | :---: |
| ![artery](https://github.com/user-attachments/assets/da9ef92e-9f4b-40af-8caa-7c7eb75685e4) | ![vein](https://github.com/user-attachments/assets/82f5be44-2f91-4044-b335-f6c4f962a9fe) |
| *Arterial network* | *Unconnected network* |

```MicrovascularModelling/scripts/VITAL.py``` takes these arterial and venous trees as input, connects their terminal, and runs a 1D hemodynamic simulation with a pressure-pressure boundary condition (insertion; check out ```/MicrovascularModelling/config/Cases/VITAL.json``` for config parameters). Then, using the hemodynamic solution, the network's equivalent hemodynamic resistance will be quantified (homogenisation) and printed in the terminal. 
| `VITAL.py` | 
| :---: |
| ![network](https://github.com/user-attachments/assets/befdae75-a40a-46db-8d10-a5af1eb23b71) |
| *Connected network* |

***Optional:*** To visualise the simulation results, use Paraview. The generated output is in ```outputs/results/ratio_1.4/test_metrics/case_165/growth_0.vtk```
## Exercise 1
1. Write or adjust existing codes to execute ```run.sh``` 100 times and save the equivalent hemodynamic resistance of each microvascular network into a CSV file.
2. Plot the distribution of the resistances and determine whether we are dealing with an RVE or an SVE.

**Key files:** ```VITAL.py``` and ```run.sh``` 

## Exercise 2
1. Introduce an additive noise term to the network's radii (`MicrovascularModelling/scripts/VITAL.py`). Assume that the noise amplitude is 10% of the original radii.
2. Determine the statistics of the equivalent hemodynamic resistances (similar to the previous exercise).

***Important note:*** Unlike the previous exercise, use a fixed network (do not run `artery` and `vein`). 
## Exercise 3
Well done! So far, you have investigated the effect of uncertainty in connectivity (***Exercise 1***) and parameters (***Exercise 2***) of microvascular networks on their equivalent hemodynamic resistance.
Now, determine the statistics of the equivalent hemodynamic resistances in the presence of both effects. 
