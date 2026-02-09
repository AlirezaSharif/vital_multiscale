# vital_multiscale
## Installation
After installing Docker on your machine, enter the following commands to create an image containing all the dependencies you need to run the tutorial:
```shell
sudo docker pull alirezasharifzadeh/vital_multiscale:v1
```
After cloning the current repository on your local machine, run the following command:
```shell
sudo docker run --name vital --rm -v path/to/the/cloned/repo:/code -it alirezasharifzadeh/vital_multiscale:v1
```
## How does it work?
By running the following command in your Docker environment, arterial (./artery) and venous (./vein) trees inside a specified cube (ex_3D.vtk) will be generated with random seeds. "MicrovascularModelling/scripts/VITAL.py" takes these venous trees as input, connects their terminal, and runs a 1D hemodynamic simulation with a pressure-pressure boundary condition (insertion; check out "/MicrovascularModelling/config/Cases/VITAL.json" for config parameters). Then, using the hemodynamic solution, the network's equivalent resistance will be quantified (homogenisation) and printed in the terminal.   
```shell
./run.sh
```
## Exercise 1
Write a code to execute "run.sh" 100 times and save the equivalent resistances into a CSV file. 
Next, plot the distribution of resistances and determine whether we are dealing with an RVE or an SVE.

## Exercise 2
Unlike the previous exercise, use a fixed network (do not run "artery" and "vein"). Instead, introduce an additive noise term to the network's radii (MicrovascularModelling/scripts/VITAL.py). Assume a noise amplitude is 10% of the original radius.
Similar to the previous exercise, determine the statistics of the equivalent resistance. 
