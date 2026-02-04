# vital_multiscale
## Installation
After installing Docker on your machine, enter the following commands to create an image containing all the dependencies you need to run the tutorial:
```shell
sudo docker pull alirezasharifzadeh/vital_multiscale:latest
```
After cloning the current repository on your local machine, go to the its directory and run the following:
```shell
sudo docker run --name vital --rm -v path/to/root:/code -it alirezasharifzadeh/vital_multiscale:latest
```
## Running the program
```shell
cd CCO
./example_1
cd ..
cd MicrovascularModelling
python -m scripts.VITAL
```
