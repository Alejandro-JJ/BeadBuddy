# BeadBuddy
## Installation and usage
In order to use BeadBuddy, you need to :
* download the content of this repository to your local machine
```
cd folder/to/clone-ito/
git clone git://github.com/Alejandro-JJ/BeadBuddy.git
```
* create the necessary conda environment
```
conda env create --file env.yml
conda activate EnvBB
```
* launch the User Interface
```
python BeadBuddy.py
```


## Graphical User Interface of BeadBuddy
BeadBuddy offers an intuitive Graphical User Interface (GUI) designed in python for the analysis of fluorescent force sensors inside biological tissue. It offers a combination of:

* Powerful segmentation based on the GPU-accelerated capabilities of py-clesperanto

* Fast spherical harmonics expansion based on pyshtools

* Pre-computed analytical solutions to the elastic problem as published in Krüger et al., Biophysical Journal (2024)

The main function is the PyQt5 environment BeadBuddy.py, which can be called from the console in the root directory 
```
python BeadBuddy.py
```
After choosing one of their available Graphics Cards, the user can load a .tiff file, segment it and analyze the found force sensors. 

## Analytical solutions to the elastic problem
Following the derivation by Krüger et al., a user can generate its own analytical solutions for the stress distribution on a deformed sphere up to an arbitrary SH order. The solutions can be generated using the functions provided in DerivationStressTensor.py, specifically the function
```
GenerateSolution(lmax, savepath)
```

## Post-process of the data




