# BeadBuddy
BeadBuddy offers an intuitive Graphical User Interface (GUI) designed in python for the analysis of fluorescent force sensors inside biological tissue. It offers a combination of:

* Powerful segmentation based on the GPU-accelerated capabilities of py-clesperanto

* Fast spherical harmonics expansion based on pyshtools

* Pre-computed analytical solutions to the elastic problem as published in Krüger et al., Biophysical Journal (2024)

## Installation and usage
In order to use BeadBuddy, you need to :
* download the content of this repository to your local machine
```
cd folder/to/clone-into/
git clone git://github.com/Alejandro-JJ/BeadBuddy.git
```
* create the necessary conda environment
```
conda env create --file env.yml
conda activate EnvBB
```
* launch the User Interface inside the environment
```
python BeadBuddy.py
```
* in case you want to save all outputs from the analysis, you can also launch
```
python BeadBuddy_outputs.py
```

## Graphical User Interface of BeadBuddy
The main function is the PyQt5 environment BeadBuddy.py, which can be called from the console in the root directory of your download as described above.
After choosing one of their available Graphics Cards, the user can load a .tiff file, segment it and analyze the found force sensors, generating high quality plot of the bodies as well as 2D and 3D projections of the deformation, radius and radial stresses. 
A typical analysis comprises the different steps:
* Select one of the availables GPUs in your machine after launching calling the GUI
* Go to File... --> Open TIFF and choose the image to be analyzed
* Input the segmentation parameters and try to segment. This might takle several iterations until the proper parameters are found:
	- BackGround noise: the size (in pixel) of a Gaussian smoothing kernel to reduce noise in the image
	- Threshold: The pixel value from which the initial watershed algotihm will start detecting fluorescent signal
	- Spot S: the estimated size (in pixels) of the objects to be segmented
	- Outline S: the separation of the detected objects. The lower this value, the more fragmented the segmentation will be.
* Scrolling on the second plotting window you can check the segmentation as compared to the original data
* Once a satisfactory segmentation has been found, you can input the pixel size of the acquisition and the Spherical Harmonics order up to which the analytical solution will be used. The higher the order, the more irregularities on the surface can be resolved and the biogger the computation time. 
* The user can now click on each of the bodies detected and check the solution by clicking "Analyze BEAD". If the checkbox "External plots" is ticked, two new windows will appear showing 2D and 3D projections of the radius and the stress tensor. 
* If the whole content of the image needs to be analyzed, the user can click on "Analyze ALL", after which a folder will be generated in the same location of the image, and the results wiil be saved. Additionally to the SH Tables, the stress map and the major compression axis, the user can choose beforehand (by marking the checkboxes) to also save the labelled pictures from the segmentation and the 3D plots.

## Analytical solutions to the elastic problem
Following the derivation by Krüger et al., a user can generate its own analytical solutions for the stress distribution on a deformed sphere up to an arbitrary SH order. By default, we include the pre-computed solutions up to an order l<sub> max</sub>=10 in the folder GeneralSolutions. New solutions for higher orders can be generated using the functions provided in DerivationStressTensor.py, specifically the function
```
GenerateSolution(lmax, savepath)
```
In order for BeadBuddy to access this new solutions, its correswponding dill file must be stored in GeneralSolutions too following the naming convention. 

## Post-process of the data
**To be implemented**




