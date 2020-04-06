# Year 4 project
### "Synthetic Training Data For Morphological Classification Of RadioGalaxies With Deep Convolutional Neural Networks"
### Jacob Morrison, University of Birmingham, April 2020

**Note:** 

Most of the code in this project/repository is written as Jupyter Notebooks. When training neural network models GPUs are required for a reasonable computation time. The jupyter notebook format allows for the code to be opened in Google Colab to make use of Google's free GPU service. However when using Colab special permissions must be given to gain access to local file storage. I use the option of mounting my Google Drive to be able to access my local file storage, however this requires a login. Therefore this code cannot be run straight out of the box, and the user will need to make sure that the program has access to the required files when ran in Google Colab.

Other code, namely DCGAN, is very computationally expensive and is written with the intent of running the code on the Blue BEAR HPC provided by the University of Birmingham. Many visual code outputs are supressed in this code, with the main outputs of the code instead being saved to external files.

## Description of code files:

### 1. Webscrape FIRST source FITS images

This notebook automatically downloads all of the desired sources the FIRST radio survey cutout server. Source coordinates are retreived from source catalogs, which are loaded from saved .txt files. 

### 2. View downloaded images, and manually select final data sample


