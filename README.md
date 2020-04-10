# Year 4 project
### "Synthetic Training Data For Morphological Classification Of RadioGalaxies With Deep Convolutional Neural Networks"
### Jacob Morrison, University of Birmingham, April 2020

**Note:** 

Most of the code in this project/repository is written as Jupyter Notebooks. When training neural network models GPUs are required for a reasonable computation time. The jupyter notebook format allows for the code to be opened in Google Colab to make use of Google's free GPU service. However when using Colab special permissions must be given to gain access to local file storage. I use the option of mounting my Google Drive to be able to access my local file storage, however this requires a login. Therefore this code cannot be run straight out of the box, and the user will need to make sure that the program has access to the required files when ran in Google Colab.

Other code, namely DCGAN, is very computationally expensive and is written with the intent of running the code on the Blue BEAR HPC provided by the University of Birmingham. Many visual code outputs are supressed in this code, with the main outputs of the code instead being saved to external files.

## Description of code files:

### 1. Webscrape FIRST source FITS images

This notebook automatically downloads all of the desired source images from the FIRST radio survey cutout server, to the desired download directory. Images are downloaded as 150x150 (4.5arcmin), and in the FITS file format. Source coordinates are retreived from source catalogs, which are loaded from saved .txt files. 

### 2. View downloaded images, and manually select final data sample

When selecting the final sample of images to use as training data, some of the images need to be removed. Images are removed if the catalog contains duplicates, the source is too big to fit in the 150x150 cutout, or the morphology of the source is not clear enough. This process was done manually. The final data samples for each classification (FRI, FRII, bent-tailed, compact) were saved as separate numpy arrays, in .npy files. These array files can be quite large due to the large number of images, it is possible to train a neural network by the reading the desired images straight from a directory when needed to save on memory, however the files were not large enough in this case for it to be necessary. Preloading all of the images before training a model also results in faster training than reading the images from a directory. 

### 3. Simple geometric simulation of synthetic radio galaxy images

Program to create simple synthetic radio galaxy images. Program is based on using the known geometric properties of radio galaxy images, i.e. central AGN peak, and two more spread out peaks representing the AGN's jets. Peaks are approximated using a sum of multiple 2D Gaussians. Images can be created to be any size. Whilst images are mostly obviously synthetic, they still contain the basic features of real AGN so could still be useful when training a model. In order to create a dataset of these synthetic images, random values are generated to use as the variables in the program (e.g. bending angle, peak brightness, peak spread, lobe asymmetry, lobe distance from central AGN, etc.). The probability distributions of these random variables is what determines the overall spread of features in the dataset. To generate our dataset most variables were choesn from a flat probability distribution between a minimum and maximum value, or a Gaussian distribution centred on sample value, with some defined spread. These probability distributions are quite simple, and a more realistic final dataset could probably be achieved by modelling the probability distributions on those observed for real data. 

### 4. Deep convolutional generative adversarial network (DCGAN)

DCGANs are another method of creating synthetic training data. They are a generative modelling technique, using two competing neural networks to improve the performance of both. The generator neural network aims to recieve noise and output an image resmebling the training data. Saving the trained generator model, and using randomly generated noise vectors allows us to create a dataset of the synthetic images. A training GIF is also created, showing the training / learning process of the GAN. 

### 5. Classification CNN

Combines all of the previous code to produce final results of the impact of synthetic training data on classification accuracy.


## Work not included in the final report:

### A. Regressive bending angle CNN classifier



### B. Variation autoencoder (VAE)



### C. Dataset size vs. classification performance



## Future work:

### i. Image size vs. classification performance

### ii. DCGAN for higher resolution image generation

### iii. CNN classifier model architecture optimisation




