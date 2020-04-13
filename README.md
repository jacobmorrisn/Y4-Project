# Year 4 project
### "Synthetic Training Data For Morphological Classification Of RadioGalaxies With Deep Convolutional Neural Networks"
### Jacob Morrison, University of Birmingham, April 2020

---

**Note:** 

Most of the code in this project/repository is written as Jupyter Notebooks. When training neural network models GPUs are required for a reasonable computation time. The jupyter notebook format allows for the code to be opened in Google Colab to make use of Google's free GPU service. However when using Colab special permissions must be given to gain access to local file storage. I use the option of mounting my Google Drive to be able to access my local file storage, however this requires a login. Therefore this code cannot be run straight out of the box, and the user will need to make sure that the program has access to the required files when ran in Google Colab.

Notebook code files are saved with their code output. This means some of the files can be quite large, but the output from running each code cell is shown so the code doesn not need to run in order to understand what the code does.

Other code, namely DCGAN, is very computationally expensive and is written with the intent of running the code on the Blue BEAR HPC provided by the University of Birmingham. Many visual code outputs are supressed in this code, with the main outputs of the code instead being saved to external files.

---

## Description of code files:

### 1. Webscrape FIRST source FITS images

**Code:** download_image_cutouts.ipynb

This notebook automatically downloads all of the desired source images from the FIRST radio survey cutout server, to the desired download directory. Images are downloaded as 150x150 (4.5arcmin), and in the FITS file format. Source coordinates are retreived from source catalogs, which are loaded from saved .txt files. 

### 2. View downloaded images, and manually select final data sample

**Code:** sample_selection.ipynb

When selecting the final sample of images to use as training data, some of the images need to be removed. Images are removed if the catalog contains duplicates, the source is too big to fit in the 150x150 cutout, or the morphology of the source is not clear enough. This process was done manually. The final data samples for each classification (FRI, FRII, bent-tailed, compact) were saved as separate numpy arrays, in .npy files. These array files can be quite large due to the large number of images, it is possible to train a neural network by the reading the desired images straight from a directory when needed to save on memory, however the files were not large enough in this case for it to be necessary. Preloading all of the images before training a model also results in faster training than reading the images from a directory. 

### 3. Simple geometric simulation of synthetic radio galaxy images

**Code:** geometric_agn_simulation.ipynb

Program to create simple synthetic radio galaxy images. Program is based on using the known geometric properties of radio galaxy images, i.e. central AGN peak, and two more spread out peaks representing the AGN's jets. Peaks are approximated using a sum of multiple 2D Gaussians. Images can be created to be any size. Whilst images are mostly obviously synthetic, they still contain the basic features of real AGN so could still be useful when training a model. In order to create a dataset of these synthetic images, random values are generated to use as the variables in the program (e.g. bending angle, peak brightness, peak spread, lobe asymmetry, lobe distance from central AGN, etc.). The probability distributions of these random variables is what determines the overall spread of features in the dataset. To generate our dataset most variables were choesn from a flat probability distribution between a minimum and maximum value, or a Gaussian distribution centred on sample value, with some defined spread. These probability distributions are quite simple, and a more realistic final dataset could probably be achieved by modelling the probability distributions on those observed for real data. 

### 4. Deep convolutional generative adversarial network (DCGAN)

**Code:**  
dcgan.ipynb  
dcgan_bluebear.py

DCGANs are another method of creating synthetic training data. They are a generative modelling technique, using two competing neural networks to improve the performance of both. The generator neural network aims to recieve noise and output an image resmebling the training data. Saving the trained generator model, and using randomly generated noise vectors allows us to create a dataset of the synthetic images. A training GIF is also created, showing the training / learning process of the GAN. 

### 5. Classification CNN

**Code:** classification_cnn_with_syn.ipynb

Combines all of the previous code to produce final results of the impact of synthetic training data on classification accuracy. First the real, geometric simulation and GAN data is loaded. The real data is sigma-clipped, resized, split into training and test sets, then augmented using rotations and flips. For each training dataset (only real, real and geo, real and GAN) a CNN model is trained, and the accuracy and loss of the model for the training and validation datasets at each epoch is plotted. For each training dataset a confusion matrix is also produced showing the model's predictions vs the real image labels. The CNN model used is based on the one presented in Alhassan (2017). 

---

## Work not included in the final report:

Here is some other work that I did which was not included in the final report. The main piece of work being the regressive CNN. This was not included since it wouldn't have fit in the word count of the report, and to give the report more focus on the single goal of radio galaxy classification. Other work was not included since it was somewhat unfinished / not relevant.

### A. Regressive CNN bending angle predictor

**Code:** regressive_cnn.ipynb

Here I wanted to explore a regressive CNN instead of a classification CNN. The model architecure is the same except for the final layer which only has one output, rather than four. The loss function is also different for regression problems. I used the regressive CNN to predict the bending angle of the radio galaxy. This was possible thanks to the Garon (2019) catalog which contains over 3000 radio galaxies, each labelled by their bending angle. These sources were downloaded from the FIRST cutout server to create the dataset. 

The CNN model was trained using an augmented dataset of flipped and rotated images. Metrics showing the mean squared error and mean absolute error of the model at each epoch were plotted showing the model improvement during training. Several other plots to visualise the model performance were also shown. 

Synthetic data was then also used to try to improve the model's performance. In this case the data needed to be labelled by the bending angle, GANs are not able to do this, however the geometric simulation is. Using a combined dataset of real and synthetic images was able to reduce the validation error of the model, however the training error was increased.

### B. Semester 1 work

I have created a folder containing the work achieved in semester 1. Most of the work here is a bit redundant now, since this work was improved upon during semester 2.

---

## Future work:

### i. Image size vs. classification performance

In order to use the GAN synthetic images in the CNN classifier the real images needed to be resized from 150x150 to 56x56 to match the size of the GAN images. This is far from ideal since a lot of spatial information of the sources is lost when downsampling. The final accuracy of the model was also only able to achieve ~91% accuracies, whereas accuracies ~98% were reported in Alhassan (2017) - this is most likely due to the downsampling. Want to explore the classification performance achieved when resizing the training images to different sizes.

### ii. DCGAN for higher resolution image generation

Generate higher resolution GAN images. This is mainly achieved through changing the generator and discriminator network architectures to accomate larger images.

### iii. CNN classifier model architecture optimisation

The classification CNN architecture used in this study was based of the achitecture proposed in Alhassan (2017), since it was proven to achieve high classification accuracies for this problem. However, here is a more rigorous approach is used to find the optimal model archiecture for this problem.

Start with a simple CNN, and continue to add convolution and pooling layers until the model starts to overfit. Overfitting can always be reduced at the end using regularization techniques. Use classic networks as inspiration when building the model, i.e. follow the same conv-pool-conv trends, as well as trends in channel and filter sizes. Then create a sample of several different CNN architecures. Train each and select the top performing one or two archiectures to hyper-parameter optimisation on (top-performing = best validation accuracy). 

Hyperparameter optimisation can be implemented using the hyperas library (simple hyperopt wrapper for keras). We need to define some functions to describe which parameters we want to optimise and the range of these parameters to test, i.e. the dropout rates, number of channels, strides, kernal size, learning rate. The best performing runs of the model can then be used to find the optimal values for each hyperparameter, and to define the final CNN architecture. 

### iv. Variation autoencoder (VAE)

VAEs are another generative modelling technique, similar to GANs, which can be used to create synthetic data that resembles the underlying distribution of the training dataset.

### v. Dataset size vs. classification performance

Explore how the number of images in a training dataset affects the classification performance of the network. First explore how the number of augmentations affects the model accuracy, in order to find out how much a dataset should be augmented by in order to optimise performance. Then keep the number of augmentations constant but vary the number of real images that are in the training set to be augmented, this can be used as a measure of how much real labelled data is needed to successfully train a classification model.


