'''
dcgan final:

[-1,1] normalisation, tanh generator activation function, -1 rotation padding
choose training dataset / object type
create training gif, training metric plots (loss and accuracy)
output trained generator model to create fake dataset (or checkpoint of trained model)

Steps:

1. Assign program parameters (e.g. epochs, object_class)
2. Load + sigma clip + resize + normalise + augment dataset (from saved .npy file)
3. Define gen, disc, training loop
4. Train gan
5. Save trained model
6. Create gif and loss plot
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
print(tf.__version__)
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
import os
from tensorflow.keras import layers
import time
import sys
import imageio
from astropy import stats
from PIL import Image

print('\nSTART')

## Variables

# As long as this is different to previous ones, won't overwrite any saved data
object_class = 'good_fr1' # used to label save location and files for gif, images, generator model
obj_images_loc = '../../data/good/good_fr1.npy' # location to find numpy array with object images
image_save_loc = './temp-images/'+object_class # location to save training images
if not os.path.isdir(image_save_loc): os.mkdir(image_save_loc) # make training images folder if not already one
batch_size = 256
epochs = 500

time0 = time.time()

## Data preparation

# Load raw image data
train_images = np.load(obj_images_loc, allow_pickle=True)
num_images, size, _ = train_images.shape

# Sigma clip images
sigma = 3
for i in range(num_images):
    _, median, std = stats.sigma_clipped_stats(train_images[i], sigma=sigma)
    train_images[i][train_images[i] < median+sigma*std] = 0

# Resize images to 56x56
train_images_resized = np.empty((num_images, 56, 56))
for i in range(num_images):
    image = Image.fromarray(train_images[i]) # convert from numpy array to Image object
    train_images_resized[i] = image.resize((56,56))
train_images = train_images_resized

# Normalise images to [-1,1]
for i in range(num_images):
    train_images[i] = 2*(train_images[i]-np.min(train_images[i]))/np.ptp(train_images[i])-1

# Augment images with rotations and flips
def augment_data(data,size):
    rotations = size//len(data) # rotations per image
    angles = np.linspace(0, 360, rotations)
    act_size = rotations*len(data)
    xpix, ypix = data.shape[1:]
    training_set = np.empty((act_size, xpix, ypix))
    for i in range(len(data)):
        for j in range(len(angles)):
            if j % 2 == 0: training_set[i*len(angles)+j,:,:] = rotate(np.fliplr(data[i,:,:]), angles[j], reshape=False, cval=-1)
            else: training_set[i*len(angles)+j,:,:] = rotate(data[i,:,:], angles[j], reshape=False, cval=-1)
    return training_set

train_images = augment_data(train_images, 60000)

# Respahe data array to 4D, and type float32
train_images = train_images.reshape(train_images.shape[0], 56, 56, 1).astype('float32')

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(batch_size)

print('Images augmented from {} to {}'.format(num_images, BUFFER_SIZE))
time1 = time.time()
print('Time for data preparation: {:02d}:{:02d}'.format(*divmod(round(time1-time0), 60)))

## Create the generator and discriminator models

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 512)))
    assert model.output_shape == (None, 7, 7, 512) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 56, 56, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[56, 56, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

## Define the loss functions and optimizers for both models

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

## Define the training loop

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled". -> improves performance
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    start_time = time.time()
    epoch = 0
    save_images(generator, epoch, seed, image_save_loc)

    # Lists to store loss function values for each epoch
    train_g_loss_results, train_d_loss_results = [], []

    while epoch < epochs:
        epoch += 1
        epoch_start = time.time()

        # Functions to compute the mean loss for all batches in an epoch
        epoch_g_loss_avg = tf.keras.metrics.Mean()
        epoch_d_loss_avg = tf.keras.metrics.Mean()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_g_loss_avg(gen_loss)
            epoch_d_loss_avg(disc_loss)

        train_g_loss_results.append(epoch_g_loss_avg.result())
        train_d_loss_results.append(epoch_d_loss_avg.result())

        # Produce images for the GIF as we go
        save_images(generator, epoch, seed, image_save_loc)

    # Generate after the final epoch
    save_images(generator, epochs, seed, image_save_loc)
    print('Total time for {}->{} epochs was {:02d}:{:02d}'.format(0, epochs, *divmod(round(time.time()-start_time), 60)))
    print('Average time per epoch: {:02d}:{:02d}'.format(*divmod(round((time.time()-start_time)/epochs), 60)))

    return train_g_loss_results, train_d_loss_results

def save_images(model, epoch, test_input, save_loc):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(16,4))
    for i in range(predictions.shape[0]):
        plt.subplot(2, 8, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='inferno')
        if i==0:
            plt.text(0, -5, 'epoch: {}'.format(epoch), ha='left', va='bottom', fontsize=12)
        plt.axis('off')
    plt.savefig(save_loc+'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

## Setup training parameters

generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim]) # random seed for training gif

time2 = time.time()
print('Time to load variables: {:02d}:{:02d}'.format(*divmod(round(time2-time1), 60)))

## Train the model

gen_loss, disc_loss = train(train_dataset, epochs)

generator.save(object_class+'_gen_model') # save the generator model

time3 = time.time()
print('Time for training: {:02d}:{:02d}'.format(*divmod(round(time3-time2), 60)))

## Generate training GIF and loss plot (training metrics)

# Create gif
anim_file = './gifs/{}.gif'.format(object_class)
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(os.path.join(image_save_loc,'image*.png'))
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

# Create mp4 from gif (moviepy module not available on BB for 3.7.4...)
#clip = mp.VideoFileClip(anim_file)
#clip.write_videofile(anim_file[:-3]+'mp4')

# Create loss plot
fig = plt.figure(figsize=(8,6))
n = len(gen_loss)
plt.plot(range(n),gen_loss, label='gen')
plt.plot(range(n),disc_loss, label='disc')
plt.legend()
plt.title(''.format(object_class))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./loss-plots/{}_loss.png'.format(object_class))
plt.close()

# Save loss arrays for future figure customization
loss = np.array([gen_loss, disc_loss])
np.save('./loss-plots/{}_loss.npy'.format(object_class), loss)

time4 = time.time()
print('Time for creating training metrics: {:02d}:{:02d}'.format(*divmod(round(time4-time3), 60)))

print('END')
