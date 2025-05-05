# Image Generation with Conditional Variational Autoencoder (C-VAE) and Conditional Generative Adversarial Network (C-GAN)

## Overview

In this project, I aim to create a set of images of Kuzushiji Japanese characters via the Kuzushiji-49 dataset using 2 graphical models which are Conditional-Variational Autoencoder(C-VAE), and Conditional-Generative Adversarial Network(C-GAN) to build and train them. The Japanese character design will have 49 units. 

## Datasets

The datasets I will use are the Kuzushiji-49 dataset which includes:
- Depiction of the training
- The Training List
- The test drawings
- The Test Characters

## Conditional-Variational Autoencoder (C-VAE)

The conditional-variation auto-encoder also called C-VAE is an extension of the conditional graphical model of the variable auto-encoder designed to produce conditional data in a label-like explicit output An example of such a C-VAE is in training time can be generated following some given conditions.

The structure of the C-VAE consists of Encoders and Decoders.
- The encoder sends the image input and its mono-hot-encoded label to where the day presses it.
- The decoder retrieves the hidden vector from the hidden space and produces reconstructed images.

The training details for this model are Loss Functions and Training Durations. The training durations are 20 epochs which are used for consistency where more epochs can help in making the model become consistent in its predictions by seeing training data many times, learning depth which allows the model to learn deeper, and generalisation which helps the model to generalise better to unseen data. The Loss Function has the reconstruction loss where the difference between the original and rebuilt images are measured, and it also has the KL divergence where the difference between the learned latent distribution and the prior distribution.


## Conditional-Generative Adversarial Network (C-GAN)

Conditional-generative adversary network also known as C-GAN is a conditional graphical model used in deep learning that is also an autoencoder of generative adversary network designed to generate conditional data on additional input such as class labels The figure looks like what is true.

The C-GAN system includes a generator and a discriminator.
- The generator takes the noise vector and the one-hot encoded label and combines them and passes them through the fully connected layers which then produces an image that corresponds to the layers.
- The discriminator takes the image and the single hot encoded characters, connects them and goes through all the connecting layers, making it possible for the images to be real or fake.

The training parameters for this model are the loss function and training time. The training period was 50 epochs and as before, used for stability, depth of learning, and generalization to help better generalize model Loss function is BCE or BinaryCross Entropy loss used to measure the difference between prediction and actual labels and performance between discriminator and generator in general Also called Adversarial loss for analysis.


## Comparison between C-VAE and C-GAN

The qualitative comparison between C-VAE and C-GAN shows that C-VAE produces images with smooth transitions and they show images that are relatively coarse and
lack fine details while C-GAN produces images accurate and comprehensive types, and effectively captures the complexity of data distribution.

The quantitative comparison between C-VAE and C-GAN is that C-VAE measures the reproducibility of the sample better and the lower variation loss indicates better
performance. While C-GAN measures the success of the generator in deceiving the discriminator and a lower enemy loss indicates better image quality.

The metrics presented for C-GAN ranged from 1.49 to 2.56 for adversary generator loss, and from 0.22 to 0.42 for adversary discriminator loss while the C-VAE metrics of reconstruction losses ranged from 161.07 to 200.15.


## Challenges, Interesting Observations and Improvements

The challenges taken so far are the time taken to draw and they were done for CVAE and C-GAN and that is why I had to reduce the number of times just to save some time. Results for both with limited periods were successful, but at the expense of obtaining blurred images and near-realistic images for C-VAE and C-GAN respectively.

An interesting fact about doing this project is that I was able to observe that the number of epochs I added changed according to the results, especially when
working in C-GAN. Originally, I had 10 epochs, which give me an output of 49 same images as I believe it was incomplete. Changing epochs to 50 gave me a clearer and almost realistic looking output with 49 different images. I theorise, if I had more than 50 epochs, the output would more accurately realistic.

In my opinion, improvements for the 2 models should be implementing more compound layers of architectures to improve the quality of images and improving the interpretability while maintaining a high-quality image.
