## StyleGAN - for Style Generation in fashion

Original paper can be found [here](https://arxiv.org/pdf/1812.04948.pdf), the original source code [here](https://github.com/NVlabs/stylegan), and a demonstration of a few properties in [this video](https://www.youtube.com/watch?v=kSLJriaOumA&feature=youtu.be), and an [alternate implementation](https://colab.research.google.com/drive/1wsxc0lZzu1wQv6Wy3iJrA1wAt3B27SKQ#scrollTo=C1ME1QLCtme5) on colab

This code is exactly the same as the source code with the training configuration altered to suit the need of Style Generation in fashion, making it a simpler interface to train on a custom dataset.  

Folders fakes, generated, snapshots and script generate_random_samples.py are additional and were additionally written for  additional convenience.

## Training

The code has been customized to accept a custom dataset all of uniform square size with the size being a power of 2 (1024, 512, 256...) upto 1024.
*The code configuration can be further customized by uncommenting and commenting certain configurations in ./training/training_loop.py*

The code accepts only .tfrecord files which the whole training data needs to be converted to using dataset_tool.py 

    >> python dataset_tool.py create_from_images datasets/smalls/ ../fashion/
    >> # Where datasets/smalls is the destination of .tfrecord files and ../fashion/ is the source dataset

After converting the dataset, change the script /training/training_loop.py to accept a network snapshot from ./snapshots/ 
./snapshots/ currently has 2 snapshots, one of trained faces (11155) and one of trained fashion images (11845), either of which could be used as initialization for training

After making these changes-- 

    >> python train.py

--will begin training

The model while training will constantly save network snapshots and a grid of fakes in ./snapshots/ and ./fakes/ respectively

## Generation

Once a model is satisfactorily trained, the training can be stopped and a list of generated images from randomly sampled (gaussian) latents can be found in ./generated/ on running 

    >> python generate_random_samples.py

The script can also be edited for the number of generated samples and the resolution of the generated samples

Almost every script is written in a specific GPU configuration and will show an error if training is attempted otherwise (on CPU)

