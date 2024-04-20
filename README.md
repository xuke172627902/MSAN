# MSAN
models and codes for Multi-scale strip-shaped convolution attention network for lightweight image super-resolution (MSAN)

The code is constructed based on BasicSR. Before any testing or reproducing, make sure the installation and the datasets preparation are done correctly.

To keep the workspace clean and simple, only test.py, train.py and your_arch.py are needed here and then you are good to go.

environment：
Python >= 3.8.0
Pyotch >= 1.8.1
torchvision >=0.16.1
basicsr = 1.4.2

dataset:
  train_Data:
  DIV2K(800 images for training and 100 images for validation)
  Flicker2K(2650 images)

  test_data:
  Set5, Set14, BSDS100, Urban100, Manga109
All datasets could be found in https://paperswithcode.com/datasets.

More preparation for training datasets: 
See https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/data for more details

training and testing
For training:
you can run the testing demo with
>>> CUDA_VISIBLE_DEVICES=0 python code/train.py -opt options/train/MSAN_X2.yml
For testing:
you can run the testing demo with
>>> CUDA_VISIBLE_DEVICES=0 python code/test.py -opt options/test/MSAN_X2.yml

