# goods-counter
count goods in images

## How to run

```shell
Location of project:/home/cad/git/goods-counter
Location of dataset:/home/cad/dataset
Location of saved cnn model:/home/cad/git/goods-counter/model

#train a cnn model
#2 seconds per epoch on a GTX 1080 GPU.
python train-goods-th.py --positive "path to positive training set" --negative "path to negative training set"

#apply cnn model and sliding window to a image
python sliding_window.py --image "path to image"

Need Install

Keras：
$ sudo pip install keras

Opencv：
$ sudo apt-get install libopencv-dev python-opencv

h5py(required if you use model saving/loading functions)：
$ sudo pip install cython
$ sudo apt-get install libhdf5-dev
$ sudo pip install h5py

```

## Train

Neural Network Architechture

## Test

Run on data set

## Result

Result for each version list here.
