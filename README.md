# goods-counter
count goods in images

## How to run

```shell
#train a cnn model
python train-goods.py --positive "path to positive training set" --negative "path to negative training set"

#apply cnn model and sliding window to a image
python sliding_window.py --image "path to image"

Need Install
tensorflow:
# Ubuntu/Linux 64-bit, CPU only:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7. Requires CUDA toolkit 7.5 and CuDNN v4.
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

Keras：
$ sudo pip install keras

If you have run Keras at least once, you will find the Keras configuration file at: 
~/.keras/keras.json
If it isn't there, you can create it.
It probably looks like this:{"epsilon": 1e-07, "floatx": "float32", "backend": "theano"}
Simply change the field backend to "tensorflow", and Keras will use the new configuration next time you run any Keras code.

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
