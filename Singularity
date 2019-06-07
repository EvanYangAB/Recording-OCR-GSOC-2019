Bootstrap: docker
From: ubuntu:latest

%setup
  # runs on host - the path to the image is $SINGULARITY_ROOTFS

%post
  # post-setup script


  # additional packages
  RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
  apt-get install -y python3
  apt-get install -y libsm6 libxext6
  pip install selenium
  pip install moviepy
  pip install lmdb
  pip install opencv-contrib-python
  pip install cryptography

  pip install numpy
  pip install tensorflow
  pip install scipy
  pip install Pillow
  pip install cython
  pip install matplotlib
  pip install scikit-image
  pip install keras>=2.0.8
  pip install opencv-python
  pip install h5py
  pip install imgaug
  pip install IPython[all]

%runscript
  # executes with the singularity run command
  # delete this section to use existing docker ENTRYPOINT command

%test
  # test that script is a success