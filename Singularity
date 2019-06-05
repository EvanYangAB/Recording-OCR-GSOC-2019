Bootstrap: docker
From: tensorflow/tensorflow:2.0.0a0-gpu-py3

%environment
  # use bash as default shell
  SHELL=/bin/bash
  export SHELL

%setup
  # runs on host - the path to the image is $SINGULARITY_ROOTFS

%post
  # post-setup script

  # load environment variables
  . /environment

  # use bash as default shell
  echo 'SHELL=/bin/bash' >> /environment

  # make environment file executable
  chmod +x /environment

  # default mount paths
  mkdir /scratch /data 

  # additional packages
  apt-get update
  apt-get install -y python-tk
  apt-get install -y libsm6 libxext6
  pip install selenium
  pip install moviepy
  pip install lmdb
  pip install opencv-contrib-python
  pip install cryptography

  pip install numpy
  pip install scipy
  pip install Pillow
  pip install cython
  pip install matplotlib
  pip install scikit-image
  pip install tensorflow>=1.3.0
  pip install keras>=2.0.8
  pip install opencv-python
  pip install h5py
  pip install imgaug
  pip install IPython[all]
  pip install Pillow

%runscript
  # executes with the singularity run command
  # delete this section to use existing docker ENTRYPOINT command

%test
  # test that script is a success
