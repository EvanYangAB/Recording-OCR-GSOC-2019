Bootstrap: docker
From: paddlepaddle/deep_speech:latest-gpu

%setup
  # runs on host - the path to the image is $SINGULARITY_ROOTFS

%post
  # post-setup script


  # additional packages
  apt-get update
  apt-get install -y python3
  apt-get install -y libsm6 libxext6
  pip3 install selenium
  pip3 install moviepy
  pip3 install lmdb
  pip3 install opencv-contrib-python
  pip3 install cryptography

  pip3 install numpy
  pip3 install scipy
  pip3 install Pillow
  pip3 install cython
  pip3 install matplotlib
  pip3 install scikit-image
  pip3 install keras>=2.0.8
  pip3 install opencv-python
  pip3 install h5py
  pip3 install imgaug
  pip3 install IPython[all]

%runscript
  # executes with the singularity run command
  # delete this section to use existing docker ENTRYPOINT command

%test
  # test that script is a success