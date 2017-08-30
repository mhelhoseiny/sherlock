This code is for training sherlock models in this paper

Mohamed Elhoseiny, Scott Cohen, Walter Chang, Brian Price, Ahmed Elgamma,  Sherlock: Scalable Fact Learning in Images, AAAI, 2017

This documents presented the Datasets and the Code Installation steps

DATASETS
=========

1) 6DS benchmark   https://dl.dropboxusercontent.com/u/33950950/sherlock_project/data/6DS_dataset.zip (2.4 GB)
	(28,000 images, 186 unique facts) 
	wit the training and testing splits
	Fact Recognition Top 1 Accuracy (our method): 69.63% 
	Fact Recognition MAP/MAP100 (our method): 34.86%/ 50.68%


2)  LSC (Large Scale benchmark) 
  (814K images, 202K unique  facts) 
	part 1,LSC_dataset.tar.gz.aa  https://dl.dropboxusercontent.com/u/33950950/sherlock_project/data/LSC_dataset.tar.gz.aa (11.72 GB):
	part 2, LSC_dataset.tar.gz.ab https://dl.dropboxusercontent.com/u/33950950/sherlock_project/data/LSC_dataset.tar.gz.ab (8.73 GB):
	After download, run
cat LSC_dataset.tar.gz.* > LSC_dataset.tar.gz
	Then extract LSC_dataset.tar.gz
with the training and testing splits

	Fact Recognition Top 1 Accuracy (our method): 16.39%
	Fact Recognition MAP/MAP100 (our method): 1.0%


CODE INSTALLATION ()
====================

1) Install Caffe Pre-requisities (Don not install Caffe itself since we will build the sherlock (our) version of caffe)
   Follow the isstructions here to install all the depenednceis depending on your OS verison.
   caffe.berkeleyvision.org/installation.html
   
   
   This code was tested on Linux.  

2) Build caffe_sherlock: This is a fork of caffe where additional layers are implemented for the sherlock loss function
   a) Make sure the caffe dependencies are installed 
   b) If you need to install openCV from source, there is a version under caffe_setup that you can install by these steps
       cd caffe_setup/opencv2.X.X.X/
       mkdir release
       cd release 
        cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..
       make
       make install              
   c) edit sherlock/caffe_setup/caffe_sherlock/Makefile.config and set the right paths for the following
       CUDA_DIR := /public/apps/cuda/8.0/
       MATLAB_DIR  := /usr/local/MATLAB/R2017a
       ANACONDA_HOME := /public/apps/anaconda2/4.3.1
       I used anaconda but you can used standard python

   d) at sherlock/caffe_setup/caffe_sherlock run make all, make pycaffe (for python interface) , make matcaffe (for matlab interface)   


3) Edit these three paths PATHS in sherlock/experiments_scripts/train_sherlock_model1_wc_6DS_VGG16_cuda8.sh
export ROOT_DIR=$HOME/sherlock
export Caffe_DEP_PATH=/usr/local/lib/
export CUDA8_PATH=/public/apps/cuda/8.0/lib64/

Make sure that these paths are referring to the root folder of the sherlock code, depenencies of caffe and CUDA8_PATH


4) download init_models
   cd ./models/
   ./download_init_models.sh 


5) download 6DS lmdb
   cd ./data
   ./download_data.sh


6)download cudnn8

cd ./caffe_setup/cudnn8
./download_cudnn8.sh

<<<<<<< HEAD
7) run 6DS training  
=======
7) run  
>>>>>>> e793af7505ce09659619729eeec6d0f050d5dfcb

cd sherlock
run ./experiments_scripts/train_sherlock_model1_wc_6DS_VGG16_cuda8.sh


This will train sherlock Model in the paper for thr 6DS dataset

8) evaluate trained model / downloaded model

A) You can either wait until the training done or download a pretrained model by the following
cd ./models/
./download_6DS_model1_wc.sh

B) evaluated the 6DS model. This is done by runnning the matlab evaluation code. Note that caffe matlab interface need to be installed to run the evaluation.











  
