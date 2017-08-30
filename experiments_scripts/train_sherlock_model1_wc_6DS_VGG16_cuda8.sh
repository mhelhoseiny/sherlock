export ROOT_DIR=$HOME/sherlock
export Caffe_DEP_PATH=/usr/local/lib/
export CUDA8_PATH=/public/apps/cuda/8.0/lib64/
export CUDNN8_PATH=$ROOT_DIR/caffe_setup/cudnn8/lib64/
export LD_LIBRARY_PATH=$Caffe_DEP_PATH:$CUDA8_PATH:$CUDNN8_PATH
export MODELS_FOLDER=$ROOT_DIR/models
$ROOT_DIR/caffe_setup/caffe_sherlock/build/tools/caffe.bin train --solver $MODELS_FOLDER/sherlock_VGG16_6DS_model1_wc/solver.prototxt --weights $MODELS_FOLDER/model1_init/model1_VGG16_init.caffemodel -gpu 0

