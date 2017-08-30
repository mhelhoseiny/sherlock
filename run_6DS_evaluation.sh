export MATLAB_DIR=/usr/local/MATLAB/R2017a/
export ANACONDA2_PATH=/public/apps/anaconda2/4.3.1/lib
export CUDA8_PATH=/public/apps/cuda/8.0/lib64/
export CUDNN8_PATH=$HOME/sherlock/caffe_setup/cudnn8/lib64/
export Caffe_DEP_PATH=/usr/local/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANACONDA2_PATH:$Caffe_DEP_PATH:$CUDA8_PATH:$CUDNN8_PATH
echo $LD_LIBRARY_PATH
$MATLAB_DIR/bin/matlab -nodesktop -nodisplay -r "addpath('./eval'); run('eval_6DS_dataset');"