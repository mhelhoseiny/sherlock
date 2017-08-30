function [scores,scores2, list_im] = matcaffe_batch_detail2(list_im, model_def_file, model_file, use_gpu)
% scores = matcaffe_batch(list_im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   list_im  list of images files
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   dim x num_images ILSVRC output vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  scores = matcaffe_batch({'peppers.png','onion.png'});
%  scores = matcaffe_batch('list_images.txt', 1);
if nargin < 1
  % For test purposes
  list_im = {'peppers.png','onions.png'};
end
if ischar(list_im)
    %Assume it is a file contaning the list of images
    filename = list_im;
    list_im = read_cell(filename);
end
% Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
batch_size = 10;
%dim = 1000;
disp(list_im)
if mod(length(list_im),batch_size)
    warning(['Assuming batches of ' num2str(batch_size) ' images rest will be filled with zeros'])
end

% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init(use_gpu, model_def_file, model_file,true);
else
  matcaffe_init();
end

d = load('ilsvrc_2012_mean');
IMAGE_MEAN = d.image_mean;

% prepare input

num_images = length(list_im);
scoresInit = false;
%scores = zeros(dim,num_images,'single');
num_batches = ceil(length(list_im)/batch_size);
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    tic
    input_data = prepare_batch(list_im(range),IMAGE_MEAN,batch_size);
    toc, tic
    fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
    output_data = caffe('forward', {input_data});
    toc
    outdatacpy = output_data;
    output_data = squeeze(output_data{1});
    output_datap = squeeze(outdatacpy{2});
    if(size(output_data,3)==1&&size(output_data,4)==1) % output layer
        if(scoresInit==false)
            dim = size(output_data,1);
            scores = zeros(dim,num_images,'single');
            dimp = size(output_datap,1);
            scores2 = zeros(dimp,num_images,'single');
            scoresInit= true;
        end
        scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
        scores2(:,range) = output_datap(:,mod(range-1,batch_size)+1);
    else
        output_data2 = reshape(output_data, [numel(output_data)/size(output_data,4), size(output_data,4)]);
        output_datap2 = reshape(output_datap, [numel(output_datap)/size(output_datap,4), size(output_datap,4)]);
        if(scoresInit==false)
            dim = size(output_data2,1);
            scores = zeros(dim,num_images,'single');
            scoresInit= true;
        end
        scores(:,range) = output_data2(:,mod(range-1,batch_size)+1);
        scores2(:,range) = output_datap2(:,mod(range-1,batch_size)+1);
        
    end
    %scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
    toc(batchtic)
end
toc(initic);

if exist('filename', 'var')
    save([filename '.probs.mat'],'list_im','scores','-v7.3');
end



