%Author: Mohamed Elhoseiny, Summer 2015
function [XEmbedding, TEmbedding] = GetTandXEmbeddingforCaffeNetworkDIfferentCrops(NetworkType, modelfile, imageprotofile, tupleprotofile, trainData, testData, CROPPED_DIM, batchsize,CropNums,gpu_id)

if(~exist('gpu_id', 'var'))
    gpu_id=0;
end



if(isfield(trainData, 'unique_tuple_features_mean'))
    tmean = trainData.unique_tuple_features_mean;
end


if(~exist('CROPPED_DIM', 'var'))
    CROPPED_DIM = 227;
end

if(~exist('batchsize', 'var'))
    batchsize = 10;
end




%CropNums = 1:10;
%CropNums = 5;
XEmbedding = [];
for i=1:numel(CropNums)
    i
    if(exist('tmean', 'var'))
        [XEmbedding_i, TEmbedding] =GetTandXEmbeddingforCaffeNetwork(testData, NetworkType, modelfile, imageprotofile, tupleprotofile,CropNums(i), tmean,CROPPED_DIM, batchsize,gpu_id);
    else
        [XEmbedding_i, TEmbedding] =GetTandXEmbeddingforCaffeNetwork(testData, NetworkType, modelfile, imageprotofile, tupleprotofile,CropNums(i), [], CROPPED_DIM, batchsize,gpu_id);
    end
    
    Eucloss = mean(sqrt(sum((XEmbedding_i-TEmbedding).^2,2)));
    
    if(isempty(XEmbedding))
        XEmbedding = XEmbedding_i;
    else
        XEmbedding = XEmbedding+XEmbedding_i;
    end
end
XEmbedding = XEmbedding/numel(CropNums);