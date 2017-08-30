% Author: Mohamed ELhoseiny
function [X_Embedding, T_Embedding] = GetTandXEmbeddingforCaffeNetwork(Data, NetworkName, modelfile, imageprotofile, tupleprotofile, CropNumber, t_Mean, CROPPED_DIM, batchsize,gpu_id)

list_im = Data.im_names;

if(~exist('CropNumber', 'var'))
    CropNumber = 5;
end
if(~exist('gpu_id', 'var'))
    gpu_id=0;
end
tupleId2IndMap = containers.Map(Data.unique_tuple_ids, [1: numel(Data.unique_tuple_ids)] );

if(strcmp(NetworkName, 'Model0')||strcmp(NetworkName, 'model0')||strcmp(NetworkName, 'model0_wc')||strcmp(NetworkName, 'model1_wc')||strcmp(NetworkName, 'model1')||strcmp(NetworkName, 'model1_800')||strcmp(NetworkName, 'model1_800_wc')||strcmp(NetworkName, 'VGG16'))
    if(~exist('CROPPED_DIM', 'var'))
        CROPPED_DIM = 227;
    end

    [X_Embedding,list_im_x, problems_chunk_X] = matcaffe_batch_detail(list_im, imageprotofile, modelfile, CROPPED_DIM, 1,CropNumber, batchsize,gpu_id);

    X_Embedding = X_Embedding';
    if(isempty(tupleprotofile))
        T_Embedding = zeros(size(X_Embedding,1), size(Data.unique_tuple_features,2));
        for i=1:size(T_Embedding,1)
            ind_uniq_i = tupleId2IndMap(Data.tuple_ids(i));
            T_Embedding(i,:) = Data.unique_tuple_features(ind_uniq_i,:);
        end
    else
        unique_tuple_features_embed = matcaffe_extract_tuple_features( Data.unique_tuple_features , tupleprotofile, modelfile,batchsize);
        unique_tuple_features_embed = unique_tuple_features_embed';
        T_Embedding = zeros((numel(list_im)), size(Data.unique_tuple_features,2));
        for i=1:size(T_Embedding,1)
            ind_uniq_i = tupleId2IndMap(Data.tuple_ids(i));
            T_Embedding(i,:) = unique_tuple_features_embed(ind_uniq_i,:);
        end
        %T_Embedding = matcaffe_extract_tuple_features(T_Embedding , tupleprotofile, modelfile,batchsize);
        %T_Embedding = T_Embedding';
    
    
    end
    
    if(exist('t_Mean', 'var')&&~isempty(t_Mean))
        T_Embedding = bsxfun(@minus,T_Embedding,t_Mean);
    end
% elseif(strcmp(NetworkName, 'Model0_two_layer'))
%     if(~exist('CROPPED_DIM', 'var'))
%         CROPPED_DIM = 227;
%     end
% 
%     [X_Embedding,list_im_x, problems_chunk_X] = matcaffe_batch_detail(list_im, imageprotofile, modelfile, CROPPED_DIM, 1,CropNumber, batchsize);
% 
%     X_Embedding = X_Embedding';
%     
%     if(exist('t_Mean', 'var'))
%         T_Features= bsxfun(@minus, Data.unique_tuple_features ,t_Mean);
%     else
%         T_Features = Data.unique_tuple_features;
%     end
%     
%     T_UniqueEmbedding = matcaffe_extract_tuple_features(T_Features , tupleprotofile, modelfile);
%     T_UniqueEmbedding = T_UniqueEmbedding';
%     T_Embedding = zeros(size(X_Embedding,1), size(T_UniqueEmbedding,2));
%     for i=1:size(T_Embedding,1)
%         ind_uniq_i = tupleId2IndMap(Data.tuple_ids(i));
%         T_Embedding(i,:) = T_UniqueEmbedding(ind_uniq_i,:);
%     end
%     
% else %if(strcmp(NetworkName, 'Model0_one_layer'))
%     if(~exist('CROPPED_DIM', 'var'))
%         CROPPED_DIM = 227;
%     end
%     [X_Embedding,list_im_x, problems_chunk_X] = matcaffe_batch_detail(list_im, imageprotofile, modelfile, CROPPED_DIM, 1,CropNumber, batchsize);
% 
%     X_Embedding = X_Embedding';
%     
%     if(exist('t_Mean', 'var'))
%         T_Features= bsxfun(@minus, Data.unique_tuple_features ,t_Mean);
%     else
%         T_Features = Data.unique_tuple_features;
%     end
%     
%     T_UniqueEmbedding = matcaffe_extract_tuple_features(  T_Features, tupleprotofile, modelfile, batchsize);
%     T_UniqueEmbedding = T_UniqueEmbedding';
%     T_Embedding = zeros(size(X_Embedding,1), size(T_UniqueEmbedding,2));
%     for i=1:size(T_Embedding,1)
%         ind_uniq_i = tupleId2IndMap(Data.tuple_ids(i));
%         T_Embedding(i,:) = T_UniqueEmbedding(ind_uniq_i,:);
%     end

%else
%   error([ 'unsupported Network ',  NetworkName]) 
end