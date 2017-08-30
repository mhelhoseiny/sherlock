function eval_6DS_dataset(model_of_interest)


if(~exist('model_of_interest', 'var'))
    model_of_interest = 'model1_wc';
end

%% Dataset setting 
sixDS_dataset_folder = 'datasets/6DS_dataset';

evalDir = 'eval';
DemoDataFlag = false;
caffe_dolder  = 'caffe_setup/caffe_sherlock';
%addpath('/home/elhosein/Mohamed/ThirdPartyCode/matpy/');

models_folder = 'models';
data_folder =  'data';

caffe_methods_folder = 'eval/caffe_methods';

deploy_parent_folder = 'deploy';
Results_folder = 'Results';

addpath(caffe_methods_folder);
addpath('eval/metrics')


%%
dataset_setting = 'INTERACT_PASCAL12_SPORT_VisualPhrases_PPMI_Stanford40';

%% Architecture 
NetworkArch = 'VGG16'; 
CROPPED_DIM = 224;
batchsize = 5;

%% deploy mode
%deploy_mode = 'deploy_10';
%deploy_mode = 'deploy_75';
%deploy_mode = 'deploy_140';

if(~exist('deploy_mode', 'var'))
    deploy_mode = 'deploy_10';
end

if(strcmp(deploy_mode, 'deploy_10'))
    deploy_mode_batchsize = 10;
elseif(strcmp(deploy_mode, 'deploy_140'))
    deploy_mode_batchsize = 140;
elseif(strcmp(deploy_mode, 'deploy_75'))
    deploy_mode_batchsize = 75;
end

%% crops  


%% params
addpath([caffe_dolder, '/matlab/+caffe/imagenet']);
addpath([caffe_dolder,'/matlab/'])


if(~exist('gpu_id', 'var'))
    gpu_id=0;
end


if(~exist('model_of_interest','var'))
    model_of_interest = 'model1_wc';
end
%% Add directories
addpath(caffe_methods_folder);
addpath(evalDir);
addpath(fullfile(evalDir,'APcode/'));
addpath(fullfile(evalDir,'AUCcode/'));


%%%





%specify_iteration_number=5000;
if(~exist('specify_iteration_number', 'var'))
    specify_iteration_number=-1;
end



found_models = get_latest_models(models_folder,fullfile(deploy_parent_folder, deploy_mode),'', specify_iteration_number);

% for deploy_bigger
for i=1:numel(found_models)
  found_models{i}.batch_size = deploy_mode_batchsize;
end






setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/local/lib:/public/apps/cuda/8.0/lib64/:/private/home/elhoseiny/sherlock/caffe_setup/cudnn8/lib64/']);
 
p = getenv('LD_LIBRARY_PATH');
p_split = strsplit(p, ':');
for i=1:numel(p_split)
    if(~isempty(p_split{i}))
        addpath(p_split{i});
    end
end
%%

outputdir = [data_folder, '/', dataset_setting, '_WVEC_SUBMEAN_LMBD_DIR'];


% if(maxNumImages~=Inf)
%     outputdir = [outputdir, '_', num2str(maxNumImages)];
%     splitSpecs_str = [dataset_setting, '_', num2str(maxNumImages)];
% end


train_data_mat_file = fullfile(sixDS_dataset_folder, 'train_data.mat');
test_data_mat_file = fullfile(sixDS_dataset_folder, 'test_data.mat');

%% 

load(test_data_mat_file);
load(train_data_mat_file);
if(DemoDataFlag)

    load(train_data_mat_file);
    Demo_Data(train_data, tupleMMap);
end

%% augment_dataset_full_path 

for i=1:numel(test_data.im_names)
   test_data.im_names{i} = fullfile(sixDS_dataset_folder,  test_data.im_names{i}); 
end
 
for i=1:numel(train_data.im_names)
   train_data.im_names{i} = fullfile(sixDS_dataset_folder,  train_data.im_names{i}); 
end

%% TODO: put here that trained caffe model that you would like to test for this particular setting

Model_Name = model_of_interest;

splitSpecs_str = dataset_setting;
splitSpecs_str_for_save = dataset_setting;



    %splitspecs.setting = 'PASCAL12';



%% find Model
found_model = false;
for i=1:numel(found_models)
    if(strcmp(found_models{i}.dataset,splitSpecs_str)&&strcmp(found_models{i}.model_type, Model_Name)&&strcmp(found_models{i}.v_encoder, NetworkArch))
      
        Caffe_Model_tuple__deployfile = found_models{i}.l_deploy_file ;

        Caffe_Model_image_deployfile = found_models{i}.v_deploy_file;
        batchsize = found_models{i}.batch_size;
        Caffe_Model_Name = found_models{i}.model_file;


        Caffe_Model_Training_Iterations = found_models{i}.max_iteration;
        found_model = true;

        break;
    end
end
if(~found_model)
    error('can not find model');
end
    %Evaluate_Network(Caffe_Model_Name, Caffe_Model_tuple__deployfile, Caffe_Model_image_deployfile, train_data_mat_file, test_data_mat_file);

  
bn_str = '';
if(strcmp(NetworkArch, 'VGG16'))
    if(~exist('specify_iteration_number', 'var')||specify_iteration_number==-1)
        Embedding_save_File = fullfile(Results_folder, [splitSpecs_str_for_save ,'_','VGG16_', Model_Name,bn_str, '_submean_Embeddings.mat' ]);
        Result_save_file = fullfile(Results_folder, [splitSpecs_str_for_save ,'_','VGG16_', Model_Name,bn_str, '_submean.mat' ]);
        Resultnotharsh_save_file = fullfile(Results_folder, [splitSpecs_str_for_save ,'_','VGG16_', Model_Name,bn_str, '_submean_notHarsh.mat' ]);
        Resultnotharshv2_save_file = fullfile(Results_folder, [splitSpecs_str_for_save ,'_','VGG16_', Model_Name,bn_str, '_submean_notHarsh_v2.mat' ]);
    else
        Embedding_save_File = fullfile(Results_folder, [splitSpecs_str_for_save ,'_','VGG16_', Model_Name,bn_str, '_submean_Embeddings_iter_', num2str(specify_iteration_number) ,'.mat' ]);
        Result_save_file = fullfile(Results_folder, [splitSpecs_str_for_save ,'_','VGG16_', Model_Name,bn_str, '_submean_iter_', num2str(specify_iteration_number) ,'.mat' ]);
        Resultnotharsh_save_file = fullfile(Results_folder, [splitSpecs_str_for_save ,'_','VGG16_', Model_Name,bn_str, '_submean_notHarsh_iter_', num2str(specify_iteration_number) ,'.mat' ]);
        Resultnotharshv2_save_file = fullfile(Results_folder, [splitSpecs_str_for_save ,'_','VGG16_', Model_Name,bn_str, '_submean_notHarsh_v2_iter_', num2str(specify_iteration_number) ,'.mat' ]);
    end
end
    

Embedding_Xcrops = [1:10];



if(~exist(Embedding_save_File, 'file'))
 [XEmbedding, TEmbedding] = GetTandXEmbeddingforCaffeNetworkDIfferentCrops(Model_Name, Caffe_Model_Name,  Caffe_Model_image_deployfile, Caffe_Model_tuple__deployfile, train_data, test_data, CROPPED_DIM, batchsize,Embedding_Xcrops,gpu_id);
 save(Embedding_save_File, 'XEmbedding', 'TEmbedding', 'Embedding_Xcrops'); 
else
 load(Embedding_save_File);
end
            
  



 if(~exist(Result_save_file, 'file'))
    testData = load(test_data_mat_file);
    testData = testData.test_data;
    metrics = {'cos', 'dot', 'euc'};
    Results = cell(1,3);
    for k=1:numel(metrics)
        
           metric = metrics{k};
           %resultsfname = fullfile(ExperFolder, ['split_',split_i_str_forfname,'_',metric , '_Result.mat']);
           Results{k} = GetSherlockResults(XEmbedding, TEmbedding, testData, metric);
    end
    save(Result_save_file, 'metrics', 'Results');
 else
             
             load(Result_save_file);
         end
          if(~exist(Resultnotharsh_save_file, 'file'))
            testData = load(test_data_mat_file);
            testData = testData.test_data;
            metrics = {'cos', 'dot', 'euc'};
            Results = cell(1,3);
            for k=1:numel(metrics)
                   metric = metrics{k};
                   %resultsfname = fullfile(ExperFolder, ['split_',split_i_str_forfname,'_',metric , '_Result.mat']);
                   Results{k} = GetSherlockResults_notharsh(XEmbedding, TEmbedding, testData, metric);
            end
            save(Resultnotharsh_save_file, 'metrics', 'Results', 'Caffe_Model_Training_Iterations', 'Caffe_Model_Name');
         else
             
             load(Resultnotharsh_save_file);
          end
         
          if(~exist(Resultnotharshv2_save_file, 'file'))
            testData = load(test_data_mat_file);
            testData=testData.test_data;
            metrics = {'cos', 'dot', 'euc'};
            Results = cell(1,3);
            for k=1:numel(metrics)
                   metric = metrics{k};
                   %resultsfname = fullfile(ExperFolder, ['split_',split_i_str_forfname,'_',metric , '_Result.mat']);
                   Results{k} = GetSherlockResults_notharsh_v2(XEmbedding, TEmbedding, testData, metric);
            end
            save(Resultnotharshv2_save_file, 'metrics', 'Results', 'Caffe_Model_Training_Iterations', 'Caffe_Model_Name');
         else
             
             load(Resultnotharshv2_save_file);
          end
         
             
    
    end



