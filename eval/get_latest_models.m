function found_models = get_latest_models(models_folder, deploy_folder, opt_inc_filter,specify_iteration_number)


found_models = {};
if(~exist('opt_inc_filter', 'var'))
    opt_inc_filter = '';
end

if(isempty(opt_inc_filter))
trained_models_dirs = dir(fullfile(models_folder, '*_VGG16*'));
else
   trained_models_dirs1 = dir(fullfile(models_folder, ['*_VGG16*', opt_inc_filter, '*']));
   trained_models_dirs2 = dir(fullfile(models_folder, ['*_VGG16*', opt_inc_filter, '*']));
   if(numel(trained_models_dirs2)>numel(trained_models_dirs1))
       trained_models_dirs = trained_models_dirs2;
   else
       trained_models_dirs = trained_models_dirs1;
   end
end
    
    

All_known_datasets_for_training= {'VisualPhrases','Stanford40', 'PASCAL12', '6DS', '12DS2000'};

all_models_of_interest = {'model0_wc','model0','model1_800_wc','model1_wc', 'model1'}; 

encoderTypes_of_interest = {'AlexNet', 'VGG16'} ;
deploy_x_VGG16_file_map = containers.Map();

for i=1:numel(all_models_of_interest)
    deploy_x_VGG16_file_map(all_models_of_interest{i}) = fullfile(deploy_folder, ['deploy_', all_models_of_interest{i},'_VGG16.prototxt']);
    assert(exist(deploy_x_VGG16_file_map(all_models_of_interest{i}) , 'file')~=0)
end

for i = 1:numel(trained_models_dirs)
    if trained_models_dirs(i).isdir && ~strcmpi(trained_models_dirs(i).name, '.') && ~strcmpi (trained_models_dirs(i).name, '..') 
        model_training_folder_name = trained_models_dirs(i).name;
        
        dataset_i = [];
        
        for k=1:numel(All_known_datasets_for_training)
            k_ij = findstr(model_training_folder_name, All_known_datasets_for_training{k});
            if(~isempty(k_ij))
                dataset_i = All_known_datasets_for_training{k};
            end
        end
        if(isempty(dataset_i))
            continue;
        end
        model_i =[];
        for k=1:numel(all_models_of_interest)
            k_ij = findstr(model_training_folder_name, all_models_of_interest{k});
            if(~isempty(k_ij))
                model_i = all_models_of_interest{k};
                break;
            end
        end
        
        if(isempty(model_i))
            continue;
        end
        
        encoder_i = [];
        
        for k=1:numel(encoderTypes_of_interest)
            k_ij = findstr(model_training_folder_name, encoderTypes_of_interest{k});
            if(~isempty(k_ij))
                encoder_i = encoderTypes_of_interest{k};
                break;
            end
        end
        if(isempty(encoder_i))
            continue;
        end
        %sherlock_VGG16_PASCAL12_model1_wc_rank_train_iter_4350.caffemodel
        if(~exist('specify_iteration_number', 'var')||specify_iteration_number==-1)
             dir_i = dir(fullfile(models_folder, model_training_folder_name, '*.caffemodel'));
        else
             dir_i = dir(fullfile(models_folder, model_training_folder_name, ['*_iter_',num2str(specify_iteration_number),'.caffemodel']));
        end
        
        if(~isempty(dir_i))
           
            max_iter_j = -1;
            max_iter = -1;
            for j=1:numel(dir_i);
                find_str = 'iter_';
                k_ij = findstr(dir_i(j).name, 'iter_');
                iter_str = dir_i(j).name(k_ij+numel(find_str):numel(dir_i(j).name)-numel('.caffemodel'));
                iter_ij = str2num(iter_str);
                if(iter_ij>max_iter)
                    max_iter = iter_ij;
                    max_iter_j = j;
                end
                
            end
            if(max_iter~=-1)
                sel_file = fullfile(models_folder, model_training_folder_name, dir_i(max_iter_j).name);

                if(strcmp( dataset_i , '6DS'))
                    dataset_i  ='INTERACT_PASCAL12_SPORT_VisualPhrases_PPMI_Stanford40';
                elseif(strcmp( dataset_i , '12DS2000'))
                    dataset_i  ='INTERACT_PASCAL12_SPORT_VisualPhrases_PPMI_Stanford40_SceneGraph_COCO_Objects_coco_train14_coco_val14_flickr30k_2000';
                end
                mdl_entry.dataset = dataset_i;
                mdl_entry.model_file= sel_file;
                mdl_entry.model_type = model_i;
                mdl_entry.max_iteration = max_iter;
                mdl_entry.v_encoder=encoder_i;
                if(strcmp(mdl_entry.v_encoder, 'VGG16'))
                    mdl_entry.v_deploy_file = deploy_x_VGG16_file_map(model_i);
                    if(strcmp(opt_inc_filter,'_rank_2layer'))
                         mdl_entry.l_deploy_file = fullfile(deploy_folder, 'deploy_model1_wc_rank_l2_TEmbedding.prototxt');
                    elseif(strcmp(opt_inc_filter,'_rank_1layer'))
                         mdl_entry.l_deploy_file = fullfile(deploy_folder, 'deploy_model1_wc_rank_l1_TEmbedding.prototxt');
                    else
                         mdl_entry.l_deploy_file = [];
                    end
                   
                    mdl_entry.batch_size = 10;
                else
                    error('only VGG16 is supported');
                end
                found_models{end+1} = mdl_entry;
            end
            
        end
        
        
        
        
    end
end


