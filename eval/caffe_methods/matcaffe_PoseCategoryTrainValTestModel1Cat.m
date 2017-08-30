function matcaffe_PoseCategoryTrainValTestModel1Cat( post_fix, outFolder, largercrop)
if(largercrop)
    inputTraintxt = '/home/elhoseiny/Mohamed/deepLearning/deepCatPoseProject/code/splitsonmydisk/train_leave2out.txt';
    inputtestTxt = '/home/elhoseiny/Mohamed/deepLearning/deepCatPoseProject/code/splitsonmydisk/test_leave2out.txt';
    inputValidateTxt = '/home/elhoseiny/Mohamed/deepLearning/deepCatPoseProject/code/splitsonmydisk/validation_leave2out.txt';
else
    inputTraintxt = '/home/elhoseiny/Mohamed/deepLearning/deepCatPoseProject/code/splitsonmydisk_crop/train_leave2out.txt';
    inputtestTxt = '/home/elhoseiny/Mohamed/deepLearning/deepCatPoseProject/code/splitsonmydisk_crop/test_leave2out.txt';
    inputValidateTxt = '/home/elhoseiny/Mohamed/deepLearning/deepCatPoseProject/code/splitsonmydisk_crop/validation_leave2out.txt';
end
%model_def_file = '../../models/bvlc_reference_caffenet/deploy.prototxt';
model_file = '../../models/finetune_rgbdDatsetCat_v2All/finetune_rgbdDatsetCat_v2All_iter_20000.caffemodel';
layer = post_fix;
if(strcmp(layer, 'fc8_rgbdC'))
    %dim = 1000; %final output
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy.prototxt';
elseif(strcmp(layer, 'fc7'))
    %dim = 4096; %fc7
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy_fc7.prototxt';
elseif(strcmp(layer, 'fc6'))
    %dim = 4096; % fc6
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy_fc6.prototxt';
elseif(strcmp(layer, 'pool5'))
    %dim =9216;%pool5
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy_pool5.prototxt';
elseif(strcmp(layer, 'conv5'))
    %dim =43264;%conv5
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy_conv5.prototxt';
elseif(strcmp(layer, 'conv4'))
    %dim =9216;%conv4
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy_conv4.prototxt';
elseif(strcmp(layer, 'conv3'))
    %dim =9216;%conv3
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy_conv3.prototxt';
elseif(strcmp(layer, 'pool2'))
    %dim = 43264; %pool2
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy_pool2.prototxt';
elseif(strcmp(layer, 'pool1'))
    %dim= 69984; %pool1
    model_def_file = '../../models/finetune_rgbdDatsetCat_v2All/deploy_pool1.prototxt';
end
use_gpu= 1;
fidtraintxt = fopen(inputTraintxt);
fidtesttxt = fopen(inputtestTxt);
fidvalidationtxt =  fopen(inputValidateTxt);
trainData = textscan(fidtraintxt, '%s %d %d %f');
testData = textscan(fidtesttxt, '%s %d %d %f');
validationData = textscan(fidvalidationtxt, '%s %d %d %f');
fclose(fidtraintxt);
fclose(fidtesttxt);
fclose(fidvalidationtxt);

list_im = trainData{1};
labels= trainData{2};
instances= trainData{3};
poses = trainData{4};
[scores,list_im1] = matcaffe_batch_detail(trainData{1}, model_def_file, model_file, use_gpu);

save([outFolder, 'train_',post_fix, '.mat'], 'list_im', 'scores','labels', 'instances', 'poses','-v7.3');

list_im = testData{1};
labels= testData{2};
instances= testData{3};
poses = testData{4};
[scores,list_im1] = matcaffe_batch_detail(testData{1}, model_def_file, model_file, use_gpu);
save([outFolder, 'test_',post_fix, '.mat'], 'list_im', 'scores','labels', 'instances', 'poses','-v7.3');

list_im = validationData{1};
labels= validationData{2};
instances= validationData{3};
poses = validationData{4};
[scores,list_im1] = matcaffe_batch_detail(validationData{1}, model_def_file, model_file, use_gpu);

save([outFolder, 'validation_',post_fix, '.mat'], 'list_im', 'scores','labels', 'instances', 'poses','-v7.3');


end