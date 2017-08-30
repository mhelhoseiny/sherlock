% ------------------------------------------------------------------------
function [images, problems] = prepare_batch(image_files,IMAGE_MEAN,batch_size, CROPPED_DIM, cropNumber)
% ------------------------------------------------------------------------
if nargin < 2
    d = load('ilsvrc_2012_mean');
    IMAGE_MEAN = d.mean_data; 
end
num_images = length(image_files);
if nargin < 3
    batch_size = num_images;
end

IMAGE_DIM = 256;
if(~exist('CROPPED_DIM', 'var'))
    CROPPED_DIM = 227;
end

if(~exist('cropNumber', 'var'))
    cropNumber=5;
end
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
center = floor(indices(2) / 2)+1;

num_images = length(image_files);
images = zeros(CROPPED_DIM,CROPPED_DIM,3,batch_size,'single');
problems = zeros(1, batch_size,'single');
if(num_images>1)
    parfor i=1:num_images
        % read file
        %fprintf('%c Preparing %s\n',13,image_files{i});
        try
            im = imread(image_files{i});
            % resize to fixed input size
            im = single(im);
            im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
            % Transform GRAY to RGB
            if size(im,3) == 1
                im = cat(3,im,im,im);
            end

            %% Orginal Code
            % permute from RGB to BGR (IMAGE_MEAN is already BGR)


            %% New code
            im_data = im(:, :, [3, 2, 1]);
            im_data = permute(im_data, [2, 1, 3]);
            im_data = im_data - IMAGE_MEAN;

            if(cropNumber<1||cropNumber>10)
               error('Invalid Crop Number'); 
            end
            if(cropNumber==5)
                 %im = im(:,:,[3 2 1]) - IMAGE_MEAN;

                % Crop the center of the image
                %images(:,:,:,i) = permute(im(center:center+CROPPED_DIM-1,...
                %     center:center+CROPPED_DIM-1,:),[2 1 3]);
                images(:,:,:,i) = im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
            elseif(cropNumber==10)
                images(:,:,:,i) = im_data(center+CROPPED_DIM-1:-1:center,center:center+CROPPED_DIM-1,:);
                % images(:,:,:,i) = im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
                %images(:, :, :, i) = images(end:-1:1, :, :, n);
            else
                if(cropNumber>5)
                    n = cropNumber-5;
                else
                    n = cropNumber;
                end
                i1_ind = floor((n-1)/2)+1;
                j1_ind = mod(n-1,2)+1;
                i1 = indices(i1_ind);
                j1 = indices(j1_ind);

                if(cropNumber>5)
                    images(:, :, :, i) = im_data(i1:i1+CROPPED_DIM-1, j1:j1+CROPPED_DIM-1, :);
                else
                    images(:, :, :, i) = im_data(i1+CROPPED_DIM-1:-1:i1, j1:j1+CROPPED_DIM-1, :);
                end

            end
        catch
            problems(i)=1;
            warning('Problems with file',image_files{i});
        end

    end
    
else
    
   for i=1:num_images
    % read file
    %fprintf('%c Preparing %s\n',13,image_files{i});
    try
        im = imread(image_files{i});
        % resize to fixed input size
        im = single(im);
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        % Transform GRAY to RGB
        if size(im,3) == 1
            im = cat(3,im,im,im);
        end
        
        %% Orginal Code
        % permute from RGB to BGR (IMAGE_MEAN is already BGR)
       
        
        %% New code
        im_data = im(:, :, [3, 2, 1]);
        im_data = permute(im_data, [2, 1, 3]);
        im_data = im_data - IMAGE_MEAN;
        
        if(cropNumber<1||cropNumber>10)
           error('Invalid Crop Number'); 
        end
        if(cropNumber==5)
             %im = im(:,:,[3 2 1]) - IMAGE_MEAN;
        
            % Crop the center of the image
            %images(:,:,:,i) = permute(im(center:center+CROPPED_DIM-1,...
            %     center:center+CROPPED_DIM-1,:),[2 1 3]);
            images(:,:,:,i) = im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
        elseif(cropNumber==10)
            images(:,:,:,i) = im_data(center+CROPPED_DIM-1:-1:center,center:center+CROPPED_DIM-1,:);
            % images(:,:,:,i) = im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
            %images(:, :, :, i) = images(end:-1:1, :, :, n);
        else
            if(cropNumber>5)
                n = cropNumber-5;
            else
                n = cropNumber;
            end
            i1_ind = floor((n-1)/2)+1;
            j1_ind = mod(n-1,2)+1;
            i1 = indices(i1_ind);
            j1 = indices(j1_ind);
           
            if(cropNumber>5)
                images(:, :, :, i) = im_data(i1:i1+CROPPED_DIM-1, j1:j1+CROPPED_DIM-1, :);
            else
                images(:, :, :, i) = im_data(i1+CROPPED_DIM-1:-1:i1, j1:j1+CROPPED_DIM-1, :);
            end
             
        end
    catch
        problems(i)=1;
        warning('Problems with file',image_files{i});
    end
   end
     
end