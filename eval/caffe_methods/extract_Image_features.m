function [chunk_files, ind_ranges] = extract_Image_features(list_im, chunksize,  model_file,model_def_file, outputfolder, outfilenames_prefix, CROPPED_DIM)

    num_chuncks = ceil(numel(list_im)/chunksize);
    chunk_files = {};
    ind_ranges = [];
    for i=1:num_chuncks
        i
        startInd_chunk = (i-1)*chunksize+1;
        endInd_chunk = min(startInd_chunk+chunksize-1, numel(list_im));
        
        list_im_chunk = list_im(startInd_chunk:endInd_chunk);
        [scores_chunk,list_im_chunk1, problems_chunk] = matcaffe_batch_detail(list_im_chunk, model_def_file, model_file, CROPPED_DIM, 1);
        
        if(sum(problems_chunk)>0)
            x=1;
        end
        chunk_file_i = fullfile(outputfolder, [outfilenames_prefix, '_',num2str(i) ,'.mat']);
        save(chunk_file_i, 'list_im_chunk', 'scores_chunk', 'problems_chunk', '-v7.3');
        chunk_files{end+1} = chunk_file_i;
        ind_ranges = [ind_ranges; startInd_chunk, endInd_chunk];
    end