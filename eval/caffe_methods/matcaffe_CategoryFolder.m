function matcaffe_CategoryFolder(catsFolder, dstFolder, layer)
    dirList = dir(catsFolder);
    
    for i = 1:numel(dirList)
        diri =  dirList(i);
        if(strcmp(diri.name, '.')==0&&strcmp(diri.name, '..')==0&&strcmp(diri.name(1), '.')==0)
            %diri.name
                 diri.name
                 dstmatfilename = diri.name;
                 dstfilemat = fullfile(dstFolder, dstmatfilename);
               
                 if(~exist([dstfilemat, '.mat'], 'file'))
                     imnamesfname = fullfile(catsFolder,diri.name,'imnames.txt');
                     cmd = ['find ', fullfile(catsFolder,diri.name, '*.jpg'), '  -type f -exec echo {} \; > ',imnamesfname ];
                     system(cmd);

                     %dstfilemat = fullfile(dstFolder, dstmatfilename);
                     matcaffe_batchFolder(imnamesfname, [dstfilemat, '.mat'], 1, layer);
                 else
                     fprintf([ dstfilemat, '.mat already exists', '\n']);
                 end
                 
            
            %pause;
        end
    end

end