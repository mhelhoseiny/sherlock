function matcaffe_batchFolder(imgListfilename, dstfilename, use_gpu, layer)

 [scores,list_im] = matcaffe_batchLayer(imgListfilename, use_gpu, layer);

 save(dstfilename, 'scores', 'list_im');
 
end