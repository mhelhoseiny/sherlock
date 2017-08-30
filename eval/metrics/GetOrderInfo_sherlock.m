   
function [num_S_0,num_P_0,num_O_0,Is_FirstOrder, Is_SecondOrder, Is_ThirdOrder,T_unique_embeddings,ic, ia] = GetOrderInfo_sherlock(TestData) 
   
 tupleId2IndMap = containers.Map(TestData.unique_tuple_ids, [1: numel(TestData.unique_tuple_ids)] );
   
    T_embedding = TestData.unique_tuple_features(get_val_array(tupleId2IndMap,TestData.tuple_ids),:);

    [T_unique_embeddings,ia, ic] = unique(T_embedding,'rows');
   
   %tbl_im_membership = tabulate(ic);
   T_unique_embeddings_plus_mean = bsxfun(@plus,T_unique_embeddings,TestData.unique_tuple_features_mean);
  
   T_unique_embeddings_plus_mean_Sq = T_unique_embeddings_plus_mean.^2;
   T_unique_embeddings_plus_mean_S_equal0  = sum(T_unique_embeddings_plus_mean_Sq(:, 1:300),2)==0;
   T_unique_embeddings_plus_mean_P_equal0  = sum(T_unique_embeddings_plus_mean_Sq(:, 301:600),2)==0;
   T_unique_embeddings_plus_mean_O_equal0  = sum(T_unique_embeddings_plus_mean_Sq(:, 601:900),2)==0;
   
   num_S_0 = sum(T_unique_embeddings_plus_mean_S_equal0)
   num_P_0 = sum(T_unique_embeddings_plus_mean_P_equal0)
   num_O_0 = sum(T_unique_embeddings_plus_mean_O_equal0)
   
   K=100;
   tic;
   
   Is_FirstOrder = T_unique_embeddings_plus_mean_S_equal0==0&T_unique_embeddings_plus_mean_P_equal0==1&T_unique_embeddings_plus_mean_O_equal0==1;
   
   Is_SecondOrder = T_unique_embeddings_plus_mean_S_equal0==0&T_unique_embeddings_plus_mean_P_equal0==0&T_unique_embeddings_plus_mean_O_equal0==1;
   
   
   Is_ThirdOrder = T_unique_embeddings_plus_mean_S_equal0==0&T_unique_embeddings_plus_mean_P_equal0==0&T_unique_embeddings_plus_mean_O_equal0==0;
 
   function valset = get_val_array(map_object, key_set)
    valset = zeros(1,numel(key_set));
    for i=1:numel(key_set)
        valset(i) = map_object(key_set(i));
    end