   
function [Is_FirstOrder, Is_SecondOrder, Is_ThirdOrder,T_unique_embeddings,ic, ia, T_unique_S_embeddings,ia_S, S_Group,T_unique_P_embeddings,ia_P, P_Group,T_unique_O_embeddings,ia_O, O_Group] = GetOrderInfo_sherlock_ECCV16(TestData) 
   
   tupleId2IndMap = containers.Map(TestData.unique_tuple_ids, [1: numel(TestData.unique_tuple_ids)] );
   
    T_embedding = TestData.unique_tuple_features(get_val_array(tupleId2IndMap,TestData.tuple_ids),:);

    [T_unique_embeddings,ia, ic] = unique(T_embedding,'rows');
   
   %tbl_im_membership = tabulate(ic);
   T_unique_embeddings_plus_mean = bsxfun(@plus,T_unique_embeddings,TestData.unique_tuple_features_mean);
  
   T_unique_embeddings_plus_mean_Sq = T_unique_embeddings_plus_mean.^2;
   T_unique_embeddings_plus_mean_S_equal0  = sum(T_unique_embeddings_plus_mean_Sq(:, 1:300),2)==0;
   T_unique_embeddings_plus_mean_P_equal0  = sum(T_unique_embeddings_plus_mean_Sq(:, 301:600),2)==0;
   T_unique_embeddings_plus_mean_O_equal0  = sum(T_unique_embeddings_plus_mean_Sq(:, 601:900),2)==0;
   
   %num_S_0 = sum(T_unique_embeddings_plus_mean_S_equal0)
   %num_P_0 = sum(T_unique_embeddings_plus_mean_P_equal0)
   %num_O_0 = sum(T_unique_embeddings_plus_mean_O_equal0)
   
   %K=100;
   %tic;
   
   Is_FirstOrder = T_unique_embeddings_plus_mean_S_equal0==0&T_unique_embeddings_plus_mean_P_equal0==1&T_unique_embeddings_plus_mean_O_equal0==1;
   
   Is_SecondOrder = T_unique_embeddings_plus_mean_S_equal0==0&T_unique_embeddings_plus_mean_P_equal0==0&T_unique_embeddings_plus_mean_O_equal0==1;
   
   
   Is_ThirdOrder = T_unique_embeddings_plus_mean_S_equal0==0&T_unique_embeddings_plus_mean_P_equal0==0&T_unique_embeddings_plus_mean_O_equal0==0;
 
   
   T_unique_embeddings_S = T_unique_embeddings(:,1:300);
   [T_unique_S_embeddings,ia_S, S_Group] = unique(T_unique_embeddings_S,'rows');
     
   T_unique_embeddings_P = T_unique_embeddings(:,301:600);
   [T_unique_P_embeddings,ia_P, P_Group] = unique(T_unique_embeddings_P,'rows');
   
   
   T_unique_embeddings_SP = T_unique_embeddings(:,601:900);
   [T_unique_O_embeddings,ia_O, O_Group] = unique(T_unique_embeddings_SP,'rows');
   
   
function valset = get_val_array(map_object, key_set)
    valset = zeros(1,numel(key_set));
    for i=1:numel(key_set)
        valset(i) = map_object(key_set(i));
    end