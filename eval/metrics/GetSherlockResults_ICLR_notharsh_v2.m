% Author: Mohamed Elhoseiny, Summer 2015
        
function Result = GetSherlockResults_ICLR_notharsh_v2(X_embedding, T_embedding, XT_err, TestData,flag_flexible_map)
 
% this is a correction since the computed embedding has the mean subtracted
% twice
 %T_embedding = bsxfun(@plus,T_embedding,TestData.unique_tuple_features_mean);
 % now mean subtracted once
    %% Retrieve Images for a Tuple Metric
    if(~exist('flag_flexible_map', 'var'))
        
        flag_flexible_map = false;
    end
    [unique_test_ids, ia_test_ids, ic_testids] = unique(TestData.tuple_ids);
    
    
    [unique_embeddings,ia, ic] = unique(T_embedding,'rows');
    
    
    
   T_unique_embeddings_plus_mean = bsxfun(@plus,unique_embeddings,TestData.unique_tuple_features_mean);
  
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
   
   ThirdOrderGroups = ia;
      
   T_unique_embeddings_S = unique_embeddings(:,1:300);
   [T_unique_S_embeddings,ia_S, ic_S] = unique(T_unique_embeddings_S,'rows');
     
   T_unique_embeddings_SP = unique_embeddings(:,1:600);
   [T_unique_SP_embeddings,ia_PO, ic_PO] = unique(T_unique_embeddings_SP,'rows');
   
   FirstOrderGroups = ic_S;
   SecondOrderGroups =ic_PO;
   
    
    AUCs = [];
    APs = [];
    AP10s = [];
    AP100s = [];
    APs_v2 = [];
    gtTuplesforImages = cell(1, numel(TestData.tuple_ids));
    gtTuplesforImages_duplicate_id = cell(1, numel(TestData.tuple_ids));
    assert(size(unique_embeddings,2)==900);
    
    SimToAllTuples = zeros(size(X_embedding,1), numel(ia_test_ids));
    flagEX=true;
    
    %Result.relatedtuple_ids = zeros(1,size(unique_embeddings,1));
    
    Result.numrelatedtuple_ids =  zeros(1,size(unique_embeddings,1));
    Result.relatedtuple_ids = cell(1,size(unique_embeddings,1));
        
        
    for i=1:size(unique_embeddings,1)
        
      
        
        if(Is_ThirdOrder(i))
            T_embedding_i = T_embedding(ia(i),:);
            %X_embedding_scores_i = GetX_Scores(X_embedding,T_embedding_i,metric);
            group_i = (ThirdOrderGroups == ThirdOrderGroups(i));
            related_thirdorder_fact = ThirdOrderGroups == ThirdOrderGroups(i);
            related_secondorder_fact = Is_SecondOrder&(SecondOrderGroups==SecondOrderGroups(i));
            related_firstorder_fact =Is_FirstOrder&(FirstOrderGroups==FirstOrderGroups(i));
            tuple_ids_thd_Exact = unique(TestData.tuple_ids(ic==ic(ia(related_thirdorder_fact))));
            ia_snd = ia(related_secondorder_fact);
            if(~flagEX||isempty(ia_snd))
                tuple_ids_second_Exact = [];
            else
                tuple_ids_second_Exact = unique(TestData.tuple_ids(ic==ic(ia_snd)));
            end
            
             ia_fst = ia(related_firstorder_fact);
            if(~flagEX||isempty(ia_fst))
                tuple_ids_first_Exact = [];  
            else
                tuple_ids_first_Exact = unique(TestData.tuple_ids(ic==ic(ia_fst)));
            end
            tuple_ids_Exact = [tuple_ids_thd_Exact,tuple_ids_second_Exact, tuple_ids_first_Exact];
            tuple_ids_duplicates = [];
            if(numel(tuple_ids_thd_Exact)>0)
                tuple_ids_duplicates = [tuple_ids_duplicates, find(related_thirdorder_fact)*ones(1,numel(tuple_ids_thd_Exact))];
            end
            
            if(flagEX&&numel(tuple_ids_second_Exact)>0)
                tuple_ids_duplicates = [tuple_ids_duplicates, find(related_secondorder_fact)*ones(1, numel(tuple_ids_second_Exact))];
            end
            
            if(flagEX&&numel(related_firstorder_fact)>0)
                tuple_ids_duplicates = [tuple_ids_duplicates, find(related_firstorder_fact)*ones(1, numel(tuple_ids_first_Exact))];
            end
      
        elseif(Is_SecondOrder(i))
            T_embedding_i = T_embedding(ia(i),1:600);
            %X_embedding_scores_i = GetX_Scores(X_embedding(:,1:600),T_embedding_i,metric);
            group_i = (SecondOrderGroups == SecondOrderGroups(i));
            related_thirdorder_fact = ThirdOrderGroups == ThirdOrderGroups(i);
             related_firstorder_fact =Is_FirstOrder&(FirstOrderGroups==FirstOrderGroups(i));
           
            tuple_ids_thd_Exact = unique(TestData.tuple_ids(ic==ic(ia(related_thirdorder_fact))));
            
            ia_fst = ia(related_firstorder_fact);
            
             if(~flagEX||isempty(ia_fst))
                tuple_ids_first_Exact = [];  
             else
                tuple_ids_first_Exact = unique(TestData.tuple_ids(ic==ic(ia_fst)));
             end
            
            tuple_ids_Exact = [tuple_ids_thd_Exact,tuple_ids_first_Exact];
            
              tuple_ids_duplicates = [];
            if(numel(tuple_ids_thd_Exact)>0)
                tuple_ids_duplicates = [tuple_ids_duplicates, find(related_thirdorder_fact)*ones(1,numel(tuple_ids_thd_Exact))];
            end
            
            if(flagEX&&numel(tuple_ids_first_Exact)>0)
                tuple_ids_duplicates = [tuple_ids_duplicates, find(related_firstorder_fact)*ones(1, numel(tuple_ids_first_Exact))];
            end
            
        else
            T_embedding_i = T_embedding(ia(i),1:300);
            %X_embedding_scores_i = GetX_Scores(X_embedding(:,1:300),T_embedding_i,metric);
            related_thirdorder_fact = ThirdOrderGroups == ThirdOrderGroups(i);
            group_i = (FirstOrderGroups == FirstOrderGroups(i));
            tuple_ids_Exact = unique(TestData.tuple_ids(ic==ic(ia(ThirdOrderGroups == ThirdOrderGroups(i)))));
            tuple_ids_thd_Exact = tuple_ids_Exact;
            tuple_ids_duplicates = [find(related_thirdorder_fact)*ones(1,numel(tuple_ids_Exact))];
        end
        Result.numrelatedtuple_ids(i) = numel(tuple_ids_Exact);
        Result.relatedtuple_ids{i} = (tuple_ids_Exact);
        tuple_ids_thorder = unique(TestData.tuple_ids(ic==ic(ia(ThirdOrderGroups == ThirdOrderGroups(i)))));
        X_embedding_scores_i = -XT_err(:,TestData.unique_tuple_ids==tuple_ids_thorder(1));
        
        [~, tuple_ind_i] = ismember(tuple_ids_thorder,unique_test_ids);
        for kind=1:numel(tuple_ind_i)
            SimToAllTuples(:,tuple_ind_i(kind)) =  X_embedding_scores_i;
        end
        assert(numel(tuple_ids_Exact)==numel(tuple_ids_duplicates));
%         if(strcmp(metric, 'dot'))
%            
%             X_embedding_scores_i = X_embedding*T_embedding_i';
%         elseif(strcmp(metric, 'cos'))
%             X_embedding_scores_i = 1- pdist2(X_embedding, T_embedding_i, 'cos'); 
%         elseif(strcmp(metric, 'euc'))
%             eucDist = pdist2(X_embedding, T_embedding_i, 'euclidean');
%             sigma=  median(eucDist);
%             X_embedding_scores_i = exp(-eucDist.^2/(2*sigma^2 ) ); 
%         else
%             error('incorrect metric');
%         end

         %tuple_ids_mask = ic==i;

         if( flag_flexible_map)
             pos_id_ind = find(group_i); % uncomment for more flexible mAP
         else
         pos_id_ind = find(related_thirdorder_fact);
         
         end
         
         tuple_ids_mask = (zeros(size(ic))==1);
         for i11 = 1:numel(pos_id_ind)
             tuple_ids_mask =tuple_ids_mask| ic==ic(ia(pos_id_ind(i11)));
         end
         unique_tuple_ids_i = unique(TestData.tuple_ids(tuple_ids_mask));
         if(size(unique_tuple_ids_i,1)>1)
             unique_tuple_ids_i = unique_tuple_ids_i';
         end
         X_embedding_gt_i = zeros(size(TestData.tuple_ids))==1;
         
         
         for k=1:numel(unique_tuple_ids_i)
              X_embedding_gt_i = X_embedding_gt_i|(TestData.tuple_ids==unique_tuple_ids_i(k));
         end
       
         
         
         
        
         
         X_embedding_gt_i = X_embedding_gt_i*2-1;
         
         [MAP_i,  pn] = calcAP(X_embedding_scores_i, X_embedding_gt_i, 0);
         [MAP_v2_i] = calcAP_v2(X_embedding_scores_i, X_embedding_gt_i);
         [AUC_i ,xs, ys] = colAUC(X_embedding_scores_i,X_embedding_gt_i,'ROC');
         AP10s = [AP10s,  calcAP(X_embedding_scores_i, X_embedding_gt_i, 10)] ;
         AP100s = [AP100s, calcAP(X_embedding_scores_i, X_embedding_gt_i, 100)];
         AUCs = [AUCs, AUC_i];
         APs = [APs, MAP_i];
         APs_v2 = [APs_v2, MAP_v2_i];
         
         X_embedding_gt_forKExi = zeros(size(TestData.tuple_ids))==1;
         for k=1:numel(tuple_ids_thd_Exact)
              X_embedding_gt_forKExi = X_embedding_gt_forKExi|(TestData.tuple_ids==tuple_ids_thd_Exact(k));
         end
         
             %tuple_ids_Exact = [tuple_ids_thd_Exact,tuple_ids_second_Exact, tuple_ids_first_Exact];
           % tuple_ids_duplicates 
         tuple_i_pos_ind = find(X_embedding_gt_forKExi);
         for j=1:numel(tuple_i_pos_ind)
             gtTuplesforImages{tuple_i_pos_ind(j)} = [gtTuplesforImages{tuple_i_pos_ind(j)}, tuple_ids_Exact];
             gtTuplesforImages_duplicate_id{tuple_i_pos_ind(j)} = [gtTuplesforImages_duplicate_id{tuple_i_pos_ind(j)}, tuple_ids_duplicates]; 
            
         end
         
         
    end
    Result.AUCs = AUCs;
    Result.APs = APs;
    Result.APs_v2 = APs_v2;
    Result.AP10s  = AP10s; 
    Result.AP100s = AP100s;
    
    
    Result.mAUC = mean(AUCs);
    Result.mAP = mean(APs);
    Result.mAP_v2 = mean(APs_v2);
    Result.mAP10s = mean(AP10s);
    Result.mAP100s = mean(AP100s);
    
    Result.tuple_ids = unique_test_ids;
    
    %% Metric for Knowledge extractions
    
    %SimToAllTuples = X_embedding*T_embedding(ia,:)';
    %SimToAllTuples = GetX_Scores(X_embedding,T_embedding(ia_test_ids,:),metric);
    
  
    
    [sSimToAllTuples,SimToAllTuples_ind]  = sort(SimToAllTuples, 2, 'descend');
    
    %K=10;
    if(true)
        avgKnowledgeDetRatio_K1  = 0;
        avgKnowledgeDetRatio_K5  = 0;
        avgKnowledgeDetRatio_K10 = 0;
        avgKnowledgeDetRatio_MRR= 0;  
        
        Result.KnowledgeDetRatios_K10 = [];
        Result.KnowledgeDetRatios_K5 = [];
        Result.KnowledgeDetRatios_K1 = [];
        Result.KnowledgeDetRatios_MRR  = [];
        Result.numTrueTuples = zeros(1, size(X_embedding,1));
        Result.Is_FirstOrder=Is_FirstOrder;
        Result.Is_SecondOrder  = Is_SecondOrder;
        Result.Is_ThirdOrder  = Is_ThirdOrder;
        
        for i=1:size(X_embedding,1)

            assert(numel(gtTuplesforImages{i})==numel(gtTuplesforImages_duplicate_id{i}))
            [trueTupleIds,ia_gt, ic_gt]  = unique(gtTuplesforImages{i});
            Result.numTrueTuples(i) = numel(trueTupleIds);
            trueTuplesforImages_duplicate_id_i = gtTuplesforImages_duplicate_id{i};
            trueTuplesforImages_duplicate_id_i = trueTuplesforImages_duplicate_id_i(ia_gt);
            
            [unique_emb_tuple_ids, em_ia , em_ic ] = unique(trueTuplesforImages_duplicate_id_i);
            numduplicates = numel(trueTupleIds)-numel(unique_emb_tuple_ids);
            ranked_retrieved_ids =unique_test_ids(SimToAllTuples_ind(i,:));
            [ismemberAll,ismemberAll_ind]  = ismember(trueTupleIds, ranked_retrieved_ids);
            trueTupleIds_covered_flag = zeros(1, numel(trueTupleIds));
            
            MRR_i = 0;
            [ismemberAll_ind_s,ismemberAll_ind_s_ind] = sort(ismemberAll_ind);
            offset=0;
            numAll = 0;
            numAllOthers = 0;
            sum_isMember10 = 0;
            sum_isMember5 = 0;
            sum_isMember1 = 0;
            
            num_of_unique_facts = numel(unique(trueTuplesforImages_duplicate_id_i));
            
            for j_ind=1:numel(trueTupleIds)
                j = ismemberAll_ind_s_ind(j_ind);
                if(trueTupleIds_covered_flag(j)==0)
               
                        
                    duplicates_found=find(trueTuplesforImages_duplicate_id_i==trueTuplesforImages_duplicate_id_i(j));
                    for k=1:numel(duplicates_found)
                        trueTupleIds_covered_flag(duplicates_found(k))=1;
                    end
                    if(ismemberAll_ind(j)-offset<=1+num_of_unique_facts-1)
                        sum_isMember1 =sum_isMember1+1;
                    end
                    if(ismemberAll_ind(j)-offset<=5+num_of_unique_facts-1)
                        sum_isMember5 = sum_isMember5+1;
                    end
                    if(ismemberAll_ind(j)-offset<=10+num_of_unique_facts-1)
                        sum_isMember10 = sum_isMember10+1;
                    end
                    numAllOthers = numAllOthers+1;
                        
                    if(ismemberAll(j))
                     
                    
                        MRR_i = MRR_i+1/(ismemberAll_ind(j)-offset);
                        numAll = numAll+1;
                        offset = offset+1;
                    end
                else
                    offset = offset+1;
                end
            end
            if(numAll==0)
                MRR_i =0;
            else
                MRR_i =MRR_i/numAll;
            end
            %ismember10= (ismemberAll_ind<=(10+numel(trueTupleIds)-1+numduplicates));
            %ismember5= (ismemberAll_ind<=(5+numel(trueTupleIds)-1+numduplicates));
            %ismember1 = (ismemberAll_ind<=(1+numel(trueTupleIds)-1+numduplicates));

                
            %sum_isMember10 = sum((ismemberAll_ind<=(10+numel(trueTupleIds)-1+numduplicates)));
            %sum_isMember5 =  sum((ismemberAll_ind<=(5+numel(trueTupleIds)-1+numduplicates)));
            %sum_isMember1=  sum((ismemberAll_ind<=(1+numel(trueTupleIds)-1+numduplicates)));
            
            detRatio10_i = sum_isMember10/numAllOthers;
            detRatio5_i = sum_isMember5/numAllOthers;
            detRatio1_i = sum_isMember1/numAllOthers; 

            Result.KnowledgeDetRatios_K10 = [Result.KnowledgeDetRatios_K10, detRatio10_i];
            Result.KnowledgeDetRatios_K5 = [Result.KnowledgeDetRatios_K5,detRatio5_i ];
            Result.KnowledgeDetRatios_K1 = [Result.KnowledgeDetRatios_K1,detRatio1_i ];
            Result.KnowledgeDetRatios_MRR  = [Result.KnowledgeDetRatios_MRR, MRR_i];

            avgKnowledgeDetRatio_K10 = avgKnowledgeDetRatio_K10+detRatio10_i;
            avgKnowledgeDetRatio_K5 = avgKnowledgeDetRatio_K5+detRatio5_i;
            avgKnowledgeDetRatio_K1 = avgKnowledgeDetRatio_K1+detRatio1_i;
            avgKnowledgeDetRatio_MRR =avgKnowledgeDetRatio_MRR+ MRR_i;
        end
        avgKnowledgeDetRatio_K10 =avgKnowledgeDetRatio_K10/size(X_embedding,1);
        avgKnowledgeDetRatio_K5 =avgKnowledgeDetRatio_K5/size(X_embedding,1);
        avgKnowledgeDetRatio_K1 =avgKnowledgeDetRatio_K1/size(X_embedding,1);
        avgKnowledgeDetRatio_MRR =avgKnowledgeDetRatio_MRR/size(X_embedding,1);
        
        
        Result.KnowledgeMeanDetRatio_K10 = avgKnowledgeDetRatio_K10;
        Result.KnowledgeMeanDetRatio_K5 = avgKnowledgeDetRatio_K5;
        Result.KnowledgeMeanDetRatio_K1 = avgKnowledgeDetRatio_K1;
        Result.KnowledgeMeanDetRatio_MRR  = avgKnowledgeDetRatio_MRR;
        
    end
    
