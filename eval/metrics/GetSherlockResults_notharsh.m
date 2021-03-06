% Author: Mohamed Elhoseiny, Summer 2015
        
function Result = GetSherlockResults_notharsh(X_embedding, T_embedding, TestData, metric)
 
% this is a correction since the computed embedding has the mean subtracted
% twice
 T_embedding = bsxfun(@plus,T_embedding,TestData.unique_tuple_features_mean);
 % now mean subtracted once
    %% Retrieve Images for a Tuple Metric
    [unique_test_ids, ia_test_ids, ic_testids] = unique(TestData.tuple_ids);
    
    
    [unique_embeddings,ia, ic] = unique(T_embedding,'rows');
    
    AUCs = [];
    APs = [];
    AP10s = [];
    AP100s = [];
    APs_v2 = [];
    gtTuplesforImages = cell(1, numel(TestData.tuple_ids));
    gtTuplesforImages_duplicate_id = cell(1, numel(TestData.tuple_ids));
    for i=1:size(unique_embeddings,1)
        T_embedding_i = T_embedding(ia(i),:);
        X_embedding_scores_i = GetX_Scores(X_embedding,T_embedding_i,metric);
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
        
         unique_tuple_ids_i = unique(TestData.tuple_ids(ic==i));
         if(size(unique_tuple_ids_i,1)>1)
             unique_tuple_ids_i = unique_tuple_ids_i';
         end
         X_embedding_gt_i = zeros(size(TestData.tuple_ids))==1;
         for k=1:numel(unique_tuple_ids_i)
              X_embedding_gt_i = X_embedding_gt_i|(TestData.tuple_ids==unique_tuple_ids_i(k));
         end
       
         
         
         
         tuple_i_pos_ind = find(X_embedding_gt_i);
         
         X_embedding_gt_i = X_embedding_gt_i*2-1;
         
         [MAP_i,  pn] = calcAP(X_embedding_scores_i, X_embedding_gt_i, 0);
         [MAP_v2_i] = calcAP_v2(X_embedding_scores_i, X_embedding_gt_i);
         [AUC_i ,xs, ys] = colAUC(X_embedding_scores_i,X_embedding_gt_i,'ROC');                        
         AP10s = [AP10s,  calcAP(X_embedding_scores_i, X_embedding_gt_i, 10)] ;
         AP100s = [AP100s, calcAP(X_embedding_scores_i, X_embedding_gt_i, 100)];
         AUCs = [AUCs, AUC_i];
         APs = [APs, MAP_i];
         APs_v2 = [APs_v2, MAP_v2_i];
         for j=1:numel(tuple_i_pos_ind)
             gtTuplesforImages{tuple_i_pos_ind(j)} = [gtTuplesforImages{tuple_i_pos_ind(j)}, unique_tuple_ids_i];
             gtTuplesforImages_duplicate_id{tuple_i_pos_ind(j)} = [gtTuplesforImages_duplicate_id{tuple_i_pos_ind(j)}, i*ones(size(unique_tuple_ids_i))]; 
            
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
    SimToAllTuples = GetX_Scores(X_embedding,T_embedding(ia_test_ids,:),metric);
    
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
        
        
        for i=1:size(X_embedding,1)

            trueTupleIds = gtTuplesforImages{i};
            trueTuplesforImages_duplicate_id_i = gtTuplesforImages_duplicate_id{i};
            
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
            
            for j_ind=1:numel(trueTupleIds)
                j = ismemberAll_ind_s_ind(j_ind);
                if(trueTupleIds_covered_flag(j)==0)
               
                        
                    duplicates_found=find(trueTuplesforImages_duplicate_id_i==trueTuplesforImages_duplicate_id_i(j));
                    for k=1:numel(duplicates_found)
                        trueTupleIds_covered_flag(duplicates_found(k))=1;
                    end
                    if(ismemberAll_ind(j)-offset<=1+numel(trueTupleIds)-1)
                        sum_isMember1 =sum_isMember1+1;
                    end
                    if(ismemberAll_ind(j)-offset<=5+numel(trueTupleIds)-1)
                        sum_isMember5 = sum_isMember5+1;
                    end
                    if(ismemberAll_ind(j)-offset<=10+numel(trueTupleIds)-1)
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
    
