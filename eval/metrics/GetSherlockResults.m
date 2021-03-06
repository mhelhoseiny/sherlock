% Author: Mohamed Elhoseiny, Summer 2015
        
function Result = GetSherlockResults(X_embedding, T_embedding, TestData, metric)
    
% this is a correction since the computed embedding has the mean subtracted
% twice
 T_embedding = bsxfun(@plus,T_embedding,TestData.unique_tuple_features_mean);
 % now mean subtracted once
    %% Retrieve Images for a Tuple Metric
    [unique_test_ids, ia, ic] = unique(TestData.tuple_ids);
    AUCs = [];
    APs = [];
    AP10s = [];
    AP100s = [];
    APs_v2 = [];
    gtTuplesforImages = cell(1, numel(TestData.tuple_ids));
    for i=1:numel(unique_test_ids)
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
        
         X_embedding_gt_i =  TestData.tuple_ids==unique_test_ids(i);
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
             gtTuplesforImages{tuple_i_pos_ind(j)} = [gtTuplesforImages{tuple_i_pos_ind(j)}, unique_test_ids(i)]; 
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
    SimToAllTuples = GetX_Scores(X_embedding,T_embedding(ia,:),metric);
    
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

            ranked_retrieved_ids =unique_test_ids(SimToAllTuples_ind(i,:));
            [ismemberAll,ismemberAll_ind]  = ismember(trueTupleIds, ranked_retrieved_ids);
            
            MRR_i = 0;
            [ismemberAll_ind_s,ismemberAll_ind_s_ind] = sort(ismemberAll_ind);
            offset=0;
            for j_ind=1:numel(trueTupleIds)
                j = ismemberAll_ind_s_ind(j_ind);
                if(ismemberAll(j))
                    MRR_i = MRR_i+1/(ismemberAll_ind(j)-offset);
                    offset = offset+1;
                end
            end
            
            ismember10= (ismemberAll_ind<=(10+numel(trueTupleIds)-1));
            ismember5= (ismemberAll_ind<=(5+numel(trueTupleIds)-1));
            ismember1 = (ismemberAll_ind<=(1+numel(trueTupleIds)-1));

            detRatio10_i = sum(ismember10)/numel(trueTupleIds);
            detRatio5_i = sum(ismember5)/numel(trueTupleIds);
            detRatio1_i = sum(ismember1)/numel(trueTupleIds); 

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
    
