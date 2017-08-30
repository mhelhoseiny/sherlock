% Author: Mohamed Elhoseiny, Summer 2015
function X_embedding_scores_i = GetX_Scores(X_embedding,T_embedding_i, metric )
    if(strcmp(metric, 'dot'))
        X_embedding_scores_i = X_embedding*T_embedding_i';
    elseif(strcmp(metric, 'cos'))
        X_embedding_scores_i = 1- pdist2(X_embedding, T_embedding_i, 'cos'); 
    elseif(strcmp(metric, 'euc'))
        eucDist = pdist2(X_embedding, T_embedding_i, 'euclidean');
        sigma=  median(eucDist(:));
        X_embedding_scores_i = exp(-eucDist.^2/(2*sigma^2 ) ); 
    else
        error('incorrect metric');
    end
    
