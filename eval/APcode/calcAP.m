%%
%% Compute Average Precision
%% @decisionScores: classification confidence
%% @gtlabels: ground truth labels
function [ap pn] = calcAP(decisionScores, gtlabels,nRetrieval)

if (nRetrieval == 0) nRetrieval = numel(gtlabels); end

[vals idx] = sort(decisionScores,'descend');
precisions = [];
nTP = 0; %% number of true positives
for i=1:nRetrieval%numel(gtlabels)
   
    if(gtlabels(idx(i))==1)
        nTP = nTP + 1;
        precisions = [precisions nTP/i];
    end
end
if nTP ~= 0 
    ap = sum(precisions)/nTP;
    pn = precisions(end);
else
    ap = 0;
    pn = 0;
end
