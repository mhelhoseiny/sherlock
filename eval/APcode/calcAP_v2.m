%%
%% Compute Average Precision for MED13 evaluation
%% @decisionScores: a vector of unordered decision scores
%% @gtlabels: a vector of binary ground truth with +1 as positive
%% @gammar: constant ratio
%% Formula: AP = 1/E * sum_i=1^i=E ( gammar* R_i/rho_i ), where E is the total 
%%          number of positive, R_i is the recall at the i-th true positive, 
%%          rho_i is the rank of 
function AP = calcAP_v2(decisionScores, gtlabels, gammar)

if nargin <= 2 
    gammar = 0.001;
end

[vals idx] = sort(decisionScores,'descend');

sumPrecision = 0;
nTP = numel(find(gtlabels==1));
nRtn = numel(idx);
nTPi = 0;
for ir=1:nRtn
    
    %% Only deal with true positive
    if ( gtlabels(idx(ir)) == 1 ) 
        
        nTPi = nTPi + 1;
        Ri =  nTPi/nTP;
        rohi = ir/nRtn;
        sumPrecision = sumPrecision + gammar*(Ri/rohi);
    end
end
AP = sumPrecision/nTP;

return;