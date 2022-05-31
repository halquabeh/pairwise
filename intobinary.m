function ppLabel = intobinary(orgLabel)
% Cite @ SOLAM
    ppLabel = orgLabel;
    uLab = unique(orgLabel);
    uNum = length(uLab);
    if uNum > 2
        uSort = [1:uNum];
        %% positive class
        pIdx = [];
        for k = 1:floor(uNum/2)
            tI = find(orgLabel == uLab(uSort(1, k), 1));
            pIdx = [pIdx, tI'];
        end
        %% negative class
        nIdx = [];
        for k = (uNum - floor(uNum/2)):uNum
            tI = find(orgLabel == uLab(uSort(1, k), 1));
            nIdx = [nIdx, tI'];
        end
        
        %%post-processing
        ppLabel(pIdx) = 1;
        ppLabel(nIdx) = -1;
    end
    
end