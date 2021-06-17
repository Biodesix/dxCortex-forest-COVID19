function out_entropy = get_class_entropy(obj, in_idxs)
    % single entropy calc
    proportions = zeros(obj.nclasses,1);
    for idx = in_idxs
        proportions(obj.definitions(idx)) = proportions(obj.definitions(idx)) + 1;
    end
    out_entropy = 0;
    for iclass = 1:obj.nclasses
        this_p = proportions(iclass)/length(in_idxs);
        if this_p > 0
            out_entropy = out_entropy - this_p * log(this_p);
        end
    end
end