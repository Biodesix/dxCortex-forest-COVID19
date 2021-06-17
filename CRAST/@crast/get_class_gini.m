function out_entropy = get_class_entropy(obj, in_idxs)
    % single entropy calc
    proportions = zeros(obj.nclasses,1);
    for idx = in_idxs
        proportions(obj.definitions(idx)) = proportions(obj.definitions(idx)) + 1;
    end
    out_entropy = 1;
    for iclass = 1:obj.nclasses
        this_p = proportions(iclass)/length(in_idxs);
        out_entropy = out_entropy - this_p * this_p;
    end
end