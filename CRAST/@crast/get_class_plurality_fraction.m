function [out_class, out_p] = get_class_plurality_fraction(obj, in_idxs)
    proportions = zeros(obj.nclasses,1);
    for idx = in_idxs
        proportions(obj.definitions(idx)) = proportions(obj.definitions(idx)) + 1;
    end
    out_p = 0;
    out_class = -1;
    idxs = randsample(obj.nclasses, obj.nclasses)'; % if the classes are equally proportional, this will assign a random class
    for iclass = idxs
        this_p = proportions(iclass)/length(in_idxs);
        if this_p > out_p
            out_p = this_p;
            out_class = iclass;
        end
    end
end
