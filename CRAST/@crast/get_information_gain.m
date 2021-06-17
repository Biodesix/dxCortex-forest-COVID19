function out_ig = get_information_gain(obj, gr1_idxs, gr2_idxs, parent_node_ID)
    % information gain from a split
    if strcmp(obj.loss_function, 'class_entropy')
        s1 = obj.get_class_entropy(gr1_idxs);
        s2 = obj.get_class_entropy(gr2_idxs);
    elseif strcmp(obj.loss_function, 'gini_index')
        s1 = obj.get_class_gini(gr1_idxs);
        s2 = obj.get_class_gini(gr2_idxs);
    end
    if obj.use_weighted_ave_info_gain
        n1 = length(gr1_idxs);
        n2 = length(gr2_idxs);
        ntot = n1 + n2;
        s_ave = n1/ntot*s1 + n2/ntot*s2;
    else
        s_ave = (s1 + s2)/2;
    end
    sp = obj.nodes{parent_node_ID}.entropy;
    out_ig = sp - s_ave;
end