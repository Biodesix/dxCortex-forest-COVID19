function [out_l2, was_flipped] = get_l2(obj, gr1_idxs, gr2_idxs)
    was_flipped = false;
    pred1 = obj.get_mean_pred(gr1_idxs);
    pred2 = obj.get_mean_pred(gr2_idxs);

    err1 = 0;
    for idx = gr1_idxs
        err1 = err1 + (pred1 - obj.data(idx).definition) * (pred1 - obj.data(idx).definition);
    end

    err2 = 0;
    for idx = gr2_idxs
        err2 = err2 + (pred2 - obj.data(idx).definition) * (pred2 - obj.data(idx).definition);
    end

    ng1 = length(gr1_idxs);
    ng2 = length(gr2_idxs);

    out_l2 = ng1/(ng1+ng2)*err1 + ng2/(ng1+ng2)*err2;
end