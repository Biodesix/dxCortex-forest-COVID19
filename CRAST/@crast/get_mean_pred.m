function out_pred = get_mean_pred(in_idxs)
    out_pred = 0;
    for idx = in_idxs
        out_pred = out_pred + obj.data(idx).definition;
    end
    out_pred = out_pred/length(in_idxs);
end