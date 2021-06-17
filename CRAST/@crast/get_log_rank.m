function [out_HR, was_flipped] = get_log_rank(obj, gr1_idxs, gr2_idxs)
    % get current log rank with defined labels
    if obj.use_total_HR
        gr1_idxs = [gr1_idxs, obj.current_gr1_idxs];
        gr2_idxs = [gr2_idxs, obj.current_gr2_idxs];
    end

    out_HR = logrank_survival_tree(obj.logrank_data(gr1_idxs,:), obj.logrank_data(gr2_idxs,:));
    was_flipped = false;
    if out_HR < 1
        was_flipped = true;
        out_HR = logrank_survival_tree(obj.logrank_data(gr2_idxs,:), obj.logrank_data(gr1_idxs,:));
    end
end
