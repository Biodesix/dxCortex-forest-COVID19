function [out_diff, was_flipped] = get_rmst_diff(obj, gr1_idxs, gr2_idxs)
    % get current log rank with defined labels
    if obj.use_total_HR
        gr1_idxs = [gr1_idxs, obj.current_gr1_idxs];
        gr2_idxs = [gr2_idxs, obj.current_gr2_idxs];
    end

    rmst1 = rmst(obj.logrank_data(gr1_idxs,1), obj.logrank_data(gr1_idxs,2), obj.rmst_start_time, obj.rmst_end_time);
    rmst2 = rmst(obj.logrank_data(gr2_idxs,1), obj.logrank_data(gr2_idxs,2), obj.rmst_start_time, obj.rmst_end_time);
    out_diff = rmst2 - rmst1;

    was_flipped = false;
    if out_diff < 0
        was_flipped = true;
        out_diff = -1*out_diff;
    end
end
