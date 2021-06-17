function [gr1_idxs, gr2_idxs] = split_data_categorical(obj, in_idxs, feature_idx, in_grouping, was_flipped)
    % TODO: in_grouping needs to be a list of definitions that belong to gr1 in that grouping
    gr1_idxs = [];
    gr2_idxs = [];
    for sample_idx = in_idxs
        was_group1 = false;
        for def = in_grouping
            if obj.data(sample_idx).features(feature_idx) == def
                gr1_idxs = [gr1_idxs, sample_idx];
                was_group1 = true;
                break;
            end
        end
        if ~was_group1
            gr2_idxs = [gr2_idxs, sample_idx];
        end
        if was_flipped
            % reverse groups
            temp_idxs = gr1_idxs;
            gr1_idxs = gr2_idxs;
            gr2_idxs = temp_idxs;
        end
    end
end
