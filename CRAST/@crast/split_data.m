function [gr1_idxs, gr2_idxs] = split_data(obj, in_idxs, feature_idx, threshold, was_flipped)
    gr1_idxs = [];
    gr2_idxs = [];
    for sample_idx = in_idxs
        if was_flipped
            if obj.data(sample_idx).features(feature_idx) >= threshold
                gr1_idxs = [gr1_idxs, sample_idx];
            else
                gr2_idxs = [gr2_idxs, sample_idx];
            end
        else
            if obj.data(sample_idx).features(feature_idx) < threshold
                gr1_idxs = [gr1_idxs, sample_idx];
            else
                gr2_idxs = [gr2_idxs, sample_idx];
            end
        end
    end
end
