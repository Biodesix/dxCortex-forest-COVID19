function score = predict_single(obj, in_features)
    leaf_idx = 1;
    next_idx = 1;
    while next_idx > 0
        leaf_idx = next_idx;
        next_idx = obj.nodes{next_idx}.predict(in_features);
    end
    score = obj.nodes{leaf_idx}.final_score;
end
