function split_node(obj, parent_node_ID)
    obj.nsplits = obj.nsplits + 1;
    if obj.nsplits > obj.max_splits || obj.nodes{parent_node_ID}.depth >= obj.max_depth
        return
    end

    % sample features
    % feature_use_idxs = randsample(obj.rand_stream, obj.nfeatures, obj.nfeature_sample)';
    if obj.nfeatures == obj.nfeature_sample
        feature_use_idxs = randsample(obj.rand_stream, obj.nfeatures, obj.nfeature_sample)';
    else
        feature_use_idxs = [];
        order_idxs = randsample(obj.rand_stream, obj.nfeatures, obj.nfeatures)';
        n_features_sampled = 0;
        max_prob = max(obj.feature_probs);
        while length(feature_use_idxs) <= obj.nfeature_sample
            use_idx = -1;
            for iidx = 1:length(order_idxs)
                if obj.feature_probs(order_idxs(iidx)) > rand()*max_prob
                    feature_use_idxs = [feature_use_idxs, order_idxs(iidx)];
                    use_idx = iidx;
                    break;
                end
            end
            if use_idx > 0
                order_idxs(use_idx) = [];
            end
        end
    end

    if obj.force_logit_in_feature_sample
        % add 1 in for logit and then reshuffle
        feature_use_idxs = [1, feature_use_idxs];
        rand_idxs = randsample(obj.rand_stream, length(feature_use_idxs), length(feature_use_idxs));
        feature_use_idxs = feature_use_idxs(rand_idxs);
    end

    % get optimal split 
    optimal_feature_idx = -1;
    optimal_feature_cutoff = 0;
    optimal_was_categorical = false;
    optimal_grouping_idx = -1;

    % classification tree 
    max_info_gain = 0;

    % regression tree
    min_l2 = 1e12;

    % survival tree parameters
    max_log_rank = obj.min_HR;
    optimal_was_flipped = false;

    for ifeature = feature_use_idxs
        if obj.is_feature_categorical{ifeature}
            these_groupings = obj.categorical_feature_groupings{ifeature};
            ngroupings = length(these_groupings);
            grouping_idxs = [];
            if ngroupings > 50
                grouping_idxs = randsample(obj.rand_stream, ngroupings, 50)';
            else
                grouping_idxs = randsample(obj.rand_stream, ngroupings, ngroupings)';
            end

            for igrouping = grouping_idxs
                [gr1_idxs, gr2_idxs] = obj.split_data_categorical(obj.nodes{parent_node_ID}.sample_idxs, ifeature, these_groupings{igrouping}, false); 

                if length(gr1_idxs) < obj.min_leaf_size || length(gr2_idxs) < obj.min_leaf_size
                    continue;
                end

                if strcmp(obj.loss_function, 'HR')
                    [log_rank, was_flipped] = obj.get_log_rank(gr1_idxs, gr2_idxs);

                    if log_rank > max_log_rank
                        max_log_rank = log_rank;
                        optimal_feature_idx = ifeature;
                        optimal_grouping_idx = igrouping;
                        optimal_was_flipped = was_flipped;
                        optimal_was_categorical = true;
                    end
                elseif strcmp(obj.loss_function, 'l2')
                    [l2, was_flipped] = obj.get_l2(gr1_idxs, gr2_idxs);

                    if l2 < min_l2
                        min_l2 = l2;
                        optimal_feature_idx = ifeature;
                        optimal_grouping_idx = igrouping;
                        optimal_was_flipped = was_flipped;
                        optimal_was_categorical = true;
                    end
                elseif strcmp(obj.loss_function, 'RMST')
                    [log_rank, was_flipped] = obj.get_rmst_diff(gr1_idxs, gr2_idxs);

                    if log_rank > max_log_rank
                        max_log_rank = log_rank;
                        optimal_feature_idx = ifeature;
                        optimal_grouping_idx = igrouping;
                        optimal_was_flipped = was_flipped;
                        optimal_was_categorical = true;
                    end
                elseif strcmp(obj.loss_function, 'class_entropy') || strcmp(obj.loss_function, 'gini_index')
                    info_gain = obj.get_information_gain(gr1_idxs, gr2_idxs, parent_node_ID);
                    if info_gain > max_info_gain
                        max_info_gain = info_gain;
                        optimal_feature_idx = ifeature;
                        optimal_grouping_idx = igrouping;
                        optimal_was_categorical = true;
                    end
                end
            end
        else
            for feature_val = obj.feature_values{ifeature}
                [gr1_idxs, gr2_idxs] = obj.split_data(obj.nodes{parent_node_ID}.sample_idxs, ifeature, feature_val, false);

                if length(gr1_idxs) < obj.min_leaf_size || length(gr2_idxs) < obj.min_leaf_size
                    continue;
                end

                if strcmp(obj.loss_function, 'HR')
                    [log_rank, was_flipped] = obj.get_log_rank(gr1_idxs, gr2_idxs);

                    if log_rank > max_log_rank
                        max_log_rank = log_rank;
                        optimal_feature_idx = ifeature;
                        optimal_feature_cutoff = feature_val;
                        optimal_was_flipped = was_flipped;
                        optimal_was_categorical = false;
                    end
                elseif strcmp(obj.loss_function, 'l2')
                    [l2, was_flipped] = obj.get_l2(gr1_idxs, gr2_idxs);

                    if l2 < min_l2
                        min_l2 = l2;
                        optimal_feature_idx = ifeature;
                        optimal_feature_cutoff = feature_val;
                        optimal_was_flipped = was_flipped;
                        optimal_was_categorical = false;
                    end
                elseif strcmp(obj.loss_function, 'RMST')
                    [log_rank, was_flipped] = obj.get_rmst_diff(gr1_idxs, gr2_idxs);

                    if log_rank > max_log_rank
                        max_log_rank = log_rank;
                        optimal_feature_idx = ifeature;
                        optimal_feature_cutoff = feature_val;
                        optimal_was_flipped = was_flipped;
                        optimal_was_categorical = false;
                    end
                elseif strcmp(obj.loss_function, 'class_entropy') || strcmp(obj.loss_function, 'gini_index')
                    info_gain = obj.get_information_gain(gr1_idxs, gr2_idxs, parent_node_ID);
                    if info_gain > max_info_gain
                        max_info_gain = info_gain;
                        optimal_feature_idx = ifeature;
                        optimal_feature_cutoff = feature_val;
                        optimal_was_categorical = false;
                    end
                end
            end
        end
    end

    if optimal_feature_idx > 0

        % parent node feautre info
        obj.nodes{parent_node_ID}.feature_idx = optimal_feature_idx;
        obj.nodes{parent_node_ID}.was_flipped = optimal_was_flipped; % TODO: is this used anywhere else anymore??

        if optimal_was_categorical
            these_groupings = obj.categorical_feature_groupings{optimal_feature_idx};
            [gr1_idxs, gr2_idxs] = obj.split_data_categorical(obj.nodes{parent_node_ID}.sample_idxs, optimal_feature_idx, these_groupings{optimal_grouping_idx}, optimal_was_flipped); 
            obj.nodes{parent_node_ID}.grouping = these_groupings{optimal_grouping_idx};
        else
            [gr1_idxs, gr2_idxs] = obj.split_data(obj.nodes{parent_node_ID}.sample_idxs, optimal_feature_idx, optimal_feature_cutoff, optimal_was_flipped);
            obj.nodes{parent_node_ID}.cutoff = optimal_feature_cutoff;
        end

        % left node
        new_node_left = node();
        new_node_left.ID = length(obj.nodes) + 1;
        new_node_left.class = 0;
        new_node_left.depth = obj.nodes{parent_node_ID}.depth + 1;
        new_node_left.sample_idxs = gr1_idxs;
        obj.nodes{parent_node_ID}.child_left = new_node_left.ID;
        if strcmp(obj.loss_function, 'HR')
            new_node_left.score = obj.nodes{parent_node_ID}.score / max_log_rank;
        elseif strcmp(obj.loss_function, 'RMST')
            % new_node_left.score = obj.nodes{parent_node_ID}.score - max_log_rank;
            new_node_left.score = rmst(obj.logrank_data(gr1_idxs,1), obj.logrank_data(gr1_idxs,2), obj.rmst_start_time, obj.rmst_end_time);
        elseif strcmp(obj.loss_function, 'class_entropy') || strcmp(obj.loss_function, 'gini_index')
            new_node_left.entropy = obj.get_class_entropy(gr1_idxs);
            [this_class, this_score] = obj.get_class_plurality_fraction(gr1_idxs);
            new_node_left.class = obj.definitions_mask(this_class);
            if obj.use_binary_class_score
                if obj.definitions_mask(this_class) == 0;
                    new_node_left.score = 1.0 - this_score;
                else
                    new_node_left.score = this_score;
                end
            end
            if obj.use_running_binary_class_score
                if obj.definitions_mask(this_class) == 0;
                    new_node_left.score = obj.nodes{parent_node_ID}.score * this_score;
                else
                    new_node_left.score = obj.nodes{parent_node_ID}.score / this_score;
                end
            end
        elseif strcmp(obj.loss_function, 'ROC') || strcmp(obj.loss_function, 'running_ROC')
            % TODO: why is default class gr0?  might want to check on this for survival trees
            new_node_left.class = 1;
        elseif strcmp(obj.loss_function, 'l2') 
            new_node_left.score = obj.get_mean_pred(gr1_idxs);
        end
        obj.nodes{end+1} = new_node_left;

        % right node
        new_node_right = node();
        new_node_right.ID = length(obj.nodes) + 1;
        new_node_right.class = 1;
        new_node_right.depth = obj.nodes{parent_node_ID}.depth + 1;
        new_node_right.sample_idxs = gr2_idxs;
        obj.nodes{parent_node_ID}.child_right = new_node_right.ID;
        if strcmp(obj.loss_function, 'HR')
            new_node_right.score = obj.nodes{parent_node_ID}.score * max_log_rank;
        elseif strcmp(obj.loss_function, 'RMST')
            % new_node_right.score = obj.nodes{parent_node_ID}.score + max_log_rank;
            new_node_right.score = rmst(obj.logrank_data(gr2_idxs,1), obj.logrank_data(gr2_idxs,2), obj.rmst_start_time, obj.rmst_end_time);
        elseif strcmp(obj.loss_function, 'class_entropy') || strcmp(obj.loss_function, 'gini_index')
            new_node_right.entropy = obj.get_class_entropy(gr2_idxs);
            [this_class, this_score] = obj.get_class_plurality_fraction(gr2_idxs);
            new_node_right.class = obj.definitions_mask(this_class);
            if obj.use_binary_class_score
                if obj.definitions_mask(this_class) == 0;
                    new_node_right.score = 1.0 - this_score;
                else
                    new_node_right.score = this_score;
                end
            end
            if obj.use_running_binary_class_score
                if obj.definitions_mask(this_class) == 0;
                    new_node_right.score = obj.nodes{parent_node_ID}.score * this_score;
                else
                    new_node_right.score = obj.nodes{parent_node_ID}.score / this_score;
                end
            end
        elseif strcmp(obj.loss_function, 'ROC') || strcmp(obj.loss_function, 'running_ROC')
            % TODO: why is default class gr1?  might want to check on this for survival trees
            new_node_right.class = 0;
        elseif strcmp(obj.loss_function, 'l2') 
            new_node_right.score = obj.get_mean_pred(gr2_idxs);
        end
        obj.nodes{end+1} = new_node_right;

        if obj.use_total_HR
            obj.min_HR = max_log_rank;
            if obj.min_leaf_size > 5
                obj.min_leaf_size = floor(obj.min_leaf_size*obj.min_leaf_size_decay_factor);
            end
            obj.current_gr1_idxs = [obj.current_gr1_idxs, gr1_idxs];
            obj.current_gr2_idxs = [obj.current_gr2_idxs, gr2_idxs];
            if optimal_was_flipped
                temp_idxs = obj.current_gr1_idxs;
                obj.current_gr1_idxs = obj.current_gr2_idxs;
                obj.current_gr2_idxs = obj.current_gr1_idxs;
                for inode = 1:length(obj.nodes)
                    if obj.nodes{inode}.class > 0.5
                        obj.nodes{inode}.class = 0;
                    else
                        obj.nodes{inode}.class = 1;
                    end
                end
            end
        end

        split_node(obj, new_node_left.ID);
        split_node(obj, new_node_right.ID);
    end
end
