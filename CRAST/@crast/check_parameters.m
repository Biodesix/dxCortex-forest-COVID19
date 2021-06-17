function check_parameters(obj)
    if ~isnumeric(obj.max_splits)
        if obj.max_splits < 1
            error('invalid parameter')
        end
    end
    if ~isnumeric(obj.min_leaf_size)
        if obj.min_leaf_size < 1
            error('invalid parameter')
        end
    end
    if ~isnumeric(obj.max_depth)
        if obj.max_depth < 1
            error('invalid parameter')
        end
    end
    if ~isnumeric(obj.nfeature_sample)
        if obj.nfeature_sample < 1
            error('invalid parameter')
        end
    end
    if ~islogical(obj.force_logit_in_feature_sample)
        error('invalid parameter')
    end
    if ~islogical(obj.use_weighted_ave_info_gain)
        error('invalid parameter')
    end
    if ~isnumeric(obj.nfeature_values)
        if obj.nfeature_values < 1
            error('invalid parameter')
        end
    end
    if ~(strcmp(obj.leaf_score, 'left-right') || strcmp(obj.leaf_score, 'HR') || strcmp(obj.leaf_score, 'percentile_all') || strcmp(obj.leaf_score, 'percentile_terminal') || strcmp(obj.leaf_score, 'event'))
        error('invalid parameter')
    end
    if ~(strcmp(obj.feature_value_sampling, 'percentile') || strcmp(obj.feature_value_sampling, 'uniform'))
        error('invalid parameter')
    end
    if ~(strcmp(obj.loss_function, 'HR') || strcmp(obj.loss_function, 'class_entropy') || strcmp(obj.loss_function, 'gini_index') || strcmp(obj.loss_function, 'RMST') || strcmp(obj.loss_function, 'ROC') || strcmp(obj.loss_function, 'running_ROC') || strcmp(obj.loss_function, 'l2'))
        error('invalid parameter')
    end
    if ~isnumeric(obj.min_HR)
        if obj.min_HR < 1
            error('invalid parameter')
        end
    end
    if ~islogical(obj.use_total_HR)
        error('invalid parameter')
    end
end