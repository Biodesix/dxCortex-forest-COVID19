function set_parameters(obj)
    % TODO: set these with paired argurments
    obj.max_splits = 1e6;
    obj.min_leaf_size = 5;
    obj.max_depth = 20;
    obj.nfeature_sample = floor(sqrt(obj.nfeatures));
    obj.force_logit_in_feature_sample = false;
    obj.nfeature_values = 10; % number of quantiles to calculate for possible feature values

    % obj.loss_function = 'class_entropy';

    % survival tree specific parameters
    obj.loss_function = 'HR';
    obj.leaf_score = 'percentile'; 
    obj.min_HR = 1.0;
    % leaf score options are:
    % event: fraction of events in the terminal leaf
    % left-right: number of left turns vs right turns down the tree to get to the terminal leaf
    % HR: 0 if leaf running HR score is less than 1, 1 otherwise
    % percentile: percentile of the HR score w.r.t. all possible scores in the tree
    % n.b. leaf scores are overwritten with the corresponding logit described above for all cases except for HR
    obj.use_total_HR = false; % experimental, suggested to not use 'true', if true, should probably have a highre min_HR (maybe around 2)
end
