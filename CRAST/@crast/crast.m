%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Thomas Campbell
% Date: April 17, 2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description:                                                            
%-------------------------------------------------------------------------
% Classification, Regression, and Survival Tree (CRAST): this class takes in
% a set of c_samples and grows a classification, regression,
% or survival tree by recursivly splitting the data in a greedy fashion.  The
% differences between the classification, regression and survival types boil
% down to the metric considered when choosing an optimal split at each node.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs:
%-------------------------------------------------------------------------
% array of c_samples and a slurry of parameters passed to the constructor by name, value pair
% parameters:
%             - max_splits: maximum number of splits to allow in the tree, default 1e6
%             - max_depth: maximum depth of the tree, default 100
%             - nfeature_sample: the number of features to sample at each split, default floor(sqrt(<number of features>))
%             - nfeature_values: for continuous features, the values are sampled to decide on a set of values to consider 
%               at each split.  this parameter controls the number of possible values to consider. default 10
%             - feature_value_sampling: which feature value sampling algorithm to use, default is 'percentile', other option is 
%               'uniform'.  for 'percentile', the values will be equally spaced by 
%               the features observed density in the training set.  For example, the default is 10, this will select the value of the feature
%               at the 10th, 20th, ... 90th percentlie as the possible values.  For 'uniform' the feature values are equally spaced from the
%               min to the max observed feature value.
%             - force_logit_in_feature_sample: flag to force the feature sample to include the logit in cases where covariates or other
%               data is being combined with a single logit.  default false
%             - loss_function: this controls what is optimized when choosing the optimal split at each node which in turn controls which 
%               type of tree we are growing. current options are 'HR' for survival trees, 'RMST' for restricted mean survival time trees, 
%               'class_entropy' for entropy based classification trees, 'gini_index' for CART like classification trees, and 'l2' for 
%               regression trees.
%               default 'class_entropy'
%             - leaf_score (default percentile_all): 
%                            - event: fraction of events in the terminal leaf
%                            - left-right: number of left turns vs right turns down the tree to get to the terminal leaf
%                            - HR: 0 if leaf running HR score is less than 1, 1 otherwise
%                            - percentile_all: percentile of the HR score w.r.t. all possible scores in the tree
%                            - percentile_terminal: percentile of the HR score w.r.t. all possible scores at terminal leafs in the tree
%                n.b. leaf scores are overwritten with the corresponding logit described above for all cases except for HR, default is 
%                percentile
%             - min_HR: for survival trees, the minimum HR (or 1/HR) for a split to be considered valid. default 1.0
%             - use_total_HR: experimental flag for survival trees.  Considers a HR between the running total Gr1 and Gr2 at each split
%               instead of considering the HR between only the daughter groups.  default false
%             - rmst_start_time: start time for the rmst integral
%             - rmst_end_time: end time for the rmst integral
%             - use_pearson_feature_weighting: weights features in feature sampling by their pearson correlation to the endpoint def (NO LONGER SUPPORTED)
%             - use_binary_class_score: for binary classification trees, use the proportion of samples in terminal leaf as the a leaf score
%             - use_running_binary_class_score: running score that starts at 1 and gets mulitplied by p or 1/p (class 0 or 1)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Outputs:                                                                
%-------------------------------------------------------------------------
% a trained CRAST
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Intended Use:
%-------------------------------------------------------------------------
% There is an easy interface with the combination fitter, but this class is
% also suitable to stand alone applications.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dependencies:
%-------------------------------------------------------------------------
% logrank_survival_tree.m, node.m, c_sample.m
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef crast < handle
    properties
        nodes % list of nodes that comprise the trained tree
        data % should be an array of data points with fields: features, survival_data, survival_censor
        nfeatures % number of features in the data
        feature_values % cell array of row vectors of allowed feature values for each feature when splitting each node
        nsplits % current number of splits

        % survival tree specific
        current_gr1_idxs % running tally of the poor perfoming group as the tree grows for when using an 'overall_HR' survival tree
        current_gr2_idxs % running tally of the good perfoming group as the tree grows for when using an 'overall_HR' survival tree
        logrank_data % data for logrank function

        % classification tree specific
        nclasses
        definitions %array of definitions for quick entropy calc
        definitions_mask

        % parameters
        loss_function % determines what type of tree we are using.  HR (survival), class_entropy (classification), l2 (regression)
        max_splits % maximum allowed number of splits
        min_leaf_size % minimum sample population in a resulting child node for the split to be considered
        nfeature_sample % number of features to sample at each split for making RFs instead of bagged trees
        max_depth % maximum allowed depth for the tree to grow
        force_logit_in_feature_sample % when combining covariates whether or not to force the mass spec logit to be included in every feature sample
        feature_value_sampling % which feature value sampling algorithm to use
        nfeature_values % number of values for each feature to consider when searching for splits
        use_binary_class_score % for binary classification trees, use the proportion of samples in terminal leaf as the a leaf score
        use_running_binary_class_score % for binary classifiacation trees, see def above

        % survival tree specific parameters
        min_HR % minimum hazard ratio between to child nodes for a split to be considered
        use_total_HR % whether or not we are using the overall-HR survival tree algorithm
        leaf_score% which leaf score algorithm to use when using a leaf score based algorithm

        is_feature_categorical
        categorical_feature_groupings
        feature_probs % prior probabilities for a feature to be drawn in a feature sample

        use_weighted_ave_info_gain % whether or not to weight the average in gini or entropy info gain
        rand_stream % independent rand stream for this crast

        rmst_start_time % start time for rmst integral
        rmst_end_time % end time for rmst integral
    end

    methods 

        set_parameters(obj) % sets the parameters for growing the tree
        fit_tree(obj) % fits a tree on the data passed to the constructor
        logits = predict_tree(obj, data_in) % uses the fit tree to predict on data_in

        % predict on a single sample
        score = predict_single(obj, in_features)

        split_node(obj, parent_node_ID) % this is the main recursive splitting function that grows the tree

        get_feature_values(obj, is_feature_categorical_in) % gets and trims possible feature values to be considered when splitting the data

        [gr1_idxs, gr2_idxs] = split_data(obj, in_idxs, feature_idx, threshold, was_flipped) % splits the data

        [out_HR, was_flipped] = get_log_rank(obj, gr1_idxs, gr2_idxs) % gets the logrank HR between the poor and good performing group

        out_entropy = get_class_entropy(obj, in_idxs)
        [out_class, out_p] = get_class_plurality_fraction(obj, in_idxs)
        out_ig = get_information_gain(obj, gr1_idxs, gr2_idxs, parent_node_ID)
        logit = predict_single_from_class_plurality(obj, in_features)

        set_categorical_feature(obj, in_feature_idx) % set individual feature to categorical by index
        detect_categorical(obj, max_n) % detect categorical features as those with less than max_n distinct feature values
        [gr1_idxs, gr2_idxs] = split_data_categorical(obj, in_idxs, feature_idx, in_grouping, was_flipped)

        % for transforming raw node scores to those used for prediction
        score_tree(obj)
        left_right_score(obj, node_id)
        get_final_leaf_scores(obj)

        function obj = crast(data_in, varargin)

            obj.nodes = {};
            obj.nsplits = 0;
            obj.data = data_in;
            obj.nfeatures = length(data_in(1).features);
            obj.feature_values = cell(obj.nfeatures,1);
            obj.feature_probs = (1.0/obj.nfeatures)*ones(1, obj.nfeatures);

            obj.is_feature_categorical = cell(1,obj.nfeatures);
            for ifeature = 1:obj.nfeatures
                obj.is_feature_categorical{ifeature} = false;
            end
            obj.categorical_feature_groupings = cell(1,obj.nfeatures);

            p = inputParser;
            addParameter(p, 'max_splits', 1e6);
            addParameter(p, 'min_leaf_size', 5);
            addParameter(p, 'max_depth', 100);
            addParameter(p, 'nfeature_sample', floor(sqrt(obj.nfeatures)));
            addParameter(p, 'force_logit_in_feature_sample', false);
            addParameter(p, 'feature_value_sampling', 'percentile');
            addParameter(p, 'nfeature_values', 10);
            addParameter(p, 'loss_function', 'class_entropy');
            addParameter(p, 'leaf_score', 'percentile_terminal');
            addParameter(p, 'min_HR', 1.0);
            addParameter(p, 'use_total_HR', false);
            addParameter(p, 'is_feature_categorical', {});
            addParameter(p, 'use_weighted_ave_info_gain', false);
            addParameter(p, 'force_reproducable', false);
            addParameter(p, 'use_binary_class_score', false);
            addParameter(p, 'use_running_binary_class_score', false);
            addParameter(p, 'rmst_start_time', 0);
            addParameter(p, 'rmst_end_time', max([obj.data().survival_data]'));
            addParameter(p, 'use_pearson_feature_weighting', false);
            parse(p, varargin{:});
            obj.max_splits = p.Results.max_splits;
            obj.min_leaf_size = p.Results.min_leaf_size;
            obj.max_depth = p.Results.max_depth;
            obj.nfeature_sample = p.Results.nfeature_sample;
            obj.force_logit_in_feature_sample = p.Results.force_logit_in_feature_sample;
            obj.nfeature_values = p.Results.nfeature_values;
            obj.loss_function = p.Results.loss_function;
            obj.leaf_score = p.Results.leaf_score;
            obj.min_HR = p.Results.min_HR;
            obj.use_total_HR = p.Results.use_total_HR;
            obj.feature_value_sampling = p.Results.feature_value_sampling;
            is_feature_categorical_in = p.Results.is_feature_categorical;
            obj.use_weighted_ave_info_gain = p.Results.use_weighted_ave_info_gain;
            use_default_seed = p.Results.force_reproducable;
            obj.use_binary_class_score = p.Results.use_binary_class_score;
            obj.use_running_binary_class_score = p.Results.use_running_binary_class_score;
            obj.rmst_start_time = p.Results.rmst_start_time;
            obj.rmst_end_time = p.Results.rmst_end_time;
            use_pearson_feature_weighting = p.Results.use_pearson_feature_weighting;
            obj.check_parameters();

            if use_default_seed
                obj.rand_stream = RandStream('mt19937ar','Seed',123456);
            else
                obj.rand_stream = RandStream.getGlobalStream;
            end


            if strcmp(obj.loss_function, 'HR')
                obj.current_gr1_idxs = [];
                obj.current_gr2_idxs = [];

                survival_data = zeros(length(data_in),1);
                survival_censor = zeros(length(data_in),1);

                % matlab's cox fit censor definition is flipped from ours
                for isample = 1:length(data_in)
                    survival_data(isample) = obj.data(isample).survival_data;
                    this_censor = obj.data(isample).survival_censor;
                    if this_censor > 0.5
                        survival_censor(isample) = 0;
                    else
                        survival_censor(isample) = 1;
                    end
                end
                % data for logrank function
                obj.logrank_data = [survival_data, survival_censor];
            end

            if strcmp(obj.loss_function, 'RMST')
                obj.current_gr1_idxs = [];
                obj.current_gr2_idxs = [];

                survival_data = [obj.data().survival_data]';
                survival_censor = [obj.data().survival_censor]';

                % data for RMST function
                obj.logrank_data = [survival_data, survival_censor];
            end


            if strcmp(obj.loss_function, 'class_entropy') || strcmp(obj.loss_function, 'gini_index')
                nsamples = length(obj.data);
                obj.definitions = -1*ones(nsamples,1);
                found_defs = [];
                for isample = 1:nsamples
                    this_def = obj.data(isample).definition;
                    is_new = true;
                    for def = found_defs
                        if def == this_def
                            is_new = false;
                            break;
                        end
                    end
                    if is_new
                        found_defs = [found_defs, this_def];
                    end
                end
                obj.definitions_mask = found_defs;
                if obj.use_binary_class_score || obj.use_running_binary_class_score
                    if length(found_defs) ~= 2
                        error('only use binary class scores for binary classification problems');
                    end
                    defs_bad = false;
                    if ~(found_defs(1) == 1 || found_defs(2) == 1)
                        defs_bad = true;
                    end
                    if ~(found_defs(1) == 0 || found_defs(2) == 0)
                        defs_bad = true;
                    end
                    if defs_bad
                        error('definitions must be either 0 or 1 for binary class scores')
                    end
                end
                    
                obj.nclasses = length(found_defs);
                % convert found defintions to natural numbers with no skips
                for isample = 1:nsamples
                    this_def = obj.data(isample).definition;
                    for idef = 1:obj.nclasses
                        if this_def == found_defs(idef)
                            obj.definitions(isample) = idef;
                            break;
                        end
                    end
                    if obj.definitions(isample) == -1;
                        error('unable to assign definitions, check typing around this')
                    end
                end
            end

            obj.get_feature_values(is_feature_categorical_in);
            for ifeature = 1:length(is_feature_categorical_in)
                if is_feature_categorical_in{ifeature}
                    obj.set_categorical_feature(ifeature);
                end
            end
            obj.trim_feature_values();
        end
    end
end