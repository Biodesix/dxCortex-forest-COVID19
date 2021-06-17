classdef sampling_engine < handle
    % this class reads in data sets from csv files, generates bag sampling for a dxCortex bagged classifier.  it handles data preparation
    % also generates mini-bag sampling used when combining additional attributes using decision trees.  Finally, it allows for out of bag 
    % classification when training a bagged model.

    properties
        data % [nsamples, nftrs] numeric array of feature data
        defs % [nsamples, 1] numeric array of definitons
        sample_IDs % {nsamples, 1} cell array of string sample ids
        headers % {nfeatures, 1} cell array of feature names

        % for out of bag predicting
        curr_ib_idxs % current in bag indecies
        curr_oob_idxs % current out of bag indecies
        oob_idxs % out of bag idxs for all bags, [nsamples, nbags] numeric array, 0 means sample was in bag, 1 means out of bag
        oob_labels % out of bag labels

        % for mini-bags and trees
        tree_data % array of c_samples for the trees
        tree_headers % name of features used in tree data
        tree_ftrs_categorical % keeps track of which features are categorical to pass to tree
        saved_curr_ib_idxs % place to save the in bag idecies when doing mini-bag sampling which will overwrtie curr_ib_idxs
        saved_curr_oob_idxs % place to save the out of bag idecies when doing mini-bag sampling which will overwrtie curr_oob_idxs

        % for validation data
        val_data % [nsamples, nftrs] numeric array of feature data
        val_defs % [nsamples, 1] numeric array of definitons (optional)
        val_sample_IDs % {nsamples, 1} cell array of string sample ids
        val_headers % {nfeatures, 1} cell array of feature names
        val_tree_data % array of c_samples for the trees

    end

    methods
        function obj = sampling_engine()
            obj.data = [];
            obj.defs = [];
            obj.sample_IDs = {};
            obj.headers = {};
            obj.curr_oob_idxs = [];
            obj.curr_ib_idxs = [];
            obj.oob_idxs = [];
            obj.oob_labels = [];
            obj.val_data = [];
            obj.val_defs = [];
            obj.val_sample_IDs = {};
            obj.val_headers = {};
        end

        function read_data(obj, in_path, id_name, def_name)
            in_cell = readcell(in_path);
            headers = in_cell(1,:);
            in_cell = in_cell(2:end,:);

            id_idx = -1;
            def_idx = -1;
            obj.headers = {};
            ftr_idxs = [];
            for iheader = 1:length(headers)
                if strcmp(headers{iheader}, id_name)
                    id_idx = iheader;
                elseif strcmp(headers{iheader}, def_name)
                    def_idx = iheader;
                else
                    obj.headers{end+1} = headers{iheader};
                    ftr_idxs = [ftr_idxs, iheader];
                end
            end

            if id_idx < 0 || def_idx < 0
                error('could not find column for sample id or definition');
            end

            nsamples = size(in_cell, 1);
            nfeatrues = length(ftr_idxs);

            obj.data = -9999*ones(nsamples, nfeatrues);
            obj.sample_IDs = cell(nsamples,1);
            obj.defs = -9999*ones(nsamples,1);

            for isample = 1:nsamples
                obj.sample_IDs{isample} = string(in_cell{isample, id_idx});
                obj.defs(isample) = str2double(string(in_cell{isample, def_idx}));
                for ifeature = 1:nfeatrues
                    obj.data(isample, ifeature) = str2double(string(in_cell{isample, ftr_idxs(ifeature)}));
                end
            end
        end

        function generate_bag_sampling(obj, train_fraction)
            % sample the data for a new bag
            gr1_idxs = [];
            gr2_idxs = [];

            nsamples = length(obj.defs);

            for isample = 1:nsamples
                if obj.defs(isample) == 1
                    gr1_idxs = [gr1_idxs, isample];
                else
                    gr2_idxs = [gr2_idxs, isample];
                end
            end

            if length(gr1_idxs) < length(gr2_idxs)
                n_to_sample = floor(train_fraction * length(gr1_idxs));
            else
                n_to_sample = floor(train_fraction * length(gr2_idxs));
            end

            obj.curr_ib_idxs = [randsample(gr1_idxs, n_to_sample), randsample(gr2_idxs, n_to_sample)];


            obj.curr_oob_idxs = [];
            for ii = 1:nsamples
                in_bag = false;
                for idx = obj.curr_ib_idxs
                    if ii == idx
                        in_bag = true;
                        break;
                    end
                end
                if ~in_bag
                    obj.curr_oob_idxs = [obj.curr_oob_idxs, ii];
                end
            end
        end

        function out_data = get_train_data(obj, ftrs_to_use)
            % get training data for this bag
            ftr_idxs = [];
            for iftr = 1:length(obj.headers)
                use = false;
                for jftr = 1:length(ftrs_to_use)
                    if strcmp(obj.headers{iftr}, ftrs_to_use{jftr})
                        use = true;
                        break;
                    end
                end
                if use
                    ftr_idxs = [ftr_idxs, iftr];
                end
            end
            out_data.data = obj.data(obj.curr_ib_idxs,ftr_idxs);
            out_data.defs = obj.defs(obj.curr_ib_idxs);
        end

        function out_data = get_test_data(obj, ftrs_to_use)
            % get test data for this bag
            ftr_idxs = [];
            for iftr = 1:length(obj.headers)
                use = false;
                for jftr = 1:length(ftrs_to_use)
                    if strcmp(obj.headers{iftr}, ftrs_to_use{jftr})
                        use = true;
                        break;
                    end
                end
                if use
                    ftr_idxs = [ftr_idxs, iftr];
                end
            end
            out_data.data = obj.data(obj.curr_oob_idxs,ftr_idxs);
            out_data.defs = obj.defs(obj.curr_oob_idxs);
        end

        function set_bag_labels(obj, labels)
            % for keeping track of OOB labels
            nsamples = length(obj.defs);
            % push back oob idxs
            oob_idxs = zeros(nsamples,1);
            for idx = obj.curr_oob_idxs
                oob_idxs(idx) = 1;
            end
            obj.oob_idxs = [obj.oob_idxs, oob_idxs];

            % push back labels for this bag
            these_labels = -9999*ones(nsamples,1);
            for ilabel = 1:length(labels)
                these_labels(obj.curr_oob_idxs(ilabel)) = labels(ilabel);
            end
            obj.oob_labels = [obj.oob_labels, these_labels];
        end

        function labels = oob_predict(obj)
            % get the oob predictions
            [nsamples, nbags] = size(obj.oob_idxs);
            labels = -9999*ones(nsamples,1);
            for isample = 1:nsamples
                n0s = 0;
                n1s = 0;
                for ibag = 1:nbags
                    if obj.oob_idxs(isample, ibag) > 0
                        if obj.oob_labels(isample, ibag) > 0.5
                            n1s = n1s + 1;
                        else
                            n0s = n0s + 1;
                        end
                    end
                end
                if (n0s > 0 || n1s > 0)
                    if n1s > n0s
                        labels(isample) = 1;
                    else
                        labels(isample) = 0;
                    end
                end
            end
        end

        % tree / mini-bag specific methods

        function prep_data_for_trees(obj, labels, tree_ftrs, tree_ftrs_categorical)
            % this method takes the dxCortex labels, adds them as a feature, and then calculates the categorical interaction terms
            % used in the study.  It also stores the data as an array of c_sample structs which the CRAST class will use.

            if length(labels) ~= length(obj.curr_oob_idxs)
                error('sample number mismatch: please pass only the labels for the out of bag samples to this method')
            end

            % save current main bag sampling so we can overwrite curr_ib and curr_oob
            obj.saved_curr_ib_idxs = obj.curr_ib_idxs;
            obj.saved_curr_oob_idxs = obj.curr_oob_idxs;

            % get the idxs of the features we need for the tree
            ftr_idxs = [];
            for iftr = 1:length(tree_ftrs)
                idx = -1;
                for jftr = 1:length(obj.headers)
                    if strcmp(obj.headers{jftr}, tree_ftrs{iftr})
                        idx = jftr;
                        break
                    end
                end
                if idx < 0
                    error(sprintf('could not find feature %s', tree_ftrs{iftr}))
                end
                ftr_idxs = [ftr_idxs, idx];
            end

            nsamples = size(obj.data, 1);

            tree_data(nsamples) = c_sample;
            for isample = 1:length(obj.saved_curr_oob_idxs);
                sample_idx = obj.saved_curr_oob_idxs(isample);
                tree_data(sample_idx).definition = obj.defs(sample_idx);
                these_ftrs = [];
                these_ftrs = [these_ftrs, labels(isample)];
                for idx = ftr_idxs
                    these_ftrs = [these_ftrs, obj.data(sample_idx, idx)];
                end
                tree_data(sample_idx).features = these_ftrs;
            end

            obj.tree_data = tree_data;

            obj.tree_headers = {};
            obj.tree_headers{end+1} = 'dxCortex_label';
            for iftr = 1:length(tree_ftrs)
                obj.tree_headers{end+1} = tree_ftrs{iftr};
            end

            obj.tree_ftrs_categorical = {};
            obj.tree_ftrs_categorical{end+1} = true;
            for iftr = 1:length(tree_ftrs_categorical)
                obj.tree_ftrs_categorical{end+1} = tree_ftrs_categorical{iftr};
            end

            % interaction terms
            nfeatures = length(obj.tree_headers);
            
            for ifeature = 1:nfeatures
                if ~obj.tree_ftrs_categorical{ifeature}
                    continue;
                end
                for jfeature = 1:nfeatures
                    if ifeature >= jfeature
                        continue
                    end
                    if ~obj.tree_ftrs_categorical{jfeature}
                        continue;
                    end

                    name_str = obj.tree_headers{ifeature};
                    name_str = strcat(name_str, '_int_');
                    name_str = strcat(name_str, obj.tree_headers{jfeature});
                    obj.tree_headers{end+1} = name_str;
                    obj.tree_ftrs_categorical{end+1} = true;
                    for isample = obj.saved_curr_oob_idxs
                        this_int_term = obj.tree_data(isample).features(ifeature) + obj.tree_data(isample).features(jfeature)*20;
                        obj.tree_data(isample).features = [obj.tree_data(isample).features, this_int_term];
                    end
                end
            end
        end

        function generate_minibag_sampling(obj, train_fraction)
            % sample the available out of bag data from the first normal bag sampling
            gr1_idxs = [];
            gr2_idxs = [];

            nsamples = length(obj.defs);

            for isample = obj.saved_curr_oob_idxs
                if obj.defs(isample) == 1
                    gr1_idxs = [gr1_idxs, isample];
                else
                    gr2_idxs = [gr2_idxs, isample];
                end
            end

            if length(gr1_idxs) < length(gr2_idxs)
                n_to_sample = floor(train_fraction * length(gr1_idxs));
            else
                n_to_sample = floor(train_fraction * length(gr2_idxs));
            end

            obj.curr_ib_idxs = [randsample(gr1_idxs, n_to_sample), randsample(gr2_idxs, n_to_sample)];


            obj.curr_oob_idxs = [];
            for ii = 1:nsamples
                in_bag = false;
                for idx = [obj.curr_ib_idxs, obj.saved_curr_ib_idxs]
                    if ii == idx
                        in_bag = true;
                        break;
                    end
                end
                if ~in_bag
                    obj.curr_oob_idxs = [obj.curr_oob_idxs, ii];
                end
            end
        end

        function out_data = get_train_data_tree(obj)
            % get the tree ready training data for this mini-bag
            out_data = obj.tree_data(obj.curr_ib_idxs);
        end

        function out_data = get_test_data_tree(obj)
            % get the tree ready test data for this mini-bag
            out_data = obj.tree_data(obj.curr_oob_idxs);
        end

        % for reading validation data
        function read_val_data(obj, in_path, id_name, def_name)
            in_cell = readcell(in_path);
            headers = in_cell(1,:);
            in_cell = in_cell(2:end,:);

            id_idx = -1;
            def_idx = -1;
            obj.val_headers = {};
            ftr_idxs = [];
            for iheader = 1:length(headers)
                if strcmp(headers{iheader}, id_name)
                    id_idx = iheader;
                elseif strcmp(headers{iheader}, def_name)
                    def_idx = iheader;
                else
                    obj.val_headers{end+1} = headers{iheader};
                    ftr_idxs = [ftr_idxs, iheader];
                end
            end

            if id_idx < 0
                error('could not find column for sample id');
            end
            if def_idx < 0
                warning('could not find column for definition')
            end

            nsamples = size(in_cell, 1);
            nfeatrues = length(ftr_idxs);

            obj.val_data = -9999*ones(nsamples, nfeatrues);
            obj.val_sample_IDs = cell(nsamples,1);
            obj.val_defs = -9999*ones(nsamples,1);

            for isample = 1:nsamples
                obj.val_sample_IDs{isample} = string(in_cell{isample, id_idx});
                if def_idx > 0
                    obj.val_defs(isample) = str2double(string(in_cell{isample, def_idx}));
                end
                for ifeature = 1:nfeatrues
                    obj.val_data(isample, ifeature) = str2double(string(in_cell{isample, ftr_idxs(ifeature)}));
                end
            end
        end

        function prep_val_data_for_trees(obj, labels, tree_ftrs, tree_ftrs_categorical_in)
            % add the dxCortex labels as features and calculate interaction terms for validation data
            nsamples = size(obj.val_data, 1);
            if length(labels) ~= nsamples
                error('nsamples mismatch')
            end

            % get the idxs of the features we need for the tree
            ftr_idxs = [];
            for iftr = 1:length(tree_ftrs)
                idx = -1;
                for jftr = 1:length(obj.headers)
                    if strcmp(obj.headers{jftr}, tree_ftrs{iftr})
                        idx = jftr;
                        break
                    end
                end
                if idx < 0
                    error(sprintf('could not find feature %s', tree_ftrs{iftr}))
                end
                ftr_idxs = [ftr_idxs, idx];
            end

            tree_data(nsamples) = c_sample;
            for isample = 1:nsamples
                tree_data(isample).definition = obj.val_defs(isample);
                these_ftrs = [];
                these_ftrs = [these_ftrs, labels(isample)];
                for idx = ftr_idxs
                    these_ftrs = [these_ftrs, obj.data(isample, idx)];
                end
                tree_data(isample).features = these_ftrs;
            end

            obj.val_tree_data = tree_data;

            tree_headers = {};
            tree_headers{end+1} = 'dxCortex_label';
            for iftr = 1:length(tree_ftrs)
                tree_headers{end+1} = tree_ftrs{iftr};
            end

            tree_ftrs_categorical = {};
            tree_ftrs_categorical{end+1} = true;
            for iftr = 1:length(tree_ftrs_categorical_in)
                tree_ftrs_categorical{end+1} = tree_ftrs_categorical_in{iftr};
            end

            % interaction terms
            nfeatures = length(tree_headers);
            
            for ifeature = 1:nfeatures
                if ~tree_ftrs_categorical{ifeature}
                    continue;
                end
                for jfeature = 1:nfeatures
                    if ifeature >= jfeature
                        continue
                    end
                    if ~tree_ftrs_categorical{jfeature}
                        continue;
                    end

                    for isample = 1:nsamples
                        this_int_term = obj.val_tree_data(isample).features(ifeature) + obj.val_tree_data(isample).features(jfeature)*20;
                        obj.val_tree_data(isample).features = [obj.val_tree_data(isample).features, this_int_term];
                    end
                end
            end
        end

        function out_data = get_val_data(obj, ftrs_to_use)
            % return validation data
            ftr_idxs = [];
            for iftr = 1:length(obj.val_headers)
                use = false;
                for jftr = 1:length(ftrs_to_use)
                    if strcmp(obj.val_headers{iftr}, ftrs_to_use{jftr})
                        use = true;
                        break;
                    end
                end
                if use
                    ftr_idxs = [ftr_idxs, iftr];
                end
            end
            out_data.data = obj.val_data(:,ftr_idxs);
            out_data.defs = obj.val_defs;
        end

        function out_data = get_val_data_tree(obj)
            % access the validation data for the trees
            out_data = obj.val_tree_data;
        end

    end
end