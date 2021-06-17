function fit_tree(obj)
    % initialize indecies to all data passed to constructor
    init_idxs = zeros(1,length(obj.data));
    for isample = 1:length(obj.data)
        init_idxs(isample) = isample;
    end
    % set off recursive splitting
    new_node = node();
    new_node.ID = length(obj.nodes) + 1;
    new_node.depth = 0;
    new_node.score = 1.0;
    new_node.final_score = 1.0;
    if strcmp(obj.loss_function, 'RMST')
        new_node.score = 0.0;
    end
    new_node.sample_idxs = init_idxs;
    if strcmp(obj.loss_function, 'class_entropy')
        new_node.entropy = obj.get_class_entropy(init_idxs);
    end
    if strcmp(obj.loss_function, 'gini_index')
        new_node.entropy = obj.get_class_gini(init_idxs);
    end
    obj.nodes{end+1} = new_node;

    obj.split_node(new_node.ID);

    % transform scores depending on leaf score parameters
    obj.score_tree();
end
