function score_tree(obj)
    if strcmp(obj.loss_function, 'HR') || strcmp(obj.loss_function, 'RMST')
        if obj.use_total_HR
            for inode = 1:length(obj.nodes)
                obj.nodes{inode}.final_score = obj.nodes{inode}.class;
            end
        else
            if strcmp(obj.leaf_score, 'left-right')
                warning('this is untested');
                obj.left_right_score(obj.nodes{1}.ID);
            else
                obj.get_final_leaf_scores();
            end
        end
    elseif strcmp(obj.loss_function, 'class_entropy') || strcmp(obj.loss_function, 'gini_index') || strcmp(obj.loss_function, 'ROC') || strcmp(obj.loss_function, 'running_ROC')
        if obj.use_binary_class_score
            obj.get_final_leaf_scores();
        else
            for inode = 1:length(obj.nodes)
                obj.nodes{inode}.final_score = obj.nodes{inode}.class;
            end
        end
    else
        for inode = 1:length(obj.nodes)
            obj.nodes{inode}.final_score = obj.nodes{inode}.score;
        end
    end
end