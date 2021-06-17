function get_final_leaf_scores(obj)

    if obj.use_binary_class_score
        for inode = 1:length(obj.nodes)
            obj.nodes{inode}.final_score = obj.nodes{inode}.score;
        end
        return;
    end

    if strcmp(obj.leaf_score, 'event')
        % final node score is proportion of events
        for inode = 1:length(obj.nodes)
            this_node_idxs = obj.nodes{inode}.sample_idxs;
            nevents = 0;
            for idx = this_node_idxs
                if obj.data(idx).survival_censor == 1
                    nevents = nevents + 1;
                end
            end
            obj.nodes{inode}.final_score = nevents/length(this_node_idxs);
        end
    end

    if strcmp(obj.leaf_score, 'percentile_all')
        % re-write scores as a percentlie for all nodes w.r.t. all nodes
        scores = [];
        for inode = 1:length(obj.nodes)
            scores = [scores; obj.nodes{inode}.score];
        end

        for iscore = 1:length(scores)
            tscore = scores(iscore);
            nless = sum(scores < tscore);
            nequal = sum(scores == tscore);
            obj.nodes{iscore}.final_score = (nless + 0.5*nequal) / length(scores);
        end
    end

    if strcmp(obj.leaf_score, 'percentile_terminal')
        % re-write scores as a percentlie for only terminal nodes w.r.t only terminal nodes
        scores = [];
        terminal_idxs = [];
        for inode = 1:length(obj.nodes)
            if obj.nodes{inode}.child_left < 0
                scores = [scores; obj.nodes{inode}.score];
                terminal_idxs = [terminal_idxs, inode];
            end
        end

        for iscore = 1:length(scores)
            tscore = scores(iscore);
            nless = sum(scores < tscore);
            nequal = sum(scores == tscore);
            obj.nodes{terminal_idxs(iscore)}.final_score = (nless + 0.5*nequal) / length(scores);
        end
    end

    if strcmp(obj.leaf_score, 'HR')
        for inode = 1:length(obj.nodes)
            if obj.nodes{inode}.score < 1
                obj.nodes{inode}.final_score = 0;
            else
                obj.nodes{inode}.final_score = 1;
            end
        end
    end
end
