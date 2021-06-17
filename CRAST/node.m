%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Thomas Campbell
% Date: March 5, 2020
% Revison log:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description:                                                            
%-------------------------------------------------------------------------
% a trained CRAST is just a connected list of nodes.  This is the class for 
% a single node in the tree.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs:
%-------------------------------------------------------------------------
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Outputs:                                                                
%-------------------------------------------------------------------------
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Intended Use:
%-------------------------------------------------------------------------
% users that wish to really customize algrithmic details of the CRAST model may
% want to add new properties and methods here.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dependencies:
%-------------------------------------------------------------------------
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef node < handle
    properties
        ID % unique ID of the node in the tree
        sample_idxs % which training samples made it to this node
        child_left % ID of the node coressponding to the poor preforming group after the optimal split is found
        child_right % ID of the node coressponding to the good preforming group after the optimal split is found

        feature_idx % index of the feature this node will use to split its sample
        cutoff % cutoff value on the chosen feature to split the its sample
        was_flipped % whether the data was split with a greater than or less than

        class % class of this node for classifying new data
        depth % depth of this node in the tree
        score % score at this node for score based tree methods 
        entropy % class entropy for classification trees
        grouping % grouping of categorical features for predicting on categorical data

        final_score % field for transformed score to be used on prediction of tree
    end

    methods 
        function obj = node()
            obj.ID = -999;
            obj.feature_idx = -999;
            obj.cutoff = -999;
            obj.child_left = -999;
            obj.child_right = -999;
            obj.class = -999;
            obj.depth = -999;
            obj.score = -999;
            obj.sample_idxs = [];
            obj.was_flipped = false;
            obj.entropy = 999;
            obj.grouping = [];
            obj.final_score = -999;
        end

        function out_node_ID = predict(obj, features_in)
            % give the appropriate child ID for a sample
            if obj.feature_idx < 0
                out_node_ID = -999;
                return;
            end

            if length(obj.grouping) == 0
                if obj.was_flipped
                    if features_in(obj.feature_idx) >= obj.cutoff
                        out_node_ID = obj.child_left;
                    else
                        out_node_ID = obj.child_right;
                    end
                else
                    if features_in(obj.feature_idx) < obj.cutoff
                        out_node_ID = obj.child_left;
                    else
                        out_node_ID = obj.child_right;
                    end
                end
            else
                was_group1 = false;
                for gg = obj.grouping
                    if features_in(obj.feature_idx) == gg
                        was_group1 = true;
                        break;
                    end
                end
                if obj.was_flipped
                    if was_group1
                        was_group1 = false;
                    else
                        was_group1 = true;
                    end
                end
                if was_group1
                    out_node_ID = obj.child_left;
                else
                    out_node_ID = obj.child_right;
                end
            end
        end

    end
end