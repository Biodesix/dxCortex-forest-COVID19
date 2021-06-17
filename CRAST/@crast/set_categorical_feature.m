function set_categorical_feature(obj, in_feature_idx)
    % manually sets feature indexed by in_feature_idx to categorical
    obj.is_feature_categorical{in_feature_idx} = true;

    these_feature_values = obj.feature_values{in_feature_idx};
    n_categories = length(these_feature_values);
    if n_categories > 20
        msg = sprintf('number of categories is greater than 20... This will be slow.  Are you sure feature %d is categorical?', in_feature_idx);
        warning(msg);
    end
    groupings = {};
    % for even n: take up to n/2 inclusive, but only half of the last grouping rows
    % for odd n, only take (n-1)/2
    if mod(n_categories,2)
        for icategory = 1:(n_categories-1)/2
            these_groupings = nchoosek(1:n_categories,icategory);
            [ng, mg] = size(these_groupings);
            for irow = 1:ng
                these_ftr_groupings = [];
                for icol = 1:mg
                    these_ftr_groupings = [these_ftr_groupings, these_feature_values(these_groupings(irow,icol))];
                end
                groupings{end+1} = these_ftr_groupings;
            end
        end
    else
        for icategory = 1:(n_categories)/2
            these_groupings = nchoosek(1:n_categories,icategory);
            [ng, mg] = size(these_groupings);
            if icategory == n_categories/2
                % n.b only able to do this because of how matlab's nchoosek orders the output sets...
                these_groupings = these_groupings(1:ceil(ng/2),:);
                [ng, mg] = size(these_groupings);
            end
            for irow = 1:ng
                these_ftr_groupings = [];
                for icol = 1:mg
                    these_ftr_groupings = [these_ftr_groupings, these_feature_values(these_groupings(irow,icol))];
                end
                groupings{end+1} = these_ftr_groupings;
            end
        end
    end
    obj.categorical_feature_groupings{in_feature_idx} = groupings;
end