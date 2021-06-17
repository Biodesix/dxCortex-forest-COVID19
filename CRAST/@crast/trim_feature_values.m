function trim_feature_values(obj)
    % trim feature values
    if strcmp(obj.feature_value_sampling, 'percentile')
        n_pts = obj.nfeature_values;
        for ifeature = 1:obj.nfeatures
            if length(obj.feature_values{ifeature}) < n_pts || obj.is_feature_categorical{ifeature}
                continue;
            end
            obj.feature_values{ifeature} = quantile(obj.feature_values{ifeature},n_pts);
        end
    elseif strcmp(obj.feature_value_sampling, 'uniform')
        n_pts = obj.nfeature_values;
        for ifeature = 1:obj.nfeatures
            if length(obj.feature_values{ifeature}) < n_pts || obj.is_feature_categorical{ifeature}
                continue;
            end
            these_ftrvals = zeros(1,n_pts);

            min_ftr = min(obj.feature_values{ifeature});
            max_ftr = max(obj.feature_values{ifeature});
            step_ftr = (max_ftr - min_ftr)/n_pts;
            for ipt = 1:n_pts
                these_ftrvals(ipt) = min_ftr + step_ftr*(ipt-1);
            end
            obj.feature_values{ifeature} = these_ftrvals;
        end
    end
end
