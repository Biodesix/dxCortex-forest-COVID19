function get_feature_values(obj, is_feature_categorical_in)
    % determine feature values for all features
    nsamples = length(obj.data);
    for ifeature = 1:obj.nfeatures
        if length(is_feature_categorical_in) == obj.nfeatures && is_feature_categorical_in{ifeature}
            these_ftrvals = [];
            for isample = 1:nsamples
                found = false;
                for this_val = these_ftrvals
                    if isequal(obj.data(isample).features(ifeature), this_val)
                        found = true;
                        break;
                    end
                end
                if ~found
                    these_ftrvals = [these_ftrvals, obj.data(isample).features(ifeature)];
                end
            end
        else
            these_ftrvals = -999*ones(1,nsamples);
            for isample = 1:nsamples
                these_ftrvals(isample) = obj.data(isample).features(ifeature);
            end
        end
        obj.feature_values{ifeature} = these_ftrvals;
    end
end
