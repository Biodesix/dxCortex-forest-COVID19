function labels = predict_tree(obj, data_in)
    % classifiy new data
    nsamples = length(data_in);
    labels = zeros(nsamples, 1);
    for isample = 1:nsamples
        labels(isample) = obj.predict_single(data_in(isample).features);
    end
end
