classdef dxCortexBag < handle
    % this class trains a single dxCortex master classifier

    properties
        mCs % cell array of mini-classifier kNNs
        trees % cell array of trees
        MC_reg % array of the logistic regression coefficients for the master classifier
        train_data % training data with numeric training features as [nsamples, nfeatures] in train_data.data, 
                   % and numeric defs as [nsamples, 1] array in train_data.defs
    end

    methods

        function obj = dxCortexBag(train_data)
            obj.mCs = {};
            obj.trees = {};
            obj.MC_reg = [];
            obj.train_data = train_data;
        end

        function train_mCs(obj, k, max_ftrs)
            % trains the mini-classifier kNNs

            nfeatrues = size(obj.train_data.data, 2);
            
            for ii = 1:max_ftrs 
                % Try all combinations (without repeating) of size ii from features
                combos = nchoosek(1:nfeatrues, ii);
                for jj = 1:length(combos)
                    kNN = fitcknn(obj.train_data.data(:,combos(jj,:)), obj.train_data.defs, 'NumNeighbors', k, 'NSMethod', 'exhaustive', 'distance', 'euclidean');
                    
                    this_mc.ftrs = combos(jj,:);
                    this_mc.kNN = kNN;                   
                    obj.mCs{end+1} = this_mc;
                end
            end
        end

        function labels = predict_mCs(obj, data)
            % applies the mini-classifers to input data
            nsamples = size(data,1);
            nmCs = length(obj.mCs);
            
            labels = zeros(nsamples, nmCs);
            for ii = 1:nmCs
                this_mC = obj.mCs{ii};
                x_test = data(:,this_mC.ftrs);
                [labels, ~, ~] = predict(this_mC.kNN, x_test);
                labels(:, ii) = labels;
            end
        end

        function train_master_classifier(obj, ndoi, nleave_in)
            % trains the master classifier drop-out logistic regression
            nmCs = length(obj.mCs);

            m_classifications = obj.predict_mCs(obj.train_data.data);
            
            obj.MC_reg = zeros(nmCs + 1, 1);
            nsuccess = 0;
            for ii = 1:ndoi
                these_idxs = datasample(1:nmCs, nleave_in, 'Replace', false);
                X = m_classifications(:,these_idxs);        
                these_coeffs = obj.log_reg_newton_raphson(X, obj.train_data.defs);
                if (isnan(these_coeffs(1)))
                    continue;
                end
                nsuccess = nsuccess + 1;
                obj.MC_reg(1) = obj.MC_reg(1) + these_coeffs(1);
                for jj = 1:nleave_in
                    obj.MC_reg(these_idxs(jj)+1) = obj.MC_reg(these_idxs(jj)+1) + these_coeffs(jj+1);
                end
            end
            obj.MC_reg = obj.MC_reg / nsuccess;
        end

        function [probs, labels] = predict_master_classifier(obj, data, prob_thresh)
            % apllies the ensemble (mini-classifiers + regression) to input data
            mC_labels = obj.predict_mCs(data);
            exponents = obj.MC_reg(1) + mC_labels * obj.MC_reg(2:end);
            probs = 1.0 ./ (1.0 + exp(-1.0 .* exponents));
            labels_bool = probs > prob_thresh;
            labels = double(labels_bool);
        end

        function B = log_reg_newton_raphson(obj, X, y)
            % logistic regression
            xSize = size(X);
            XWithZerothCol = [ones(xSize(1),1) X];
            err = 1;
            stdError = ones(xSize(2) + 1, 1);
            iter = 0;
            maxIter = 200;
            errorTol = 1e-6;
            B = zeros(xSize(2) + 1, 1);
            while ((err > errorTol) && (iter < maxIter))
                iter = iter + 1;

                tmpD = XWithZerothCol * B;
                tmpDD = exp(tmpD);
                P = tmpDD ./ (1.0 + tmpDD);

                W = P.*(1-P);

                tmpYP = y - P;

                grad = XWithZerothCol' * tmpYP;

                xhat = W .* XWithZerothCol;

                infoM = XWithZerothCol' * xhat;

                lambda = 1e-5;

                infoMTrustReg = infoM + lambda*eye(size(infoM));

                invInfoMTrustReg = inv(infoMTrustReg);

                xDelta = invInfoMTrustReg * grad;
                stdError = sqrt(diag(invInfoMTrustReg));

                err = max(abs(xDelta));

                B = B + 0.1*xDelta;
            end
        end
    end
end