% This script gives an example of training a dxCortex classifier like those used in the 
% child classifiers of the hierarchical COVID-19 prognostic tests.  

% these are the features that will be used in the dxCortex model
dxCortex_ftrs = {
    'ed_temperature_c',
    'ed_heart_rate',
    'ed_systolic_bp',
    'ed_diastolic_bp',
    'ed_respiratory_rate',
    'ed_oxygen_saturation',
    'initial_qtc',
    'sodium',
    'potassium',
    'carbon_dioxide_bicarb',
    'bun',
    'creatinine',
    'egfr',
    'anion_gap',
    'wbc_screen',
    'hemoglobin',
    'hematocrit',
    'platelet_count',
    'initial_ldh',
    'initial_d_dimer',
    'initial_c_reactive_protein',
    'ferritin'
};

% hyper parameter for the bagging, nbags is set to 50 for this example so this script will run in a sensible amount of time 
% on a single core.  For the COVID-19 prognostic tests study, these methods implemented in an HPC environment were used with nbags=625
nbags = 50;

% construct a sampling engine to handle the bag sampling
se = sampling_engine();

% read in the data
se.read_data('./data/synthetic_training.csv', 'SampleID', 'Definition'); % definitions must be coded as 1 = positive (endpoint occured), 0 = negative for performance metrics below to make sense

% for storing the ensemble for later use
dcbs = cell(nbags,1);
for ibag = 1:nbags
    % fprintf('on bag: %d out of %d\n', ibag, nbags);

    % generate this bag's train/test split
    se.generate_bag_sampling(0.667);

    % construct a dxCortexBag with the training data for this bag
    dcb = dxCortexBag(se.get_train_data(dxCortex_ftrs));

    % train the kNN miniclassifiers with k=7, using all single features and pairs of features
    dcb.train_mCs(7, 2);

    % train the master classifier regression with 10,000 dropout iterations leaving in 10 miniclassifers in each dropout iteration
    dcb.train_master_classifier(10000, 10);

    % NOTE: 
    % the hyper-parameters used in this example are different than those used in the COVID-19 prognostic tests study.  
    % the models with these parameters will run on a single core in a sensible amount of time.  In the study,
    % these methods were implemented in an HPC environment and used different, more computationally expensive parameters: 
    % dcb.train_mCs(11, 3);
    % dcb.train_master_classifier(100000, 10);
    % (k=11, considering singles, pairs, and triplets of features for the kNNs 
    % and 100,000 drop out iterations with 10 miniclassifiers left in for each iteration)

    % save the models for predicting on validation set
    dcbs{ibag} = dcb; 

    % oob predictions to get reliable classifications on the development set
    [~, oob_labels] = dcb.predict_master_classifier(se.get_test_data(dxCortex_ftrs).data, 0.5);
    se.set_bag_labels(oob_labels);
end

% print some performance information on the development set using the out of bag labels
dev_labels = se.oob_predict;
dev_defs = se.defs;

fprintf('Development performance (OOB):\n');
fprintf('\n');

fprintf('Higher Risk Group (n=%d):\n', sum(dev_labels == 1));
fprintf('Positives: %d\n', sum(dev_defs(dev_labels == 1)));
fprintf('Negatives: %d\n', sum(dev_defs(dev_labels == 1) == 0));
fprintf('PPV: %d\n', sum(dev_defs(dev_labels == 1))/sum(dev_labels == 1));
fprintf('Recall: %d\n', sum(dev_defs(dev_labels == 1))/sum(dev_defs == 1));
fprintf('\n');

fprintf('Lower Risk Group (n=%d):\n', sum(dev_labels == 0));
fprintf('Positives: %d\n', sum(dev_defs(dev_labels == 0)));
fprintf('Negatives: %d\n', sum(dev_defs(dev_labels == 0) == 0));
fprintf('PPV: %d\n', sum(dev_defs(dev_labels == 0) == 0)/sum(dev_labels == 0));
fprintf('Recall: %d\n', sum(dev_defs(dev_labels == 0) == 0)/sum(dev_defs == 0));
fprintf('\n');

% predict on validation data

% need both sets of features, and if named differently in validation data spreadsheet,
% THE ORDER HERE MUST BE THE SAME AS WAS USED IN TRAINING.
dxCortex_ftrs = {
    'ed_temperature_c',
    'ed_heart_rate',
    'ed_systolic_bp',
    'ed_diastolic_bp',
    'ed_respiratory_rate',
    'ed_oxygen_saturation',
    'initial_qtc',
    'sodium',
    'potassium',
    'carbon_dioxide_bicarb',
    'bun',
    'creatinine',
    'egfr',
    'anion_gap',
    'wbc_screen',
    'hemoglobin',
    'hematocrit',
    'platelet_count',
    'initial_ldh',
    'initial_d_dimer',
    'initial_c_reactive_protein',
    'ferritin'
};

% read the validation data
se.read_val_data('./data/synthetic_validation.csv', 'SampleID', 'Definition'); % definitions must be coded as 1 = positive (endpoint occured), 0 = negative for performance metrics below to make sense

results = [];

for ibag = 1:length(dcbs)

    % get the dcb for this bag
    dcb = dcbs{ibag};

    % apply the master classifier to the validation data
    [~, val_labels] = dcb.predict_master_classifier(se.get_val_data(dxCortex_ftrs).data, 0.5);

    % push labels back to results
    results = [results, val_labels];
end

% results is now a [nsamples, nbags] matrix of labels, need to vote to get final labels
[nsamples, nbags] = size(results);
val_labels = -9999*ones(nsamples,1);
for isample = 1:nsamples
    n0s = 0;
    n1s = 0;
    for ibag = 1:nbags
        if results(isample, ibag) > 0.5
            n1s = n1s + 1;
        else
            n0s = n0s + 1;
        end
    end
    if n1s > n0s
        val_labels(isample) = 1;
    elseif n0s > n1s
        val_labels(isample) = 0;
    end
end

% print some performance information on the validation set 
val_defs = se.val_defs;

fprintf('Validation performance:\n');
fprintf('\n');

fprintf('Higher Risk Group (n=%d):\n', sum(val_labels == 1));
fprintf('Positives: %d\n', sum(val_defs(val_labels == 1)));
fprintf('Negatives: %d\n', sum(val_defs(val_labels == 1) == 0));
fprintf('PPV: %d\n', sum(val_defs(val_labels == 1))/sum(val_labels == 1));
fprintf('Recall: %d\n', sum(val_defs(val_labels == 1))/sum(val_defs == 1));
fprintf('\n');

fprintf('Lower Risk Group (n=%d):\n', sum(val_labels == 0));
fprintf('Positives: %d\n', sum(val_defs(val_labels == 0)));
fprintf('Negatives: %d\n', sum(val_defs(val_labels == 0) == 0));
fprintf('PPV: %d\n', sum(val_defs(val_labels == 0) == 0)/sum(val_labels == 0));
fprintf('Recall: %d\n', sum(val_defs(val_labels == 0) == 0)/sum(val_defs == 0));
fprintf('\n');