%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Thomas Campbell
% Date: March 5, 2020
% Revison log:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description:                                                            
%-------------------------------------------------------------------------
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
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dependencies:
%-------------------------------------------------------------------------
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef c_sample < handle
    % combination sample (c_sample) is the base data point representing a single sample.  it contains the features, defs, 
    % survival data, etc for a single sample

    properties
        features % row vector of feature values
        extra_data % cell array of extra data stored as strings
        survival_data % survival time
        survival_censor % survival censor
        definition % training class label
        groupname % DxCx like groupname
        logit % result logit from a combination method
        filename % unique ID
    end

    methods
        function obj = c_sample()
            obj.features = [];
            obj.extra_data = {};
            obj.survival_data = -999;
            obj.survival_censor = -999;
            obj.definition = -999;
            obj.groupname = '';
            obj.logit = -999;
            obj.filename = '';
        end
    end
end