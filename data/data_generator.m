classdef data_generator < handle
    properties
        cont_atts
        cat_atts
        def_att
    end

    methods
        function obj = data_generator(in_path, cont_ftrs, cat_ftrs, def_str)
            in_cell = readcell(in_path);
            headers = in_cell(1,:);
            in_cell = in_cell(2:end,:);
            nsamples = size(in_cell, 1);

            def_idx = -1;
            for iheader = 1:length(headers)
                if strcmp(def_str, headers{iheader})
                    def_idx = iheader;
                    break;
                end
            end
            if def_idx < 0
                error('could not find definition column')
            end
            obj.def_att.name = def_str;
            obj.def_att.npos = 0;
            obj.def_att.nneg = 0;
            defs = -9999*ones(nsamples,1);
            for isample = 1:nsamples
                defs(isample) = str2double(string(in_cell{isample, def_idx}));
                if defs(isample) == 0
                    obj.def_att.nneg = obj.def_att.nneg + 1;
                else
                    obj.def_att.npos = obj.def_att.npos + 1;
                end
            end

            obj.cont_atts = cell(length(cont_ftrs), 1);
            for iftr = 1:length(cont_ftrs)
                this_att.name = cont_ftrs{iftr};
                this_att.pos = [];
                this_att.neg = [];
                idx = -1;
                for iheader = 1:length(headers)
                    if strcmp(cont_ftrs{iftr}, headers{iheader})
                        idx = iheader;
                        break;
                    end
                end
                if idx < 0
                    error(sprintf('could not find attribute: %s', this_att.name))
                end
                for isample = 1:nsamples
                    if defs(isample) == 0
                        this_att.neg = [this_att.neg; str2double(string(in_cell{isample, idx}))];
                    else
                        this_att.pos = [this_att.pos; str2double(string(in_cell{isample, idx}))];
                    end
                end
                this_att.pos = sort(this_att.pos);
                this_att.neg = sort(this_att.neg);
                obj.cont_atts{iftr} = this_att;
            end

            obj.cat_atts = cell(length(cat_ftrs), 1);
            for iftr = 1:length(cat_ftrs)
                this_att.name = cat_ftrs{iftr};
                this_att.pos = [];
                this_att.neg = [];
                idx = -1;
                for iheader = 1:length(headers)
                    if strcmp(cat_ftrs{iftr}, headers{iheader})
                        idx = iheader;
                        break;
                    end
                end
                if idx < 0
                    error(sprintf('could not find attribute: %s', this_att.name))
                end
                for isample = 1:nsamples
                    if defs(isample) == 0
                        this_att.neg = [this_att.neg; str2double(string(in_cell{isample, idx}))];
                    else
                        this_att.pos = [this_att.pos; str2double(string(in_cell{isample, idx}))];
                    end
                end
                obj.cat_atts{iftr} = this_att;
            end
        end

        function generate(obj, nsamples, out_path)
            out_cell = cell(nsamples+1, length(obj.cat_atts)+length(obj.cont_atts)+2);
            out_cell{1,1} = 'SampleID';
            out_cell{1,2} = 'Definition';
            for iftr = 1:length(obj.cont_atts)
                out_cell{1, iftr+2} = obj.cont_atts{iftr}.name;
            end
            for iftr = 1:length(obj.cat_atts)
                out_cell{1, iftr+2+length(obj.cont_atts)} = obj.cat_atts{iftr}.name;
            end

            for isample = 1:nsamples
                is_pos = false;
                if rand() < obj.def_att.npos/(obj.def_att.npos + obj.def_att.nneg);
                    is_pos = true;
                end
                out_cell{isample+1, 1} = isample;
                if is_pos
                    out_cell{isample+1, 2} = 1;
                else
                    out_cell{isample+1, 2} = 0;
                end

                for iftr = 1:length(obj.cont_atts)
                    idx = iftr + 2;
                    this_att = obj.cont_atts{iftr};
                    if is_pos
                        draw_idx = randi(obj.def_att.npos-2) + 1;
                        this_feature = mean([this_att.pos(draw_idx-1), this_att.pos(draw_idx), this_att.pos(draw_idx+1)]);
                    else
                        draw_idx = randi(obj.def_att.nneg-2) + 1;
                        this_feature = mean([this_att.neg(draw_idx-1), this_att.neg(draw_idx), this_att.neg(draw_idx+1)]);
                    end
                    out_cell{isample+1, idx} = this_feature;
                end

                for iftr = 1:length(obj.cat_atts)
                    idx = iftr + length(obj.cont_atts) + 2;
                    if is_pos
                        draw_idx = randi(obj.def_att.npos-2) + 1;
                        this_feature = obj.cat_atts{iftr}.pos(draw_idx);
                    else
                        draw_idx = randi(obj.def_att.nneg-2) + 1;
                        this_feature = obj.cat_atts{iftr}.neg(draw_idx);
                    end
                    out_cell{isample+1, idx} = this_feature;
                end
            end
            writecell(out_cell, out_path);
        end
    end
end