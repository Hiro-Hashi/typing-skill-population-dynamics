%% Figure 7 analysis
% Reproduces the analyses and visualizations used for Figure 7.
%
% This script:
%   1. performs demixed principal component analysis (dPCA),
%   2. extracts low-dimensional latent trajectories across sessions,
%   3. quantifies latent trajectory length,
%   4. relates trajectory length to typing performance, and
%   5. compares cross-session latent geometry before and after CCA alignment.

% External dependency:
%   dPCA toolbox (see ../Fig4/external/dPCA)


%% Setup
clc
close all
clear

projectDir = string(pwd);

% dPCA
dpca_path = fullfile(projectDir, '..', 'Fig4', 'external', 'dPCA');
addpath(dpca_path);


% Load data

DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;

DataNum = numel(DaysList);

ClipD = struct();

for i = 1:DataNum
    datapath = fullfile(projectDir, '..', 'Fig5', 'data', char("day" + DaysList(i) + ".mat"));
    ClipD.("D" + DaysList(i)) = load(datapath);
end

FingerLabel = ["R Thumb Up",  "R Thumb Down",  "R Thumb In", ...
                "R Index Up",  "R Index Down",  "R Index In", ...
                "R Middle Up", "R Middle Down", "R Middle In", ...
                "R Ring Up",   "R Ring Down",   "R Ring In", ...  
                "R Pinky Up",  "R Pinky Down",  "R Pinky In", ...
                "L Thumb Up",  "L Thumb Down",  "L Thumb In", ...
                "L Index Up",  "L Index Down",  "L Index In", ...
                "L Middle Up", "L Middle Down", "L Middle In", ...
                "L Ring Up",   "L Ring Down",   "L Ring In", ...  
                "L Pinky Up",  "L Pinky Down",  "L Pinky In"];

Hand_List = ["R", "L"];
Finger_List = ["Thumb",  "Index", "Middle", "Ring",  "Pinky"];
Gesture_List = ["Up",  "Down", "In"];

% Construct combined hand/finger labels.
Hand_Finger_List = strings(1, numel(Hand_List)*numel(Finger_List));
k = 1;
for hi = 1:numel(Hand_List)

    for fi = 1:numel(Finger_List)
        Hand_Finger_List(k) = Hand_List(hi) + " " + Finger_List(fi);
        k = k+1;
    end

end

%% Run dPCA

outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end


ArrayList = ["All", "r6d", "l6d"];
ChList = [{1:384}, {257:384}, {1:256}];

taskNum = numel(FingerLabel);

for di = 1:numel(DayList2)
    
    % Select the current session.
    selectDay = DayList2(di);
    
    for ai = 1:numel(ArrayList)
        % Select the array/channel subset.
        usedArray = ArrayList(ai);
        usedCh = ChList{ArrayList == usedArray};
        chNum = numel(usedCh);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Determine recording lengths and trial counts, and extract the data for analysis.
        [all_length, all_trial, selectData] = gather_data_length_trialNum(taskNum, ClipD, selectDay);
        % Identify tasks with available data in this session.
        validInd = ~isnan(all_length); % task contains valid data
        % Resample task trajectories to a common length because durations vary across tasks.
        % tempLength = round(median(all_length(validInd))); % Alternative: session-specific target length.
        % Use a fixed target length across sessions for comparability.
        tempLength = 100;
        Emax = max(all_trial(validInd)); % maximum number of trials
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Retain a balanced Hand x Finger x Gesture task set.
        % Encode each task by hand, finger, and gesture.
        [hand, finger, gesture] = make_hand_finger_gesture();
        % Reshape task availability into the hand/finger/gesture layout.
        Keyboard_Ind = my_keyboard_layout_convert(validInd);
        % Keep only fingers for which Up, Down, and In are all available.
        exlude_finger_ind = any(Keyboard_Ind == 0, 1);
        included_HandFinger = Hand_Finger_List(~exlude_finger_ind); 
        % Keep fingers represented for both hands.
        [Finger_candidate] = extract_same_finger_RightLeft(Finger_List, included_HandFinger);
        % Select tasks forming a complete Hand x Finger x Gesture design.
        valid_index = detect_valid_index(Finger_candidate, FingerLabel, validInd);
        % Task labels used in the current analysis.
        valid_Finger_Label = FingerLabel(valid_index);
        % Convert hand, finger, and gesture labels to analysis indices.
        [hand_v, finger_v, gesture_v] = convert_valid_hfg(hand, finger, gesture, valid_index);
        H=2; % fixed
        F=numel(Finger_candidate); % variable
        G=3; % fixed
        
        % Build the dPCA input array: [channel x hand x finger x gesture x time x trial].
        [firingRates] = create_firingRates(chNum, H, F, G, tempLength, Emax, valid_index, selectData, usedCh, ...
                                            hand_v, finger_v, gesture_v);
        
        % Compute trial-averaged firing rates (PSTHs).
        firingRatesAverage = mean(firingRates, 6, "omitnan");          % [channel x hand x finger x gesture x time]
        
        % Count valid trials for each condition: [channel x hand x finger x gesture].
        trialNum = make_trial_num(chNum, H, F, G, firingRates);
        
        
        % Define dPCA marginalizations.
        % Each main effect is grouped with its interaction with time; remaining terms are grouped as interactions.
        % Parameter indices: 1 = Hand, 2 = Finger, 3 = Gesture, 4 = Time.
        combinedParams = { ...
            {1, [1 4]}, ...   % Hand and Hand x Time
            {2, [2 4]}, ...   % Finger and Finger x Time
            {3, [3 4]}, ...   % Gesture and Gesture x Time
            {4}, ...          % Time (condition-independent)
            {[1 2],[1 3],[2 3],[1 2 3],[1 2 4],[1 3 4],[2 3 4],[1 2 3 4]} % remaining interactions
        };
        margNames   = {'Hand','Finger','Gesture','Common','Interactions'};
        co = colororder("gem");
        margColours = co(1:5, :); % colors for marginalizations
        
        % Time axis.
        bin_sec = 0.01;
        time = linspace(0, (tempLength-1)*bin_sec, tempLength);    % Time starts at the beginning of the warped trajectory.
        timeEvents = [];           % Go presentation = 0s
        
        % First run dPCA without regularization.
        [W,V,whichMarg] = dpca(firingRatesAverage, 20, 'combinedParams', combinedParams);
        explVar = dpca_explainedVariance(firingRatesAverage, W, V, 'combinedParams', combinedParams);
        
        dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
            'explainedVar', explVar, 'whichMarg', whichMarg, ...
            'marginalizationNames', margNames, 'marginalizationColours', margColours, ...
            'time', time, 'timeEvents', timeEvents, 'timeMarginalization', 4, ...
            'legendSubplot', 16);
        drawnow
        % Optional figure export.
        % saveFigName = selectDay + "_dPCA_ClosedTyping_" + usedArray + "array_raw.pdf";
        % F = gcf;
        % exportgraphics(F, char(saveFigName), 'ContentType', 'vector')
        % close all
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%% Run regularized dPCA.
        ifSimultaneousRecording = true; % true because channels were recorded simultaneously within each session
        saveFileName = char(selectDay + " optLambda_FingerSweep_" + usedArray + " array.mat");
        optimalLambda = dpca_optimizeLambda(firingRatesAverage, firingRates, trialNum, ...
            'combinedParams', combinedParams, 'simultaneous', ifSimultaneousRecording, ...
            'numRep', 5, 'filename', saveFileName); % supplying 'filename' enables caching of the optimized lambda
        Cnoise = dpca_getNoiseCovariance(firingRatesAverage, firingRates, trialNum, ...
            'simultaneous', ifSimultaneousRecording);
        
        [Wr,Vr,whichMarg_r] = dpca(firingRatesAverage, 20, ...
            'combinedParams', combinedParams, 'lambda', optimalLambda, 'Cnoise', Cnoise);
        explVar_r = dpca_explainedVariance(firingRatesAverage, Wr, Vr, ...
            'combinedParams', combinedParams);
        
        dpca_plot(firingRatesAverage, Wr, Vr, @dpca_plot_default, ...
            'explainedVar', explVar_r, 'whichMarg', whichMarg_r, ...
            'marginalizationNames', margNames, 'marginalizationColours', margColours, ...
            'time', time, 'timeEvents', timeEvents, 'timeMarginalization', 4, ...
            'legendSubplot', 16);
        drawnow

        % Optional figure export.
        % saveFigName =  selectDay + "_dPCA_ClosedTyping_" + usedArray + "array_lambdaOpt_normalized_var.pdf";
        % F = gcf;
        % exportgraphics(F, char(saveFigName), 'ContentType', 'vector')
        % close all
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Save dPCA outputs.
        saveName = selectDay + "_dPCA_ClosedTyping_" + usedArray + "_normalized_var.mat";
        save(fullfile(outputDir, char(saveName)), 'firingRatesAverage', 'Wr', 'Vr', 'whichMarg_r', 'explVar_r', 'optimalLambda', 'Cnoise', 'usedArray', 'usedCh', ...
            'combinedParams','margNames','margColours','time','timeEvents','selectDay', 'FingerLabel', ...
            'valid_index', 'all_trial', 'all_length', 'trialNum')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        drawnow
    end

end


%% Fig. 7A

DaysList   = ["45", "85", "86", "92", "93", "106", "107"]; DayList2 = "D" + DaysList;

co1 = colororder("gem12"); co2 = colororder("glow12");
close all
co = cat(1, co1, co2);

% Build grouping information used for labels and plotting.
[GroupS] = make_Group_struct();

All_LatentR = struct();

ArrayList = ["All", "r6d", "l6d"];

for ai = 1:numel(ArrayList)

    currentArray = ArrayList(ai);
        
    for di = 1:numel(DayList2)
        
        selectDay = DayList2(di);

        fileName = selectDay + "_dPCA_ClosedTyping_" + currentArray + "_normalized_var.mat";
        dP = load(fullfile(projectDir, 'results', char(fileName)));
        margNames = dP.margNames; whichMatg = dP.whichMarg_r; frAve = dP.firingRatesAverage;
        valid_Ind = dP.valid_index; W = dP.Wr; timeD = dP.time; FingerLabel = dP.FingerLabel;
    
        % Store metadata for this session.
        All_LatentR.(currentArray).(selectDay).valid_Ind = valid_Ind;
        All_LatentR.(currentArray).(selectDay).W = W;
        All_LatentR.(currentArray).(selectDay).timeD = timeD;
        All_LatentR.(currentArray).(selectDay).FingerLabel = FingerLabel;
        All_LatentR.(currentArray).(selectDay).margNames = margNames;
        All_LatentR.(currentArray).(selectDay).whichMatg = whichMatg;
    
        % Flatten PSTHs into task order.
        flattenPSTH = my_flattenPSTH_make(frAve);
        [taskNum, dataNum, ~] = size(flattenPSTH);
        
        for mi = 1:numel(margNames)
            
            current_Marg = string(margNames{mi});
            current_MargInd = find(strcmp(margNames, current_Marg), 1);
            % Retain the first two components for this marginalization.
            upper_ind = find(whichMatg == current_MargInd, 2);
            
            latentData = nan(taskNum, dataNum, 2); % compare only the first two dPCs
            for i = 1:numel(upper_ind)
                currentInd = upper_ind(i);
                allProj = make_all_project_data(currentInd, W, flattenPSTH);
                latentData(:, :, i) = allProj';
            end
    
            % Store the full latent space before subgrouping.
            All_LatentR.(currentArray).(selectDay).(current_Marg).Latent = latentData;
            
            % Obtain subgroup definitions and legend labels.
            try
                currentGroup = GroupS.(current_Marg).componentGroup;
                legendName = GroupS.(current_Marg).legendName;
            catch
                currentGroup = GroupS.("Allf").componentGroup;
                legendName = GroupS.("Allf").legendName;
            end
            
            for gi = 1:numel(currentGroup)
            
                current_subGroup = currentGroup{gi};
                ind = ismember(FingerLabel(valid_Ind), current_subGroup);
                if sum(ind) == 0 % no task belongs to this subgroup
                    continue
                end
            
                dpca1 = (squeeze(latentData(ind, :, 1))); % dPCA1 taskNum * data * 1
                dpca2 = (squeeze(latentData(ind, :, 2))); % dPCA2

                [h1, ~, ci1] = normfit(dpca1);
                [h2, ~, ci2] = normfit(dpca2);

                currentlegendName = erase(string(legendName{gi}), " ");
                All_LatentR.(currentArray).(selectDay).(current_Marg).(currentlegendName).ave1 = h1;
                All_LatentR.(currentArray).(selectDay).(current_Marg).(currentlegendName).ci1(1:2, :) = ci1;
                All_LatentR.(currentArray).(selectDay).(current_Marg).(currentlegendName).ave2 = h2;
                All_LatentR.(currentArray).(selectDay).(current_Marg).(currentlegendName).ci2(1:2, :) = ci2;

                % Store subgroup-specific latent trajectories.
                All_LatentR.(currentArray).(selectDay).(current_Marg).(currentlegendName).Latent = latentData(ind, :, :);

            end
                   
        end
    
    end

end

saveName = "LatentSpace_from_dPCA_across_marg_array_date.mat";
save(fullfile(projectDir, 'results', char(saveName)), 'All_LatentR')


% Gather latent trajectories across sessions.
MargList = ["Hand", "Finger", "Gesture"];
subGlist = {["RightHand", "LeftHand"], ["Thumb", "Middle"], ["Up", "Down", "In"]};

% Average trajectories across tasks within each subgroup and session.
GatherD = gathering_task_average_dpc_data(ArrayList, MargList, subGlist, DayList2, All_LatentR);

%%%%%%%%%%%% Session colors for visualization
dateNum = numel(DayList2);
cm = colormap("winter");   % default colormap contains 256 colors
idx = round(linspace(1, size(cm,1)*0.7, dateNum));
winter7 = cm(idx, :);

cm = colormap("autumn");   % default colormap contains 256 colors
idx = round(linspace(1, size(cm,1)*0.7, dateNum));
autumn7 = cm(idx, :);

cm = colormap("summer");   % default colormap contains 256 colors
idx = round(linspace(1, size(cm,1)*0.7, dateNum));
summer7 = cm(idx, :);

ColorOrder = {winter7, autumn7, summer7}; close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure
t = tiledlayout(numel(ArrayList), numel(MargList));
all_axes = gobjects(numel(ArrayList) * numel(MargList), 1);
ax_c = 1;

% 
All_changes = struct();

for ai = 1:numel(ArrayList) % loop over arrays
    currentArray = ArrayList(ai);
    for mi = 1:numel(MargList) % loop over marginalizations
        % Create the next axes.
        all_axes(ax_c) = nexttile(t, ax_c);
        ax_c = ax_c + 1;
        
        currentMarg = MargList(mi); 
        % Get subgroups for the current marginalization.
        currentSubG = subGlist{MargList == currentMarg};
        % Preallocate graphics objects for the legend.
        h_color = gobjects(numel(currentSubG)*dateNum + 2, 1);
        k = 1;
        LegendNameBox = cell(numel(currentSubG), 1);
        for si = 1:numel(currentSubG) % loop over subgroups
        
            current_cmap = ColorOrder{si};
            currentSubG_sub = currentSubG(si);
            LegendNameBox{si} = DayList2 + " " + currentSubG_sub;
            %dPC1
            p1 = GatherD.(currentArray).(currentMarg).(currentSubG_sub).ave1;
            ci11 = GatherD.(currentArray).(currentMarg).(currentSubG_sub).ci11;
            ci12 = GatherD.(currentArray).(currentMarg).(currentSubG_sub).ci12;
            %dPC2
            p2 = GatherD.(currentArray).(currentMarg).(currentSubG_sub).ave2;
            ci21 = GatherD.(currentArray).(currentMarg).(currentSubG_sub).ci21;
            ci22 = GatherD.(currentArray).(currentMarg).(currentSubG_sub).ci22;
            
            centroid = []; % collect trajectory centroids
            for i = 1:dateNum % loop over sessions
                % Plot the within-session mean trajectory for this subgroup.
                plot(p1{i}, p2{i}, 'LineWidth', 3, 'Color',current_cmap(i, :))
                hold on, grid on
                % Mark trajectory start and end points.
                scatter(p1{i}(1), p2{i}(1), 50, 'o', 'filled', 'k')
                scatter(p1{i}(end), p2{i}(end), 100, '+', 'k')

                %%%%%%%%%%%%%%%%%%%%%% Compute trajectory length in the dPC1-dPC2 plane.
                curveLen = calculate_length_dpc([p1{i}', p2{i}']);
                All_changes.(currentArray).(currentMarg).curveLen.(currentSubG_sub)(i) = curveLen;

                %%%%%%%%%%%%%%%%%%%%%% Compute the trajectory centroid in the dPC1-dPC2 plane.
                centroid(i, :) = mean([p1{i}', p2{i}'], 1);
        
                h_color(k) = plot(nan, nan, 'LineWidth', 3, 'Color',current_cmap(i, :));
                k = k + 1;
            end

            % Quantify centroid shifts between adjacent sessions.
            stepMag = [];
            stepDir = [];
            for i = 1:dateNum -1
                % 2-D displacement vector
                v = centroid(i+1, :) - centroid(i, :);  % 2-D displacement vector
                % displacement magnitude between sessions
                stepMag(i) = norm(v);                   % Euclidean displacement
                % direction from session i to session i+1
                [temp_ang, deg360] = calculate_angle_vector(v);

                stepDir(1, i) = deg360;         % direction in the dPC1-dPC2 plane (degrees, 0-360)
                stepDir(2, i) = temp_ang;         % direction in the dPC1-dPC2 plane (radians)

            end
            
            All_changes.(currentArray).(currentMarg).(currentSubG_sub).stepMag = stepMag;
            All_changes.(currentArray).(currentMarg).(currentSubG_sub).stepDir = stepDir;
        
        end
        
        h_color(k) = scatter(nan, nan, 50, 'o', 'filled', 'k');
        h_color(end) = scatter(nan, nan,  100, '+', 'k');
        
        legendName = cat(2, LegendNameBox{:});
        legendName = cat(2, legendName, ["Start", "End"]);
        
        legend(h_color, legendName, 'Location', 'bestoutside', 'AutoUpdate','off', 'Box','off', 'FontSize', 8)
        box off
        
        xlabel('dPC1'), ylabel('dPC2')
    
        titleName = currentArray + " array, " + currentMarg;
        title(titleName, 'FontSize',14)
       
    end

end


%% Figure 7B

co = colororder("gem");

anaArray = ["All", "r6d", "l6d"];
MargList = ["Hand", "Finger", "Gesture"];
subGlist = {["RightHand", "LeftHand"], ["Thumb", "Middle"], ["Up", "Down", "In"]};

% Collect dPC1-dPC2 trajectory lengths for individual tasks.
LengthR = gathering_lengthData(anaArray, MargList, subGlist, DayList2, All_LatentR);

% Visualize trajectory length across sessions.
figure
tiledlayout(1, 3, 'TileSpacing','loose', 'Padding','loose')

Average_Length = struct(); % retain session means for correlations with performance
for ai = 1:numel(anaArray)
    nexttile
    currentArray = anaArray(ai);

    all_x = [];
    all_y = []; % session index
    all_z = []; % marginalization index
    all_ave = [];
    all_ci1 = [];
    all_ci2 = [];

    for mi = 1:numel(MargList)

        currentMarg = MargList(mi);
        currentSubG = currentMarg;
        
        selectD = LengthR.(currentArray).(currentMarg).(currentSubG); % may be returned as a cell or numeric array
        % Convert the data to long format and compute summary statistics.
        [X, Y, ave, ci1, ci2] = calculate_x_y_data_for_boxchart(selectD, numel(DayList2));
        
        Z = repmat(mi, numel(X), 1);

        all_x = cat(1, all_x, X);
        all_y = cat(1, all_y, Y);
        all_z = cat(1, all_z, Z);
        all_ave = cat(2, all_ave, ave);
        all_ci1 = cat(2, all_ci1, ci1);
        all_ci2 = cat(2, all_ci2, ci2);

    end

    all_ave_tbl = array2table(all_ave, 'VariableNames', MargList);
    Average_Length.(currentArray) = all_ave_tbl;

    % Plot session means and 95% confidence intervals.
    % violinplot(all_y, all_x, 'GroupByColor',all_z)
    hold on
    h_color = gobjects(size(all_ave, 2), 1);
    p_all = nan(size(all_ave, 2), 1);
    LegendNames = strings(size(all_ave, 2), 1);
    for pi = 1:size(all_ave, 2)
        % Plot the 95% confidence interval.
        fill([1:numel(DayList2), flip(1:numel(DayList2))], [all_ci1(:, pi); flip(all_ci2(:, pi))]', co(pi, :), 'FaceAlpha',0.2, 'EdgeColor','none')
        plot(1:numel(DayList2), all_ave(:, pi), 'o-', 'LineWidth',1.5, 'Color', co(pi, :))
        h_color(pi) = plot(nan, nan, 'o-', 'LineWidth',1.5, 'Color', co(pi, :));

        % Test for differences across sessions.
        p = kruskalwallis(all_x(all_z==pi), all_y(all_z == pi), 'off');
        if p > 0.05
            textP = sprintf('(p = %.2f, KW test)', p);
        elseif p > 0.001
            textP = sprintf('(*p = %.3f, KW test)', p);
        else
            textP = sprintf('(*p < 0.001, KW test)');
        end
        LegendNames(pi) = MargList(pi) + string(textP);
    end

    xticks(1:numel(DayList2)), xticklabels(DayList2), xlim([0 numel(DayList2) + 1])
    ylabel({'Latent trajectory length', '(dPC1–2)'})
    legend(h_color, LegendNames, 'Box','off', 'AutoUpdate','off', 'Location','northwest', 'FontSize',10)
    titleName = currentArray + " array";
    title(titleName, 'FontSize',14)

    % Add headroom to reduce overlap between the legend and plotted data.
    ax = gca; Ylimv = ax.YLim; maxYv = max(Ylimv);
    ylim([min(Ylimv), maxYv*1.2])

end

%% Figure 7C
loadName = "RNN_performance_metrical.mat";
perfD = load(fullfile(projectDir, 'data', char(loadName)));

anaArray = ["All"];
MargList = ["Hand","Finger","Gesture"];

co = colororder("gem");      % colors

D_list = {perfD.meanD(:), perfD.meanD_rCER(:)};
L_List_Name = ["Typing speed (CPM)", "Typing error rate (CER/RNN)"];

figure
t = tiledlayout(1, 2);

for datai = 1:numel(D_list)
    D = D_list{datai};          % performance metric as a column vector
    

    for ai = 1:numel(anaArray)
        currentArray = anaArray(ai);
        
        nexttile
        hold on
        
        h_color = gobjects(numel(MargList), 1);
        legendNameList = cell(numel(MargList), 1);
        for mi = 1:numel(MargList)
            marg = MargList(mi);
        
            % ---- x: latent trajectory length across sessions ----
            x = Average_Length.(currentArray ).(marg)(:);
        
            % ---- scatter plot ----
            scatter(x, D, 100, co(mi,:), 'o', ...
                'filled','MarkerEdgeColor','none', ...
                'DisplayName', marg)
        
            % ---- Linear regression ----
            mdl = fitlm(x, D);
            b = mdl.Coefficients.Estimate;  % [intercept; slope]
        
            % Plot the fitted regression line.
            xx = linspace(min(x), max(x), 100);
            yy = b(1) + b(2)*xx;
            plot(xx, yy, 'Color', co(mi,:), 'LineWidth',3)
        
            % ---- Pearson correlation ----
            [r, p] = corr(x, D);
            if p > 0.05
                dispT = sprintf('%s:r = %.2f (p = %.2f)', marg, r, p);
            elseif p >= 0.001
                dispT = sprintf('%s:r = %.2f (*p = %.3f)', marg, r, p);
            else
                dispT = sprintf('%s:r = %.2f (*p < 0.001)',marg, r);
            end
        
            legendNameList{mi} = dispT;
            % Optional: display r and p directly on the axes.
            % text(5.8, 80-(mi-1)*3, dispT, 'Color', [0 0 0], 'FontSize', 10)
        
            h_color(mi) = plot(nan, nan, '-', 'Color', co(mi, :), 'LineWidth',2);
        end
        
        legend(h_color, legendNameList, 'AutoUpdate','off', 'Box','off', 'Location', 'bestoutside')
        
        xlabel('Latent trajectory length') 
        xlim([0 9])
        ylabel(L_List_Name(datai))
        ax = gca; ylimv = ax.YLim; ylim([ylimv(1)*0.9, ylimv(2)*1.1])
        title(currentArray)
        grid on
    end

end

%% Figure 7D

LD = load(fullfile(projectDir, 'results', 'LatentSpace_from_dPCA_across_marg_array_date.mat'));

% Load variables required for cross-session comparisons.
FingerLabelRef = LD.All_LatentR.All.D45.FingerLabel;
LabelNum = numel(FingerLabelRef);
MargList = ["Hand", "Finger", "Gesture"];
DaysList   = [45, 85, 86, 92, 93, 106, 107];
DayList2 = "D" + string(DaysList);
% Generate all pairwise combinations of recording sessions.
dayNum = numel(DaysList);
comList = nchoosek(1:dayNum, 2); combNum = size(comList, 1);
comList_day1 = DaysList(comList(:, 1))'; comList_day2 = DaysList(comList(:, 2))';
DurationList = comList_day2 - comList_day1;

ArrayList = ["r6d", "l6d"];

% Compute cross-session correlations and dPC1-dPC2 trajectory lengths; Spearman correlation is used.
[R_Length_all] = create_dpc_r_and_length_row_and_aligned_duration(ArrayList, MargList, comList, DaysList, DayList2, LD, LabelNum);
usedDiffList = ["diffAb"]; % choose "diff" or "diffAb"
co = colororder("gem");      % colors
usedTable = "Length";

for ai = 1:numel(ArrayList)
    anaArray = ArrayList(ai);
    figure
    t = tiledlayout(numel(usedDiffList), numel(MargList)*2, 'TileSpacing','loose', 'Padding','loose');
    for diffti = 1:numel(usedDiffList)
    
        usedDiff = usedDiffList(diffti); % choose "diff" or "diffAb"
    
        for mi = 1:numel(MargList)
            
            nexttile
            anaMarg = MargList(mi);
    
            legendName = cell(1, 2);
            % Extract unaligned trajectory lengths.
            [day_cat, data_cat1] = day_data_cat(R_Length_all.(anaArray).(anaMarg).(usedTable), "L", 1); % Day 1 trajectory length
            [~,       data_cat2] = day_data_cat(R_Length_all.(anaArray).(anaMarg).(usedTable), "L", 2); % Day 2 trajectory length
            diff_data = (data_cat1- data_cat2); % Day 1 precedes Day 2; signed difference is Day 1 minus Day 2.
            diff_data_abs = abs(data_cat1- data_cat2);
            % Store both signed and absolute between-session differences.
            DiffLengthUn = struct(); DiffLengthUn.diff = diff_data; DiffLengthUn.diffAb = diff_data_abs;
            scatter(day_cat, DiffLengthUn.(usedDiff), 50, 'o', 'filled', 'MarkerFaceColor', co(1, :), 'MarkerEdgeColor','none', 'MarkerFaceAlpha',0.3)
            [xx, yy, r, p, dispT] = fit_lm_and_corr(day_cat, DiffLengthUn.(usedDiff));
            hold on
            plot(xx, yy, '-', 'LineWidth',3, 'Color', co(1, :))
            legendName{1} = "UnAligned: " + string(dispT);
            
            % Extract CCA-aligned trajectory lengths.
            [~, data_cat1] = day_data_cat(R_Length_all.(anaArray).(anaMarg).(usedTable), "Laligned", 1); % Day 1 trajectory length
            [~, data_cat2] = day_data_cat(R_Length_all.(anaArray).(anaMarg).(usedTable), "Laligned", 2); % Day 2 trajectory length
            diff_data = (data_cat1- data_cat2); % Day 1 precedes Day 2; signed difference is Day 1 minus Day 2.
            diff_data_abs = abs(data_cat1- data_cat2);
            % Store both signed and absolute between-session differences.
            DiffLengthAl = struct(); DiffLengthAl.diff = diff_data; DiffLengthAl.diffAb = diff_data_abs;
            scatter(day_cat, DiffLengthAl.(usedDiff), 50, 'o', 'filled', 'MarkerFaceColor', co(2, :), 'MarkerEdgeColor','none', 'MarkerFaceAlpha',0.3)
            [xx, yy, r, p, dispT] = fit_lm_and_corr(day_cat, DiffLengthAl.(usedDiff));
            hold on
            plot(xx, yy, '-', 'LineWidth',3, 'Color', co(2, :))
            legendName{2} = "Aligned: " + string(dispT);
            
            if usedDiff == "diff"
                ylabelName = {"Change in latent trajectory length", "(Day1 - Day2)"};
            elseif usedDiff == "diffAb"
                ylabelName = {"Absolute difference in",  "latent trajectory length"};
            end
            ylabel(ylabelName)
            xlabel('Inter-day interval (days)')
            xlim([-5 70])
            ax = gca; yLimV = ax.YLim; ylim_min = min(yLimV) - abs(diff(yLimV))/5;
            ylim([ylim_min, max(yLimV)])
            
            h_color= gobjects(2, 1);
            h_color(1) = plot(nan, nan, '-', 'LineWidth',3, 'Color', co(1, :));
            h_color(2) = plot(nan, nan, '-', 'LineWidth',3, 'Color', co(2, :));
            legend(h_color, legendName, 'Box','off', 'Location', 'southwest')
            
            titleName = anaArray + " array, " + anaMarg;
            title(titleName, 'FontSize', 14)
            
            % Compare unaligned and aligned differences using histograms.
            nexttile
            histogram(DiffLengthUn.(usedDiff), 'FaceAlpha',0.5, 'EdgeColor','none', 'FaceColor',co(1, :), 'BinWidth',0.5)
            hold on, histogram(DiffLengthAl.(usedDiff), 'FaceAlpha',0.5, 'EdgeColor','none', 'FaceColor',co(2, :), 'BinWidth',0.5)
            % Overlay the median and interquartile range for each distribution.
            ax = gca; ylimval = ax.YLim; yPos = max(ylimval); ylim([0, yPos*1.1])

            h = median(DiffLengthUn.(usedDiff)); ci = quantile(DiffLengthUn.(usedDiff), [0.25, 0.75]);
            scatter(h, yPos*0.97, 25, co(1,:), 'o', 'filled', 'MarkerEdgeColor','none')
            plot(ci, [yPos*0.97, yPos*0.97], '-', 'Color', co(1, :), 'LineWidth',2)

            h = median(DiffLengthAl.(usedDiff)); ci = quantile(DiffLengthAl.(usedDiff), [0.25, 0.75]);
            scatter(h, yPos, 25, co(2,:), 'o', 'filled', 'MarkerEdgeColor','none')
            plot(ci, [yPos, yPos], '-', 'Color', co(2, :), 'LineWidth',2)
            
            % Use a paired Wilcoxon signed-rank test because the observations are matched.
            p_sr = signrank(DiffLengthUn.(usedDiff), DiffLengthAl.(usedDiff));
            
            h_color = gobjects(3, 1);
            h_color(1) = histogram(nan, 'FaceAlpha',0.5, 'EdgeColor','none', 'FaceColor',co(1, :), 'BinWidth',0.5);
            h_color(2) = histogram(nan, 'FaceAlpha',0.5, 'EdgeColor','none', 'FaceColor',co(2, :), 'BinWidth',0.5);
            h_color(3) = plot(nan, nan, 'Color', [1 1 1]);
            
            
            if p_sr > 0.05
                dispT = sprintf('p = %.2f (paired Wilcoxon)', p_sr);
            elseif p_sr >= 0.001
                dispT = sprintf('*p = %.3f (paired Wilcoxon)', p_sr);
            else
                dispT = sprintf('*p < 0.001 (paired Wilcoxon)');
            end
            
            legendName = {"Unaligned", "Aligned", dispT};
            legend(h_color, legendName, 'Box','off', 'Location', 'best')
            
            box off
            title(titleName, 'FontSize', 14)
            xlabel(ylabelName)
            ylabel('Count')
    
        end
    end

end




%% Local functions

function [all_length, all_trial, selectData] = gather_data_length_trialNum(taskNum, ClipD, selectDay)

    all_length = nan(taskNum, 1);
    all_trial = nan(taskNum, 1);
    selectData = ClipD.(selectDay).refined_Warping_zFR; 
    
    for taski = 1:taskNum
        currentData = selectData.("T" + string(taski));
        if isempty(currentData)
            continue
        end
        all_length(taski, :) = size(currentData, 2);
        all_trial(taski, :) = size(currentData, 1);
    end

end

function [hand, finger, gesture] = make_hand_finger_gesture()
    % ---- Map task index to (hand, finger, gesture) ----
    task = (1:30).';
    hand    = ones(30,1);  
    hand(task>=16)=2;                     % 1:right, 2:left
    within  = task - (hand - 1) * 15;                                 % 1..15 indices
    finger  = ceil(within/3);                                         % 1:thumb .. 5:pinky
    gesture = mod(within-1,3)+1;                                      % 1:Up, 2:Down, 3:In

end

function [KeyboardLayout] = my_keyboard_layout_convert(validInd)


    KeyboardLayout = nan(3, 10);
    for i = 1:10
        sInd = (i-1)*3+1;
        eInd = i*3;
        KeyboardLayout(:, i) = validInd(sInd:eInd);
    end


end

function [Finger_candidate] = extract_same_finger_RightLeft(Finger_List, included_HandFinger)
    Finger_candidate = strings();
    k = 1;
    for fi = Finger_List
        currentRightFinger = "R " + fi;
        currentLeftFinger = "L " + fi;
        rightFingerInd = ismember(currentRightFinger, included_HandFinger);
        leftFingerInd = ismember(currentLeftFinger, included_HandFinger);
    
        if rightFingerInd && leftFingerInd
            Finger_candidate(k) = fi;
            k = k + 1;
        end
    end
end

function [valid_index] = detect_valid_index(Finger_candidate, FingerLabel, validInd)
    commonF_ind = [];
    for i = 1:numel(Finger_candidate)
        ind = contains(FingerLabel, Finger_candidate(i));
        commonF_ind = cat(2, commonF_ind, ind');
    end

    valid_index = sum(commonF_ind, 2) > 0 & validInd;
end

function [hand_v, finger_v, gesture_v] = convert_valid_hfg(hand, finger, gesture, valid_index)

    hand_v = hand;
    hand_v(~valid_index) = nan;

    gesture_v = gesture;
    gesture_v(~valid_index) = nan;

    finger_candidate = finger(valid_index);
    unique_vals = unique(finger_candidate); 

    finger_v = finger;
    finger_v(~valid_index) = nan;
    for i = 1:numel(unique_vals)
    
        ind = finger_v == unique_vals(i);
        finger_v(ind) = i;
    end

end

function [firingRates] = create_firingRates(chNum, H, F, G, tempLength, Emax, valid_index, selectData, usedCh, hand_v, finger_v, gesture_v)

    taskNum = numel(valid_index);

    firingRates = nan(chNum, H, F, G, tempLength, Emax);

    for taski = 1:taskNum
    
        if ~valid_index(taski)
            continue
        end
    
        currentClipData = selectData.("T" + string(taski));
        trialNum = size(currentClipData, 1);
    
        temp_all_d = nan(chNum, tempLength, trialNum);
        for triali = 1:trialNum
            selectD = squeeze(currentClipData(triali, :, usedCh));
            Y = stretch_template_linear(selectD, tempLength);
            temp_all_d(:, :, triali) = Y';
            
        end
    
        currentH = hand_v(taski);
        currentF = finger_v(taski);
        currentG = gesture_v(taski);
    
        firingRates(:, currentH, currentF, currentG, :, 1:trialNum) = temp_all_d;
    
    end

end

function [trialNum] = make_trial_num(chNum, H, F, G, firingRates)
    trialNum = zeros(chNum,H,F,G);
    for h=1:H
        for f=1:F 
            for g=1:G
                x = squeeze(firingRates(1,h,f,g,1,:));         
                trialNum(:,h,f,g) = sum(~isnan(x));
            end
        end
    end

end

function Y = stretch_template_linear(X, Lout)

    [Tin, ~] = size(X); 
    if Lout == Tin
        Y = X; 
        return; 
    end
    xi = linspace(1, Tin, Lout);

    Y  = interp1(1:Tin, X, xi, 'linear'); 
end

function [GroupS] = make_Group_struct()

    [R_Thumb, R_Index, R_Middle, R_Ring, R_Pinky, L_Thumb, L_Index, L_Middle, L_Ring, L_Pinky, ...
        Right_F, Left_F, Up_F, Down_F, In_F] = make_group_data();
    
    
    GroupS = struct();

    GroupS.("Hand").componentGroup = {Right_F, Left_F}; 
    GroupS.("Hand").legendName = {'Right Hand', 'Left Hand'};
    GroupS.("Hand").LabelName = "R/L Hand";

    GroupS.("Finger").componentGroup = {[R_Thumb, L_Thumb], [R_Index, L_Index], [R_Middle, L_Middle], [R_Ring, L_Ring], [R_Pinky, L_Pinky]}; 
    GroupS.("Finger").legendName = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'};
    GroupS.("Finger").LabelName = "Each finger";

    GroupS.("Gesture").componentGroup = {Up_F, Down_F, In_F}; 
    GroupS.("Gesture").legendName = {'Up', 'Down', 'In'};
    GroupS.("Gesture").LabelName = "UpDwIn";

    GroupS.("Allf").componentGroup = {R_Thumb, R_Index, R_Middle, R_Ring, R_Pinky, L_Thumb, L_Index, L_Middle, L_Ring, L_Pinky}; 
    GroupS.("Allf").legendName = {'R Thumb', 'R Index', 'R Middle', 'R Ring', 'R Pinky', 'L Thumb', 'L Index', 'L Middle', 'L Ring', 'L Pinky'};
    GroupS.("Allf").LabelName = "All finger";

end

function [R_Thumb, R_Index, R_Middle, R_Ring, R_Pinky, L_Thumb, L_Index, L_Middle, L_Ring, L_Pinky, Right_F, Left_F, ...
    Up_F, Down_F, In_F] = make_group_data()

    R_Thumb  = ["R Thumb Up",  "R Thumb Down",  "R Thumb In"];
    R_Index  = ["R Index Up",  "R Index Down",  "R Index In"];
    R_Middle = ["R Middle Up", "R Middle Down", "R Middle In"];
    R_Ring   = ["R Ring Up",   "R Ring Down",   "R Ring In"];
    R_Pinky = ["R Pinky Up",  "R Pinky Down",  "R Pinky In"];
    L_Thumb  = ["L Thumb Up",  "L Thumb Down",  "L Thumb In"];
    L_Index  = ["L Index Up",  "L Index Down",  "L Index In"];
    L_Middle = ["L Middle Up", "L Middle Down", "L Middle In"];
    L_Ring   = ["L Ring Up",   "L Ring Down",   "L Ring In"];
    L_Pinky = ["L Pinky Up",  "L Pinky Down",  "L Pinky In"];
    
    Right_F = ["R Thumb Up",  "R Thumb Down",  "R Thumb In", "R Index Up",  "R Index Down",  "R Index In", "R Middle Up", "R Middle Down", "R Middle In", ...
        "R Ring Up",   "R Ring Down",   "R Ring In", "R Pinky Up",  "R Pinky Down",  "R Pinky In"];
    Left_F = ["L Thumb Up",  "L Thumb Down",  "L Thumb In", "L Index Up",  "L Index Down",  "L Index In", "L Middle Up", "L Middle Down", "L Middle In", ...
        "L Ring Up",   "L Ring Down",   "L Ring In", "L Pinky Up",  "L Pinky Down",  "L Pinky In"];
    
    Up_F = ["R Thumb Up","R Index Up","R Middle Up","R Ring Up","R Pinky Up","L Thumb Up","L Index Up","L Middle Up","L Ring Up","L Pinky Up"];
    Down_F = ["R Thumb Down","R Index Down","R Middle Down","R Ring Down","R Pinky Down","L Thumb Down","L Index Down","L Middle Down","L Ring Down","L Pinky Down"];
    In_F = ["R Thumb In","R Index In","R Middle In","R Ring In","R Pinky In","L Thumb In","L Index In","L Middle In","L Ring In","L Pinky In"];

end

function [flattenPSTH] = my_flattenPSTH_make(inputData)

    [chNum, hnum, fnum, gnum, dataNum] = size(inputData);
    taski = 1;
    flattenPSTH = nan(hnum*fnum*gnum, dataNum, chNum);
    for h = 1:hnum
        for f = 1:fnum
            for g = 1:gnum
                currentData = squeeze(inputData(:, h, f, g, :));
                flattenPSTH(taski, :, :) = currentData';
                taski = taski + 1;
            end
        end
    end

end

function [allProj] = make_all_project_data(selectComponent, W, flattenPSTH)

    taskNum = size(flattenPSTH, 1);

    Wr = W(:, selectComponent);
    
    allProj = [];
    for selectTask = 1:taskNum
    
        selectData = squeeze(flattenPSTH(selectTask, :, :));
        
        proj = selectData * Wr;
    
        allProj = cat(2, allProj, proj);
        
    end

end

function [GatherD] = gathering_task_average_dpc_data(ArrayList, MargList, subGlist, DayList2, All_LatentR)

    GatherD = struct();
    for ai = 1:numel(ArrayList)
        currentArray = ArrayList(ai);
    
        for mi = 1:numel(MargList) 
            currentMarg = MargList(mi);
            currentSubG = subGlist{MargList == currentMarg};
    
            for si = 1:numel(currentSubG)
                currentSubGname = currentSubG(si);
    
                % dPC1
                ave_dpc1 = cell(numel(DayList2),1);
                ci1_dpc1 = cell(numel(DayList2),1);
                ci2_dpc1 = cell(numel(DayList2),1);
    
                %dPC2
                ave_dpc2 = cell(numel(DayList2),1);
                ci1_dpc2 = cell(numel(DayList2),1);
                ci2_dpc2 = cell(numel(DayList2),1);
    
                for di = 1:numel(DayList2)
                    currentDate = DayList2(di);
                    ave_dpc1{di} = All_LatentR.(currentArray).(currentDate).(currentMarg).(currentSubGname).ave1;
                    ci1_dpc1{di} = All_LatentR.(currentArray).(currentDate).(currentMarg).(currentSubGname).ci1(1, :);
                    ci2_dpc1{di} = All_LatentR.(currentArray).(currentDate).(currentMarg).(currentSubGname).ci1(2, :);
    
                    ave_dpc2{di} = All_LatentR.(currentArray).(currentDate).(currentMarg).(currentSubGname).ave2;
                    ci1_dpc2{di} = All_LatentR.(currentArray).(currentDate).(currentMarg).(currentSubGname).ci2(1, :);
                    ci2_dpc2{di} = All_LatentR.(currentArray).(currentDate).(currentMarg).(currentSubGname).ci2(2, :);
    
                end
    
                GatherD.(currentArray).(currentMarg).(currentSubGname).ave1 = ave_dpc1;
                GatherD.(currentArray).(currentMarg).(currentSubGname).ci11 = ci1_dpc1;
                GatherD.(currentArray).(currentMarg).(currentSubGname).ci12 = ci2_dpc1;
    
                GatherD.(currentArray).(currentMarg).(currentSubGname).ave2 = ave_dpc2;
                GatherD.(currentArray).(currentMarg).(currentSubGname).ci21 = ci1_dpc2;
                GatherD.(currentArray).(currentMarg).(currentSubGname).ci22 = ci2_dpc2;
            end
    
        end
    
    end
end

function [curveLen] = calculate_length_dpc(traj)

    diffs = diff(traj, 1, 1);                      % (T-1) x 2
    segmentLen = sqrt(sum(diffs.^2, 2));           
    curveLen = sum(segmentLen);                    
end

function [temp_ang, deg360] = calculate_angle_vector(v)
    temp_ang = atan2(v(2), v(1));
    deg = rad2deg(temp_ang);        % -180 to 180 degrees
    deg360 = mod(deg, 360);        % convert to 0-360 degrees 
end

function [LengthR] = gathering_lengthData(anaArray, MargList, subGlist, DayList2, All_LatentR)

    LengthR = struct();
    for ai = 1:numel(anaArray)
        currentArray = anaArray(ai);
        for mi = 1:numel(MargList)
            currentMarg = MargList(mi);
            selectSubG = subGlist{MargList == currentMarg};
    
            lengthMarg = [];
            for si = 1:numel(selectSubG)
                currentSubG = selectSubG(si);
    
                lengthDate = cell(numel(DayList2), 1);
                for di = 1:numel(DayList2)
                    currentDay = DayList2(di);
                    latentD = All_LatentR.(currentArray).(currentDay).(currentMarg).(currentSubG).Latent;
    
                    if ndims(latentD) ~= 3
                        error('Expected latentD to be a 3-D array: task x time x component.')
                    end
                    nsize = size(latentD, 1);
                    
                    lengthD = [];
                    for ni = 1:nsize
                        clipdata = squeeze(latentD(ni, :, :));
                        lengthD(end+1) = calculate_length_dpc(clipdata);
                    end
                    
                    lengthDate{di} = lengthD;
    
                end
    
                if isempty(lengthMarg)
                    lengthMarg = lengthDate;
                else
                    lengthMarg = cat(2, lengthMarg, lengthDate);
                end
            end
    
            cat_length = cell(numel(DayList2), 1);
            % Concatenate data across all subgroups.
            for di = 1:numel(DayList2)
                cat_length{di} = cat(2, lengthMarg{di, :});
            end
            
            lengthMarg = cat(2, lengthMarg, cat_length);
            lengthMarg = cell2table(lengthMarg, 'VariableNames', [selectSubG, currentMarg]); lengthMarg.Properties.RowNames = DayList2;
            LengthR.(currentArray).(currentMarg) = lengthMarg;
        end
    
    end


end

function [X, Y, ave, ci1, ci2] = calculate_x_y_data_for_boxchart(selectD, DateNum)
    X = [];
    Y = [];
    ave = [];
    ci1 = [];
    ci2 = [];
    for di = 1:DateNum
    
        try 
            D = selectD{di};
        catch 
            D = selectD(di, :);
        end
        dnum = numel(D);
        x_data = reshape(D, dnum, 1);
        y_data = repmat(di, dnum, 1);
    
        X = cat(1, X, x_data);
        Y = cat(1, Y, y_data);
    
        [h, ~, ci] = normfit(x_data);
        ave(end+1) = h;
        ci1(end+1) = ci(1);
        ci2(end+1) = ci(2);
    
    end

    ave = reshape(ave, numel(ave), 1);
    ci1 = reshape(ci1, numel(ci1), 1);
    ci2 = reshape(ci2, numel(ci2), 1);
end

function [R_Length_all] = create_dpc_r_and_length_row_and_aligned_duration(ArrayList, MargList, comList, DaysList, DayList2, LD, LabelNum)
    R_Length_all = struct();
    
    comList_day1 = DaysList(comList(:, 1))';
    comList_day2 = DaysList(comList(:, 2))';
    combNum = size(comList, 1);
    DurationList = comList_day2 - comList_day1;

    for ai = 1:numel(ArrayList)
 
        AnalyzeArray = ArrayList(ai);
    
        for mi = 1:numel(MargList)

            AnalyzeMarg = MargList(mi);
            
            R_acrossDay = cell(combNum, 1);
            Length_acrossDay = cell(combNum, 1);
            for combi = 1:combNum

                currentComb = comList(combi, :);
                Day1 = DayList2(currentComb(1));
                Day2 = DayList2(currentComb(2));
                latent1 = LD.All_LatentR.(AnalyzeArray).(Day1).(AnalyzeMarg).Latent;
                latent2 = LD.All_LatentR.(AnalyzeArray).(Day2).(AnalyzeMarg).Latent;
            
                [validTask1, validTask2] = get_same_tasknumber(LD, AnalyzeArray, Day1, Day2, LabelNum);
                   
                r_all = [];
                r_all_aligned = [];
            
                length_all = [];
                length_all_aligned = [];
            
       
                for ti = 1:numel(validTask1)
                    
                    l1 = squeeze(latent1(validTask1(ti), :, :)); 
                    l2 = squeeze(latent2(validTask2(ti), :, :));
            
                    
                    r = abs(diag(corr(l1, l2, 'Type', 'Spearman')));
                    r_all = cat(2, r_all, r); 
                    
                   
                    len1 = calculate_length_dpc(l1);
                    len2 = calculate_length_dpc(l2);
                    length_all = cat(2, length_all, [len1;len2]);
            
                    
                    [L1_aligned, L2_aligned, A, B, ~] = align_latent_cca(l1, l2);
                    
                    r = abs(diag(corr(L1_aligned, L2_aligned, 'Type', 'Spearman')));
                    r_all_aligned = cat(2, r_all_aligned, r);
            
                    
                    len_al_1 = calculate_length_dpc(L1_aligned);
                    len_al_2 = calculate_length_dpc(L2_aligned);
                    length_all_aligned = cat(2, length_all_aligned, [len_al_1; len_al_2]);
                      
                end
            
            
                R_acrossDay{combi, 1} = r_all;
                R_acrossDay{combi, 2} = r_all_aligned;
            
                Length_acrossDay{combi, 1} = length_all;
                Length_acrossDay{combi, 2} = length_all_aligned;
            
            end
            
            
            R_tbl = table(comList_day1, comList_day2, DurationList, R_acrossDay(:, 1), R_acrossDay(:, 2), ...
                'VariableNames',{'Day1', 'Day2', 'Duration', 'R', 'Raligned'});
            
            Length_tbl = table(comList_day1, comList_day2, DurationList, Length_acrossDay(:, 1), Length_acrossDay(:, 2), ...
                'VariableNames',{'Day1', 'Day2', 'Duration', 'L', 'Laligned'});
    
            R_Length_all.(AnalyzeArray).(AnalyzeMarg).corrR = R_tbl;
            R_Length_all.(AnalyzeArray).(AnalyzeMarg).Length = Length_tbl;
        end
    
    end


end

function [validTask1, validTask2] = get_same_tasknumber(LD, AnalyzeArray, Day1, Day2, LabelNum)

    validTask = LD.All_LatentR.(AnalyzeArray).(Day1).valid_Ind & LD.All_LatentR.(AnalyzeArray).(Day2).valid_Ind;

    taskTemp1 = nan(LabelNum, 1); 
    taskTemp2 = nan(LabelNum, 1);

    taskTemp1(LD.All_LatentR.(AnalyzeArray).(Day1).valid_Ind) = 1:sum(LD.All_LatentR.(AnalyzeArray).(Day1).valid_Ind);
    validTask1 = taskTemp1(validTask);

    taskTemp2(LD.All_LatentR.(AnalyzeArray).(Day2).valid_Ind) = 1:sum(LD.All_LatentR.(AnalyzeArray).(Day2).valid_Ind);
    validTask2 = taskTemp2(validTask);

    if numel(validTask1) ~= numel(validTask2)
        error('The task extraction is incorrect.')
    end
end


function [L1_aligned, L2_aligned, A, B, r] = align_latent_cca(L1, L2)
%ALIGN_LATENT_CCA Align two latent trajectories using canonical correlation analysis.
%
%   [L1_ALIGNED, L2_ALIGNED, A, B, R] = ALIGN_LATENT_CCA(L1, L2)
%   aligns two latent trajectories using canonical correlation analysis
%   (CCA).
%
%   Inputs
%   ------
%   L1, L2 : T-by-D numeric matrices
%       Latent trajectories to be aligned, where T is the number of time
%       points and D is the number of latent dimensions. L1 and L2 must
%       have the same size.
%
%   Outputs
%   -------
%   L1_aligned, L2_aligned : T-by-D matrices
%       Latent trajectories projected onto their respective canonical
%       coordinate systems.
%
%   A, B : D-by-D matrices
%       Canonical coefficient matrices for L1 and L2, respectively.
%
%   r : D-by-1 vector
%       Canonical correlations between corresponding dimensions.

    % Center each latent dimension across time.
    X = L1 - mean(L1, 1);
    Y = L2 - mean(L2, 1);

    % Perform canonical correlation analysis.
    [A, B, r] = canoncorr(X, Y);

    % Project the latent trajectories onto the canonical coordinate systems.
    L1_aligned = X * A;
    L2_aligned = Y * B;

end

function [day_cat, data_cat] = day_data_cat(usedTable, selectRow, dPC_num)
%DAY_DATA_CAT Concatenate pairwise-session data and corresponding intervals.
%
%   DAY_CAT contains the inter-session interval repeated for each value in
%   DATA_CAT. SELECTROW specifies the table variable containing the data,
%   and DPC_NUM selects the requested dPC row.

    numRow = height(usedTable);
    day_cat = [];
    data_cat = [];
    for ri = 1:numRow
    
        rowD = usedTable.(selectRow){ri}(dPC_num, :);
        rowD = rowD(:); 
        data_cat = cat(1, data_cat, rowD);
    
        dayD = repmat(usedTable.Duration(ri), numel(rowD), 1);
        day_cat = cat(1, day_cat, dayD);
    
    end

end

function [xx, yy, r, p, dispT] = fit_lm_and_corr(x, D)
%FIT_LM_AND_CORR Fit a linear model and compute a Pearson correlation.
%
%   XX and YY define the fitted regression line. R and P are the Pearson
%   correlation coefficient and its p-value. DISPT is a formatted string
%   used in figure legends.

    mdl = fitlm(x, D);
    b = mdl.Coefficients.Estimate;  % [intercept; slope]
    
    xx = linspace(min(x), max(x), 100);
    yy = b(1) + b(2)*xx;
    
    [r, p] = corr(x, D);
    if p > 0.05
        dispT = sprintf('r = %.2f (p = %.2f)', r, p);
    elseif p >= 0.001
        dispT = sprintf('r = %.2f (*p = %.3f)', r, p);
    else
        dispT = sprintf('r = %.2f (*p < 0.001)',r);
    end

end
