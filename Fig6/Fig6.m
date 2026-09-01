%% Figure 6: dimensionality and connectivity analyses
% This script reproduces the analyses and visualizations shown in Figure 6.

% See README.md for installation instructions, data organization, and
% software requirements.

%% Setup
clc
close all
clear

projectDir = string(pwd);

% Load data

DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;

DataNum = numel(DaysList);

ClipD = struct();

for i = 1:DataNum
    datapath = fullfile(projectDir, '..', 'Fig5', 'data', char("day" + DaysList(i) + ".mat"));
    ClipD.("D" + DaysList(i)) = load(datapath);
end

%% Fig. 6A

ArrayList = ["r6d", "l6d"];
ChList = [{257:384}, {1:256}];
% Time-warp PSTHs to a common template length. Results are similar for tempLen >= 20.
tempLen = 20;

% Define task groupings analogous to marginalizations in dPCA.
HandTask = {1:15, 16:30}; HandTaskName = ["Right", "Left"];
FingerTask = {[1:3, 16:18], [4:6, 19:21], [7:9, 22:24], [10:12, 25:27], [13:15, 28:30]}; FingerTaskName = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
GestureTask = {1:3:28, 2:3:29, 3:3:30}; GestureTaskName = ["Up", "Down", "In"];

MargiList = ["Hand", "Finger", "Gesture"];
TaskList = {HandTask, FingerTask, GestureTask};
TaskNameList = {HandTaskName, FingerTaskName, GestureTaskName};

Result_PCnum = struct();

% Start a parallel pool if one is not already running.
if isempty(gcp('nocreate'))
    parpool(12);  % number of workers used in the original analysis
end

for mi = 1:numel(MargiList)
    CurrentMarg = MargiList(mi);
    
    currentTask = TaskList{MargiList==CurrentMarg};
    currentTaskName = TaskNameList{MargiList==CurrentMarg};
    currentTaskNum = numel(currentTask);
    
    for si = 1:currentTaskNum
        selectTaskInd = currentTask{si}; % task indices included in the current group
        selectTaskName = currentTaskName{si};
        
        ResultDim = nan(numel(DayList2), numel(ArrayList));
        for ai = 1:numel(ArrayList)
        
            usedArray = ArrayList(ai);
            usedCh = ChList{ArrayList == usedArray};
            chNum = numel(usedCh);
        
            for di = 1:numel(DayList2)
                currentDate = DayList2(di);
                currentData = ClipD.(currentDate).refined_Warping_zFR;
        
                % Compute PSTHs and time-warp them to the common template length.
                [PSTH, PSTH_orig, validInd] = create_PSTH_with_templateLength(ClipD, currentDate, tempLen, usedCh, selectTaskInd);
         
                %%%%%%%%%%% Step 1. PCA of the observed data
                X = [];
                for ti = 1:size(PSTH, 1)
                    X = cat(1, X, squeeze(PSTH(ti, :, :)));
                end
                % Run PCA.
                [~, ~, latent] = pca(X, 'Centered', true);  % eigenvalue for each PC
                
                %%%%%%%%%%% Step 2. Estimate the noise floor from trial-to-trial variability.
                nIter  = 1000;            % 1,000 iterations, matching the reference analysis
                noiseEigenvals = nan(chNum, nIter);    % noise eigenvalues [PC dimension x iteration]
                                
                parfor it = 1:nIter % parallelized across iterations
                    noiseMat = [];  % [N_noise_sample × chNum]
                    for ti = 1:30
                        if ~validInd(ti) % skip unavailable tasks
                            continue
                        end
                        % Randomly split trials, compute PSTHs, and estimate noise.
                        [diffFR] = calculate_noise_data(currentData, ti, usedCh, tempLen);
                        % Concatenate samples along rows, matching the observed-data matrix.
                        noiseMat = [noiseMat; diffFR];   
                    end
                
                    if isempty(noiseMat) % skip if no noise samples are available
                        continue
                    end
                
                    % Run PCA on the noise matrix.
                    [~, ~, latent_noise] = pca(noiseMat, 'Centered', true);
                
                    % Use a temporary vector for parfor-compatible assignment.
                    tmp = nan(chNum, 1);
                    len = min(numel(latent_noise), chNum);
                    tmp(1:len) = latent_noise;
                
                    noiseEigenvals(:, it) = tmp;
                
                end
                
                % Use the 99th percentile of the noise eigenvalue distribution as the noise ceiling for each PC.
                [noiseVar] = get_noiseVar(noiseEigenvals, 99);
                
                %%%%%%%%%%% Step 3. Determine the number of PCs explaining 95% of noise-corrected signal variance.
                % Estimate signal variance by subtracting the corresponding noise ceiling from each observed eigenvalue.
                nPC = numel(latent);
                signalEigen = latent - noiseVar(1:nPC);
                signalEigen(signalEigen < 0) = 0;  % truncate values below the noise floor at zero
                signalEigenRatio = signalEigen / sum(signalEigen); % fraction of noise-corrected signal variance
                cumSignalExplained = cumsum(signalEigenRatio); % cumulative explained signal variance
                % Number of PCs required to explain 95% of signal variance.
                dim95 = find(cumSignalExplained >= 0.95, 1, 'first');
                
                fprintf('Array %s, %s: estimated dimensionality (95%% of signal variance) = %d\n', ...
                    usedArray, currentDate, dim95);
        
                ResultDim(di, ai) = dim95;
            end
        
        end
        
        ResultDim = array2table(ResultDim, 'VariableNames',ArrayList, 'RowNames',DayList2);

        Result_PCnum.(CurrentMarg).(selectTaskName) = ResultDim;

    end

end

% Save results
outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

saveName = "PCA_signal_dims_defined_by_noisePCA_accordingMarg.mat";
save(fullfile(outputDir, char(saveName)), 'Result_PCnum');

figure
co = colororder("gem");
t = tiledlayout(1, 2, 'TileSpacing','loose','Padding','loose');

% MargiList = ["Hand", "Finger", "Gesture"];
MargiList = ["Hand"];
HandTaskName = ["Right", "Left"];
FingerTaskName = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
GestureTaskName = ["Up", "Down", "In"];
% TaskNameList = {HandTaskName, FingerTaskName, GestureTaskName};
TaskNameList = {HandTaskName};

for mi = 1:numel(MargiList)
    currentMarg = MargiList(mi);
    margName = TaskNameList{MargiList == currentMarg};
    margNum = numel(margName);
    
    startTile = 1;
    for si = 1:margNum
        nexttile(t, startTile)
        startTile = startTile + 1;
        currentSubName = margName{si};
         
        ResultDim = Result_PCnum.(currentMarg).(currentSubName);
        % Right 6d results.
        dr = ResultDim.r6d;
        % Left 6d results.
        dl = ResultDim.l6d;

        titleName = currentMarg + ": " + currentSubName;
        plot_histogram_right_and_left(dr, dl, co, titleName)
    end

end

%% Fig. 6B
% Estimate within-session intra-array connectivity.
% Because the GLM is evaluated within session, use cross-validation.
tempLen = 100;
pcDim = 10; % fixed number of predictor PCs

% 10-fold cross validation
K = 10;

ArrayList = ["All", "r6d", "l6d"];
chList = {1:384, 257:384, 1:256};
% Analyze right- and left-hand task groups separately.
TaskList = {1:30, 1:15, 16:30}; 
HandTaskName = ["All", "Right", "Left"];

% Start a parallel pool if one is not already running.
if isempty(gcp('nocreate'))
    parpool(14);  % number of workers used in the original analysis
end

Result_intraArray = struct();
c = 1;
for ai = 1:numel(ArrayList)
    usedArray = ArrayList(ai);
    usedCh = chList{ArrayList == usedArray};
    for di = 1:numel(DayList2)
        currentDate = DayList2(di);      
               
        onedayD = ClipD.(currentDate).refined_Warping_zFR;

        for taski = 1:numel(TaskList)
            currentTask = TaskList{taski};
            currentTaskName = HandTaskName(taski);

            % Select task-modulated channels using a Kruskal-Wallis test within the current task set.
            [sig_idx] = significant_ch_get_by_KW(onedayD, usedCh, currentTask);
            
            % Compute PSTHs and time-warp them to the common template length.
            [PSTH, PSTH_orig, validInd] = create_PSTH_with_templateLength(ClipD, currentDate, tempLen, usedCh, currentTask);
            
            % Construct the data matrix used for PCA.
            X = cat_all_data_before_pca(PSTH, sig_idx);
            
            % Predict each target channel from the leading latent dimensions of the remaining channels.
            R_results = nan(K, sum(sig_idx));
            parfor targetCh = 1:sum(sig_idx) % target channel; parallelized
                trainCh = 1:sum(sig_idx);
                trainCh(targetCh) = [];
                
                trainSignal = X(:, trainCh);
                y = X(:, targetCh);

                
                zscoreInd = false;
                nrmse_ind = true;


                R2_list = my_GLM_train_kfold(trainSignal, y, K, pcDim, zscoreInd, nrmse_ind); % Legacy variable name; values are NRMSE when nrmse_ind = true.
                
                R_results(:, targetCh) = R2_list;
            
            end

            Result_intraArray.(usedArray).(currentDate).(currentTaskName).results_row = R_results;
            Result_intraArray.(usedArray).(currentDate).(currentTaskName).results_mean = mean(R_results);
            Result_intraArray.(usedArray).(currentDate).(currentTaskName).usedCh = usedCh;
            Result_intraArray.(usedArray).(currentDate).(currentTaskName).sigCh = sig_idx;

            fprintf("%d/%d\n", c, numel(ArrayList)*numel(DayList2)*numel(TaskList))
            c = c + 1;

        end
    end
end


% Save results
outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

saveName = "Intra_array_GLM_prediction_withinDay_nrmse.mat";
save(fullfile(outputDir, char(saveName)), 'Result_intraArray');

% Visualization
DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;
dateNum = numel(DayList2);

HandTaskName = ["All", "Right", "Left"];
ArrayList = ["r6d", "l6d"];
ArrayNames = ["Right 6d", "Left 6d"];

co = colororder("gem");

figure
t = tiledlayout(1, 2, 'TileSpacing','loose', 'Padding','loose');

for ai = 1:numel(ArrayList)
    currentArray = ArrayList(ai);
    nexttile

    legendNames = cell(1, numel(HandTaskName)-1);
    for taski = 2:numel(HandTaskName)
        currentTask = HandTaskName(taski);
        
        % Aggregate intra-array connectivity estimates.
        [ave_data, ci_data, x_all, y_all] = get_ave_ci_data_intra_array(DayList2, Result_intraArray, currentArray, currentTask);
        
        fill([1:dateNum, flip(1:dateNum)], [ci_data(1, :), flip(ci_data(2, :))], co(taski-1, :), 'FaceAlpha',0.2, 'EdgeColor','none')
        hold on
        plot(1:dateNum, ave_data, 'o-', 'Color',co(taski-1, :), 'LineWidth',2)
        % Test for differences across groups.
        [p, ~, stats] = kruskalwallis(x_all, y_all, 'off');

        if p > 0.05
            dispP = sprintf('p=%.2f(KW test)', p);
        elseif p >= 0.001
            dispP = sprintf('*p=%.3f(KW test)', p);
        else
            dispP = sprintf('*p<0.001(KW test)');
        end
        legendNames{taski-1} = HandTaskName(taski) + " hand; " + string(dispP);
    end

    h_color = gobjects(2, 1); h_color(1) = plot(nan, nan, 'o-', 'Color',co(1, :), 'LineWidth',2); h_color(2) = plot(nan, nan, 'o-', 'Color',co(2, :), 'LineWidth',2);
    legend(h_color, legendNames, 'Box','off', 'Location','best', 'AutoUpdate','off')

    box off

    xticks(1:dateNum), xticklabels(DayList2), xlim([0 dateNum+1])
    yticks(0:0.1:0.6)
    ylim([0.38 0.65])
    title(ArrayNames(ai), 'FontSize', 14)
    ylabel({'Intra-array connectivity', '(Normalized RMSE)'})

end

%% Fig. 6D

% Generate all pairs of different recording days.
dayNum = numel(DayList2);
comList = nchoosek(1:dayNum, 2); 
% comList(end+1:end+7, :) = repmat(1:numel(DayList2), 2, 1)'; % Uncomment to include same-day pairs (duration = 0)
comList_day1 = double(DaysList(comList(:, 1)))'; comList_day2 = double(DaysList(comList(:, 2)))';
DurationList = comList_day2 - comList_day1;

tempLen= 100;
pcDim = 10; % fixed number of predictor PCs

ArrayList = ["r6d", "l6d"];
chList = {257:384, 1:256};

HandTaskName = ["All", "Right", "Left"];
TaskList = {1:30, 1:15, 16:30}; 

DirectionList = ["Backward", "Forward"];

% Start a parallel pool if one is not already running.
if isempty(gcp('nocreate'))
    parpool(14);  % number of workers used in the original analysis
end

IntraArray_AcrossDay = struct();
c = 1;
for ai = 1:numel(ArrayList)
    currentArray = ArrayList(ai);
    usedCh = chList{ArrayList==currentArray};
    
    for taski = 1:numel(HandTaskName)
        currentTask = HandTaskName(taski);
        currentTaskNum = TaskList{HandTaskName == currentTask};
        
        % Retain channels that are significant by Kruskal-Wallis test on every recording day.
        all_sig_ch = get_KWsigCh_all_across_day(Result_intraArray, DayList2, currentArray, currentTask);
        
        % Retain tasks that are available on every recording day.
        [all_common_ind, all_ind] = common_task_ind_make(ClipD, DayList2);
        % Update the task indices used in the analysis.
        currentTaskNum = currentTaskNum(ismember(currentTaskNum, find(all_common_ind)));
        
        % Concatenate data across days and construct day-specific matrices.
        [across_allData, each_DateData] = cat_allday_psth(DayList2, ClipD, tempLen, usedCh, currentTaskNum, all_sig_ch);
        
        for direci = 1:numel(DirectionList)
        
            currentDirection = DirectionList(direci);
        
            All_R2 = cell(size(comList, 1), 1);
            TrainDateR = strings(size(comList, 1), 1);
            PredDateR = strings(size(comList, 1), 1);
            Durations = nan(size(comList, 1), 1);

            % Iterate over recording-day pairs.
            for combi = 1:size(comList, 1)
                % Select the training and prediction days.
                currentComb = comList(combi, :);
            
                if currentDirection  == "Backward"
                    % Backward prediction: train on the later day and predict the earlier day.
                    trainingDateInd = currentComb(2); % train on the later session
                    predDateInd = currentComb(1); % predict the earlier session
                elseif currentDirection  == "Forward"
                    % Forward prediction: train on the earlier day and predict the later day.
                    trainingDateInd = currentComb(1); % train on the earlier session
                    predDateInd = currentComb(2); % predict the later session
                end
                
                TrainingDay = DayList2(trainingDateInd); PredDay = DayList2(predDateInd);
                Duration = DurationList(combi);
            
                TrainingOrigData = each_DateData{trainingDateInd};
                PredOrigData = each_DateData{predDateInd};
                
                R_result = nan(sum(all_sig_ch), 1);
                parfor TargetCh = 1:sum(all_sig_ch) % iterate over target channels in parallel
                
                    % Train on one day and predict activity on the other day.
                    [y_test, y_pred] = glm_train_pred_acrossDay(TrainingOrigData, PredOrigData, TargetCh, across_allData, pcDim);

                    %%%%%%% Compute normalized RMSE.
                    R_result(TargetCh) = sqrt(mean((y_test - y_pred).^2)) / std(y_test);
                    
                end
            
                All_R2{combi} = R_result;
                TrainDateR(combi) = TrainingDay;
                PredDateR(combi) = PredDay;
                Durations(combi) = Duration;
                
                % Display progress.
                fprintf('%d/%d\n', c, numel(ArrayList)*numel(HandTaskName)*numel(DirectionList)*size(comList, 1))
                c = c + 1;
            
            end
            
            R_GLM = table(All_R2, TrainDateR, PredDateR, Durations, 'VariableNames',{'R2', 'TrainingDate', 'PredDate', 'Durations'});

            IntraArray_AcrossDay.(currentArray).(currentTask).(currentDirection) = R_GLM;
        
        end
    
    end

end

% Save results
outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

saveName = "Intra_array_GLM_prediction_AcrossDay_nrmse.mat";
save(fullfile(outputDir, char(saveName)), 'IntraArray_AcrossDay');

% Visualization

ArrayList = ["r6d", "l6d"];
ArrayListName = ["Right 6d", "Left 6d"];
HandTaskName = ["All", "Right", "Left"];
DirectionList = ["Backward"];
DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;

co = colororder("gem");

figure
t = tiledlayout(1, 2, 'TileSpacing','loose', 'Padding','loose');
 
currentDirec = "Backward";

% Plot hand-specific results for the right 6d array.
nexttile(t, 1)
% Select array.
currentArray = "r6d";
currentTask = [];
Loop_Data = ["Right", "Left"];
LegendNameList = ["Right Hand", "Left Hand"];
[total_R_list, legendNames, h_color] = scatter_and_linearReg_plot(Loop_Data, LegendNameList, currentArray, currentTask, currentDirec, IntraArray_AcrossDay, co);
ax = gca; cla(ax)
% Plot the median and interquartile range for the two groups.
two_group_medina_quantile_plot(total_R_list, co, LegendNameList);
xtickangle (45)
ylim([0 4.4]), yticks([0:4])
ylabel({'Cross-day backward prediction accuracy (NRMSE)', 'Intra-array connectivity'})
title('Right 6d array', 'FontSize',14)

% Plot hand-specific results for the left 6d array.
nexttile(t, 2)
% Select array.
currentArray = "l6d";
currentTask = [];
Loop_Data = ["Right", "Left"];
LegendNameList = ["Right Hand", "Left Hand"];
[total_R_list, legendNames, h_color] = scatter_and_linearReg_plot(Loop_Data, LegendNameList, currentArray, currentTask, currentDirec, IntraArray_AcrossDay, co);
ax = gca; cla(ax)
% Plot the median and interquartile range for the two groups.
two_group_medina_quantile_plot(total_R_list, co, LegendNameList);
xtickangle (45)
ylim([0 4.4]), yticks([0:4])
ylabel({'Cross-day backward prediction accuracy (NRMSE)', 'Intra-array connectivity'})
title('Left 6d array', 'FontSize',14)


%% Fig. 6C

ArrayList = ["r6d", "l6d"];
ArrayListName = ["Right 6d", "Left 6d"];
HandTaskName = ["All", "Right", "Left"];
DirectionList = ["Backward", "Forward"];
DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;

co = colororder("gem");

figure
t = tiledlayout(1, 2, 'TileSpacing','loose', 'Padding','loose');

nexttile(t, 1)
% r6d, All hand
currentTask = "All";
currentArray = "r6d";
Loop_Data = DirectionList;
LegendNameList = DirectionList;
currentDirec = [];  % Direction is the field varied by Loop_Data.
[total_R_list, legendNames, h_color] = scatter_and_linearReg_plot(Loop_Data, LegendNameList, currentArray, currentTask, currentDirec, IntraArray_AcrossDay, co);
ax = gca; cla(ax)
% Plot the median and interquartile range for the two groups.
two_group_medina_quantile_plot(total_R_list, co, LegendNameList, true);
xtickangle (45)
ylim([0 3.2]), yticks([0:4])
ylabel({'Cross-day prediction accuracy (NRMSE)', 'Intra-array connectivity'})
title('Right 6d array', 'FontSize',12)

nexttile(t, 2)
% l6d, All hand
currentTask = "All";
currentArray = "l6d";
Loop_Data = DirectionList;
LegendNameList = DirectionList;
currentDirec = [];  % Direction is the field varied by Loop_Data.
[total_R_list, legendNames, h_color] = scatter_and_linearReg_plot(Loop_Data, LegendNameList, currentArray, currentTask, currentDirec, IntraArray_AcrossDay, co);
ax = gca; cla(ax)
% Plot the median and interquartile range for the two groups.
two_group_medina_quantile_plot(total_R_list, co, LegendNameList, true);
xtickangle (45)
ylim([0 3.2]), yticks([0:4])
ylabel({'Cross-day prediction accuracy (NRMSE)', 'Intra-array connectivity'})
title('Left 6d array', 'FontSize',12)

%% Fig. 6F
% Estimate within-session inter-array connectivity.
% Train on one 6d array and predict activity in the contralateral 6d array.

tempLen = 100;
pcDim = 10; % fixed number of predictor PCs

% 10-fold cross validation
K = 10;

% Analyze right- and left-hand task groups separately.
TaskList = {1:30, 1:15, 16:30}; 
HandTaskName = ["All", "Right", "Left"];
% RL: train on right 6d and predict left 6d; LR: the reverse.
PredictDirection = ["RL", "LR"]; 

% Start a parallel pool if one is not already running.
if isempty(gcp('nocreate'))
    parpool(14);  % number of workers used in the original analysis
end

Result_interArray = struct();
c = 1;
for direci = 1:numel(PredictDirection)
    currentDirec = PredictDirection(direci);
    
    for di = 1:numel(DayList2)
        currentDate = DayList2(di);      
               
        onedayD = ClipD.(currentDate).refined_Warping_zFR;
        
        for taski = 1:numel(TaskList)
            currentTask = TaskList{taski};
            currentTaskName = HandTaskName(taski);
        
            usedCh_R = 257:384; % Right 6d
            usedCh_L = 1:256; % Left 6d
            % Select task-modulated channels using a Kruskal-Wallis test within the current task set.
            sig_idx_R = significant_ch_get_by_KW(onedayD, usedCh_R, currentTask);
            sig_idx_L = significant_ch_get_by_KW(onedayD, usedCh_L, currentTask);
            
            % Compute PSTHs and time-warp them to the common template length.
            [PSTH_R, PSTH_orig_R, validInd_R] = create_PSTH_with_templateLength(ClipD, currentDate, tempLen, usedCh_R, currentTask);
            [PSTH_L, PSTH_orig_L, validInd_L] = create_PSTH_with_templateLength(ClipD, currentDate, tempLen, usedCh_L, currentTask);
            
            % Construct the data matrix used for PCA.
            X_R = cat_all_data_before_pca(PSTH_R, sig_idx_R);
            X_L = cat_all_data_before_pca(PSTH_L, sig_idx_L);
        
            if currentDirec == "RL"
                % Train on right 6d and predict left 6d.
                sig_idx = sig_idx_L;
                trainSignal = X_R;
                predictSignal = X_L;
            elseif currentDirec == "LR"
                % Train on left 6d and predict right 6d.
                sig_idx = sig_idx_R;
                trainSignal = X_L;
                predictSignal = X_R;
            end
            
            % Predict each target channel from the leading latent dimensions of the remaining channels.
            R_results = nan(K, sum(sig_idx));
            for targetCh = 1:sum(sig_idx) % target channel; parallelized
                y = predictSignal(:, targetCh);
        
                % Derive predictors from neural activity excluding the target channel when applicable.            
                % Run K-fold GLM evaluation; PCA is fit within each training fold.
                
 
                zscoreInd = false; nrmse_ind = true;

                R2_list = my_GLM_train_kfold(trainSignal, y, K, pcDim, zscoreInd, nrmse_ind); % Legacy variable name; values are NRMSE when nrmse_ind = true.
                
                R_results(:, targetCh) = R2_list;
            
            end
        
            Result_interArray.(currentDirec).(currentDate).(currentTaskName).results_row = R_results;
            Result_interArray.(currentDirec).(currentDate).(currentTaskName).results_mean = mean(R_results);
            Result_interArray.(currentDirec).(currentDate).(currentTaskName).sigCh_R = sig_idx_R;
            Result_interArray.(currentDirec).(currentDate).(currentTaskName).sigCh_L = sig_idx_L;
        
            fprintf("%d/%d\n", c, numel(PredictDirection)*numel(DayList2)*numel(TaskList))
            c = c + 1;
        
        end
    
    end

end

% Save results
outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

saveName = "Inter_array_GLM_prediction_withinDay_nrmse.mat";
save(fullfile(outputDir, char(saveName)), 'Result_interArray');


% Visualization
DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;
dateNum = numel(DayList2);

HandTaskName = ["All", "Right", "Left"];
DirectionList = ["RL", "LR"];
DirectionNames = ["R → L", "L → R"];

co = colororder("gem");

figure
t = tiledlayout(1, 2, 'TileSpacing','loose', 'Padding','loose');
for taski = 2:numel(HandTaskName)
    nexttile
    currentTask = HandTaskName(taski);
    legendNames = cell(1, numel(DirectionList));

    for direci = 1:numel(DirectionList)
    
        currentDirection = DirectionList(direci);
        % Aggregate intra-array connectivity estimates.
        [ave_data, ci_data, x_all, y_all] = get_ave_ci_data_intra_array(DayList2, Result_interArray, currentDirection, currentTask);
        
        fill([1:dateNum, flip(1:dateNum)], [ci_data(1, :), flip(ci_data(2, :))], co(direci, :), 'FaceAlpha',0.2, 'EdgeColor','none')
        hold on
        plot(1:dateNum, ave_data, 'o-', 'Color',co(direci, :), 'LineWidth',2)
        % Test for differences across groups.
        [p, ~, stats] = kruskalwallis(x_all, y_all, 'off');
    
        if p > 0.05
            dispP = sprintf('p=%.2f(KW test)', p);
        elseif p >= 0.001
            dispP = sprintf('*p=%.3f(KW test)', p);
        else
            dispP = sprintf('*p<0.001(KW test)');
        end
        legendNames{direci} = DirectionNames(direci) + " " + string(dispP);
    end
    
    h_color = gobjects(2, 1); h_color(1) = plot(nan, nan, 'o-', 'Color',co(1, :), 'LineWidth',2); h_color(2) = plot(nan, nan, 'o-', 'Color',co(2, :), 'LineWidth',2);
    legend(h_color, legendNames, 'Box','off', 'Location','best', 'AutoUpdate','off')
    box off
    xticks(1:dateNum), xticklabels(DayList2), xlim([0 dateNum+1])
    yticks(0.4:0.1:1)
    ylim([0.4 0.76])
    title(currentTask + " hand")
    ylabel({'Inter-array connectivity', '(cross-validated NRMSE)'})

end

%% Fig. 6E

dataPath = fullfile(projectDir, 'results', 'Intra_array_GLM_prediction_withinDay_nrmse.mat');
load(dataPath)

DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;
DaysList  = double(DaysList);

% Generate all pairs of different recording days.
dayNum = numel(DayList2);
comList_day1 = repmat(1:7', 7, 1); comList_day1 = comList_day1(:);
comList_day2 = repmat(1:7, 1, 7)';
comList = [comList_day1, comList_day2];
DurationList = abs(DaysList(comList_day2) - DaysList(comList_day1))';

tempLen = 100;
pcDim = 10; % fixed number of predictor PCs

HandTaskName = [ "Right", "Left"];
TaskList = {1:15, 16:30}; 

InterArray_AcrossDay_CCA = struct();
c = 1;

    
for taski = 1:numel(HandTaskName)
    currentTask = HandTaskName(taski);
    currentTaskNum = TaskList{HandTaskName == currentTask};
    
    % Retain channels that are significant by Kruskal-Wallis test on every recording day.
    all_sig_ch_R = get_KWsigCh_all_across_day(Result_intraArray, DayList2, "r6d", currentTask);
    all_sig_ch_L = get_KWsigCh_all_across_day(Result_intraArray, DayList2, "l6d", currentTask);
    
    % Retain tasks that are available on every recording day.
    [all_common_ind, all_ind] = common_task_ind_make(ClipD, DayList2);
    % Update the task indices used in the analysis.
    currentTaskNum = currentTaskNum(ismember(currentTaskNum, find(all_common_ind)));
    
    % Concatenate data across days and construct day-specific matrices.
    [across_allData_R, each_DateData_R] = cat_allday_psth(DayList2, ClipD, tempLen, 257:384, currentTaskNum, all_sig_ch_R);
    [across_allData_L, each_DateData_L] = cat_allday_psth(DayList2, ClipD, tempLen, 1:256, currentTaskNum, all_sig_ch_L);
    

    All_r = cell(size(comList, 1), 1);
    All_r_mean = nan(size(comList, 1), 1);
    DateR = strings(size(comList, 1), 1);
    DateL = strings(size(comList, 1), 1);

    % Iterate over recording-day pairs.
    for combi = 1:size(comList, 1)
        currentComb = comList(combi, :);

        day1 = currentComb(1);
        day2 = currentComb(2);

        data1 = each_DateData_R{day1};
        data2 = each_DateData_L{day2};

        [~, pca1] = pca(data1);
        pca1_10 = pca1(:, 1:pcDim);


        [~, pca2] = pca(data2);
        pca2_10 = pca2(:, 1:pcDim);       

        % Compute canonical correlations.
        [~,~,r] = canoncorr(pca1_10, pca2_10);

        All_r{combi} = r;
        All_r_mean(combi) = mean(r); % mean canonical correlation
        DateR(combi) = DayList2(day1);
        DateL(combi) = DayList2(day2);
            
    end
    
    R_cca = table(All_r, All_r_mean, DateR, DateL, DurationList, 'VariableNames',{'CCAr', 'CCA_rmean', 'r6d', 'l6d', 'Durations'});

    InterArray_AcrossDay_CCA.(currentTask) = R_cca;

end

% Visualization
HandTaskName = ["Right", "Left"]; % pooling both hands obscures the hand-specific pattern
HandTaskNameLegend = ["Right hand", "Left hand"];

DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;

co = colororder("gem");
% Number of leading canonical correlations used for summary statistics.
CCA_higiDim = 8; % the leading eight components account for approximately 90% of summed r^2
used_CCA_Dim = min(CCA_higiDim, 10);

DurationD = InterArray_AcrossDay_CCA.Right.Durations;
sameDateInd = DurationD == 0;
sameDateRow = find(sameDateInd);
dateNum = numel(sameDateRow);

figure
t = tiledlayout(1, 2, 'TileSpacing','loose', 'Padding','loose');

legendNames = strings();
for taski = 1:numel(HandTaskName)
    currentTask = HandTaskName(taski);
    
    all_z = [];
    ave_d = [];
    ci1 = [];
    ci2 = [];
    for rowi = 1:dateNum
    
        r_data = InterArray_AcrossDay_CCA.(currentTask).CCAr{sameDateRow(rowi)};

        %%%%%%%%%%%% The leading eight canonical correlations account for approximately 90% of summed r^2.

        % Restrict the summary to leading components because higher dimensions are increasingly noise-dominated.
        r_data_clip = r_data(1:used_CCA_Dim); 
        
        % Estimate the mean and 95% confidence interval by bootstrap.
        [mean_est, ci95] = bootstrap_CI(r_data_clip);
    
        all_z = [all_z, r_data(:)];
        ave_d(end+1) = mean_est;
        ci1(end+1) = ci95(1);
        ci2(end+1) = ci95(2);
    
    end
    
    nexttile(t, 1);
    fill([1:dateNum, flip(1:dateNum)], [ci1, flip(ci2)],  co(taski, :), 'FaceAlpha',0.2, 'EdgeColor','none')
    hold on
    plot(1:dateNum, ave_d, 'o-', 'LineWidth',2, 'Color', co(taski, :))
    xticks(1:dateNum), xlim([0 dateNum+1]), xticklabels(DayList2)
    box off
    ylabel('Canonical correlation (r)')
    titleNames = sprintf('top %d components', CCA_higiDim);
    title({'Inter-hemispheric coupling (CCA)', titleNames})
    yticks(0.8:0.05:1), ylim([0.82 1.01])
    
    % Test for differences across groups.
    [p, ~, stats] = friedman(all_z, 1, 'off');

    if p > 0.05
        dispT = sprintf('(p = %.2f, Friedman test)', p);
    elseif p >= 0.001
        dispT = sprintf('(*p = %.3f, Friedman test)', p);
    else
        dispT = sprintf('(*p < 0.001, Friedman test)');
    end

    legendNames(taski) = HandTaskNameLegend(taski) + " " + string(dispT);

    meanRanks   = stats.meanranks;   % 1 x number of days
    nexttile(t, 2);
    plot(1:dateNum, meanRanks, 'o-', 'Color', co(taski, :), 'LineWidth', 2);  % x: day, y: mean rank
    hold on
    xticks(1:dateNum), xlim([0 dateNum+1]), xticklabels(DayList2)
    box off
    ylabel('Friedman rank')
    title({'Friedman test ranks across days', ''})
end

h_color1 = gobjects(numel(HandTaskName), 1);
h_color2 = gobjects(numel(HandTaskName), 1);

for taski = 1:numel(HandTaskName)
    nexttile(t, 1);
    h_color1(taski) = plot(nan, nan, 'o-', 'LineWidth',2, 'Color', co(taski, :));
    nexttile(t, 2);
    h_color2(taski) = plot(nan, nan, 'o-', 'LineWidth',2, 'Color', co(taski, :));
end

legend(h_color1, legendNames, 'Box','off', 'AutoUpdate','off', 'Location','bestoutside')
legend(h_color2, HandTaskNameLegend, 'Box','off', 'AutoUpdate','off', 'Location','bestoutside')


%% Local functions

function [PSTH, PSTH_orig, validInd] = create_PSTH_with_templateLength(ClipD, currentDate, tempLen, usedCh, selectTaskInd)
    
    if nargin < 5
        selectTaskInd = 1:30;
    end

    currentData = ClipD.(currentDate).refined_Warping_zFR;
    chNum = numel(usedCh);

    PSTH_orig = cell(30, 1);
    validInd = zeros(30, 1);
    for ti = 1:30
        selectD = currentData.("T" + string(ti));
        if isempty(selectD)
            continue
        elseif size(selectD, 1) < 10 
            continue
        elseif ~ismember(ti, selectTaskInd) 
            continue
        end

        selectD = selectD(:, :, usedCh); 

        psthD = squeeze(mean(selectD, 1)); 
        PSTH_orig{ti} = stretch_template_linear(psthD,tempLen);

        validInd(ti) = 1;

    end
    validInd = logical(validInd);

    PSTH = nan(sum(validInd), tempLen, chNum);
    k = 1;
    for ti = 1:numel(validInd)
        if validInd(ti)
            PSTH(k, :, :) = PSTH_orig{ti}(:, :);
            k = k + 1;
        end
    end

end

function diffFR = calculate_noise_data(currentData, ti, usedCh, tempLen)

    % Extract all trials for the current task
    singleTask = currentData.("T" + string(ti));
    singleTask = singleTask(:, :, usedCh); 
    nTrial = size(singleTask, 1);

    % Randomly pair trials
    idx = randperm(nTrial);
    if mod(numel(idx), 2) == 1
        idx(end) = []; 
    end
    idx1 = idx(1:2:end);
    idx2 = idx(2:2:end);

     % Compute PSTHs from the two randomly divided trial groups
    tr1 = squeeze(singleTask(idx1, :, :)); % [nPair × time × ch]
    tr2 = squeeze(singleTask(idx2, :, :));
    tr1psth = squeeze(mean(tr1, 1)); % PSTH from the first trial group
    tr2psth = squeeze(mean(tr2, 1)); % PSTH from the second trial group

    % Time-warp both PSTHs to a common length
    tr1_w = stretch_template_linear(tr1psth, tempLen); % [tempLen × chNum]
    tr2_w = stretch_template_linear(tr2psth, tempLen);

    % Compute the difference between the two PSTHs as an estimate of
    % trial-to-trial variability (noise)
    %
    % Note: The original method computes trial-wise differences and then
    % averages them. Here, we instead compute the difference between PSTHs,
    % as the trial-wise implementation did not provide stable results for
    % the present dataset.
    diffFR = tr1_w - tr2_w;          % [tempLen × chNum]


end

function noiseVar = get_noiseVar(noiseEigenvals, percentNum)

    chNum = size(noiseEigenvals, 1);
    noiseVar = nan(chNum, 1);
    for d = 1:chNum
        thisDimVals = noiseEigenvals(d, :);
        thisDimVals = thisDimVals(~isnan(thisDimVals)); 
        if isempty(thisDimVals)
            noiseVar(d) = 0;
        else
            noiseVar(d) = prctile(thisDimVals, percentNum);
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

function plot_histogram_right_and_left(dr, dl, co, titlename)

    histogram(dr-0.01,'BinWidth',0.2, 'FaceColor',co(1,:))
    hold on
    histogram(dl+0.01,'BinWidth',0.2, 'FaceColor',co(2,:))

    meanR = round(mean(dr));
    meanL = round(mean(dl));

    ax = gca; YlimV = ax.YLim; ylimMax = max(YlimV);
    
    if meanR > meanL
        scatter(meanR, ylimMax*1.1, 50, 'o', 'filled', 'MarkerFaceColor',co(1, :), 'MarkerEdgeColor','none')
        scatter(meanL, ylimMax*1.05, 50, 'o', 'filled', 'MarkerFaceColor',co(2, :), 'MarkerEdgeColor','none')
    else
        scatter(meanR, ylimMax*1.05, 50, 'o', 'filled', 'MarkerFaceColor',co(1, :), 'MarkerEdgeColor','none')
        scatter(meanL, ylimMax*1.1, 50, 'o', 'filled', 'MarkerFaceColor',co(2, :), 'MarkerEdgeColor','none')
    end
    xticks(0:1:12)
    yticks(0:1:floor(ylimMax*1.1)), ylim([0 ylimMax*1.2])
    box off
    
    p = signrank(dr, dl); 
    if p > 0.05
        dispP = sprintf('p = %.2f', p);
    elseif p > 0.001
        dispP = sprintf('*p = %.3f', p);
    else
        dispP = sprintf('*p <0.001');
    end
    
    h_color = gobjects(4, 1);
    h_color(1) = histogram(nan,'BinWidth',0.2, 'FaceColor',co(1,:));
    h_color(2) = histogram(nan,'BinWidth',0.2, 'FaceColor',co(2,:));
    h_color(3) = scatter(nan, nan, 1, 'o', 'MarkerEdgeColor','none', 'MarkerFaceColor','none');
    h_color(4) = scatter(nan, nan, 1, 'o', 'MarkerEdgeColor','none', 'MarkerFaceColor','none');
    legend(h_color, {'Right 6d', 'Left 6d', dispP,  '(Wilcoxon signed-rank)'}, 'Box','off', 'AutoUpdate','off', 'Location','northwest', 'FontSize',10)
    
    xlabel('Number of significant PCs')
    ylabel('Counts')
    title({'Noise-corrected dimensionality', char(titlename)})

end

function sig_idx = significant_ch_get_by_KW(onedayD, usedCh, currentTask)

    validInd = zeros(30, 1);
    for i = 1:30
        currentfname = "T" + string(i);
        if isempty(onedayD.(currentfname))
            continue
        elseif size(onedayD.(currentfname), 1) < 10 
            continue
        else
            validInd(i) = 1;
        end
    end
    validInd = logical(validInd);
   
    [x_all, y_all, ~] = create_x_y_data_for_svm(validInd, onedayD);

    current_task_ind = ismember(y_all, currentTask);

    [~, sig_idx] = refined_xdata_by_KW(usedCh, x_all(current_task_ind, :), y_all(current_task_ind, :));
end

function X = cat_all_data_before_pca(PSTH, sig_idx)

    sigPSTH = PSTH(:, :, sig_idx); 
    X = [];
    for ti = 1:size(sigPSTH, 1)
        X = cat(1, X, squeeze(sigPSTH(ti, :, :)));
    end

end

function R2_list = my_GLM_train_kfold(trainSignal, y, K, pcDim, zscoreInd, nrmseInd)

    if nargin < 5
        zscoreInd = false;
        nrmseInd = false;
    elseif nargin < 6
        nrmseInd = false;
    end

    cv = cvpartition(length(y), 'KFold', K);
    R2_list = nan(K,1);
    
    for k = 1:K
        trainIdx = training(cv, k);
        testIdx  = test(cv, k);

 
        [coeff, signals, ~, ~, ~, mu] = pca(trainSignal(trainIdx, :), 'Centered', true);
        X_train = signals(:, 1:pcDim);                 
        y_train = y(trainIdx);
    

        testData  = trainSignal(testIdx, :);
        testData_centered = testData - mu; 
        X_test = testData_centered*coeff;
        X_test = X_test(:, 1:pcDim);
        y_test  = y(testIdx);
    

        mdl = fitglm(X_train, y_train, 'linear');
    
        y_pred = predict(mdl, X_test);

        if nrmseInd 
            result_val = sqrt(mean((y_test - y_pred).^2)) / std(y_test);
        else

            
            if zscoreInd
                y_pred = zscore(y_pred);
                y_test = zscore(y_test);
            end
    
            SS_res = sum((y_test - y_pred).^2);
            SS_tot = sum((y_test - mean(y_test)).^2);
            result_val = 1 - SS_res/SS_tot;

        end

        R2_list(k) = result_val;
    end

end

function [x_all, y_all, validInd] = create_x_y_data_for_svm(validInd, onedayD)
    x_all = [];
    y_all = [];
    for i = 1:30
        if validInd(i)
            currentData = onedayD.("T" + string(i));
            taskNum = size(currentData, 1);
            if taskNum < 10
                validInd(i) = false; 
                continue
            end
            for ti = 1:taskNum
                one_data = squeeze(currentData(ti, :, :));
                
                mean_d = mean(one_data, 1);
                x_all = cat(1, x_all, mean_d);
                y_all = cat(1, y_all, i);
            end
    
        end
    
    end
end

function [sigX, sig_idx] = refined_xdata_by_KW(usedCh, x_all, y_all)
    chNum = numel(usedCh);
    
    all_p = [];
    for chi = 1:chNum
        currentCh = usedCh(chi);
        oneCh_data = x_all(:, currentCh);
    
        [p, ~, stats] = kruskalwallis(oneCh_data, y_all, 'off');
        all_p(end+1) = p;
    end

    [crit_p, sig_idx] = fdr_bh(all_p(:), 0.05);
    
    clippedX = x_all(:, usedCh); 
    sigX = clippedX(:, sig_idx); 

end

function [crit_p, sig_idx] = fdr_bh(pvals, q)
    
    m = length(pvals);
    [p_sorted, sort_idx] = sort(pvals);
    thresholds = ((1:m)' / m) * q;
    
    below_thresh = p_sorted <= thresholds;
    if any(below_thresh)
        max_idx = find(below_thresh, 1, 'last');
        crit_p = p_sorted(max_idx); 
        sig_idx = pvals <= crit_p;
    else
        sig_idx = false(size(pvals));
        crit_p = 1;
    end

end

function [ave_data, ci_data, x_all, y_all] = get_ave_ci_data_intra_array(DayList2, Result_intraArray, currentArray, currentTask)
    
    dateNum = numel(DayList2);

    ave_data = nan(1, dateNum);
    ci_data = nan(2, dateNum);
    
    x_all = [];
    y_all = [];
    
    for di = 1:dateNum
        currentDate = DayList2(di);
    
        cData = Result_intraArray.(currentArray).(currentDate).(currentTask).results_mean;
        [h, ~, ci] = normfit(cData);
    
        ave_data(di) = h;
        ci_data(:, di) = ci;

        
        x_all = cat(1, x_all, cData(:));
        y_all = cat(1, y_all, repmat(di, numel(cData), 1));
    end

end

function all_sig_ch = get_KWsigCh_all_across_day(Result_intraArray, DayList2, currentArray, currentTask)

    all_sigCh = [];
    for di = 1:numel(DayList2)
        sigCh = Result_intraArray.(currentArray).(DayList2(di)).(currentTask).sigCh;
        all_sigCh = cat(2, all_sigCh, sigCh);
    end
    
    all_sig_ch = all(logical(all_sigCh), 2);

end

function [all_common_ind, all_ind] = common_task_ind_make(ClipD, DayList2)
    all_ind = [];
    for di = 1:numel(DayList2)
        currentData = ClipD.(DayList2(di)).refined_Warping_zFR;
    
        indVal = zeros(30, 1);
        for taski = 1:30
            if isempty(currentData.("T" + string(taski)))
                continue
            elseif size(currentData.("T" + string(taski)), 1) < 10 
                continue
            end
    
            indVal(taski) = 1;
    
        end
    
        all_ind = cat(2, all_ind, indVal);
    
    end
    all_ind = logical(all_ind);
    all_common_ind = all(all_ind, 2);

end

function [across_allData, each_DateData] = cat_allday_psth(DayList2, ClipD, tempLen, usedCh, currentTaskNum, all_sig_ch)
    across_allData = [];
    each_DateData = cell(numel(DayList2), 1);
    for di = 1:numel(DayList2)
        currentDate = DayList2(di);
        
        [PSTH, PSTH_orig, validInd] = create_PSTH_with_templateLength(ClipD, currentDate, tempLen, usedCh, currentTaskNum);
        

        sigPSTH = PSTH(:, :, all_sig_ch); 
        X = [];
        for ti = 1:size(sigPSTH, 1)
            X = cat(1, X, squeeze(sigPSTH(ti, :, :)));
        end
    
        each_DateData{di} = X;
    
        across_allData = cat(1, across_allData, X);
    end

end


function [y_test, y_pred] = glm_train_pred_acrossDay(TrainingOrigData, PredOrigData, TargetCh, across_allData, pcDim)

    y = TrainingOrigData(:, TargetCh);


    modified_across_data = across_allData;
    modified_across_data(:, TargetCh) = [];
    [coeff, ~, ~, ~, ~, mu] = pca(modified_across_data);

    trainData = TrainingOrigData;
    trainData(:, TargetCh) = [];
    trainData = trainData - mu;

    trainPC = trainData*coeff;
    trainPC = trainPC(:, 1:pcDim);
    

    mdl = fitglm(trainPC, y, 'linear');
    

    y_test = PredOrigData(:, TargetCh);

    testData = PredOrigData;
    testData(:, TargetCh) = [];
    testData = testData - mu;
    testPC = testData*coeff;
    testPC = testPC(:, 1:pcDim);

    y_pred = predict(mdl, testPC);
end

function [total_R_list, legendNames, h_color] = scatter_and_linearReg_plot(Loop_Data, LegendNameList, currentArray, currentTask, currentDirec, IntraArray_AcrossDay, co)

    legendNames = cell(numel(Loop_Data), 1);
    h_color = gobjects(numel(Loop_Data), 1);
    total_R_list = cell(numel(Loop_Data), 1);

    for li = 1:numel(Loop_Data)
    
        currentLoop = Loop_Data(li);

        if isempty(currentArray)
            currentData = IntraArray_AcrossDay.(currentLoop).(currentTask).(currentDirec);
        elseif isempty(currentTask)
            currentData = IntraArray_AcrossDay.(currentArray).(currentLoop).(currentDirec);
        elseif isempty(currentDirec)
            currentData = IntraArray_AcrossDay.(currentArray).(currentTask).(currentLoop);
        end
       
        [totalR, DurationAll] = gather_r2_duration_data(currentData);
        total_R_list{li} = totalR;

        scatter(DurationAll, totalR, 50, 'o', 'filled', 'MarkerFaceColor', co(li, :), 'MarkerEdgeColor','none', 'MarkerFaceAlpha',0.3)
        [xx, yy, r, p, dispT] = fit_lm_and_corr(DurationAll, totalR);
        hold on

        plot(xx, yy, '-', 'LineWidth',2, 'Color', co(li, :))
        legendNames{li} = LegendNameList(li) + ": " + dispT;
        h_color(li) = plot(nan, nan, '-', 'LineWidth',2, 'Color', co(li, :));
    end


end

function [totalR, DurationAll] = gather_r2_duration_data(currentData)
    totalR = [];
    DurationAll = [];
    for comi = 1:height(currentData)
    
        clipD = currentData.R2{comi}; 
        clipNum = numel(clipD);
    
        totalR = [totalR; clipD];
    
        DurationAll = [DurationAll; repmat(currentData.Durations(comi), clipNum, 1)];
    end

end

function [xx, yy, r, p, dispT] = fit_lm_and_corr(x, D)

    mdl = fitlm(x, D);
    b = mdl.Coefficients.Estimate;  
    

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

function two_group_medina_quantile_plot(total_R_list, co, LegendNameList, signrankInd)

    if nargin < 4
        signrankInd = false;
    end
    for gi = 1:numel(total_R_list)
        currentData = total_R_list{gi};
        medianV = median(currentData);
        qdata = quantile(currentData, [0.25 0.75]);
        
        scatter(gi, medianV, 100, co(gi, :), 'filled', 'MarkerEdgeColor','none')
        hold on
        plot([gi gi], qdata, 'Color',co(gi, :), 'LineWidth',3)
    end
    xticks(1:numel(total_R_list)), xlim([0 numel(total_R_list)+1])
    xticklabels(LegendNameList)
    

    if signrankInd
        p = signrank(total_R_list{1}, total_R_list{2});
        if p > 0.05
            dispT = sprintf('p = %.2f(Wilcoxon sign-rank test)', p);
        elseif p >= 0.001
            dispT = sprintf('*p = %.3f(Wilcoxon sign-rank test)', p);
        else
            dispT = sprintf('*p < 0.001(Wilcoxon sign-rank test)');
        end
    else
        p = ranksum(total_R_list{1}, total_R_list{2});
        if p > 0.05
            dispT = sprintf('p = %.2f(Wilcoxon rank-sum test)', p);
        elseif p >= 0.001
            dispT = sprintf('*p = %.3f(Wilcoxon rank-sum test)', p);
        else
            dispT = sprintf('*p < 0.001(Wilcoxon rank-sum test)');
        end
    
    end


    plot([1 2], [0.1, 0.1], '-', 'Color', [0.5 0.5 0.5], 'LineWidth',1)
    text(1.05, 0.3, dispT, 'HorizontalAlignment','left')

end

function [mean_est, ci95] = bootstrap_CI(data, B)

    if nargin < 2
        B = 10000;
    end

    n = numel(data);
    boot_means = zeros(B,1);

    for b = 1:B
        idx = randi(n, n, 1);             
        sample = data(idx);               
        boot_means(b) = mean(sample);     
    end

    mean_est = mean(boot_means);
    ci95 = prctile(boot_means,[2.5 97.5]);   
end