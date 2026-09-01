%% Setup
% See README.md for installation instructions and software requirements.
clear;
clc;
close all;

% Reset the MATLAB search path.
restoredefaultpath
rehash toolboxcache

%%%%%%%% Note:  Select the project directory as the current working directory before running this script.
projectDir = string(pwd);

% Add FieldTrip to the MATLAB path
addpath(projectDir + "/external/fieldtrip");

% Initialize FieldTrip
ft_defaults

% Load the finger sweep dataset.
segD = load(projectDir + "/data/t18FingerSweepData.mat");

% Generate finger movement labels.
[OriginalLabel, convertedLabels, myOrder, myOrder_short] = my_label_converter();

%% Generate firing rate data
TX_segmentedD = {segD.TXpre, segD.TXgo};

% Number of bins before and after the Go cue
minDelayDur = 150; 
minGoDur = 240;

% Analysis window parameters
time_window_data = 50; % Number of bins included in each analysis window
sliding_data = 10; % Step size (bins) for the sliding window
bin_sec = 0.01;  % Duration of one bin (10 ms)

S_time = minDelayDur * -bin_sec;
E_time = minGoDur*bin_sec;

% Generate analysis time windows
[time_char, time_window_list, total_analized_bin_num] = create_AnaT_data(S_time, E_time, time_window_data, bin_sec, sliding_data);

% Generate 4-D neural data (Task × Trial × Time × Channel)
% Trials are reordered according to myOrder.
[inputFeaturesTX, TX_DeGo, ErrorTrial_tx, Transcriptions] = Four_D_data_make(TX_segmentedD, minDelayDur, minGoDur, segD, myOrder, convertedLabels, OriginalLabel);

% Convert threshold crossings (ncTX) to firing rates
sigma_bin = 3; % Gaussian kernel SD (30 ms)

% Compute firing rates after segmenting the neural data
inputFeaturesFR = my_FR_make(sigma_bin, inputFeaturesTX, bin_sec);

%% Generate baseline firing rate data
TX_segmentedD_base = {segD.TXpre, segD.TXpre};
minDelayDur_base = 0; 
minGoDur_base = 50;

% Generate baseline neural data
[inputFeaturesTX_Base, TX_DeGo_base, ErrorTrial_tx_base, Transcriptions_base] = Four_D_data_make(TX_segmentedD_base, minDelayDur_base, minGoDur_base, segD, myOrder, convertedLabels, OriginalLabel);

% Compute baseline firing rates
inputFeaturesFR_Base = my_FR_make(sigma_bin, inputFeaturesTX_Base, bin_sec);

%% Identify significant electrodes using spatiotemporal cluster analysis


Array_Allocation = ["v6d", "d6d", "r6d"];
Array_ChNum= {1:128, 129:256, 257:384};
taskNum = numel(myOrder);


% Channel indices are converted from the original Blackrock numbering to
% the post-mapping channel order during feature extraction.
% Convert them back to the original Blackrock channel numbering.
PostmapCh = {1:32, 33:64, 65:96, 97:128, ... % v6d
            129:160, 161:192, 193:224, 225:256, ... % d6d
            257:288, 289:320, 321:352, 353:384}; % r6d
BRCh = {1:32, 97:128, 33:64, 65:96, ... 
        161:192, 193:224, 129:160, 225:256, ... 
        289:320, 321:352, 257:288, 353:384}; 


Results = struct;
for i =1:numel(Array_Allocation)
    fieldName = Array_Allocation{i};
    Results.(fieldName) = cell(taskNum, 2);
end


inputData = inputFeaturesFR;
baseData = inputFeaturesFR_Base;


% Convert channel indices from the post-mapping order to the original
% Blackrock channel numbering.
anaD = ch_convert_from_postmap_to_BR(inputData, PostmapCh, BRCh);
baseD = ch_convert_from_postmap_to_BR(baseData, PostmapCh, BRCh);

% Channel indices corresponding to each subarray
subarray1_ind = [1:32, 97:128]; 
subarray2_ind = 33:96; 

fs = 100;  % sampling rate

% Total number of iterations for the progress bar
totalSteps = numel(Array_Allocation) * 2 * taskNum;
step = 0;

% Initialize the progress bar
h = waitbar(0, 'Spatiotemporal cluster analysis running...');

for arrayi = 1:numel(Array_Allocation)
    usedArrayNum = arrayi;
    fieldName = Array_Allocation(arrayi);
    for subarrayi = 1:2
        used_subarray = subarrayi;
        if used_subarray == 1
            input_subarray = subarray1_ind;
        elseif used_subarray == 2
            input_subarray = subarray2_ind;
        end   

        for taski = 1:taskNum
            task = taski;            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%% Step1: Convert the neural data into the FieldTrip format
            [data_bl, data_act] = base_active_data_make(baseD, anaD, S_time, E_time, Array_ChNum, usedArrayNum, input_subarray, task, fs);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%% Step2: Define neighboring channels
            % The d6d subarrays are rotated by 180°, whereas the
            % v6d and r6d subarrays are not rotated.
            if Array_Allocation(usedArrayNum) == "v6d" || Array_Allocation(usedArrayNum) == "r6d"
                neighbours = my_neighbours_make(used_subarray, false, Array_ChNum, usedArrayNum);
            elseif Array_Allocation(usedArrayNum) == "d6d" 
                neighbours = my_neighbours_make(used_subarray, true, Array_ChNum, usedArrayNum);
            end
                        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Step3: Perform the spatiotemporal cluster analysis       
            [stat, ~] = do_spatiotemporal_cluster_ana(data_act, data_bl, neighbours);
            Results.(fieldName){taski, subarrayi} = stat;

            %%%%%%%%%%%%%%%%%%%%%% Update the progress bar
            step = step + 1;
            waitbar(step / totalSteps, h, sprintf('Analyzing %s, Subarray %d, Task %d/%d', ...
                Array_Allocation{arrayi}, subarrayi, taski, taskNum));
            
        end
    end

end

close(h);

%% Save the spatiotemporal cluster analysis results
% Create the output directory if it does not exist
outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'SpatiotemporalClusterResults.mat'), ...
    'Results', ...
    'myOrder', ...
    'Array_Allocation', ...
    'Array_ChNum', ...
    'BRCh', ...
    '-v7.3');

%% Fig.1D

%%%%%%%%%%%%%%%%%%%%%%  Fixed parameters    %%%%%%%%%%%%%%%%%%%%%%  
Array_Allocation = ["v6d", "d6d", "r6d"]; 

% Row indices for each array and subarray
% Rows: arrays (v6d, d6d, r6d)
% Columns: Subarray 1 and Subarray 2
Array_yaxis   = {(1:8), (10:17); (29:36), (20:27); (48:55), (39:46)}; 

% Indicates whether each subarray is rotated by 180°
Array_180turn = {false, false;   true,     true;   false,   false};

subarray1_ind = [1:32, 97:128]';
subarray2_ind = (33:96)'; 
total_subarray_ind = {subarray1_ind, subarray2_ind};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


co = colororder("gem");

DispTaskList = myOrder;
DispTaskNum = numel(DispTaskList);

% Time interval (s) used for visualization
analizedTrange = [0 1]; 

t = tiledlayout(4, 8, 'TileSpacing','compact', 'Padding','compact');
for dispi = 1:DispTaskNum
    selectTask  = DispTaskList(dispi);
   
    [Results_updown] = my_Results_up_and_down_get(Array_Allocation, dispi, Results, analizedTrange);
    
    nexttile(t, dispi)

    % Draw the six subarrays
    display_my_six_subarray

    % Arrays to be displayed
    array_select = [1, 2, 3];

    circleScale = 0.5;

    % [Up, Down]
    % [1 0]: Up only
    % [0 1]: Down only
    % [1 1]: Both
    plot_select = [1, 1]; 
    my_up_or_down_ch_plot(plot_select, circleScale, array_select, Array_Allocation, Results_updown, total_subarray_ind, Array_180turn, Array_yaxis)
    title([char(selectTask), ' ', num2str(analizedTrange(1)), '-', num2str(analizedTrange(2)), 's'], 'FontSize',8)

end

% Draw the legend
nexttile(t, dispi+1)
% Draw the six subarrays
display_my_six_subarray
hold on
circleScale = 0.5;
scatter(1, 1, 10*circleScale, co(2, :), "filled", 'MarkerEdgeColor','none')
scatter(1, 3, 50*circleScale, co(2, :), "filled", 'MarkerEdgeColor','none')
scatter(1, 5, 100*circleScale, co(2, :), "filled", 'MarkerEdgeColor','none')
text(2, 1, '10', 'FontSize',8, 'FontName','Helvetica')
text(2, 3, '50', 'FontSize',8, 'FontName','Helvetica')
text(2, 5, '100', 'FontSize',8, 'FontName','Helvetica')

%% Fig.1E

%%%%%%%%%%%%%%%%%%%% Convert to Blackrock channel numbering %%%%%%%%%%%%%%%%%%%%
PostmapCh = {1:32, 33:64, 65:96, 97:128, ...        % v6d
            129:160, 161:192, 193:224, 225:256, ... % d6d
            257:288, 289:320, 321:352, 353:384};    % r6d
BRCh = {1:32, 97:128, 33:64, 65:96, ...         % v6d: Subarrays 1 and 2 are not rotated
        161:192, 193:224, 129:160, 225:256, ... % d6d: Subarrays 1 and 2 are rotated by 180°
        289:320, 321:352, 257:288, 353:384};    % r6d: Subarrays 1 and 2 are not rotated

inputFeaturesFR_BR = ch_convert_from_postmap_to_BR(inputFeaturesFR, PostmapCh, BRCh);


%%%%%%%%%%%%%%%%%%%% Generate Figure 1E %%%%%%%%%%%%%%%%%%%%
F = figure;
t = tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','compact');
axAll = gobjects(1, 2);

%%%% Plot PSTHs from two representative channels for the same task %%%%
color_value = colororder("gem12");
SelectTask = "R Middle Up";
selectTaskInd = find(SelectTask == myOrder);
UpChNum = 374;    % Representative channel showing a significant increase
DownChNum = 360;  % Representative channel showing a significant decrease
legendName = ["Blackrock Ch " + string(UpChNum), "Blackrock Ch " + string(DownChNum)];
titleName = SelectTask + " with r6d sub1";

axAll(1) = nexttile(t, 1);

timed = linspace(S_time, E_time, size(inputFeaturesFR_BR, 3));

%%%% Plot the channel showing a significant increase
[psth_d, ci_Data] = make_psth_data(inputFeaturesFR_BR, SelectTask, UpChNum, myOrder);
plot(timed, psth_d, 'Color', color_value(2, :), 'LineWidth', 2)
hold on
fill([timed, flip(timed)], [ci_Data(1, :), flip(ci_Data(2, :))], color_value(2, :), 'FaceAlpha',0.2, 'EdgeColor','none')
xline(0, '-', 'LineWidth',1, 'Color', [0.5 0.5 0.5])
% Mark time points belonging to significant spatiotemporal clusters
my_sigpoint_scatter(UpChNum, 200, color_value(2, :), true, Results, selectTaskInd, Array_Allocation, Array_ChNum, total_subarray_ind); % 第４引数はUpの場合,true

%%%% Plot the channel showing a significant decrease
[psth_d, ci_Data] = make_psth_data(inputFeaturesFR_BR, SelectTask, DownChNum, myOrder);
plot(timed, psth_d, 'Color', color_value(1, :), 'LineWidth', 2)
fill([timed, flip(timed)], [ci_Data(1, :), flip(ci_Data(2, :))], color_value(1, :), 'FaceAlpha',0.2, 'EdgeColor','none')
% Mark time points belonging to significant spatiotemporal clusters
my_sigpoint_scatter(DownChNum, 0, color_value(1, :), false, Results, selectTaskInd, Array_Allocation, Array_ChNum, total_subarray_ind); % 第４引数はUpの場合,true

ylim([-10 210]), yticks(0:40:200)
xlim([-1.5 2]), xticks(-1:2)

%%% Legend
h_color = gobjects(2, 1);

h_color(1) = scatter(nan, nan, 50, color_value(2,:), 'filled');
h_color(2) = scatter(nan, nan, 50, color_value(1,:), 'filled');
L = legend(h_color, legendName, 'Location','bestoutside', 'Box', 'off');
L.FontSize = 10;

box off
xlabel('Time (s) from Go (t=0)', 'FontName','Helvetica')
ylabel('Firing rate (Hz)', 'FontName','Helvetica', 'FontSize',12)
title(titleName)

%%%% Plot PSTHs from multiple tasks for a representative channel %%%%
co1 = colororder("gem");
co2 = colororder("glow");
color_value = [co1(1, :); co2(1, :); co1(2, :); co2(2, :)];

FixedCh = 360;
% Tasks to be displayed
TaskLisst = ["R Middle Up", "R Middle Down", "L Middle Up", "L Middle Down"];
numTasks = numel(TaskLisst);

axAll(2) = nexttile(t, 2);
for taski = 1:numTasks
    currentTask = TaskLisst(taski);
    [psth_d, ci_Data] = make_psth_data(inputFeaturesFR_BR, currentTask, FixedCh, myOrder);
    plot(timed, psth_d, 'Color', color_value(taski, :), 'LineWidth', 2)
    hold on
    
end
xline(0, '-', 'LineWidth',1, 'Color', [0.5 0.5 0.5])

h_color = gobjects(numTasks, 1);
for i = 1:numTasks
    h_color(i) = scatter(nan, nan, 50, color_value(i,:), 'filled');
end
L = legend(h_color, TaskLisst, 'Location','bestoutside', 'Box', 'off');
L.FontSize = 10;

xlabel('Time (s) from Go (t=0)', 'FontName','Helvetica')
ylabel('Firing rate (Hz)', 'FontName','Helvetica', 'FontSize',12)
titleName = "Blackrock Ch" + string(FixedCh) + " in r6d sub1";
title(titleName)
box off
ylim([0 135]), yticks(0:40:200)
xlim([-1.5 2]), xticks(-1:2)
set(findall(F,'Type','axes'), 'FontName','Helvetica', 'FontSize', 12);

%% Fig.1F
% Generate normalized mean t-values plots for Figure 1F
% Calculate normalized mean t-values over time for right- and left-hand movement groups

Right_F = ["R Thumb Up", "R Thumb Down", "R Thumb In", ...
    "R Index Up", "R Index Down", "R Index In", ...
    "R Middle Up", "R Middle Down", "R Middle In", ...
    "R Ring Up", "R Ring Down", "R Ring In", ...
    "R Pinky Up", "R Pinky Down", "R Pinky In"];

Left_F = ["L Thumb Up", "L Thumb Down", "L Thumb In", ...
    "L Index Up", "L Index Down", "L Index In", ...
    "L Middle Up", "L Middle Down", "L Middle In", ...
    "L Ring Up", "L Ring Down", "L Ring In", ...
    "L Pinky Up", "L Pinky Down", "L Pinky In"];

% Group tasks into right- and left-hand movements
GroupLisName = "Right and Left";
analysed_group = {Right_F, Left_F}; 
groupName = ["Right", "Left"];

GroupNum = numel(analysed_group);

ResultUD = struct();

Array_Allocation = ["v6d", "d6d", "r6d"];
arrayNum = numel(Array_Allocation);

% Process each array
for arrayi = 1:arrayNum
    currentArray = Array_Allocation(arrayi);
    % Process each movement group
    for gi = 1:GroupNum
        currentG = analysed_group{gi};
        % Identify task indices belonging to the current movement group
        current_taskInd = find(contains(myOrder, currentG));
        
        % Collect significant-channel results across tasks
        totalUp = [];
        totalDown = [];
        totalUp_tstat = [];
        totalDown_tstat = [];
        for taski = 1:numel(current_taskInd)
            currentIndTask = current_taskInd(taski);
        
             % Extract significant channels and their t-statistics
            [upData, downData, cuTime, upData_tstat, downData_tstat] = total_sig_ch_extraction_eachArray(Results, currentArray, currentIndTask);
        
            totalUp = [totalUp; upData];
            totalDown = [totalDown; downData];
            totalUp_tstat = [totalUp_tstat; upData_tstat];
            totalDown_tstat = [totalDown_tstat; downData_tstat];
        end
        
        % Normalize positive t-values by the number of significant channels
        [mean_tnorm_up, mean_tnorm_up_ci, mean_t_up, mean_t_up_ci] = mean_t_and_normalized_t_get(totalUp, totalUp_tstat);
        % Smooth the time series using a Gaussian kernel with reflection padding
        sigma = 0.02;
        mean_tnorm_up_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_up');
        mean_tnorm_up_ci1_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_up_ci(:, 1)');
        mean_tnorm_up_ci2_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_up_ci(:, 2)');

        % Normalize negative t-values by the number of significant channels
        [mean_tnorm_down, mean_tnorm_down_ci, mean_t_down, mean_t_up_down] = mean_t_and_normalized_t_get(totalDown, totalDown_tstat);
        % Smooth the time series using a Gaussian kernel with reflection padding
        mean_tnorm_down_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_down');
        mean_tnorm_down_ci1_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_down_ci(:, 1)');
        mean_tnorm_down_ci2_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_down_ci(:, 2)');

        % Store the results
        ResultUD.(currentArray).(groupName(gi)).totalUp = totalUp;
        ResultUD.(currentArray).(groupName(gi)).totalDown = totalDown;
        ResultUD.(currentArray).(groupName(gi)).totalUp_tstat = totalUp_tstat;
        ResultUD.(currentArray).(groupName(gi)).totalDown_tstat = totalDown_tstat;
        ResultUD.(currentArray).(groupName(gi)).aveUpS = mean_tnorm_up_smooth;
        ResultUD.(currentArray).(groupName(gi)).ciUpS = [mean_tnorm_up_ci1_smooth; mean_tnorm_up_ci2_smooth];    
        ResultUD.(currentArray).(groupName(gi)).aveDownS = mean_tnorm_down_smooth;
        ResultUD.(currentArray).(groupName(gi)).ciDownS = [mean_tnorm_down_ci1_smooth; mean_tnorm_down_ci2_smooth];   
    end

end

% Combine v6d and d6d results to obtain the left 6d (l6d) result
for gi = 1:GroupNum
    currentG = groupName(gi);
    totalUp = [ResultUD.v6d.(currentG).totalUp; ResultUD.d6d.(currentG).totalUp];
    totalDown = [ResultUD.v6d.(currentG).totalDown; ResultUD.d6d.(currentG).totalDown];

    totalUp_tstat = [ResultUD.v6d.(currentG).totalUp_tstat; ResultUD.d6d.(currentG).totalUp_tstat];
    totalDown_tstat = [ResultUD.v6d.(currentG).totalDown_tstat; ResultUD.d6d.(currentG).totalDown_tstat];


    [mean_tnorm_up, mean_tnorm_up_ci, mean_t_up, mean_t_up_ci] = mean_t_and_normalized_t_get(totalUp, totalUp_tstat);

    sigma = 0.02;
    mean_tnorm_up_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_up');
    mean_tnorm_up_ci1_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_up_ci(:, 1)');
    mean_tnorm_up_ci2_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_up_ci(:, 2)');


    [mean_tnorm_down, mean_tnorm_down_ci, mean_t_down, mean_t_up_down] = mean_t_and_normalized_t_get(totalDown, totalDown_tstat);

    mean_tnorm_down_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_down');
    mean_tnorm_down_ci1_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_down_ci(:, 1)');
    mean_tnorm_down_ci2_smooth = make_signal_smooth_with_G(bin_sec, sigma,  mean_tnorm_down_ci(:, 2)');


    ResultUD.l6d.(groupName(gi)).totalUp = totalUp;
    ResultUD.l6d.(groupName(gi)).totalDown = totalDown;
    ResultUD.l6d.(groupName(gi)).totalUp_tstat = totalUp_tstat;
    ResultUD.l6d.(groupName(gi)).totalDown_tstat = totalDown_tstat;
    ResultUD.l6d.(groupName(gi)).aveUpS = mean_tnorm_up_smooth;
    ResultUD.l6d.(groupName(gi)).ciUpS = [mean_tnorm_up_ci1_smooth; mean_tnorm_up_ci2_smooth];    
    ResultUD.l6d.(groupName(gi)).aveDownS = mean_tnorm_down_smooth;
    ResultUD.l6d.(groupName(gi)).ciDownS = [mean_tnorm_down_ci1_smooth; mean_tnorm_down_ci2_smooth]; 
end

co1 = colororder("gem");

color_value = [co1(2, :); co1(7, :)];
color_value2 = [co1(1, :); co1(4, :)];

% Plot normalized mean t-values
F = figure;
fieldNameList = fieldnames(ResultUD);
fNum = numel(fieldNameList);
t = tiledlayout(fNum, 2, 'TileSpacing','compact', 'Padding','compact');
axRight = gobjects(fNum, 1);
axLeft = gobjects(fNum, 1);
legendNames = ["Right hand", "Left hand"];
for fi = 1:fNum
    currentF = fieldNameList{fi};

    % Plot normalized mean t-values for significantly increased channels
    axRight(fi) = nexttile(t, (fi-1)*2+1);
    yline(0, '-', 'LineWidth', 1, 'Color', [0.5 0.5 0.5])
    hold on
    for gi = 1:GroupNum
        currentG = groupName(gi);
        aveS = ResultUD.(currentF).(currentG).aveUpS;
        ciS = ResultUD.(currentF).(currentG).ciUpS;
        

        plot(cuTime, aveS, 'Color', color_value(gi, :), 'LineWidth',1.5);

        fill([cuTime, flip(cuTime)], [ciS(1, :), flip(ciS(2, :))], color_value(gi, :), 'FaceAlpha',0.2, 'EdgeColor','none')
    end
    titleName = currentF + ": Sig. Up channels";
    title(char(titleName))

    %%% Legend
    h_color = gobjects(GroupNum, 1);
    for i = 1:GroupNum
        h_color(i) = scatter(nan, nan, 50, color_value(i,:), 'filled');
    end
    legend(h_color, legendNames, 'Location','bestoutside', 'Box', 'off', 'AutoUpdate', 'off', 'FontName','Helvetica', 'FontSize', 12);
    xline(0, '-', 'LineWidth', 1, 'Color', [0.5 0.5 0.5])
    ylabel('Normalized mean t-value')
    xlabel('Time (s) from Go (t=0)')
    ylim([-0.1 2.6]), yticks(0:1:2)
    box off

    % Plot normalized mean t-values for significantly decreased channels
    axLeft(fi) = nexttile(t, fi*2);
    yline(0, '-', 'LineWidth', 1, 'Color', [0.5 0.5 0.5])
    hold on
    for gi = 1:GroupNum
        currentG = groupName(gi);
        aveS = ResultUD.(currentF).(currentG).aveDownS;
        ciS = ResultUD.(currentF).(currentG).ciDownS;
        

        plot(cuTime, aveS, 'Color', color_value2(gi, :), 'LineWidth',1.5);

        fill([cuTime, flip(cuTime)], [ciS(1, :), flip(ciS(2, :))], color_value2(gi, :), 'FaceAlpha',0.2, 'EdgeColor','none')
    end
    titleName = currentF + ": Sig. Down channels";
    title(char(titleName))
    %%% Legend
    h_color = gobjects(GroupNum, 1);
    for i = 1:GroupNum
        h_color(i) = scatter(nan, nan, 50, color_value2(i,:), 'filled');
    end
    legend(h_color, legendNames, 'Location','bestoutside', 'Box', 'off', 'AutoUpdate', 'off', 'FontName','Helvetica', 'FontSize', 12);
    xline(0, '-', 'LineWidth', 1, 'Color', [0.5 0.5 0.5])
    box off
    ylabel('Normalized mean t-value')
    xlabel('Time (s) from Go (t=0)')
    ylim([-0.25 0.01]), yticks(-0.2:0.1:0)
end

linkaxes(axRight, 'xy');
linkaxes(axLeft, 'xy');
set(findall(F, 'Type', 'axes'),'FontName','Helvetica');

%% Fig.1G and H

% Load the original spatiotemporal cluster analysis results used to
% generate the dendrograms shown in the manuscript.
% These results are provided to ensure exact reproduction of the figure.
resultData = fullfile(projectDir, 'data', 'SpatiotemporalClusterResults_Orig.mat');
load(resultData)

%%% Compute Hamming distances across tasks
Array_Allocation = ["v6d", "d6d", "r6d"]; % Array configuration is consistent across sessions
arrayNum = numel(Array_Allocation);

% Generate sliding analysis time windows
S_time = -1.5; 
E_time = 2.4;
time_window_data = 100;
bin_sec = 0.01;
sliding_data = 10;
[time_char, time_window_list, timeNum] = create_AnaT_data(S_time, E_time, time_window_data, bin_sec, sliding_data);
% Generate lists of start and end times for each analysis window
S_timeL = S_time:sliding_data*bin_sec:S_time + (timeNum-1)*sliding_data*bin_sec;
E_timeL = S_time + time_window_data*bin_sec:sliding_data*bin_sec:E_time;

taskNum = numel(myOrder);

% Generate channel-location representations for Hamming distance analysis.
% Each channel is classified as:
% 0 = neither significant increase nor decrease
% 1 = significant increase only
% 2 = significant decrease only
% 3 = both increase and decrease

Results_updown_location = struct();
for timei = 1:timeNum
    analizedTrange = [S_timeL(timei), E_timeL(timei)]; 
    
    for taski = 1:taskNum
        Results_updown = my_Results_up_and_down_get(Array_Allocation, taski, Results, analizedTrange);
    
        for arrayi = 1:arrayNum
            fieldName = Array_Allocation(arrayi);

            threeClassInd = my_updown_location_make(fieldName, Results_updown);
            Results_updown_location.(fieldName){timei,taski} = threeClassInd;
        end
    
    end

    % Combine v6d and d6d to obtain the left 6d (l6d) representation
    for taski = 1:taskNum
        v6dD = Results_updown_location.v6d{timei,taski};
        d6dD = Results_updown_location.d6d{timei,taski};
        Results_updown_location.l6d{timei,taski} = [v6dD; d6dD];
    end
    
end

% Structure for storing Hamming distance matrices
Result_Hamming = struct();

fieldnameList = fieldnames(Results_updown_location);

for fi = 1:numel(fieldnameList)
    currentF = string(fieldnameList{fi});
    
    for timei = 1:timeNum
        % Compute pairwise Hamming distances from the channel-state labels.
        % A distance of 0 indicates identical patterns, whereas 1 indicates
        % completely different patterns.
        hammingResults = get_hamming_dist(timei, currentF, Results_updown_location, taskNum);
        Result_Hamming.(currentF){1, timei} = hammingResults;
    end

end


% Visualize Hamming distance matrices
F = figure;
subarrayInd = 1;

HammingDist = struct;

for fi = 3:4
    
    currentF = string(fieldnameList{fi});
    for timei = 18 % Corresponds to the 0.7-s analysis window
        subplot(1, 2, subarrayInd)
        subarrayInd = subarrayInd + 1;
        currentHamming = Result_Hamming.(currentF){1, timei};
        HammingDist.(currentF) = currentHamming;
        imagesc(currentHamming)
        clim([0 1]);          
        axis('image');

        pbaspect([1 1 1])
        xticks(1:taskNum);  yticks(1:taskNum);
        xticklabels(myOrder_short);  yticklabels(myOrder_short);
        ax = gca;
        ax.FontSize = 8;

        title(currentF + ": " + string(time_char(timei)) + " (s)", 'FontSize', 10);
        colorbar;
    end


end

% Generate dendrograms
color_value = colororder("gem");

% Define right- and left-hand movement groups
Right_F = ["R Thumb Up",  "R Thumb Down",  "R Thumb In", "R Index Up",  "R Index Down",  "R Index In", "R Middle Up", "R Middle Down", "R Middle In", ...
    "R Ring Up",   "R Ring Down",   "R Ring In", "R Pinky Up",  "R Pinky Down",  "R Pinky In"];
Left_F = ["L Thumb Up",  "L Thumb Down",  "L Thumb In", "L Index Up",  "L Index Down",  "L Index In", "L Middle Up", "L Middle Down", "L Middle In", ...
    "L Ring Up",   "L Ring Down",   "L Ring In", "L Pinky Up",  "L Pinky Down",  "L Pinky In"];

% Define movement-direction groups
R_Up_F = ["R Thumb Up","R Index Up","R Middle Up","R Ring Up","R Pinky Up"];
R_Down_F = ["R Thumb Down","R Index Down","R Middle Down","R Ring Down","R Pinky Down"];
R_In_F = ["R Thumb In","R Index In","R Middle In","R Ring In","R Pinky In"];
L_Up_F = ["L Thumb Up","L Index Up","L Middle Up","L Ring Up","L Pinky Up"];
L_Down_F = ["L Thumb Down","L Index Down","L Middle Down","L Ring Down","L Pinky Down"];
L_In_F = ["L Thumb In","L Index In","L Middle In","L Ring In","L Pinky In"];

% Right 6d: color-code clusters by right- vs. left-hand movements
dendroLabelL = {Right_F, Left_F}; 
LabelName = "RightLeft";

F = figure;
subplot(1,2,1)
currentF = "Right 6d";

currentHamming = HammingDist.r6d;
% Plot the dendrogram
my_dendrogram_plot(currentHamming, myOrder, dendroLabelL, color_value, 6);

title(currentF + ": 0.7(s)", 'FontSize', 10);

% Left 6d: color-code clusters by movement direction
dendroLabelL = {[R_Up_F,L_Up_F] [R_Down_F,L_Down_F], [R_In_F, L_In_F]}; 
LabelName = "UpDwIn";

subplot(1,2,2)
currentF = "Left 6d";

currentHamming = HammingDist.l6d;
% Plot the dendrogram
my_dendrogram_plot(currentHamming, myOrder, dendroLabelL, color_value, 6);

title(currentF + ": 0.7(s)", 'FontSize', 10);

%% Internal function



function [OriginalLabel, convertedLabels, myOrder, myOrder_short] = my_label_converter()


    OriginalLabel = ["DO NOTHING", ...
        "LEFT INDEX - Extend Upwards",   "LEFT INDEX - In to Palm",   "LEFT INDEX - Straight Downwards", ...
        "LEFT MIDDLE - Extend Upwards",  "LEFT MIDDLE - In to Palm",  "LEFT MIDDLE - Straight Downwards", ...
        "LEFT PINKY - Extend Upwards",   "LEFT PINKY - In to Palm",   "LEFT PINKY - Straight Downwards", ...
        "LEFT RING - Extend Upwards",    "LEFT RING - In to Palm",    "LEFT RING - Straight Downwards", ...
        "LEFT THUMB - Extend Upwards",   "LEFT THUMB - In to Palm",   "LEFT THUMB - Straight Downwards", ...
        "RIGHT INDEX - Extend Upwards",  "RIGHT INDEX - In to Palm",  "RIGHT INDEX - Straight Downwards", ...
        "RIGHT MIDDLE - Extend Upwards", "RIGHT MIDDLE - In to Palm", "RIGHT MIDDLE - Straight Downwards", ...
        "RIGHT PINKY - Extend Upwards",  "RIGHT PINKY - In to Palm",  "RIGHT PINKY - Straight Downwards", ...
        "RIGHT RING - Extend Upwards",   "RIGHT RING - In to Palm",   "RIGHT RING - Straight Downwards", ...
        "RIGHT THUMB - Extend Upwards",  "RIGHT THUMB - In to Palm",  "RIGHT THUMB - Straight Downwards"];


    convertedLabels = ["None", ...
    "L Index Up",  "L Index In",  "L Index Down", ...
    "L Middle Up", "L Middle In", "L Middle Down", ...
    "L Pinky Up",  "L Pinky In",  "L Pinky Down", ...
    "L Ring Up",   "L Ring In",   "L Ring Down", ...  
    "L Thumb Up",  "L Thumb In",  "L Thumb Down", ... 
    "R Index Up",  "R Index In",  "R Index Down", ...
    "R Middle Up", "R Middle In", "R Middle Down", ...
    "R Pinky Up",  "R Pinky In",  "R Pinky Down", ...
    "R Ring Up",   "R Ring In",   "R Ring Down", ...  
    "R Thumb Up",  "R Thumb In",  "R Thumb Down"];

    myOrder = ["None", ...
        "R Thumb Up",  "R Thumb Down",  "R Thumb In", ...
        "R Index Up",  "R Index Down",  "R Index In", ...
        "R Middle Up", "R Middle Down", "R Middle In", ...
        "R Ring Up",   "R Ring Down",   "R Ring In", ...  
        "R Pinky Up",  "R Pinky Down",  "R Pinky In", ...
        "L Thumb Up",  "L Thumb Down",  "L Thumb In", ...
        "L Index Up",  "L Index Down",  "L Index In", ...
        "L Middle Up", "L Middle Down", "L Middle In", ...
        "L Ring Up",   "L Ring Down",   "L Ring In", ...  
        "L Pinky Up",  "L Pinky Down",  "L Pinky In"];
    

    myOrder_short = ["None", ...
        "R T Up",  "R T Dw",  "R T In", ...
        "R I Up",  "R I Dw",  "R I In", ...
        "R M Up",  "R M Dw",  "R M In", ...
        "R R Up",  "R R Dw",  "R R In", ...  
        "R P Up",  "R P Dw",  "R P In", ...
        "L T Up",  "L T Dw",  "L T In", ...
        "L I Up",  "L I Dw",  "L I In", ...
        "L M Up",  "L M Dw",  "L M In", ...
        "L R Up",  "L R Dw",  "L R In", ...  
        "L P Up",  "L P Dw",  "L P In"];


end


function [time_char, time_window_list, total_analized_bin_num] = create_AnaT_data(S_time, E_time, time_window_data, bin_sec, sliding_data)

    total_analized_bin_num = floor(((E_time - S_time) - time_window_data*bin_sec)/(sliding_data*bin_sec) + 1e-10)+1;
    time_char = {};
    time_window_list = {};
    for i = 1:total_analized_bin_num
        current_stime = (i-1)*sliding_data*bin_sec+S_time;
        current_etime = (i-1)*sliding_data*bin_sec+S_time + time_window_data*bin_sec;
        menantime = (current_stime + current_etime)/2;
        time_char{end+1} = sprintf('%.2f',menantime );
        time_window_list{end+1} = [floor((current_stime - S_time)/bin_sec)+1: floor((current_stime - S_time)/bin_sec)+  time_window_data];
    end

end


function [inputFeaturesTX, TX_DeGo2, ErrorTrial_tx, Transcriptions] = Four_D_data_make(TX_segmentedD, minDelayDur, minGoDur, segD, myOrder, convertedLabels, OriginalLabel)


    [TX_DeGo, ErrorTrial_tx] = pre_post_Cue_data_cat(TX_segmentedD, minDelayDur, minGoDur);

    TX_DeGo2 = TX_DeGo;
    TX_DeGo2(ErrorTrial_tx) = [];

    Transcriptions = segD.transcriptions;
    Transcriptions(ErrorTrial_tx) = [];

    minimunTrial = min(countcats(categorical(Transcriptions)));
    taskNum = numel(countcats(categorical(Transcriptions)));
    if  ~numel(myOrder) == taskNum
        error('The number of tasks in "Transcription" does not match the number of tasks in "myOrder."')
    end
    [dataNum, Chnum] = size(TX_DeGo2{1});

    inputFeaturesTX = create_fourD_data(TX_DeGo2, Transcriptions, taskNum, minimunTrial, dataNum, Chnum, myOrder, convertedLabels, OriginalLabel);


end


function [TX_DeGo, ErrorTrial] = pre_post_Cue_data_cat(TX_segmentedD, minDelayDur, minGoDur)
    TX_DeGo = {};
    Error_trialL = [];
    trial_num = size(TX_segmentedD{2}, 2);
    for t = 1:trial_num
        SegD1 = TX_segmentedD{1}{t};
        SegD2 = TX_segmentedD{2}{t};
        ChNum = size(SegD1, 2);
        DataSize1 = size(SegD1, 1);
        DataSize2 = size(SegD2, 1);


        if DataSize1 - minDelayDur >= 0
            cur_firstTX = SegD1(end-minDelayDur+1:end, :);
        else

            cur_firstTX = nan(minDelayDur, ChNum);
            cur_firstTX(end - DataSize1+1:end, :) = SegD1;
            warn_message = sprintf('Trial %d predata is short', t);
            warning(warn_message)
            Error_trialL(end+1) = t;
        end
    

        if  minGoDur <= DataSize2
            cur_secondTX = SegD2(1:minGoDur, :);
        else

            cur_secondTX = nan(minGoDur, ChNum);
            cur_secondTX(1:DataSize2, :) = SegD2;
            warn_message = sprintf('Trial %d data is short', t);
            warning(warn_message)
            Error_trialL(end+1) = t;
        end
    
        TX_DeGo{end+1} = cat(1, cur_firstTX, cur_secondTX);
    end
    ErrorTrial = unique(Error_trialL);
end


function [fourDdata] = create_fourD_data(inputdata, Transcriptions, taskNum, minimunTrial, dataNum, Chnum, myOrder, convertedLabels, OriginalLabel)
    fourDdata = nan(taskNum, minimunTrial, dataNum, Chnum);

    for current_task = 1:taskNum
        currentLabel = myOrder(current_task);
        currentOriLabe = OriginalLabel(convertedLabels == currentLabel);
        currentIndList = find(Transcriptions == currentOriLabe);
        for triali = 1:minimunTrial
            selectInd = currentIndList(triali);
            fourDdata(current_task, triali, :, :) = inputdata{selectInd};
        end
    end

end


function [inputFeaturesFR] = my_FR_make(sigma_bin, inputFeaturesTXorig, bin_sec)


    inputFeaturesFR = nan(size(inputFeaturesTXorig));
    [taskNum, trialNum, ~, chNum] = size(inputFeaturesTXorig);
    for taski = 1:taskNum
        for triali = 1:trialNum
            for chi = 1:chNum
                currentData = squeeze(inputFeaturesTXorig(taski, triali, :, chi));

                fr_hz = my_gaussKernel_convert(sigma_bin, bin_sec, currentData);
                inputFeaturesFR(taski, triali, :, chi) = fr_hz;
            end
        end
    
    end

end


function fr_hz = my_gaussKernel_convert(sigma_bin, bin_sec, sigD)
    % sigma_bin : Standard deviation of the Gaussian kernel (bins)
    % bin_sec   : Duration of one time bin (s), e.g., 0.01
    % sigD      : Threshold crossing counts (ncTX) for each time bin
    
    % Generate a Gaussian kernel spanning ±3 standard deviations
    t = -round(3*sigma_bin):round(3*sigma_bin);

    % Compute and normalize the Gaussian kernel
    gaussKernel = exp(-(t.^2)/(2*sigma_bin^2));
    gaussKernel = gaussKernel / sum(gaussKernel);

    khalf = floor(length(gaussKernel)/2);

    % --- Reflect padding---
    % Reflect the left edge
    leftPad  = sigD(khalf+1:-1:2);

    % Reflect the right edge
    rightPad = sigD(end-1:-1:end-khalf);

    sigPadded = [leftPad; sigD; rightPad];

    % Perform Gaussian convolution
    tmp = conv(sigPadded, gaussKernel, 'same');

    % Remove the padded samples
    startIdx = khalf+1;
    endIdx   = startIdx + length(sigD) - 1;

    % Convert counts per bin to firing rate (Hz)
    fr_hz = tmp(startIdx:endIdx) / bin_sec;
end


function [reshapeData] = ch_convert_from_postmap_to_BR(inputFeaturesFR, PostmapCh, BRCh)
    [taskNum, trialNum, dataNum, chNum] = size(inputFeaturesFR);
    reshapeData = nan(taskNum, trialNum, dataNum, chNum);
    subarrayNum = numel(PostmapCh); 
    
    for taski = 1:taskNum
        for triali = 1:trialNum
            for subarrayi = 1:subarrayNum
                clippedData = inputFeaturesFR(taski, triali, :, PostmapCh{subarrayi});
                reshapeData(taski, triali, :, BRCh{subarrayi}) = clippedData;
            end
            
        end
    end
end


function [data_bl, data_act] = base_active_data_make(baseD, anaD, S_time_G, E_time_G, Array_ChNum, usedArrayNum, subarray1_ind, task, fs)
    [~, nTrials, nTime, ~] = size(anaD);
    Times= linspace(S_time_G, E_time_G, nTime);
    

    arrayData = anaD(:, :, :, Array_ChNum{usedArrayNum});
    clipped_arrayData = arrayData(:, :, :, subarray1_ind);
    
    arrayBaseD = baseD(:, :, :,  Array_ChNum{usedArrayNum});
    clippedBaseD = arrayBaseD(:, :, :, subarray1_ind);
    

    data_bl = [];  % baseline:
    data_act = []; % active

    for trial = 1:nTrials
    
        % active
        data_act.trial{trial} = squeeze(clipped_arrayData(task, trial, :, :))';
        data_act.time{trial}  = Times;
    
    
        % baseline 
        ave_base_d = mean(squeeze(clippedBaseD(task, trial, :, :)))'; 
        data_bl.trial{trial} = repmat(ave_base_d, 1, nTime); 
        data_bl.time{trial}  = Times;   
    
    end
    
    data_bl.label = arrayfun(@(x) sprintf('Ch%d', x), subarray1_ind + Array_ChNum{usedArrayNum}(1)-1, 'UniformOutput', false);
    data_act.label = data_bl.label;
    data_bl.fsample = fs;
    data_act.fsample = fs;
end

% Spatio-temporal cluster analysis
function [stat, cfg] = do_spatiotemporal_cluster_ana(data1, data2, neighbours)
    cfg = [];
    cfg.method           = 'montecarlo';  % Monte Carlo permutation test
    cfg.statistic        = 'depsamplesT'; % Dependent-samples t-statistic (paired t-test)
    cfg.correctm         = 'cluster';     % Cluster-based multiple-comparison correction
    cfg.clusteralpha     = 0.05;          % Cluster-forming threshold (sample-level p < 0.05)
    cfg.clusterstatistic = 'maxsum';      % Cluster statistic: maximum sum of t-values
    cfg.tail             = 0;             % Two-sided test
    cfg.clustertail      = 0;             % Two-sided cluster test
    cfg.alpha            = 0.025;         % Cluster-level significance threshold (two-sided α = 0.05)
    cfg.numrandomization = 1000;          % Number of permutations
    cfg.neighbours       = neighbours;    % Spatial neighborhood definition
    
    % Design matrix for a paired t-test
    nTrials = size(data1.trial, 2);
    nTotal = 1 * nTrials;
    cfg.design = [1:nTotal, 1:nTotal; ones(1,nTotal), 2*ones(1,nTotal)];
    cfg.uvar   = 1; % trial
    cfg.ivar   = 2; % condition (1=BL, 2=Act)
    
    % Perform the cluster-based permutation test (data1 vs. data2)
    [stat] = ft_timelockstatistics(cfg, data1, data2);

end


function neighbours = my_neighbours_make(subarraynum, turn_180, Array_ChNum, usedArrayNum)

    % subarray1
    ch_grid1 = [
     2   17   23   31   97  105  112  128;
     1   15   25   18   99  106  115  114;
     3    8   21   22   98  107  117  116;
     4   11   14   24  100  108  119  118;
     7   13   16   26  101  109  121  120;
     5   19   20   28  102  110  123  122;
     6   10   27   30  103  111  125  124;
     9   12   29   32  104  113  127  126
    ];

    % 180°-rotated version of Subarray 1
    ch_grid1_180 = [
      126 127 113 104 32  29 12 9;
      124 125 111 103 30  27 10 6;
      122 123 110 102 28  20 19 5;
      120 121 109 101 26  16 13 7;
      118 119 108 100 24  14 11 4;
      116 117 107 98  22  21 8  3;
      114 115 106 99  18  25 15 1;
      128 112 105 97  31  23 17 2
    ];

    % subarray2
    ch_grid2 = [
        65  66  81  89  33  38  51  58;
        67  68  82  90  35  40  53  57;
        69  70  84  91  37  43  50  62;
        71  72  83  92  39  47  46  60;
        73  74  86  93  41  49  56  59;
        75  76  85  94  45  48  55  64;
        77  78  87  96  34  42  54  61;
        79  80  88  95  36  44  52  63
    ];

    % 180°-rotated version of Subarray 2
    ch_grid2_180 = [
        63  52  44  36  95  88  80  79;
        61  54  42  34  96  87  78  77;
        64  55  48  45  94  85  76  75;
        59  56  49  41  93  86  74  73;
        60  46  47  39  92  83  72  71;
        62  50  43  37  91  84  70  69;
        57  53  40  35  90  82  68  67;
        58  51  38  33  89  81  66  65
    ];
    if subarraynum == 1
        if turn_180
            used_ch_grid = ch_grid1_180;
        else
            used_ch_grid = ch_grid1;
        end
    elseif subarraynum == 2
        if turn_180
            used_ch_grid = ch_grid2_180;
        else
            used_ch_grid = ch_grid2;
        end
    end

    

    [X, Y] = meshgrid(1:8, 1:8);
    X = X(:); Y = Y(:);
    ch_list = used_ch_grid(:) + Array_ChNum{usedArrayNum}(1) - 1;
    

    elec.label = cell(64,1);
    elec.elecpos = zeros(64,3);
    elec.chanpos = zeros(64,3);
    for i = 1:64
        elec.label{i} = sprintf('Ch%d', ch_list(i));  
        elec.elecpos(i,:) = [X(i), Y(i), 0];          
        elec.chanpos(i,:) = [X(i), Y(i), 0];
    end
    elec.unit = 'cm'; 
    elec.type = 'utah_subarray1';
    
    cfg_neighb = [];
    cfg_neighb.method = 'distance';
    cfg_neighb.elec = elec;
    
    cfg_neighb.neighbourdist = (0.07/0.04);  
    
    neighbours = ft_prepare_neighbours(cfg_neighb);

end


function [Results_updown] = my_Results_up_and_down_get(Array_Allocation, selectTaskInd, Results, analizedTrange)

    Results_updown = struct();
    arrayNum = numel(Array_Allocation);
    
    for arrayi = 1:arrayNum 

        FieldName = Array_Allocation(arrayi);
    
        upSum_Results1 = zeros(64, 1); downSum_Results1=zeros(64, 1); 
        upSum_Results2 =zeros(64, 1); downSum_Results2=zeros(64, 1); 
    
        for taski = 1:numel(selectTaskInd)
            current_selectTaskNum = selectTaskInd(taski);
            sub1D = Results.(FieldName){current_selectTaskNum, 1};
            sub2D = Results.(FieldName){current_selectTaskNum, 2};
    
            timeVec = sub1D.time;

            analizedTimeInd = find(timeVec >= analizedTrange(1) & timeVec <= analizedTrange(2));
    

            % subarray1
            [sub1up, sub1down] = create_up_down_both_results(sub1D, analizedTimeInd);
            upSum_Results1 = upSum_Results1 + sub1up;
            downSum_Results1 = downSum_Results1 + sub1down;
    
            % subarray2
            [sub2up, sub2down] = create_up_down_both_results(sub2D, analizedTimeInd);
            upSum_Results2 = upSum_Results2 + sub2up;
            downSum_Results2 = downSum_Results2 + sub2down;
        end
        

        Results_updown.(FieldName).up{1,1} = upSum_Results1;
        Results_updown.(FieldName).up{1,2} = upSum_Results2;
        Results_updown.(FieldName).down{1,1} = downSum_Results1;
        Results_updown.(FieldName).down{1,2} = downSum_Results2;
    
    end

end


function my_up_or_down_ch_plot(plot_select, circleScale, array_select, Array_Allocation, Results_updown, total_subarray_ind, Array_180turn, Array_yaxis)

    co = colororder("gem");

    for arrayi = array_select
        FieldName = Array_Allocation{arrayi};
    
        for subi = 1:2 
    
            currentUpChdata = Results_updown.(FieldName).up{1,subi};
            currentDownChdata = Results_updown.(FieldName).down{1,subi};
            both_ind = currentUpChdata > 0 & currentDownChdata > 0; 
    

            [used_ch_grid] = get_used_ch_grid(Array_180turn{arrayi, subi}, subi);

            current_yaxis_last = Array_yaxis{arrayi, subi}(end);
            

            if sum(plot_select) == 2

                [upScatterX, upScatterY, upCirSz] = get_subarray_col_and_row(current_yaxis_last, currentUpChdata.*~both_ind, total_subarray_ind{subi}, used_ch_grid);
                
                hold on
                scatter(upScatterX, upScatterY, upCirSz*circleScale, co(2, :), 'filled')
                
                [downScatterX, downScatterY, downCirSz] = get_subarray_col_and_row(current_yaxis_last, currentDownChdata.*~both_ind, total_subarray_ind{subi}, used_ch_grid);
                scatter(downScatterX,downScatterY, downCirSz*circleScale, co(1, :), 'filled')
                
                [bothScatterX, bothScatterY, bothCirSz] = get_subarray_col_and_row(current_yaxis_last, (currentUpChdata + currentDownChdata).*both_ind, ...
                    total_subarray_ind{subi}, used_ch_grid);
                scatter(bothScatterX, bothScatterY, bothCirSz*circleScale, co(5, :), 'filled')
            elseif plot_select(1) == 1 
                [upScatterX, upScatterY, upCirSz] = get_subarray_col_and_row(current_yaxis_last, currentUpChdata, total_subarray_ind{subi}, used_ch_grid);        
                hold on
                scatter(upScatterX, upScatterY, upCirSz*circleScale, co(2, :), 'filled')
            elseif plot_select(2) == 1        
                [downScatterX, downScatterY, downCirSz] = get_subarray_col_and_row(current_yaxis_last, currentDownChdata, total_subarray_ind{subi}, used_ch_grid);
                hold on
                scatter(downScatterX,downScatterY, downCirSz*circleScale, co(1, :), 'filled')
            end

        end
    
    end

end


function [upSum, downSum] = create_up_down_both_results(sub1D, analizedTimeInd)
    

    sub1dMask = sub1D.mask(:, analizedTimeInd);
    sub1dstat = sub1D.stat(:, analizedTimeInd);
    
    upmatrix = sub1dMask & (sub1dstat > 0);    
    downmatrix = sub1dMask & (sub1dstat < 0);  

    upSum = sum(upmatrix, 2);
    downSum = sum(downmatrix, 2);
    
end


function [used_ch_grid] = get_used_ch_grid(ture_or_false_180, subarrayNum)
    % subarray1
    ch_grid1 = [
     2   17   23   31   97  105  112  128;
     1   15   25   18   99  106  115  114;
     3    8   21   22   98  107  117  116;
     4   11   14   24  100  108  119  118;
     7   13   16   26  101  109  121  120;
     5   19   20   28  102  110  123  122;
     6   10   27   30  103  111  125  124;
     9   12   29   32  104  113  127  126
    ];
    
    % 180°-rotated version of Subarray 1
    ch_grid1_180 = [
      126   127   113   104   32   29   12   9;
      124   125   111   103   30   27   10   6;
      122   123   110   102   28   20   19   5;
      120   121   109   101   26   16   13   7;
      118   119   108   100   24   14   11   4;
      116   117   107   98    22   21   8    3;
      114   115   106   99    18   25   15   1;
      128   112   105   97    31   23   17   2
    ];
    
    % subarray2
    ch_grid2 = [
        65  66  81  89  33  38  51  58;
        67  68  82  90  35  40  53  57;
        69  70  84  91  37  43  50  62;
        71  72  83  92  39  47  46  60;
        73  74  86  93  41  49  56  59;
        75  76  85  94  45  48  55  64;
        77  78  87  96  34  42  54  61;
        79  80  88  95  36  44  52  63
    ];
    
    % 180°-rotated version of Subarray 2
    ch_grid2_180 = [
        63  52  44  36  95  88  80  79;
        61  54  42  34  96  87  78  77;
        64  55  48  45  94  85  76  75;
        59  56  49  41  93  86  74  73;
        60  46  47  39  92  83  72  71;
        62  50  43  37  91  84  70  69;
        57  53  40  35  90  82  68  67;
        58  51  38  33  89  81  66  65
    ];

    if ture_or_false_180 
        if subarrayNum == 1
            used_ch_grid = ch_grid1_180;
        elseif subarrayNum == 2
            used_ch_grid = ch_grid2_180;
        end
    else 
        if subarrayNum == 1
            used_ch_grid = ch_grid1;
        elseif subarrayNum == 2
            used_ch_grid = ch_grid2;
        end

    end


end

function display_my_six_subarray()
    [X, Y] = meshgrid(1:8, 1:55);
    x_oneVec = X(:);
    y_oneVec = Y(:);
    Ind_remove = y_oneVec == 9 | y_oneVec == 18 | y_oneVec == 19 ...
    | y_oneVec == 28 | y_oneVec == 37 | y_oneVec == 38 | y_oneVec == 47;

    scatter(x_oneVec(~Ind_remove), y_oneVec(~Ind_remove), 1, 'filled', 'k')
    box off
    xlim([0 9])
    ylim([0 56])

    ax = gca;
    ax.XAxis.Visible = 'off';
    yticks([9, 28, 48])
    yticklabels({'v6d', 'd6d', 'r6d'})
    ax.YAxis.TickLength = [0 0];
    
end


function [ScatterX, ScatterY, ScatterCircleSize] = get_subarray_col_and_row(current_yaxis_last, Sum_Results, subarray_ind, used_ch_grid)
    ScatterX = [];
    ScatterY = [];

    for i = 1:64
        if Sum_Results(i, 1) > 0
            get_ch_num = subarray_ind(i); 
            [row, col] = find(used_ch_grid == get_ch_num);
            ScatterX(end+1) = col;
            ScatterY(end+1) = current_yaxis_last - row + 1;
        end
    
    end

    ScatterCircleSize = Sum_Results(Sum_Results > 0);
end

function [psth_d, ci_Data] = make_psth_data(Input4dData, SelectTask, ChNum, myOrder)
    taskInd = (myOrder == SelectTask);
    clippedData = squeeze(Input4dData(taskInd, :, :, ChNum));
    [psth_d, ~, ci_Data] = normfit(clippedData);
end

function my_sigpoint_scatter(TargetChNum, y_values, Color_val, Up_or_down, Results, selectTaskInd, Array_Allocation, Array_ChNum, total_subarray_ind)

    [arrayName, subarrayIndex, subarrayChNum] = identify_channel_location(TargetChNum, Array_Allocation, Array_ChNum, total_subarray_ind);
    maskInd = Results.(arrayName){selectTaskInd, subarrayIndex}.mask(subarrayChNum, :);
    if Up_or_down
        statInd = Results.(arrayName){selectTaskInd, subarrayIndex}.stat(subarrayChNum, :) > 0;
    elseif ~Up_or_down
        statInd = Results.(arrayName){selectTaskInd, subarrayIndex}.stat(subarrayChNum, :) < 0;
    end
    maskIndTime = Results.(arrayName){selectTaskInd, subarrayIndex}.time;
    TrueMaskInd = maskInd & statInd ;
    maskIndY = repmat(y_values, 1, sum(TrueMaskInd));
    scatter(maskIndTime(TrueMaskInd),  maskIndY , 10, 'filled', 'MarkerFaceColor', Color_val)

end

function [arrayName, subarrayIndex, subarrayChNum] = identify_channel_location(TargetChNum, Array_Allocation, Array_ChNum, total_subarray_ind)

    arrayName = '';
    subarrayIndex = [];
    subarrayChNum = [];

    for arrayi = 1:numel(Array_Allocation)
        ch_list = Array_ChNum{arrayi};
        if ismember(TargetChNum, ch_list)
            arrayName = Array_Allocation(arrayi);
            local_ch = TargetChNum - ch_list(1) + 1;

            for subi = 1:2
                if ismember(local_ch, total_subarray_ind{subi})
                    subarrayIndex = subi;

                    sub_idx_list = total_subarray_ind{subi};
                    subarrayChNum = find(sub_idx_list == local_ch);
                    return;
                end
            end
        end
    end
end


function [upData, downData, cuTime, upData_tstat, downData_tstat] = total_sig_ch_extraction_eachArray(Results, fieldName, taski)

    % subarray1
    cuMask = Results.(fieldName){taski, 1}.mask;
    cuStat = Results.(fieldName){taski, 1}.stat;
    cuTime = Results.(fieldName){taski, 1}.time;


    cuStat_clean = cuStat;
    cuStat_clean(isnan(cuStat_clean)) = 0;

    upData1 = cuMask.*(cuStat_clean > 0); 
    upData1_tstat = cuStat_clean.*(logical(upData1));
    downData1 = cuMask.*(cuStat_clean < 0);
    downData1_tstat = cuStat_clean.*(logical(downData1));
    
    % subarray2
    cuMask = Results.(fieldName){taski, 2}.mask;
    cuStat = Results.(fieldName){taski, 2}.stat;


    cuStat_clean = cuStat;
    cuStat_clean(isnan(cuStat_clean)) = 0;
    
    upData2 = cuMask.*(cuStat_clean > 0);
    upData2_tstat = cuStat_clean.*(logical(upData2));
    downData2 = cuMask.*(cuStat_clean < 0);
    downData2_tstat = cuStat_clean.*(logical(downData2));

    upData = cat(1, upData1, upData2);
    upData_tstat = cat(1, upData1_tstat, upData2_tstat);
    downData = cat(1, downData1, downData2);
    downData_tstat = cat(1, downData1_tstat, downData2_tstat);


end


function [mean_tnorm_up, mean_tnorm_up_ci, mean_t_up, mean_t_up_ci] = mean_t_and_normalized_t_get(totalUp, totalUp_tstat)

    prop_d = mean(totalUp, 1); 
    mean_t_up = zeros(size(totalUp_tstat, 2), 1);
    mean_t_up_ci = zeros(size(totalUp_tstat, 2), 2);
    mean_tnorm_up = zeros(size(totalUp_tstat, 2), 1);
    mean_tnorm_up_ci = zeros(size(totalUp_tstat, 2), 2);

    for timei = 1:size(totalUp_tstat, 2)
        current_t_data = totalUp_tstat(:, timei);
        
        validInd = current_t_data ~= 0;
        if ~any(validInd) 
            continue
        end

        validT = current_t_data(validInd);
        [h, ~, ci] = normfit(validT);
        mean_t_up(timei) = h;
        mean_t_up_ci(timei, :) = ci;

        
        validT_norm = validT * prop_d(timei);
        [h, ~, ci] = normfit(validT_norm);
        mean_tnorm_up(timei) = h;
        mean_tnorm_up_ci(timei, :) = ci;
    end
end


function [outputsignal] = make_signal_smooth_with_G(dt, sigma, inputsignal)


    sigmaBins = round(sigma/dt);    
    x = -4*sigmaBins:4*sigmaBins;
    g = exp(-(x.^2)/(2*sigmaBins^2));
    g = g / sum(g); 

    L  = floor(numel(g)/2);
    pp = padarray(inputsignal, [0 L], 'symmetric', 'both');  
    tmp = conv(pp, g, 'same');
    outputsignal = tmp(L+1:end-L);                     

end


function [threeClassInd] = my_updown_location_make(fieldName, Results_updown)

    up1data = Results_updown.(fieldName).up{1,1};
    up2data = Results_updown.(fieldName).up{1,2};
    totalUpD = [up1data; up2data];
    
    down1data = Results_updown.(fieldName).down{1,1};
    down2data = Results_updown.(fieldName).down{1,2};
    totalDownD = [down1data; down2data];
    
    threeClassInd = zeros(size(totalUpD, 1), 1);
    upInd = totalUpD > 0;
    downInd = totalDownD > 0;
    bothInd = upInd & downInd ;
    threeClassInd(upInd & ~bothInd) = 1; 
    threeClassInd(downInd & ~bothInd) = 2; 
    threeClassInd(bothInd) = 3; 


end

function [hammingResults] = get_hamming_dist(timei, arrayName, Results_updown_location, taskNum)

    hammingResults = nan(taskNum, taskNum);
    for taski = 1:taskNum
        currentRowData = Results_updown_location.(arrayName){timei, taski};
    
        for taskl = 1:taskNum
            currentColData = Results_updown_location.(arrayName){timei, taskl};
    

            nonzero_mask = (currentRowData ~= 0) | (currentColData ~= 0);
    
            if sum(nonzero_mask) == 0
                
                hamming_dist = 0;
            else
                % masked Hamming distance
                diff_count = sum(currentRowData(nonzero_mask) ~= currentColData(nonzero_mask));
                hamming_dist = diff_count / sum(nonzero_mask);
            end
    
            hammingResults(taski, taskl) = hamming_dist;
        end
    
    end

end


function my_dendrogram_plot(distance_matrix, myOrder, dendroLabelL, Color_val, FontSizeNum)


    dist_vec = squareform(distance_matrix);  

    Z = linkage(dist_vec, 'ward');

    [H, ~, outperm] = dendrogram(Z, 0, 'Labels', myOrder, 'Orientation', 'top');

    for i = 1:length(H)
        set(H(i), 'Color', 'k')
        set(H(i), 'LineWidth', 1.5)
    end
    ax = gca;
    ax.XTickLabel = [];
    ax.LineWidth = 1.5;
    ax.FontSize = 12; ax.FontName = 'Helvetica';
    xticks = ax.XTick;
    ylimVal = ax.YLim;

    for i = 1:numel(outperm)
        idx = outperm(i);
        txt = myOrder(idx);
        color = [];
        for j = 1:numel(dendroLabelL)
            currentL = dendroLabelL{j};
            currentInd =find(ismember(myOrder, currentL));
            if ismember(idx, currentInd)
                color = Color_val(j, :);
                continue
            end    
        end

        if isempty(color)
            color = [0 0 0];
        end
        text(xticks(i), ylimVal(1) - 0.01 * diff(ylimVal), txt, ...
        'Rotation', 45, 'HorizontalAlignment', 'right', ...
        'FontSize', FontSizeNum, 'FontName', 'Helevetica', 'Color', color);
    end



end