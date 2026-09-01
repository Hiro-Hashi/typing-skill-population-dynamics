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

% This script reproduces the dPCA analyses shown for the Right 6d and
% Left 6d populations.

%% User settings
% Select the population to analyze. Change only this variable to reproduce
% the corresponding Right 6d or Left 6d results.
%   "r6d" = Right 6d (channels 257-384)
%   "l6d" = Left 6d  (combined v6d + d6d; channels 1-256)
usedArray = "r6d";   % Choose "r6d" or "l6d"

numComponents = 20;      % Number of dPCA components to compute
numLambdaRep = 5;        % Cross-validation repetitions for lambda optimization


% "Demixed principal component analysis of neural population data"
% D. Kobak, W. Brendel, C. Constantinidis, C. E. Feierstein, A. Kepecs, Z. F. Mainen, et al.
% eLife 2016 Vol. 5 Pages e10989 
% DOI: 10.7554/eLife.10989
% dPCA MATLAB code: https://github.com/machenslab/dPCA

% Add dPCA to the MATLAB path
addpath(projectDir + "/external/dPCA");

% Load the finger sweep dataset used for Figure 1.
dataFile = fullfile(projectDir, '..', ...
    'Fig1', 'data', 't18FingerSweepData.mat');
segD = load(dataFile);

% Generate finger movement labels.
[OriginalLabel, convertedLabels, myOrder, ~] = my_label_converter();



%% Prepare firing-rate data around the Go cue
TXgo_segmentedD = {segD.TXpre, segD.TXgo}; 
% Number of bins before and after the Go cue
minDelayDur = 150; 
minGoDur = 240;

% Time axis parameters
bin_sec = 0.01;  % Duration of one bin (10 ms)
S_time = -minDelayDur * bin_sec;
E_time =  minGoDur * bin_sec;

% Build a 4-D array [task x trial x time x channel] and reorder tasks according to myOrder.
[inputFeaturesTXgo, ~, ~, ~] = Four_D_data_make( ...
    TXgo_segmentedD, minDelayDur, minGoDur, segD, myOrder, convertedLabels, OriginalLabel);

% Convert threshold-crossing counts (ncTX) to firing rates.
sigma_bin = 5; % Gaussian kernel SD (5 bins = 50 ms), following Kobak et al. (2016)
% Compute firing rates after segmenting the ncTX data.
inputFeaturesFR_go = my_FR_make(sigma_bin, inputFeaturesTXgo, bin_sec);

% Estimate channel-wise baseline mean and SD from the None condition.
nonInd = find(myOrder == "None");

baseData = squeeze(inputFeaturesFR_go(nonInd, :, :, :));
[baselineTrialNum, ~, ~] = size(baseData);
allData = [];
for triali = 1:baselineTrialNum
    currentData = squeeze(baseData(triali, :, :));
    allData = cat(1, allData, currentData);
end

aveD = mean(allData, 1);
stdD = std(allData, 1);
stdD = max(stdD, eps);  

% Z-score firing rates using the None-condition baseline statistics.
[taskNum, trialNum, dataNum, chNum] = size(inputFeaturesFR_go);
inputFeaturesFR_go_z = nan(taskNum-1, trialNum, dataNum, chNum); % Exclude the None condition
k = 1;
for taski = 1:taskNum
    if taski == nonInd
        continue
    end

    for triali = 1:trialNum
        currentData = squeeze(inputFeaturesFR_go(taski, triali, :, :));
        z_currentData = (currentData - aveD)./stdD;
        inputFeaturesFR_go_z(k, triali, :, :) = z_currentData;
    end

    k = k + 1;
end

% Movement-condition labels after excluding the None condition.
validInd = (1:taskNum ~= nonInd);
FingerLabel = myOrder(validInd);


%% Run dPCA without regularization

ArrayList = ["All", ...
            "v6d1", "v6d2", "v6d", ...
            "d6d1", "d6d2", "d6d", ...
            "r6d1", "r6d2", "r6d", "l6d"];
ChList = [{1:384}, ...
          {1:64}, {65:128}, {1:128}, ...
          {129:192}, {193:256}, {129:256}, ...
          {257:320}, {321:384}, {257:384}, {1:256}];

if ~ismember(usedArray, ["r6d", "l6d"])
    error('usedArray must be either "r6d" or "l6d".');
end
usedCh = ChList{ArrayList == usedArray};


%%%% Step 1. Prepare the dPCA input (z-scored firing rates) 

% Input: inputFeaturesFR_go_z [task x trial x time x channel]
% Tasks 1-15 are right-hand movements; tasks 16-30 are left-hand movements.
% Output: firingRates [channel x hand x finger x gesture x time x trial]

% Use z-scored firing rates from the 30 movement conditions.
data = inputFeaturesFR_go_z(:, :, :, usedCh);                 % [task x trial x time x channel]
% data = inputFeaturesFR_go(validInd, :, :, usedCh);   % Alternative: use non-z-scored firing rates
[taskNum, trialNumMax, T, N] = size(data);

% Map each task to hand, finger, and gesture factors.
% ---- Task index -> (hand, finger, gesture) ----
if taskNum ~= 30
    error('Expected 30 movement conditions after excluding the None condition.');
end
task = (1:taskNum).';
hand = ones(taskNum, 1);  
hand(task>=16)=2;                     % 1:right, 2:left
within  = task - (hand - 1) * 15;                                     % index within each hand (1-15)
finger  = ceil(within/3);                                         % 1:thumb .. 5:pinky
gesture = mod(within-1,3)+1;                                      % 1:Up, 2:Down, 3:In

H=2; F=5; G=3;

% Maximum number of trials across conditions.
Emax = trialNumMax;

%%%% Step 2. Rearrange data into the tensor format required by dPCA

% Use NaN padding so unequal trial counts can be accommodated.
% Rearrange [task x trial x time x channel] to [channel x hand x finger x gesture x time x trial].
firingRates = nan(N, H, F, G, T, Emax);

for t = 1:taskNum

    h = hand(t); 
    f = finger(t); 
    g = gesture(t);

    trials_t = squeeze(data(t,:,:,:));                 % [trial x time x channel]

    if ndims(trials_t)==2 % Preserve the trial dimension if only one trial is present
        trials_t = reshape(trials_t, [1, size(trials_t)]); 
    end
    Et = size(trials_t,1); % number of trials for this condition

    % [trial x time x channel] -> [channel x time x trial]
    trials_cte = permute(trials_t, [3, 2, 1]);   % [channel x time x trial]

    % Store trials in the dPCA tensor.
    firingRates(:,h,f,g,:,1:Et) = trials_cte;
end

%%%% Step 3. Compute condition-averaged firing rates (PSTHs)
firingRatesAverage = mean(firingRates, 6, "omitnan");          % [channel x hand x finger x gesture x time]

% Number of valid trials for each condition [channel x hand x finger x gesture].
trialNum = zeros(N,H,F,G);
for h=1:H
    for f=1:F 
        for g=1:G
            x = squeeze(firingRates(1,h,f,g,1,:));             % Trial availability is identical across channels
            trialNum(:,h,f,g) = sum(~isnan(x));
        end
    end
end

%%%% Step 4. Define dPCA marginalizations
% Each main effect is combined with its interaction with time.
% Parameter indices: 1=Hand, 2=Finger, 3=Gesture, 4=Time.
combinedParams = { ...
    {1, [1 4]}, ...   % Hand and Hand x Time
    {2, [2 4]}, ...   % Finger and Finger x Time
    {3, [3 4]}, ...   % Gesture and Gesture x Time
    {4}, ...          % Time (condition-independent)
    {[1 2],[1 3],[2 3],[1 2 3],[1 2 4],[1 3 4],[2 3 4],[1 2 3 4]} % remaining interactions
};
margNames = {'Hand','Finger','Gesture','Common','Interactions'};
color_value = colororder("gem");
margColours = color_value(1:5, :);

% Time axis
time = linspace(S_time, E_time, T);    
timeEvents = 0;           % Go presentation = 0s

%%%% Step 5. Run dPCA without regularization
[W,V,whichMarg] = dpca(firingRatesAverage, numComponents, 'combinedParams', combinedParams);
explVar = dpca_explainedVariance(firingRatesAverage, W, V, 'combinedParams', combinedParams);

dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, 'whichMarg', whichMarg, ...
    'marginalizationNames', margNames, 'marginalizationColours', margColours, ...
    'time', time, 'timeEvents', timeEvents, 'timeMarginalization', 4, ...
    'legendSubplot', 16);

%% Run regularized dPCA
%%%% Step 6. Optimize regularization and run dPCA
ifSimultaneousRecording = true; % true because all channels were recorded simultaneously
outputDir = fullfile(projectDir, "results");
if ~exist(outputDir, "dir")
    mkdir(outputDir);
end
saveFileName = char(fullfile(outputDir, "optLambda_FingerSweep_" + usedArray + ".mat"));
optimalLambda = dpca_optimizeLambda(firingRatesAverage, firingRates, trialNum, ...
    'combinedParams', combinedParams, 'simultaneous', ifSimultaneousRecording, ...
    'numRep', numLambdaRep, 'filename', saveFileName); % Cache the optimized lambda to disk.
Cnoise = dpca_getNoiseCovariance(firingRatesAverage, firingRates, trialNum, ...
    'simultaneous', ifSimultaneousRecording);

[Wr,Vr,whichMarg_r] = dpca(firingRatesAverage, numComponents, ...
    'combinedParams', combinedParams, 'lambda', optimalLambda, 'Cnoise', Cnoise);
explVar_r = dpca_explainedVariance(firingRatesAverage, Wr, Vr, ...
    'combinedParams', combinedParams);

dpca_plot(firingRatesAverage, Wr, Vr, @dpca_plot_default, ...
    'explainedVar', explVar_r, 'whichMarg', whichMarg_r, ...
    'marginalizationNames', margNames, 'marginalizationColours', margColours, ...
    'time', time, 'timeEvents', timeEvents, 'timeMarginalization', 4, ...
    'legendSubplot', 16);


%% Fig. 4: Panels A and C for right 6d; Panels B and D for left 6d

% Colors used for visualization.
co = colororder("gem12");
% margNames = dP.margNames;
figure
t = tiledlayout(2, 2, 'TileSpacing','compact','Padding','compact');

%%%%%%%% AX1
nexttile(t, 1);
% Cumulative variance explained by PCA and dPCA components.
pcaD = explVar_r.cumulativePCA;
dpcaD = explVar_r.cumulativeDPCA;
dataNum = numel(pcaD);

plot(1:dataNum, pcaD, 'o-', 'LineWidth',1, 'Color',co(1, :))
hold on
plot(1:dataNum, dpcaD, 'o-', 'LineWidth',1, 'Color',co(2, :))
legend({'PCA', 'dPCA'}, 'Box', 'off', 'AutoUpdate','off', 'Location', 'best', 'FontSize', 12)
xlabel('Component')
ylabel('Explained variance (%)')
box off
yline(75, '--', 'LineWidth',1, 'Color', [0.5 0.5 0.5])
ylim([0 100]), yticks([0 25 50 75 100])
xlim([0, dataNum+1]), xticks([1 5 10 15 20])
title(usedArray + " array: cumulative explained variance", 'FontSize', 14)

%%%%%%%% AX2: Marginalized variance for each component
nexttile(t, 3);
margVar = (explVar_r.margVar)';

B = bar(margVar, 'stacked', 'FaceColor','flat', 'FaceAlpha',0.6);
for i = 1:size(margVar, 2)
    B(i).CData = co(i, :);
end
box off
legend(margNames, 'Box', 'off', 'AutoUpdate','off', 'Location', 'best', 'FontSize',12);
xlim([0, dataNum+1]), xticks([1 5 10 15 20]), xlabel('Component')
ylabel('Component variance (%)')
title(usedArray + " array: variance per component", 'FontSize', 14)


%%%%%%%% AX3: Total variance by marginalization
nexttile(t, 4);

vals = explVar_r.totalMarginalizedVar(:);
vals(vals < 0) = 0;                         % Clamp small negative numerical values to zero.
pct  = 100 * vals / sum(vals);              % Convert to percentages.

tbl = table(margNames', pct, 'VariableNames',{'Name', 'Per'});
% Pie-chart visualization.
piechart(tbl, "Per", "Name", ExplodedWedges=[1 2 3 4 5]);
colororder("gem")
title(usedArray + " array dPCA variance breakdown")

%% Fig. 4: Panels E for right 6d; Panels F for left 6d

figure

% Components shown in the manuscript differ between Right 6d and Left 6d.
% The appropriate set is selected automatically from usedArray.
switch usedArray
    case "r6d"
        ComponentList = [1 5 3 2];
    case "l6d"
        ComponentList = [4 5 1 3];
end
GroupList = ["Hand", "Finger", "Gesture", "Allf"];

tiledlayout(1, numel(ComponentList), "TileSpacing","loose", "Padding","loose")

% Generate task groups used for coloring.
[R_Thumb, R_Index, R_Middle, R_Ring, R_Pinky, L_Thumb, L_Index, L_Middle, L_Ring, L_Pinky, ...
    Right_F, Left_F, Up_F, Down_F, In_F] = make_group_data();


GroupS = struct();
% Define condition groups used to color dPCA projections.
% Right hand vs. left hand
GroupS.("Hand").componentGroup = {Right_F, Left_F}; 
GroupS.("Hand").legendName = {'Right Hand', 'Left Hand'};
GroupS.("Hand").LabelName = "R/L Hand";
% Finger identity
GroupS.("Finger").componentGroup = {[R_Thumb, L_Thumb], [R_Index, L_Index], [R_Middle, L_Middle], [R_Ring, L_Ring], [R_Pinky, L_Pinky]}; 
GroupS.("Finger").legendName = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'};
GroupS.("Finger").LabelName = "Each finger";
% Movement gesture
GroupS.("Gesture").componentGroup = {Up_F, Down_F, In_F}; 
GroupS.("Gesture").legendName = {'Up', 'Down', 'In'};
GroupS.("Gesture").LabelName = "UpDwIn";
% Individual fingers from both hands
GroupS.("Allf").componentGroup = {R_Thumb, R_Index, R_Middle, R_Ring, R_Pinky, L_Thumb, L_Index, L_Middle, L_Ring, L_Pinky}; 
GroupS.("Allf").legendName = {'R Thumb', 'R Index', 'R Middle', 'R Ring', 'R Pinky', 'L Thumb', 'L Index', 'L Middle', 'L Ring', 'L Pinky'};
GroupS.("Allf").LabelName = "All finger";

taskNum = numel(FingerLabel);
timeD = time;

% Flatten the condition-average tensor back to [task x time x channel].
[flattenPSTH] = my_flattenPSTH_make(firingRatesAverage, taskNum);

for i = 1:numel(ComponentList)

    selectComponent = ComponentList(i);

    nexttile(i);
  
    % Project all condition averages onto the selected dPC.
    [allProj] = make_all_project_data(selectComponent, Wr, flattenPSTH);
    
    % Plot condition trajectories.
    currentGroup = GroupS.(GroupList(i)).componentGroup;
    lNum = numel(currentGroup);
    for gi = 1:lNum
        currentInd = ismember(FingerLabel, currentGroup{gi});
        plot(timeD, allProj(:, currentInd), '-', 'LineWidth',1.5, 'Color',co(gi, :))
        hold on
    end
    
    h_color = gobjects(lNum, 1);
    for gi = 1:lNum
        h_color(gi) = plot(nan, nan, '-', 'LineWidth',1.5, 'Color', co(gi, :));
    end
    L = legend(h_color, GroupS.(GroupList(i)).legendName , 'Location','best', 'Box', 'off', 'FontSize',10, 'AutoUpdate','off');
    
    xlabel('Time (s) from Go (t=0)')
    ylabel('Normalized firing rate (Hz)', 'FontSize',12)
    box off
    xline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1)
    yline(0, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1)

    current_var = sum(explVar_r.margVar(:, selectComponent));
    current_main_name = margNames{whichMarg_r(selectComponent)};
    titleName = sprintf('Component #%d [%.1f%%], %s', selectComponent, current_var, current_main_name);
    title(usedArray + " array: " + string(titleName), 'FontSize',12)
    xlim([timeD(1), timeD(end)])
end

%% Fig. 4: Panels G for right 6d; Panels H for left 6d
% Variables used for latent-space visualization.
whichMargPlot = whichMarg_r;
W = Wr; 
timeD = time; 

taskNum = numel(FingerLabel); dataNum = size(flattenPSTH, 2);

figure
t = tiledlayout(1, 4, 'TileSpacing','compact', 'Padding','compact');

for mi = 1:4
        
    current_Marg = string(margNames{mi});
    current_MargInd = find(string(margNames) == current_Marg);
    % Use up to the first three components assigned to this marginalization.
    upper_ind = find(whichMargPlot == current_MargInd, 3);
    
    latentData = nan(taskNum, dataNum, 3); % Up to three dimensions are visualized
    for i = 1:numel(upper_ind)
        currentInd = upper_ind(i);
        allProj = make_all_project_data(currentInd, W, flattenPSTH);
        latentData(:, :, i) = allProj';
    end
        
    % Retrieve grouping information for plotting.
    try
        currentGroup = GroupS.(current_Marg).componentGroup;
        legendName = GroupS.(current_Marg).legendName;
    catch
        currentGroup = GroupS.("Allf").componentGroup;
        legendName = GroupS.("Allf").legendName;
    end
    
    % Visualize latent trajectories.
    nexttile(t, mi);
    
    for gi = 1:numel(currentGroup)
    
        current_subGroup = currentGroup{gi};
        ind = ismember(FingerLabel, current_subGroup);
    
        x = (squeeze(latentData(ind, :, 1)))';
        y = (squeeze(latentData(ind, :, 2)))';
        z = (squeeze(latentData(ind, :, 3)))';
        if ~any(isnan(z), 'all')
            plot3(x, y, z, 'Color',co(gi, :), 'LineWidth',2)
            hold on
            scatter3(x(1, :), y(1, :), z(1, :), 50, 'k', 'o', 'filled')
            scatter3(x(end, :), y(end, :), z(end, :), 50, 'k', '+')
        elseif ~any(isnan(y), 'all')
            plot(x, y, 'Color',co(gi, :), 'LineWidth',2)
            hold on
            scatter(x(1, :), y(1, :), 50, 'k', 'o', 'filled')
            scatter(x(end, :), y(end, :), 50, 'k', '+')
        elseif ~any(isnan(x), 'all')
            plot(timeD, x, 'Color',co(gi, :), 'LineWidth',2)
            hold on
        else
            warning('No valid component data are available for this marginalization.')
        end

    end
    box off, grid on
    
    % Legend.
    h_color = gobjects(numel(currentGroup) + 2, 1);
    for i = 1:numel(currentGroup)
        h_color(i) = plot(nan, nan, 'Color', co(i,:), 'LineWidth', 2);
    end
    h_color(numel(currentGroup) + 1) = scatter(nan, nan, 50, 'k', 'o', 'filled');
    h_color(numel(currentGroup) + 2) = scatter(nan, nan, 50, 'k', '+');
    
    legend(h_color, {legendName{:}, 'Start', 'End'}, 'Box','off','AutoUpdate','off', 'Location', 'best', 'FontSize', 10)
    
    if ~any(isnan(z), 'all')
        xlabel('dPC1'), ylabel('dPC2'), zlabel('dPC3'), view(3)
    elseif ~any(isnan(y), 'all')
        xlabel('dPC1'), ylabel('dPC2')
    elseif ~any(isnan(x), 'all')
        xlabel('Time (s)'), ylabel('dPC1'), xline(0, '-', 'LineWidth', 0.5, 'Color',[0.5 0.5 0.5])
        xlim([timeD(1), timeD(end)])
    end
    
    title(usedArray + " array, " + current_Marg)
    
end


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

function [inputFeaturesTX, TX_DeGo2, ErrorTrial_tx, Transcriptions] = Four_D_data_make(TX_segmentedD, minDelayDur, minGoDur, segD, myOrder, convertedLabels, OriginalLabel)

    [TX_DeGo, ErrorTrial_tx] = pre_post_Cue_data_cat(TX_segmentedD, minDelayDur, minGoDur);
    TX_DeGo2 = TX_DeGo;
    TX_DeGo2(ErrorTrial_tx) = [];

    Transcriptions = segD.transcriptions;
    Transcriptions(ErrorTrial_tx) = [];

    minimunTrial = min(countcats(categorical(Transcriptions)));
    taskNum = numel(countcats(categorical(Transcriptions)));
    if  ~numel(myOrder) == taskNum
        error('The number of tasks in "Transcription" does not match the number of tasks in "myOrder.')
    end
    [dataNum, Chnum] = size(TX_DeGo2{1});

    inputFeaturesTX = create_fourD_data(TX_DeGo2, Transcriptions, taskNum, minimunTrial, dataNum, Chnum, myOrder, convertedLabels, OriginalLabel);


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

    

    t = -round(3*sigma_bin):round(3*sigma_bin);

    gaussKernel = exp(-(t.^2)/(2*sigma_bin^2));
    gaussKernel = gaussKernel / sum(gaussKernel);

    khalf = floor(length(gaussKernel)/2);


    leftPad  = sigD(khalf+1:-1:2);

    rightPad = sigD(end-1:-1:end-khalf);

    sigPadded = [leftPad; sigD; rightPad];


    tmp = conv(sigPadded, gaussKernel, 'same');

    startIdx = khalf+1;
    endIdx   = startIdx + length(sigD) - 1;
    fr_hz = tmp(startIdx:endIdx) / bin_sec;
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

function [flattenPSTH] = my_flattenPSTH_make(inputData, taskNum)

    [chNum, hnum, fnum, gnum, dataNum] = size(inputData);
    taski = 1;
    flattenPSTH = nan(taskNum, dataNum, chNum);
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