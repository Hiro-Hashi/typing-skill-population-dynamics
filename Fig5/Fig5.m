%% Fig.5
% Setup
% See README.md for installation instructions and software requirements.
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
    datapath = fullfile(projectDir, 'data', char("day" + DaysList(i) + ".mat"));
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

% Keyboard labels corresponding to the Finger Sweep movement labels.
KeyboardLabel = ["Y",  "H",  "Sp", ...
    "U",  "J",  "N", ...
    "I",  "K",  "M", ...
    "O",  "L",  ",", ...
    "P",  "?",  ".", ...
    "T",  "G",  "B", ...
    "R",  "F",  "V", ...
    "E",  "D",  "C", ...
    "W",  "S",  "X", ...
    "Q",  "A",  "Z"];

%% Calculate Mahalanobis distance
% Array names and corresponding channel indices
ArrayList = ["v6d", "d6d", "r6d", "l6d", "All"];
ChList = [{1:128}, {129:256}, {257:384}, {1:256}, {1:384}];

num_task = numel(FingerLabel);

All_MD = struct();
% Loop over sessions

for di = 1:numel(DayList2)
    currentDay = DayList2(di);

    % Loop over arrays
    for ai = 1:numel(ArrayList)
        useArray = ArrayList(ai);
        analyzed_ch = ChList{ArrayList == useArray};
        ChNum = numel(analyzed_ch);

        % Compute class-wise means and covariance matrices for Mahalanobis distance; exclude tokens with fewer than 10 trials.
        [means, covs] = calculate_means_covs_for_MahaDis(analyzed_ch, num_task, ClipD, "refined_Warping_zFR", currentDay, 10);
        % Compute Mahalanobis distance
        [distance_matrix] = my_MahalanobisDis(means, covs, num_task);

        % Normalize Mahalanobis distance by the square root of the number of channels.
        distance_matrix_N = distance_matrix./sqrt(ChNum);

        All_MD.(currentDay).(useArray) = distance_matrix_N;
    end

end


outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

saveName = "normalized Mahalanobis dis results.mat";
save(fullfile(outputDir, char(saveName)), 'All_MD');
%% Fig.5A

% Display order
RowList = ["r6d", "l6d"];

num_task = numel(FingerLabel);

figure
tiledlayout(numel(RowList), numel(DayList2), 'TileSpacing','compact', 'Padding','compact')
k = 0;
for ri = 1:numel(RowList)
    currentArray = RowList(ri);

    for di = 1:numel(DayList2)
        nexttile(di+k)
        currentDay = DayList2(di);

        MD = All_MD.(currentDay).(currentArray);
        % Compute Mahalanobis distanceを描画する
        imagesc(1:num_task, 1:num_task, MD)
        axis square
        xticks(1:1:num_task)
        xticklabels(FingerLabel)
        yticks(1:1:num_task)
        yticklabels(FingerLabel)
        ax = gca; ax.FontSize = 6;
        colorbar
        title(currentDay, 'FontSize', 16, 'FontName','Helvetica')

        if di == 1
            ylabel(currentArray, 'FontSize',14, 'FontName','Helvetica')
        end

    end

    k = k + numel(DayList2);

end

%% Fig.5B

RowList = ["r6d", "l6d"];

color_value = colororder("gem12");

% Define label groups
R_Thumb  = ["R Thumb Up",  "R Thumb Down",  "R Thumb In"];
R_Index  = ["R Index Up",  "R Index Down",  "R Index In"];
R_middle = ["R Middle Up", "R Middle Down", "R Middle In"];
R_Ring   = ["R Ring Up",   "R Ring Down",   "R Ring In"];
R_Pincky = ["R Pinky Up",  "R Pinky Down",  "R Pinky In"];
L_Thumb  = ["L Thumb Up",  "L Thumb Down",  "L Thumb In"];
L_Index  = ["L Index Up",  "L Index Down",  "L Index In"];
L_middle = ["L Middle Up", "L Middle Down", "L Middle In"];
L_Ring   = ["L Ring Up",   "L Ring Down",   "L Ring In"];
L_Pincky = ["L Pinky Up",  "L Pinky Down",  "L Pinky In"];

Right_F = ["R Thumb Up",  "R Thumb Down",  "R Thumb In", "R Index Up",  "R Index Down",  "R Index In", "R Middle Up", "R Middle Down", "R Middle In", ...
    "R Ring Up",   "R Ring Down",   "R Ring In", "R Pinky Up",  "R Pinky Down",  "R Pinky In"];
Left_F = ["L Thumb Up",  "L Thumb Down",  "L Thumb In", "L Index Up",  "L Index Down",  "L Index In", "L Middle Up", "L Middle Down", "L Middle In", ...
    "L Ring Up",   "L Ring Down",   "L Ring In", "L Pinky Up",  "L Pinky Down",  "L Pinky In"];

Up_F = ["R Thumb Up","R Index Up","R Middle Up","R Ring Up","R Pinky Up","L Thumb Up","L Index Up","L Middle Up","L Ring Up","L Pinky Up"];
Down_F = ["R Thumb Down","R Index Down","R Middle Down","R Ring Down","R Pinky Down","L Thumb Down","L Index Down","L Middle Down","L Ring Down","L Pinky Down"];
In_F = ["R Thumb In","R Index In","R Middle In","R Ring In","R Pinky In","L Thumb In","L Index In","L Middle In","L Ring In","L Pinky In"];

% Select label groups used to color dendrogram labels
% Right hand vs. left hand
dendroLabelL = {Right_F, Left_F};
LabelName = "RL finger";

figure
tiledlayout(numel(RowList), numel(DayList2), 'TileSpacing','loose', 'Padding','loose')
k = 0;

currentArray = RowList(1);

for di = 1:numel(DayList2)
    nexttile(di+k)
    currentDay = DayList2(di);

    MD = All_MD.(currentDay).(currentArray);
    % Plot dendrogram
    validInd = ~isnan(MD(:, 1));
    validMD = MD(validInd, validInd);
    myOrder = FingerLabel(validInd);
    my_dendrogram_plot(validMD, myOrder, dendroLabelL, color_value, 8, 45);

    title(currentDay, 'FontSize', 16, 'FontName','Helvetica')

    if di == 1
        ylabel(currentArray, 'FontSize',16, 'FontName','Helvetica')
    end

    pbaspect([3 2 1])
end

k = k + numel(DayList2);

dendroLabelL = {Up_F, Down_F, In_F};
LabelName = "UpDwIn";

currentArray = RowList(2);

for di = 1:numel(DayList2)
    nexttile(di+k)
    currentDay = DayList2(di);

    MD = All_MD.(currentDay).(currentArray);

    validInd = ~isnan(MD(:, 1));
    validMD = MD(validInd, validInd);
    myOrder = FingerLabel(validInd);
    my_dendrogram_plot(validMD, myOrder, dendroLabelL, color_value, 8, 45);

    title(currentDay, 'FontSize', 16, 'FontName','Helvetica')

    if di == 1
        ylabel(currentArray, 'FontSize',16, 'FontName','Helvetica')
    end

    pbaspect([3 2 1])
end

%% Fig.5C

% Display order
RowList = ["All","v6d", "d6d", "l6d", "r6d"];

num_task = numel(FingerLabel);
% Collect pairwise distances across sessions for each array.
DistAll = struct();
for ri = 1:numel(RowList)
    currentArray = RowList(ri);
    all_dis = [];
    for di = 1:numel(DayList2)
        currentDay = DayList2(di);

        MD = All_MD.(currentDay).(currentArray);
        % Extract unique pairwise distances from the symmetric matrix.
        d = (squareform(MD))';
        all_dis = cat(2, all_dis, d);
    end

    DistAll.(currentArray) = all_dis;
    
end


% Initialize results structure
Results = struct();
for ri = 1:numel(RowList)
    currentArray = RowList(ri);
    md = DistAll.(currentArray);

    ave_data = [];
    ci1_data = [];
    ci2_data = [];
    corr_r = [];
    corr_p = [];
    corr_x = [];
    prevD = [];
    for i = 1:size(md, 2)
        selectD = md(:, i);
        validInd = ~isnan(selectD);

        [h, ~, ci] = normfit(selectD(validInd));
        ave_data(end+1) = h;
        ci1_data(end+1) = ci(1);
        ci2_data(end+1) = ci(2);

        if isempty(prevD)
            prevD = selectD;
        else
            % Compute Spearman correlation with the preceding session.
            validInd = ~(isnan(selectD) | isnan(prevD));
            [r, p] = corr(selectD(validInd), prevD(validInd), 'Type','Spearman');
            corr_r(end+1) = r;
            corr_p(end+1) = p;
            corr_x(end+1) = mean([i, i-1]);
            prevD = selectD;
        end


    end

    Results.(currentArray).ave = ave_data;
    Results.(currentArray).ci1 = ci1_data;
    Results.(currentArray).ci2 = ci2_data;
    Results.(currentArray).r = corr_r;
    Results.(currentArray).p = corr_p;
    Results.(currentArray).xdata = corr_x;

end

% Visualize results
figure
t = tiledlayout(1, numel(RowList), 'TileSpacing','loose', 'Padding','loose');
all_ax = gobjects(numel(RowList));
dNum = numel(DayList2);
for ai = 1:numel(RowList)
    currentArray = RowList(ai);
    all_ax(ai) = nexttile(t, ai);

    D = Results.(currentArray);
    % Plot mean normalized Mahalanobis distance and 95% CI.
    fill([1:dNum, flip(1:dNum)], [D.ci1, flip(D.ci2)], 'b', 'EdgeColor','none', 'FaceAlpha',0.2);
    hold on
    plot(1:dNum, D.ave, '*-', 'Color',color_value(1, :), 'LineWidth',2)
    ylabel({'Normalized Mahalanobis distance', '(task separation)'}, 'FontSize', 12, 'FontName','Helvetica')
    ylim([0 14])
    yyaxis right
    % Plot Spearman correlation.
    plot(D.xdata, D.r, '-', 'Color',color_value(2, :), 'LineWidth',2)
    for i = 1:numel(D.xdata)
        if D.p(i) < 0.05
            scatter(D.xdata(i), D.r(i), 100, 'red', 'filled')
        else
            scatter(D.xdata(i), D.r(i), 100, 'black', 'filled')
        end

    end

    xticks(1:dNum), xticklabels(DayList2)
    ax = gca;
    ax.YAxis(2).Color = [0 0 0];
    box off
    title(currentArray, 'FontSize', 16, 'FontName','Helvetica')
    ylabel('Spearman r between adjacent sessions', 'FontSize', 12, 'FontName','Helvetica')
    ylim([0.8 1])
    axis square

    yyaxis left
    % Compute Mahalanobis distanceの群間比較を行う
    x_data = DistAll.(currentArray);

    % Perform across-session statistical test.
    validInd = ~any(isnan(x_data), 2);
    [p, tbl, stats] = friedman(x_data(validInd, :), 1, 'off');

    if p > 0.05
        disText = sprintf('p=%.2f(Friedman test)', p);
    elseif p < 0.001
        disText = sprintf('*p<0.001(Friedman test)');
    else
        disText = sprintf('*p=%.2f(Friedman test)', p);
    end

    text(1, 1.2, disText)

end



%%% Legend
h_color = gobjects(3, 1);
h_color(1) = plot(nan, nan, '-', 'LineWidth',2, 'Color', color_value(1, :));
h_color(2) = plot(nan, nan, '-', 'LineWidth',2, 'Color', color_value(2, :));
h_color(3) = scatter(nan, nan, 100, 'red', 'filled');

L = legend(h_color, {'Mahalanobis distance', 'Session-to-session correlation', 'Red: p < 0.05'}, 'Location','best', 'Box', 'off', 'FontSize',12);

%% Fig.5E

ArrayList = ["All","v6d", "d6d", "l6d", "r6d"];
Arraynum = numel(ArrayList);

combL = nchoosek(1:Arraynum, 2);
combNum = size(combL, 1);

figure
t = tiledlayout(1, 1, 'TileSpacing','loose', 'Padding','loose');

for combi = combNum

    currentComb = combL(combi, :);

    g1 = ArrayList(currentComb(1));
    g2 =  ArrayList(currentComb(2));
    dNum = size(DistAll.(g1), 2);
    
    % Compare representational structure between regions.
    R = calculate_two_region_corr(DistAll.(g1), DistAll.(g2));
    
    nexttile
    fill([1:dNum, flip(1:dNum)], [R.ci1_data, flip(R.ci2_data)], 'b', 'EdgeColor','none', 'FaceAlpha',0.2);
    hold on
    plot(1:dNum, R.ave_data, '*-', 'Color',color_value(1, :), 'LineWidth',2)
    ylabel('Inter-area Mahalanobis distance', 'FontSize', 12, 'FontName','Helvetica')
    ylim([-1 8.5])
    xlim([0 numel(DayList2)+1])


    % Test session-wise differences.
    x_data = [];
    y_data = [];
    for i = 1:numel(R.rowData)

        if isempty(x_data)
            x_data = R.rowData{i};
            y_data = repmat(i, numel(R.rowData{i}), 1);
        else
            x_data = cat(1, x_data, R.rowData{i});
            y_data = cat(1, y_data, repmat(i, numel(R.rowData{i}), 1));
        end

    end

    [p, tbl, stats] = kruskalwallis(x_data, y_data, 'off');

    if p > 0.05
        disText = sprintf('p=%.2f(KW test)', p);
    elseif p < 0.001
        disText = sprintf('*p<0.001(KW test)');
    else
        disText = sprintf('*p=%.3f(KW test)', p);
    end

    text(1, -0.5, disText)





    yyaxis right
    % Plot Spearman correlation.
    plot(1:dNum, R.corr_r, '-', 'Color',color_value(2, :), 'LineWidth',2)
    for i = 1:dNum
        if R.corr_p(i) < 0.05
            scatter(i, R.corr_r(i), 100, 'red', 'filled')
        else
            scatter(i, R.corr_r(i), 100, 'black', 'filled')
        end
    
    end
    
    xticks(1:dNum), xticklabels(DayList2)
    ax = gca;
    ax.YAxis(2).Color = [0 0 0];
    box off
    titleName = g1 + " – " + g2;
    title(char(titleName), 'FontSize', 16, 'FontName','Helvetica')
    ylabel('Spearman r between areas', 'FontSize', 12, 'FontName','Helvetica')
    ylim([0.65 1])
    axis square

end

%%% Legend
h_color = gobjects(3, 1);
h_color(1) = plot(nan, nan, '-', 'LineWidth',2, 'Color', color_value(1, :));
h_color(2) = plot(nan, nan, '-', 'LineWidth',2, 'Color', color_value(2, :));
h_color(3) = scatter(nan, nan, 100, 'red', 'filled');

L = legend(h_color, {'Mahalanobis distance', 'Area-to-area correlation', 'Red: p < 0.05'}, 'Location','best', 'Box', 'off', 'FontSize',10);


%% Fig.5D: SVM classification

ArrayList = ["l6d", "r6d"];
chList = {1:256, 257:384};

k_fold = 10;

SVM_R = struct();
for ai = 1:numel(ArrayList)
    usedArray = ArrayList(ai);
    usedCh = chList{ArrayList == usedArray};

    svm_tempR = nan(k_fold, numel(DayList2));
    for di = 1:numel(DayList2)

        currentDate = DayList2(di);
        
        onedayD = ClipD.(currentDate).refined_Warping_zFR;
        
        validInd = zeros(30, 1);
        for i = 1:30
            currentfname = "T" + string(i);
            if ~isempty(onedayD.(currentfname))
                validInd(i) = 1;
            end
        end
        validInd = logical(validInd);
        
        % Average each trial across time to obtain feature vectors.
        % Update validInd by excluding movement classes with fewer than 10 trials.
        [x_all, y_all, validInd] = create_x_y_data_for_svm(validInd, onedayD);
        
        % Select discriminative channels using the Kruskal-Wallis test with FDR correction.
        [sigX, sig_idx] = refined_xdata_by_KW(usedCh, x_all, y_all);
        
        % Stratified k-fold cross-validation
        cv = cvpartition(y_all, 'KFold', k_fold, 'Stratify', true);
        [y_gnd_allFolds, y_pred_allFolds, mean_acc_allFolds] = my_nfold_cv_SVM(cv, sigX, y_all);
        svm_tempR(:, di) = mean_acc_allFolds(:);
        
        
        gnd = y_gnd_allFolds; pred = y_pred_allFolds;
        % Convert numeric ground-truth and predicted labels to FingerLabel categories.
        gnd = ydata_double_to_categories(gnd, FingerLabel, validInd);
        pred = ydata_double_to_categories(pred, FingerLabel, validInd);
        
        
        macroF1 = my_f1_score(gnd, pred);
        acc = mean(gnd == pred);
        
    end

    svm_tempR = array2table(svm_tempR, 'VariableNames',DayList2);

    SVM_R.(usedArray) = svm_tempR;
end


co = colororder("gem");
dateNum = numel(DayList2);
figure
h_color = gobjects(numel(ArrayList), 1);
for ai = 1:numel(ArrayList)
    usedArray = ArrayList(ai);
    
    currentData = SVM_R.(usedArray){:, :}*100;

    [h, ~, ci] = normfit(currentData);
    fill([1:dateNum, flip(1:dateNum)], [ci(1,:), flip(ci(2, :))], co(ai, :), 'FaceAlpha', 0.2, 'EdgeColor','none');
    hold on
    plot(1:dateNum, h, 'o-', 'Color', co(ai, :), 'LineWidth',2)

    h_color(ai) = plot(nan, nan, 'o-', 'Color', co(ai, :), 'LineWidth',2);
end
xticks(1:dateNum), xticklabels(DayList2), xlim([0 dateNum+1]), xlabel('Session Day')
ylabel('Classification Accuracy (%)'), ylim([50 100])
box off
legend(h_color, {'Left 6d', 'Right 6d'}, 'Box','off', 'Location','best', 'AutoUpdate','off', 'FontSize',12)
title('Session-wise SVM Classification Accuracy Across Arrays')

%% Fig.5F-G: Spatiotemporal cluster analysis using FieldTrip

% Reset the MATLAB search path.
restoredefaultpath
rehash toolboxcache

% Add FieldTrip to the MATLAB path
addpath(fullfile(projectDir,'..', 'Fig1', 'external', 'fieldtrip'));

% Initialize FieldTrip
ft_defaults

%% Spatiotemporal cluster calculation
Array_Allocation = ["v6d", "d6d", "r6d"];
Array_ChNum= {1:128, 129:256, 257:384};
taskNum = numel(FingerLabel);
aNum = numel(Array_Allocation);

% During feature extraction, Blackrock channel indices were converted to post-mapping indices.
% The mappings below convert them back to the original Blackrock channel order.
PostmapCh = {1:32, 33:64, 65:96, 97:128, ... % v6d
            129:160, 161:192, 193:224, 225:256, ... % d6d
            257:288, 289:320, 321:352, 353:384}; % r6d
BRCh = {1:32, 97:128, 33:64, 65:96, ... % v6d: subarray 1 then 2 from ventral to dorsal; no 180-degree rotation
        161:192, 193:224, 129:160, 225:256, ... % d6d: subarray 2 then 1 from ventral to dorsal; rotated 180 degrees
        289:320, 321:352, 257:288, 353:384}; % r6d: subarray 2 then 1 from ventral to dorsal; no 180-degree rotation

% Channel indices for the two subarrays
subarrayList = {[1:32, 97:128], 33:96};

% Precompute the total number of iterations for progress reporting.
totalSteps = numel(Array_Allocation) * 2 * taskNum * DataNum;
step = 0;

% Initialize progress bar
h = waitbar(0, 'Spatiotemporal cluster analysis running...');

% Initialize results structure
ResultST = struct();

for di = 1:DataNum 

    currentDay = DayList2(di);
    
    data = ClipD.(currentDay).refined_Warping_zFR;
    base = ClipD.(currentDay).base_zFR;

    % Initialize results structure
    Results = struct();
    for i =1:numel(Array_Allocation)
        fieldName = Array_Allocation{i};
        Results.(fieldName) = cell(taskNum, 2);
    end
    
    for arrayi = 1:aNum
        % Select current array
        fieldName = Array_Allocation(arrayi);
        
        % Loop over subarrays
        for subarrayi = 1:2
    
            % Loop over movement classes
            for taski = 1:taskNum
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%% Step 1: Convert data to FieldTrip format
                currentField = "T" + string(taski);
    
                data_task = data.(currentField);
                base_task = base.(currentField);
                if ~isempty(data_task)
                    % Average across time because movement-segment durations differ across classes.
                    ave_data = squeeze(mean(data_task, 2)); % trials x time x channels -> trials x channels
                    ave_base = squeeze(mean(base_task, 2)); % trials x channels
                else % if no data are available
                    Results.(fieldName){taski, subarrayi} = [];
                    %%%%%%%%%%%%%%%%%%%%%% Update progress bar
                    step = step + 1;
                    waitbar(step / totalSteps, h, sprintf('Analyzing %s, Subarray %d, Task %d/%d', ...
                            Array_Allocation{arrayi}, subarrayi, taski, taskNum));
                    continue
                end
                % Convert post-mapping channel order back to the original Blackrock order.
                br_data = my_Map_to_BR_convert(ave_data, PostmapCh, BRCh);
                br_base = my_Map_to_BR_convert(ave_base, PostmapCh, BRCh);
    
                % Extract channels for the current array and subarray.
                br_data_array = br_data(:, Array_ChNum{arrayi});
                br_base_array = br_base(:, Array_ChNum{arrayi});
                br_data_subarray = br_data_array(:, subarrayList{subarrayi});
                br_base_subarray = br_base_array(:, subarrayList{subarrayi});
    
                % Number of trials for the current movement class
                trialn = size(br_data, 1);
    
                if trialn < 10 % skip classes with fewer than 10 trials
                    Results.(fieldName){taski, subarrayi} = [];
                    %%%%%%%%%%%%%%%%%%%%%% Update progress bar
                    step = step + 1;
                    waitbar(step / totalSteps, h, sprintf('Analyzing %s, Subarray %d, Task %d/%d', ...
                            Array_Allocation{arrayi}, subarrayi, taski, taskNum));
                    continue
                end
                
                % Construct FieldTrip data structures.
                data_bl = [];  % baseline:
                data_act = []; % active
                for trial = 1:trialn
                    % ch x 1
                    tmp_act  = br_data_subarray(trial, :)';
                    tmp_base = br_base_subarray(trial, :)';
                
                    % Duplicate each channel value across two identical time samples to enable spatiotemporal cluster analysis.
                    data_act.trial{trial} = [tmp_act  tmp_act];   % ch x 2
                    data_bl.trial{trial}  = [tmp_base tmp_base];  % ch x 2
                
                    % Use two corresponding time points.
                    data_act.time{trial}  = [0 0.01];  % arbitrary time units, consistent with fsample
                    data_bl.time{trial}   = [0 0.01];
    
                end
                data_bl.label = arrayfun(@(x) sprintf('Ch%d', x), subarrayList{subarrayi} + Array_ChNum{arrayi}(1)-1, 'UniformOutput', false);
                data_act.label = data_bl.label;
                data_bl.fsample = 100;
                data_act.fsample = 100;
              
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%% Step 2: Define neighboring channels
                % Both d6d subarrays are rotated by 180 degrees; v6d and r6d are not.
                if Array_Allocation(arrayi) == "v6d" || Array_Allocation(arrayi) == "r6d"
                    neighbours = my_neighbours_make(subarrayi, false, Array_ChNum, arrayi);
                elseif Array_Allocation(arrayi) == "d6d" % set the second argument to true for a 180-degree rotation
                    neighbours = my_neighbours_make(subarrayi, true, Array_ChNum,  arrayi);
                end
                            
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Step 3: Spatiotemporal cluster-based statistical test          
                [stat, ~] = do_spatiotemporal_cluster_ana(data_act, data_bl, neighbours);
                Results.(fieldName){taski, subarrayi} = stat;
    
                %%%%%%%%%%%%%%%%%%%%%% Update progress bar
                step = step + 1;
                waitbar(step / totalSteps, h, sprintf('Analyzing %s, Subarray %d, Task %d/%d', ...
                    Array_Allocation{arrayi}, subarrayi, taski, taskNum));
                
            end
        end
    
    end

    ResultST.(currentDay) = Results;

end

close(h);

% Save results
outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

saveName = "spatiotemporal results.mat";
save(fullfile(outputDir, char(saveName)), 'ResultST', 'FingerLabel','Array_Allocation', 'Array_ChNum', 'BRCh', 'PostmapCh',...
    '-v7.3');

%%
DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DayList2 = "D" + DaysList;

Array_Allocation = ["v6d", "d6d", "r6d"];
arrayNum = numel(Array_Allocation);
TaskNum = numel(FingerLabel);
dateNum = numel(DayList2);

UpDownAll = struct();

% Aggregate results for each subarray and array; define left 6d as v6d + d6d.
for di = 1:dateNum

    currentDay = DayList2(di);
    Results = ResultST.(currentDay);

    % Initialize output variables
    v6d1u = cell(TaskNum, 1); v6d2u = cell(TaskNum, 1); v6du = cell(TaskNum, 1); 
    d6d1u = cell(TaskNum, 1); d6d2u = cell(TaskNum, 1); d6du = cell(TaskNum, 1); 
    r6d1u = cell(TaskNum, 1); r6d2u = cell(TaskNum, 1); r6du = cell(TaskNum, 1); l6du = cell(TaskNum, 1);

    v6d1d = cell(TaskNum, 1); v6d2d = cell(TaskNum, 1); v6dd = cell(TaskNum, 1); 
    d6d1d = cell(TaskNum, 1); d6d2d = cell(TaskNum, 1); d6dd = cell(TaskNum, 1); 
    r6d1d = cell(TaskNum, 1); r6d2d = cell(TaskNum, 1); r6dd = cell(TaskNum, 1); l6dd = cell(TaskNum, 1);

    % Loop over movement classes
    for taski = 1:TaskNum

        if isempty(Results.("v6d"){taski, 1})
            % Assign NaN when no input data are available.
            v6d1u{taski, 1} = nan; v6d2u{taski, 1} = nan; v6du{taski, 1} = nan; 
            d6d1u{taski, 1} = nan; d6d2u{taski, 1} = nan; d6du{taski, 1} = nan; 
            r6d1u{taski, 1} = nan; r6d2u{taski, 1} = nan; r6du{taski, 1} = nan; l6du{taski, 1} = nan;
            
            v6d1d{taski, 1} = nan; v6d2d{taski, 1} = nan; v6dd{taski, 1} = nan; 
            d6d1d{taski, 1} = nan; d6d2d{taski, 1} = nan; d6dd{taski, 1} = nan; 
            r6d1d{taski, 1} = nan; r6d2d{taski, 1} = nan; r6dd{taski, 1} = nan; l6dd{taski, 1} = nan;
            continue
        end

        Results_updown = my_Results_up_and_down_get(Array_Allocation, taski, Results, 1);
        % t-values for channels showing significant activation
        % v6d
        v6d1u{taski, 1} = Results_updown.("v6d").upS{1};
        v6d2u{taski, 1} = Results_updown.("v6d").upS{2};
        v6du{taski, 1} = cat(1, Results_updown.("v6d").upS{:});
        % d6d
        d6d1u{taski, 1} = Results_updown.("d6d").upS{1};
        d6d2u{taski, 1} = Results_updown.("d6d").upS{2};
        d6du{taski, 1} = cat(1, Results_updown.("d6d").upS{:});
        % r6d
        r6d1u{taski, 1} = Results_updown.("r6d").upS{1};
        r6d2u{taski, 1} = Results_updown.("r6d").upS{2};
        r6du{taski, 1} = cat(1, Results_updown.("r6d").upS{:});
        % left 6d (v6d + d6d)
        l6du{taski, 1} = [v6du{taski, 1}; d6du{taski, 1}];

        % t-values for channels showing significant suppression
        % v6d
        v6d1d{taski, 1} = Results_updown.("v6d").downS{1};
        v6d2d{taski, 1} = Results_updown.("v6d").downS{2};
        v6dd{taski, 1} = cat(1, Results_updown.("v6d").downS{:});
        % d6d
        d6d1d{taski, 1} = Results_updown.("d6d").downS{1};
        d6d2d{taski, 1} = Results_updown.("d6d").downS{2};
        d6dd{taski, 1} = cat(1, Results_updown.("d6d").downS{:});
        % r6d
        r6d1d{taski, 1} = Results_updown.("r6d").downS{1};
        r6d2d{taski, 1} = Results_updown.("r6d").downS{2};
        r6dd{taski, 1} = cat(1, Results_updown.("r6d").downS{:});
        % left 6d (v6d + d6d)
        l6dd{taski, 1} = [v6dd{taski, 1}; d6dd{taski, 1}];
    end

    upTable = table(v6d1u, v6d2u, v6du, d6d1u, d6d2u, d6du, r6d1u, r6d2u, r6du, l6du, 'VariableNames',{'v6d1', 'v6d2', 'v6d', 'd6d1', 'd6d2', 'd6d', ...
        'r6d1', 'r6d2', 'r6d', 'l6d'});

    downTable = table(v6d1d, v6d2d, v6dd, d6d1d, d6d2d, d6dd, r6d1d, r6d2d, r6dd, l6dd, 'VariableNames',{'v6d1', 'v6d2', 'v6d', 'd6d1', 'd6d2', 'd6d', ...
        'r6d1', 'r6d2', 'r6d', 'l6d'});

    UpDownAll.(currentDay).up_t = upTable; 
    UpDownAll.(currentDay).down_t = downTable; 

end

% Divide movement classes into right- and left-hand groups.
G1 = 1:15; % Group1
G2 = 16:30; % Group2
G1Name = "RightGesture";
G2Name = "LeftGesture";

All_tstat = struct();

for di = 1:numel(DayList2)
    currentDate = DayList2(di);

    up = UpDownAll.(currentDate).up_t;
    down = UpDownAll.(currentDate).down_t;
    
    Tstat_Result = struct();
    
    % Group 1: right-hand movements
    % Compute the proportion of significantly modulated channels, mean t-value, and their product.
    [ch_prop_u, mean_t_u, mT_u, ch_prop_d, mean_t_d, mT_d] = convert_sigChCount_meanT(G1, up, down);
    
    Tstat_Result.(G1Name).ch_prop_u = ch_prop_u;
    Tstat_Result.(G1Name).mean_t_u = mean_t_u;
    Tstat_Result.(G1Name).mT_u  = mT_u;
    
    Tstat_Result.(G1Name).ch_prop_d = ch_prop_d;
    Tstat_Result.(G1Name).mean_t_d = mean_t_d;
    Tstat_Result.(G1Name).mT_d  = mT_d;
    
    % Group 2: left-hand movements
    % Compute the proportion of significantly modulated channels, mean t-value, and their product.
    [ch_prop_u, mean_t_u, mT_u, ch_prop_d, mean_t_d, mT_d] = convert_sigChCount_meanT(G2, up, down);
    
    Tstat_Result.(G2Name).ch_prop_u = ch_prop_u;
    Tstat_Result.(G2Name).mean_t_u = mean_t_u;
    Tstat_Result.(G2Name).mT_u  = mT_u;
    
    Tstat_Result.(G2Name).ch_prop_d = ch_prop_d;
    Tstat_Result.(G2Name).mean_t_d = mean_t_d;
    Tstat_Result.(G2Name).mT_d  = mT_d;

    All_tstat.(currentDate) = Tstat_Result;
end

% Visualization
A1 = "l6d";
A2 = "r6d";
ArrayList = [A1, A2];
ArrayNameList = ["Left 6d", "Right 6d"];
dateNum = numel(DayList2);
varName = All_tstat.(DayList2(1)).RightGesture.mT_u.Properties.VariableNames;

groupNameList = ["Right Gesture: ", "Left Gesture: "];

anaFeature1 = "mT_u"; 
anaFeature2 = "mT_d"; 
anaFeature_name_List = [" — Activation (t > 0)", " — Suppression (t < 0)"];

% Compute session-wise means and confidence intervals for the selected features.
Total_across_arrayR = create_mean_and_ci_one_date(DayList2, All_tstat, G1Name, G2Name, anaFeature1, anaFeature2);


figure
tiledlayout(2, 2, 'TileSpacing','loose', 'Padding','loose')
all_axs1 = gobjects(2);
all_axs2 = gobjects(2);
% Loop over arrays
for ai = 1:numel(ArrayList)

    % Loop over activation and suppression.
    for fi = 1:2
        
        color_list = ["red", "blue"];
        if fi == 1
            all_axs1(ai) = nexttile;
        elseif fi == 2
            all_axs2(ai) = nexttile;
        end
        p_results_list = {};
        % Loop over hand groups.
        for gi = 1:2
            makerTypeList = ["o", "*"];
            ave = Total_across_arrayR.(ArrayList(ai)).("g" + string(gi) + "_ave_" + string(fi));
            ci = Total_across_arrayR.(ArrayList(ai)).("g" + string(gi) + "_ci_" + string(fi));
            rowD = Total_across_arrayR.(ArrayList(ai)).("g" + string(gi) + "_row_" + string(fi));
            if fi == 1
                fill([1:dateNum, flip(1:dateNum)], [ci(1, :), flip(ci(2, :))], 'red', 'FaceAlpha',0.1, 'EdgeColor','none')
                hold on
                plot(1:dateNum, ave, 'r-', 'LineWidth',1.5, 'Marker', char(makerTypeList(gi)))
            elseif fi == 2
                fill([1:dateNum, flip(1:dateNum)], [ci(1, :), flip(ci(2, :))], 'blue', 'FaceAlpha',0.1, 'EdgeColor','none')
                hold on
                plot(1:dateNum, ave, 'b-', 'LineWidth',1.5, 'Marker', char(makerTypeList(gi)))
            end
    
            % Test session-wise differences.
            x_data = [];
            y_data = [];
            for ri = 1:numel(rowD)
                if isempty(x_data)
                    x_data = rowD{ri};
                    y_data = repmat(ri, numel(rowD{ri}), 1);
                else
                    x_data = cat(1, x_data, rowD{ri});
                    y_data = cat(1, y_data, repmat(ri, numel(rowD{ri}), 1));            
                end
            end


            p = kruskalwallis(x_data, y_data, 'off'); % across-session comparison
            if p >= 0.05
                distext = sprintf('(\\itp\\rm = %.3f, KW test)', p);
            elseif p > 0.001
                distext = sprintf('(*\\itp\\rm = %.3f, KW test)', p);
            else
                distext = sprintf('(*\\itp\\rm < 0.001, KW test)');
            end

            p_results_list{gi} =  distext;
    
        end

        %%% Legend
        h_color = gobjects(2, 1);
        h_color(1) = plot(nan, nan,  char(makerTypeList(1)), 'LineWidth',1.5, 'Color', color_list(fi));
        h_color(2) = plot(nan, nan,  char(makerTypeList(2)), 'LineWidth',1.5, 'Color', color_list(fi));

        legendName = [groupNameList(1) + string(p_results_list{1}), groupNameList(2) + string(p_results_list{2})];  
        L = legend(h_color, legendName, 'Location','best', 'Box', 'off', 'FontSize', 10, 'Autoupdate', 'off');
    
        titleName = {ArrayNameList(ai) + " " + anaFeature_name_List(fi)};
        title(titleName)
        xticks(1:dateNum), xticklabels(DayList2)
        box off
        xlim([0 dateNum+1])
        ylabel("Normalized mean t-value")

    end

end

linkaxes(all_axs1, 'xy')
linkaxes(all_axs2, 'xy')



%% Local functions

function [means, covs] = calculate_means_covs_for_MahaDis(analyzed_ch, num_task, ClipD, usedFeatures, currentDay, minimumTrial)

ChNum = numel(analyzed_ch);
means = zeros(num_task, ChNum);
covs = cell(num_task, 1);

for taski = 1:num_task
    currentFieldName = "T" + string(taski);

    currentData = ClipD.(currentDay).(usedFeatures).(currentFieldName);

    if isempty(currentData)
        means(taski, :) = nan(1, ChNum);
        covs{taski} = nan(ChNum, ChNum);
        continue
    else

        X1 = squeeze(mean(currentData, 2));

        if size(X1, 1) < minimumTrial
            means(taski, :) = nan(1, ChNum);
            covs{taski} = nan(ChNum, ChNum);
            continue
        end

        means(taski, :) = mean(X1(:, analyzed_ch), 1);
        covs{taski} = cov(X1(:, analyzed_ch));
    end

end

end

function [distance_matrix] = my_MahalanobisDis(means, covs, num_task)
distance_matrix = nan(num_task, num_task);

ChNum = size(covs{1}, 1);

for i = 1:num_task
    mu_i = means(i, :)';

    for j = 1:num_task
        mu_j = means(j, :)';
        cov_avg = 0.5 * (covs{i} + covs{j}) + 1e-3 * eye(ChNum);
        d = sqrt((mu_i - mu_j)' * (cov_avg \ (mu_i - mu_j)));
        distance_matrix(i, j) = d;
    end
    distance_matrix(i, i) = 0;
end
end

function my_dendrogram_plot(distance_matrix, myOrder, dendroLabelL, Color_val, Fsize, RotaionAngel)

dist_vec = squareform(distance_matrix);  % 1×(nC2)

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
        currentInd = find(ismember(myOrder, currentL));
        if ismember(idx, currentInd)
            color = Color_val(j, :);
            continue
        end
    end

    if isempty(color)
        color = [0 0 0];
    end
    text(xticks(i), ylimVal(1) - 0.01 * diff(ylimVal), txt, ...
        'Rotation', RotaionAngel, 'HorizontalAlignment', 'right', ...
        'FontSize', Fsize, 'FontName', 'Helvetica', 'Color', color);
end

end

function [Results] = calculate_two_region_corr(g1, g2)
    
    dNum = size(g1, 2);

    Results = struct();
    
    Results.ave_data = [];
    Results.ci1_data = [];
    Results.ci2_data = [];
    Results.corr_r = [];
    Results.corr_p = [];
    Results.rowData = [];

    
    for di = 1:dNum
        l = g1(:, di);
        r = g2(:, di);
    
        validInd = ~(isnan(l) | isnan(r));
    
        lv = l(validInd);
        rv = r(validInd);
    
        % Absolute difference in pairwise distances
        diffD = abs(lv- rv);
        [h, ~, ci] = normfit(diffD);
        [r, p] = corr(lv, rv, 'Type','Spearman');
        Results.ave_data(end+1) = h;
        Results.ci1_data(end+1) = ci(1);
        Results.ci2_data(end+1) = ci(2);
        Results.corr_r(end+1) = r;
        Results.corr_p(end+1) = p;

        Results.rowData{di} = diffD;
    
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

function [y_gnd_allFolds, y_pred_allFolds, mean_acc_allFolds] = my_nfold_cv_SVM(cv, newAllignedDat, ydata)


    y_gnd_allFolds = [];
    y_pred_allFolds = [];
    mean_acc_allFolds = [];
   


    h = waitbar(0, sprintf('Running CV 0/%d...', cv.NumTestSets));

    c = onCleanup(@() safeCloseWaitbar(h));  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 10Fold loop
    for cvi = 1:cv.NumTestSets
        %%%%%%%%%%%%%%% Step 1: Obtain training and test indices.
        train_idx = training(cv, cvi);
        test_idx = test(cv, cvi);
        
        %%%%%%%%%%%%%%% Step 2: Estimate z-scoring parameters from the training data only.
        
        data_train = newAllignedDat(train_idx, :);


        mn = mean(data_train, 1);
        sd = std(data_train, 0, 1);

        %%%%%%%%%%%%%%% Step 3: Z-score all trials using training-set parameters.
        xdata = (newAllignedDat - mn)./sd;
    
        %%%%%%%%%%%%%%% Step 4: Train and evaluate the SVM using the standardized feature vectors.
        X_train = xdata(train_idx,:);
        y_train = ydata(train_idx);
        X_test  = xdata(test_idx,:);
        y_test  = ydata(test_idx);
    

        t = templateSVM('KernelFunction', 'linear');
        
        SVMModel = fitcecoc(X_train, y_train, 'Learners', t); 

        pred_val = predict(SVMModel, X_test);
        

        y_gnd_allFolds = [y_gnd_allFolds; y_test];
        y_pred_allFolds = [y_pred_allFolds; pred_val];
        mean_acc_allFolds = [mean_acc_allFolds; mean(pred_val == y_test)];


        waitbar(cvi / cv.NumTestSets, h, ...
            sprintf('Running CV %d/%d...', cvi, cv.NumTestSets));
    
    end

    function safeCloseWaitbar(h)
        if isvalid(h); close(h); end
    end


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

function [new_gnd] = ydata_double_to_categories(gnd, FingerLabel, validInd)
    numD = numel(gnd);
    
    new_gnd = strings(numD, 1);
    for i = 1:numD
        new_gnd(i) = FingerLabel(gnd(i));
    end
    
    new_gnd = categorical(new_gnd, FingerLabel(validInd), 'Ordinal', true);
end

function macroF1 = my_f1_score(unroll_y, y_pred_categorical)
    cm = confusionmat(unroll_y, y_pred_categorical);
    
    num_classes = size(cm, 1);
    f1_scores = zeros(num_classes, 1);
    
    for c = 1:num_classes
        TP = cm(c,c);
        FP = sum(cm(:,c)) - TP;
        FN = sum(cm(c,:)) - TP;
        precision = TP / (TP + FP + eps);
        recall    = TP / (TP + FN + eps);
        f1_scores(c) = 2 * precision * recall / (precision + recall + eps);
    end
    
    macroF1 = mean(f1_scores);
    fprintf('Macro F1-score: %.3f\n', macroF1);
end

function [new_data] = my_Map_to_BR_convert(ave_data, PostmapCh, BRCh)

    new_data = nan(size(ave_data));
    
    dataNum = numel(PostmapCh);
    for i = 1:dataNum
        currentMap = PostmapCh{i};
        currentBR = BRCh{i};
    
        new_data(:, currentBR) = ave_data(:, currentMap);
    end

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

    % subarray1 180° rotation
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

    % subarray2 - 180° rotation
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
        elec.elecpos(i,:) = [X(i), Y(i), 0];          % Z=0
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

function [stat, cfg] = do_spatiotemporal_cluster_ana(data1, data2, neighbours)
    cfg = [];
    cfg.method           = 'montecarlo'; 
    cfg.statistic        = 'depsamplesT'; 
    cfg.correctm         = 'cluster'; 
    cfg.clusteralpha     = 0.05;  
    cfg.clusterstatistic = 'maxsum'; 
    cfg.tail             = 0; 
    cfg.clustertail      = 0; 
    cfg.alpha            = 0.025;
    cfg.numrandomization = 1000; 
    cfg.neighbours       = neighbours;
    
    
    nTrials = size(data1.trial, 2);
    nTotal = 1 * nTrials;
    cfg.design = [1:nTotal, 1:nTotal; ones(1,nTotal), 2*ones(1,nTotal)];
    cfg.uvar   = 1; % trial
    cfg.ivar   = 2; % condition (1=BL, 2=Act)
    
    [stat] = ft_timelockstatistics(cfg, data1, data2); 

end

function [Results_updown] = my_Results_up_and_down_get(Array_Allocation, selectTaskInd, Results, analizedTimeInd)

    Results_updown = struct();
    arrayNum = numel(Array_Allocation);
    
    for arrayi = 1:arrayNum 
 
        FieldName = Array_Allocation(arrayi);
    
        sub1D = Results.(FieldName){selectTaskInd, 1};
        sub2D = Results.(FieldName){selectTaskInd, 2};
    
        
        % subarray1
        [sub1up, sub1down, upStats1, downStats1, rawS1] = create_up_down_both_results(sub1D, analizedTimeInd);

        % subarray2
        [sub2up, sub2down, upStats2, downStats2, rawS2] = create_up_down_both_results(sub2D, analizedTimeInd);
        
        Results_updown.(FieldName).up{1,1} = sub1up;
        Results_updown.(FieldName).up{1,2} = sub2up;
        Results_updown.(FieldName).down{1,1} = sub1down;
        Results_updown.(FieldName).down{1,2} = sub2down;    
        Results_updown.(FieldName).upS{1,1} = upStats1;
        Results_updown.(FieldName).upS{1,2} = upStats2;
        Results_updown.(FieldName).downS{1,1} = downStats1;
        Results_updown.(FieldName).downS{1,2} = downStats2;  
        Results_updown.(FieldName).rawS{1,1} = rawS1; 
        Results_updown.(FieldName).rawS{1,2} = rawS2; 
    end

end

function [upSum, downSum, upStats, downStats, sub1dstat] = create_up_down_both_results(sub1D, analizedTimeInd)
    
    sub1dMask = sub1D.mask(:, analizedTimeInd);
    sub1dstat = sub1D.stat(:, analizedTimeInd);
    
    upmatrix = sub1dMask & (sub1dstat > 0);   
    downmatrix = sub1dMask & (sub1dstat < 0);  
    
   
    upSum = sum(upmatrix, 2);
    downSum = sum(downmatrix, 2);

    
    upStats = sub1dstat(upmatrix);
    downStats = sub1dstat(downmatrix);
    
end

function [ch_prop_u, mean_t_u, mT_u, ch_prop_d, mean_t_d, mT_d] = convert_sigChCount_meanT(TaskList, up, down)

    variableName = up.Properties.VariableNames;
    vnNum = numel(variableName);

    ch_prop_u = []; 
    mean_t_u = []; 
    mT_u = []; 
    
    ch_prop_d = []; 
    mean_t_d = []; 
    mT_d = [];
        
    k = 1;
    for gi = TaskList
    
        
        if isnan(up{gi, 1}{:})
            continue
        end
    
        for vi = 1:vnNum
            
            currentArray = string(variableName{vi});
            if currentArray == "v6d" || currentArray == "d6d" || currentArray == "r6d"
                totalChNum = 64*2; 
            elseif currentArray == "l6d"
                totalChNum = 64*4;
            else
                totalChNum = 64;
            end
    
            
            currentUp = up{gi, vi}{:};
            ch_prop_u(k, vi) = numel(currentUp)./totalChNum;
            if isnan(mean(currentUp)) 
                mean_t_u(k, vi) = 0;
            else
                mean_t_u(k, vi) = mean(currentUp);
            end
            mT_u(k, vi) = ch_prop_u(k, vi)*mean_t_u(k, vi);
    
            
            currentDown = down{gi, vi}{:};
            ch_prop_d(k, vi) = numel(currentDown)./totalChNum;
            if isnan(mean(currentDown)) 
                mean_t_d(k, vi) = 0;
            else
                mean_t_d(k, vi) = mean(currentDown);
            end
            mT_d(k, vi) = ch_prop_d(k, vi)*mean_t_d(k, vi);
    
        end
    
        k = k + 1;
    
    end
    
    
    ch_prop_u = array2table(ch_prop_u, 'VariableNames', variableName);
    mean_t_u = array2table(mean_t_u, 'VariableNames', variableName);
    mT_u      = array2table(mT_u, 'VariableNames', variableName);
    
    ch_prop_d = array2table(ch_prop_d, 'VariableNames', variableName);
    mean_t_d = array2table(mean_t_d, 'VariableNames', variableName);
    mT_d      = array2table(mT_d, 'VariableNames', variableName);

end

function [Total_across_arrayR] = create_mean_and_ci_one_date(DayList2, All_tstat, G1Name, G2Name, anaFeature1, anaFeature2)

    varName = All_tstat.(DayList2(1)).(G1Name).mT_u.Properties.VariableNames;

    for ai = 1:numel(varName)
        currentArray = varName{ai};
    
        across_day_result = struct();
    
        for di = 1:numel(DayList2)
    
            currentDate = DayList2(di);
    
  
            currentR1 = All_tstat.(currentDate).(G1Name).(anaFeature1).(currentArray);
            [h, ~, ci] = normfit(currentR1);
            across_day_result.g1_ave_1(di,1) = h;
            across_day_result.g1_ci_1(1:2, di) = ci;
            across_day_result.g1_row_1{di,1} = currentR1;

    
            currentR2 = All_tstat.(currentDate).(G1Name).(anaFeature2).(currentArray);
            [h, ~, ci] = normfit(currentR2);
            across_day_result.g1_ave_2(di,1) = h;
            across_day_result.g1_ci_2(1:2, di) = ci;
            across_day_result.g1_row_2{di,1} = currentR2;
    

            currentL1 = All_tstat.(currentDate).(G2Name).(anaFeature1).(currentArray);
            [h, ~, ci] = normfit(currentL1);
            across_day_result.g2_ave_1(di,1) = h;
            across_day_result.g2_ci_1(1:2, di) = ci;
            across_day_result.g2_row_1{di,1} = currentL1;
    
            currentL2 = All_tstat.(currentDate).(G2Name).(anaFeature2).(currentArray);
            [h, ~, ci] = normfit(currentL2);
            across_day_result.g2_ave_2(di,1) = h;
            across_day_result.g2_ci_2(1:2, di) = ci;    
            across_day_result.g2_row_2{di,1} = currentL2;
    
        end
    
        Total_across_arrayR.(currentArray) = across_day_result;
    
    end

end
