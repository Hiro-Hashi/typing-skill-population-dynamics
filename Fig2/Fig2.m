%% Code related to Fig. 2

%% Forced alignment with HMM including Blank states

%%%%% Notes:
% Because the RNN predicts probabilities for 31 classes including the Rest
% state, the HMM-based labeling also uses templates for all 31 classes
% (Blank = Rest).
%
% Incorporating Blank states may improve the estimation of key onset and
% offset times.
%
% The HMM-based forced alignment procedure was adapted from the handwriting
% decoding method described in:
%
% High-performance brain-to-text communication via handwriting
% F. R. Willett, D. T. Avansino, L. R. Hochberg, J. M. Henderson, and K. V. Shenoy
% Nature 2021, Vol. 593, Issue 7858, pp. 249-254

%%%%%%%%%%%%%%%%% HMM forced-alignment procedure

% 1). Load the refined templates adapted to the session.
%     The templates are refined using forced_alignment_HMM_labeling.

% 2). Each Finger Type trial consists of three phases: Cue -> Go -> Rest.
%     Neural activity during the Rest period is used to convert firing rates
%     (FR) into z-scored firing rates (z-FR).

% 3). HMM labeling may fail when the neural data are too short relative to
%     the total duration of Mu.
%     Use adjust_length_template to adjust the template duration.
%     This step is important for successful alignment.

% 4). Using the templates and ground-truth token sequence, construct the
%     initial-state probabilities (pi0), state-transition probabilities (A),
%     and emission means (Mu; mean firing-rate vector for each state).
%     The emission mean for each state (i.e., each temporal sample within a
%     template) represents the expected firing-rate pattern to be compared
%     with the observed z-FR.
%     Implemented in build_hmm_from_templates. (Corresponds to Willett Step 1.)

% 5). Based on the Euclidean distance between the downsampled observed z-FR
%     (50-ms bins) and the mean vector Mu for each state, calculate the
%     multivariate Gaussian log-likelihood (logB).
%     logB is a [time x state] matrix representing the likelihood of the
%     observed neural activity under each HMM state.
%     Implemented in log_emission_gauss. (Corresponds to Willett Step 1.)

% 6). Constrain the temporal window in which each token can occur using
%     apply_time_window_constraint.
%     Constrain the sequence to terminate at the predefined final state using
%     apply_terminal_constraint. (Corresponds to Willett Step 2.)

% 7). Use Viterbi decoding with A, logB, and pi0 to estimate the most likely
%     sequence of HMM states over time.
%     Implemented in viterbi_log. (Corresponds to Willett Step 2.)

% 8). Temporally stretch/compress and shift each token template and perform
%     a grid search for the onset and offset times that maximize the mean
%     correlation with the observed neural activity.
%     This procedure refines the token boundaries estimated by the HMM.
%     Implemented in refine_segments_by_grid. (Corresponds to Willett Step 3.)


%% Load data

clc
close all
clear

projectDir = string(pwd);
ClosedData = fullfile(projectDir, 'data', 'day45ClosedTyping.mat');
segD = load(ClosedData);


% Load the previously generated 50-ms templates for all 31 classes,
% including the Blank class.
tempData = fullfile(projectDir, 'data', 'templateData', 'tempData.mat');
templateOrig = load(tempData);

FingerLabel = templateOrig.MyTemplateFinger.FingerLabel;
KeyboardLabel = templateOrig.MyTemplateFinger.KeyboarLabel;
originalTemplate = templateOrig.MyTemplateFinger.template50ms;

% Refined template data
RtempData = fullfile(projectDir, 'data', 'templateData', 'day45RefinedTemplate.mat');
refined_temp_d = load(RtempData);

% The refined templates do not contain the None state;
% therefore, this state is retained from the Original Templates.
refined_template = originalTemplate;
refined_template(2:31) = refined_temp_d.refined_template; 


%% Fig2B, Perform HMM labeling using templates generated from the Finger Sweep
% and visualize the results after refining the onset and offset times


% Parameters for converting neural activity to firing rates
sigma_bin = 4; % Gaussian kernel SD = 40 ms
               % (not 30 ms; corresponds to the Sentence task in Willett et al.)

bin_sec = 0.01; % Enter 0.01 s when the data are sampled in 10-ms bins

% Number of cues in the loaded session
cueNum = numel(segD.transcriptions);

% Extract firing rates during the Rest state across the session
% to calculate the Rest-state mean and standard deviation.
total_rest_FR = my_rest_state_FR_get(cueNum, segD, sigma_bin, bin_sec);

% Calculate the session-level mean and standard deviation of Rest-state FR.
% These values are used to convert firing rates during typing to z-FR.
aveFR_session = mean(total_rest_FR);
stdFR_session = std(total_rest_FR);

% Prevent division by zero when the standard deviation is zero
eps_z = 1e-6;
stdFR_session = max(stdFR_session, eps_z);

clc
close all

% Sentence #30 is shown in Fig. 2B
currentTranscript = 30;

trueLabel = segD.transcriptions{currentTranscript};
fprintf(['True sentence   : ', char(trueLabel), '\n'])


%%%%%%%%%%%%%%% Step1. Tokenize the sentence and assign template indices
[sent_idx, tokens] = true_sentence_convert_idx_includingBlank(trueLabel, KeyboardLabel);

%%%%%%%%%%%%%%% Step2. Extract ncTX during typing, convert it to FR,
% and z-score it using the mean and standard deviation of the Rest-state FR.

ncTX = segD.TXgo{currentTranscript};

FR = my_ncTX_FR_convert(ncTX, sigma_bin, bin_sec);

zFR = (FR - aveFR_session)./stdFR_session;

%%%%%%%%%%%%%%% Step3.Downsample the z-scored FR from 10-ms to 50-ms bins.
% Specify the downsampling factor.
% For example, use downRate = 5 to reduce the sampling rate by a factor of 5.
downRate = 5;
dw_zFR = down_sample_FR(zFR, downRate);
T = size(dw_zFR, 1);

%%%%%%%%%%%%%%% Step4. Select the templates used for HMM alignment
used_Template = refined_template; 

% Adjust the template lengths only when Mu is too long relative to the
% available neural data. The adjustment is determined automatically
% within the function.
%%%%%% Note: Reducing the template duration using adjust_length_template
% enables successful labeling. %%%%%%
adj_Template = adjust_length_template(used_Template, sent_idx, T, 0.2, 2);

%%%%%%%%%%%%%%% Step.5 Construct the HMM parameters.
% Generate the mean firing-rate vectors (Mu) from the templates and the
% state-transition matrix (A) from sent_idx.
[pi0, A, Mu, slices, metaD] = build_hmm_from_templates(adj_Template, sent_idx);

%%%%%%%%%%%%%%% Step.6  Calculate the multivariate Gaussian log-likelihood
% (logB) from the observed neural activity and the mean template vector (Mu)
% for each state.

% logB is a [time x state] matrix of log probability densities,
% corresponding to the log emission likelihoods.
sigma2 = 1;   % Covariance = sigma2*I (identity covariance in the referenced method).
              % Because z-scored signals are used, sigma2 is set to 1.

logB = log_emission_gauss(dw_zFR, Mu, sigma2);

%%%%%%%%%%%%%%% Step.7 Apply temporal constraints and perform Viterbi decoding
%%%%%%% Temporal-window constraint:
% Center the j-th token around (j/M)*T and allow a window of +/-0.3T.
logB = apply_time_window_constraint(logB, slices, 0.3); 

%%%%%%% Terminal-state constraint:
% Force the sequence to terminate at the final state of the last Blank token.
logB = apply_terminal_constraint(logB, metaD);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Decode the most likely state sequence
[z_path, delta, psi] = viterbi_log(pi0, A, logB);   

%%%%%%%%%%%%%%% Step.8 Local refinement of token onset and offset times
% (Willett Step 3, Methods p. 20)

% Reconstruct the tokens (excluding Blank periods) and their onset (t0)
% and offset (t1) times from the Viterbi state path.
segments = z_path_to_segments(z_path, slices, tokens);

% Using the observed activity (50-ms bins, [T x C]) and templates
% (cell array of [Ti x C]), refine the onset time and duration (stretch)
% of each token.
dt = 0.05;  % Duration of one bin (50 ms) in the data provided to the HMM

% Parameters for the grid search
refineOpt = struct( ...
    'maxShiftSec', 0.5, ...   % ±0.5 s
    'stepSec',     0.05, ...  % 0.05-s increments
    'stretchGrid', 0.4:0.0786:1.5 ...  % 0.4 ～ 1.5
);

% Refine the onset and offset times by temporally stretching/compressing
% and shifting each template.
segments_refined = refine_segments_by_grid(dw_zFR, adj_Template, sent_idx, segments, dt, refineOpt, KeyboardLabel);

%%%%%%%%%%%%%%% Step.9 Convert the refined results to a table
% Store the refined segment information in a table
TBL = my_segments_refined_converter(segments_refined, dt);

% Reconstruct the decoded character sequence
decodedS = make_decoded_copy_typing(TBL);
fprintf(['Decoded sentence: ', char(lower(decodedS)), '\n'])

% Calculate the accuracy of the HMM output relative to the ground-truth labels
% Convert the HMM output to a string array
outPut = string(TBL.Label)';

% Remove Blank tokens
refined_tokens = remove_blank_from_tokens(tokens);
[editDist, CER] = char_error_rate(refined_tokens, outPut);
disRe = sprintf('HMM: Edit Dist. %d, CER %.2f%%', round(editDist), CER*100);
disp(disRe)


%%%%%%%%%%%% Visualize the results
% Bin duration of the z-FR data (10 ms)
bin_sec_zFR = 0.01; 

figure
t = tiledlayout(2, 1, 'TileSpacing','compact', 'Padding','compact');
all_ax = gobjects(2);

% Step 1. Plot neural activity (RMS of z-FR)
all_ax(2) = nexttile(2);
timeData = my_rms_FR_plot(zFR, bin_sec_zFR);   
ax = gca;
yl = ylim;
ylim(ax, yl)

% Step 2. Plot the HMM output
token_output_plot(ax, 0.9, TBL, timeData, [0 0 1]);

% Step 3. Visualize the RNN output stored in decoder_logit_output
% Retrieve the logit output from the RNN
current_logit_output = segD.decoder_logit_output{currentTranscript};
current_logit_output_time = segD.Decoder_output_after_go_cue_sec{currentTranscript};

RNN_outputTBL = rnn_output_convert_table(current_logit_output, current_logit_output_time);

% Reconstruct the sequence decoded by the RNN
decodedRNN = make_decoded_copy_typing(RNN_outputTBL);
RNN_output = string(RNN_outputTBL.Label)';
[editDist, CER] = char_error_rate(refined_tokens, RNN_output);
disRNN = sprintf('RNN: Edit Dist. %d, CER %.2f%%', round(editDist), CER*100);
disp(disRNN)

% Plot the RNN output
token_output_plot(ax, 0.8, RNN_outputTBL, timeData, [1 0 0]);

% Display the current condition and decoding results
dispText = {char("#" + string(currentTranscript)), ...
            ['Ref: ', trueLabel], ...
            ['HMM: ', char(lower(decodedS))], ...
            ['RNN: ', char(lower(decodedRNN))], ...
            disRe, disRNN};
xlim([0, timeData(end)*1.2])
xl = xlim;
text(max(xl)*0.83, max(yl)*0.85, dispText)


% ---- Create the color legend ----
% Create dummy graphics objects containing NaN values so that they are
% displayed only in the legend and not in the plot.
h_color(1) = fill(nan, nan, [0 0 1], 'EdgeColor','none', 'FaceAlpha', 0.1);
h_color(2) = fill(nan, nan, [1 0 0], 'EdgeColor','none', 'FaceAlpha', 0.1);

LegendName = {'HMM', 'RNN'};
L = legend(h_color, LegendName, 'Location','best', 'Box', 'off');
L.FontSize = 12;
L.AutoUpdate = 'off';

% Step 4. Plot the n-gram output
all_ax(1) = nexttile(1);
ax = gca;
decodedList = ngram_output_plot(segD, currentTranscript, ax, 0.8, 0.2, 12);
xlim([0, timeData(end)*1.2])
ax.TickLength = [0 0];
ax.XAxis.Visible = 'off';
ax.YAxis.Visible = 'off';
pbaspect([100 1 1])


%% Label neural data during closed-loop typing using HMM-based forced alignment

outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

TrialDayList  = ["45", "85", "86", "92", "93", "106", "107"];

TrialNum = numel(TrialDayList);

for ti = 1:TrialNum
    
    % Data load
    dataPath = fullfile(projectDir, 'data', 'closedLoopData', char("day" + TrialDayList(ti) + ".mat"));
    segD = load(dataPath);

    %%%%%%%%%%%%%% Load the refined templates
    dataPath = fullfile(projectDir, 'data', 'templateData', char("day" + TrialDayList(ti) + "RefinedTemplate.mat"));
    refined_temp_d = load(dataPath);

    % The refined templates do not contain the None state;
    % therefore, this state is retained from the Original Templates
    refined_template = originalTemplate;
    refined_template(2:31) = refined_temp_d.refined_template; %
    
    
    sigma_bin = 4; 
    bin_sec = 0.01; 
    

    cueNum = numel(segD.transcriptions);
    

    total_rest_FR = my_rest_state_FR_get(cueNum, segD, sigma_bin, bin_sec);
    

    aveFR_session = mean(total_rest_FR);
    stdFR_session = std(total_rest_FR);

    eps_z = 1e-6;
    stdFR_session = max(stdFR_session, eps_z);
    

    Results = struct();

    for currentTranscript = 1:cueNum
    
        fprintf('Count: %d/%d\n', currentTranscript, cueNum);
         % Obtain the HMM and RNN outputs for the current sentence
        [TBL, rnnTBL, tokens] = tbl_and_rnn_tbl_make(segD, currentTranscript, sigma_bin, bin_sec, aveFR_session, stdFR_session, refined_template, KeyboardLabel);
        
        % Obtain the number of tokens predicted by the RNN and the time
        % required to type the presented sentence
        typedNum = height(rnnTBL); % Number of tokens predicted by the RNN while typing the presented sentence
        typedSec = rnnTBL.End_s(end); % Time required to type the presented sentence
        
        refined_tokens = remove_blank_from_tokens(tokens);
        % Calculate the character error rates for the HMM and RNN outputs
        [ED, CER] = char_error_rate(refined_tokens, string(TBL.Label)');
        [rED, rCER] = char_error_rate(refined_tokens, string(rnnTBL.Label)');
        
        % Compare the token intervals identified by the HMM with the RNN outputs.
        % Retain only matched tokens for which the HMM-derived onset precedes
        % the corresponding RNN-derived onset.
        % The RNN output is treated as the reference label.
        % A maximum difference of 1.5 s between the HMM and RNN token onset
        % times is allowed.
        [TP] = compare_tbl_rnntbla(TBL, rnnTBL, 0, 1.5); % Third argument: minimum required HMM lead time;
                                                         % fourth argument: maximum allowed HMM lead time
        Results.TBL{currentTranscript} = TBL;
        Results.rnnTBL{currentTranscript} = rnnTBL;
        Results.tokens{currentTranscript} = refined_tokens;
        Results.ED(currentTranscript) = ED;
        Results.CER(currentTranscript) = CER;
        Results.rED(currentTranscript) = rED;
        Results.rCER(currentTranscript) = rCER;
        Results.TP{currentTranscript} = TP;
        Results.matchTokenNum(currentTranscript) = height(TP);
        Results.matchTokenPro(currentTranscript) = height(TP)/numel(refined_tokens);
        Results.typedNum(currentTranscript) = typedNum;
        Results.typedSec(currentTranscript) = typedSec;
    
    end
    

    Results.TrialDay = "day" + TrialDayList(ti);
    
    saveName = "HMM RNN matched list day" + TrialDayList(ti) + ".mat";
    save(fullfile(outputDir, char(saveName)), 'Results');
    disp('Results have been saved.')

end


%% Fig.2C and D

% Load data
DaysList   = ["45", "85", "86", "92", "93", "106", "107"];
DataNum = numel(DaysList);

TypingTime = struct();

for i = 1:DataNum
    loadDatapath = fullfile(projectDir, 'data', 'timeData', char("day" + DaysList(i) + ".mat"));
    TypingTime.("D" + DaysList(i)) = load(loadDatapath);

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

% Labels used in the keyboard task, corresponding to the Finger Sweep labels
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

% Analyze the number and frequency of token occurrences.
% Also calculate typing duration, defined as the duration of the neural-data
% segment corresponding to each token.

color_value = colororder("gem12");


% Organize variables used for the analysis
num_token = numel(KeyboardLabel);
token_occurrence = zeros(num_token, DataNum);   % Number of extracted tokens [Token x Day]
token_type_speed_ave = nan(num_token, DataNum); % Mean typing duration for each token
token_type_speed_ci1 = nan(num_token, DataNum);
token_type_speed_ci2 = nan(num_token, DataNum);

for di = 1:DataNum
    currentField = "D" + DaysList(di);

    for ti = 1:num_token
        currentData = TypingTime.(currentField).TypingT.("T" + string(ti));
        if isempty(currentData)
            continue
        else
            token_occurrence(ti, di) = numel(currentData); % Number of occurrences of the current token
            [h, ~, ci] = normfit(currentData);             % Calculate the mean typing duration for the current token
            token_type_speed_ave(ti, di) = h;
            token_type_speed_ci1(ti, di) = ci(1);
            token_type_speed_ci2(ti, di) = ci(2);
        end
    end
end

% Calculate the proportion of each token relative to the total number
% of matched tokens within each session
token_occurrence_prop = (token_occurrence./sum(token_occurrence))*100;

DaysList2 = "D" + DaysList;

% Visualize token counts and token occurrence proportions
figure
% AX1: Total number of extracted tokens across sessions
AX1 = subplot(2, 4, 1);
AX1.FontName = 'Helvetica';
B = bar(sum(token_occurrence), 'FaceColor','flat');
B.CData = color_value(1, :);
hold on, plot(1:DataNum, sum(token_occurrence), 'ko-', 'LineWidth',1.5)
xticks(1:DataNum), xticklabels(DaysList2)
xlim([0 DataNum+1])
box off
title({'Total number of correctly', 'matched tokens per day', ''}, 'FontSize', 14, 'FontName', 'Helvetica')
ylabel('Number of matched tokens', 'FontSize',12, 'FontName', 'Helvetica')
xlabel('Session day', 'FontSize',12, 'FontName', 'Helvetica')

% AX2: Number of occurrences for each of the 30 tokens across sessions
AX2 = subplot(2, 4, 2);
AX2.FontName = 'Helvetica';
B = bar(token_occurrence, 'FaceColor','flat');
for i = 1:DataNum
    B(i).CData = color_value(i, :);
end
xticks(1:num_token)
xticklabels(KeyboardLabel)
legend(DaysList2, 'Box','off', 'Location','bestoutside', 'FontSize',10)
box off
AX2.TickLength = [0.001 0.001];
title('Token counts per day (HMM-RNN matched)', 'FontSize', 14, 'FontName', 'Helvetica')
ylabel('Number of tokens', 'FontSize',12, 'FontName', 'Helvetica')
xlabel('Token type', 'FontSize',12, 'FontName', 'Helvetica')
AX2.XAxis.FontSize = 12;


% AX3: Proportion of occurrences for each of the 30 tokens across sessions
AX3 = subplot(2, 4, 6);
AX3.FontName = 'Helvetica';
B = bar(token_occurrence_prop, 'FaceColor','flat');
for i = 1:DataNum
    B(i).CData = color_value(i, :);
end
xticks(1:num_token)
xticklabels(KeyboardLabel)
legend(DaysList2, 'Box','off', 'Location','bestoutside', 'FontSize',10)
box off
AX3.TickLength = [0.001 0.001];
title('Token proportion per day (normalized by total matched tokens)', 'FontSize', 14, 'FontName', 'Helvetica')
ylabel('Proportion (%)', 'FontSize',12, 'FontName', 'Helvetica')
xlabel('Token type', 'FontSize',12, 'FontName', 'Helvetica')
AX3.XAxis.FontSize = 12;
% Friedman test: nonparametric test for differences among three or more
% related groups
p = friedman(token_occurrence_prop, 1, 'off');
distext = sprintf('\\itp\\rm = %.2f (Friedman test)', p);
text(AX3, 24, 20, distext, 'FontName','Helvetica', 'FontSize',14)

% Adjust subplot positions
AX1.Position = [0.12    0.2    0.08    0.58];
AX2.Position = [0.25    0.6    0.6    0.3412];
AX3.Position = [0.25    0.1    0.6    0.3412];

%% Fig.2E, F, G, H, and I
color_value = colororder("gem12");
TrialDayList  = ["45", "85", "86", "92", "93", "106", "107"];
TrialNum = numel(TrialDayList);

All_HMM_RNN = struct();
for i = 1:TrialNum
    dataPath = fullfile(projectDir, 'results', char("HMM RNN matched list day" + TrialDayList(i) + ".mat"));
    All_HMM_RNN.("D" + TrialDayList(i)) = load(dataPath);

end


FieldList = fieldnames(All_HMM_RNN);
FieldNum = numel(FieldList);

figure
t = tiledlayout(1, 5, 'TileSpacing','compact', 'Padding','compact');

% Fig.2E
% HMM-CER
[x_ED, all_y] = boxchart_data_make("ED", FieldList, All_HMM_RNN);
cat_y = categorical(all_y, FieldList, 'Ordinal', true);
catList = categories(cat_y);
catNum = numel(catList);
x_CER = boxchart_data_make("CER", FieldList, All_HMM_RNN);
x_CER = x_CER * 100;
nexttile(1)
boxchart(cat_y, x_CER)
title ({'HMM forced-alignment', 'labeling'})
ylabel('CER (%)')
meanD = get_mean_based_categor(x_CER, cat_y);
hold on
plot(1:catNum, meanD, 'ko-', 'LineWidth',1)
ylim([-10 105])

% Fig.2F
% RNN-CER
x_rCER = boxchart_data_make("rCER", FieldList, All_HMM_RNN);
x_rCER = x_rCER * 100;
nexttile(2)
boxchart(cat_y, x_rCER)
title ('RNN output')
ylabel('CER (%)')
meanD_rCER = get_mean_based_categor(x_rCER, cat_y);
hold on
plot(1:catNum, meanD_rCER, 'ko-', 'LineWidth',1)
ylim([-5 68])
p = kruskalwallis(x_rCER, cat_y, 'off'); 
if p > 0.001
    distext = sprintf('\\itp\\rm = %.3f (Kruskal–Wallis test)', p);
else
    distext = sprintf('\\itp\\rm < 0.001 (Kruskal–Wallis test)');
end
text(3, 55, distext, 'FontName','Helvetica', 'FontSize',14)

% Fig.2G
% Proportion of tokens matched between the HMM and RNN outputs
x_matchTokenPrp = boxchart_data_make("matchTokenPro", FieldList, All_HMM_RNN);
x_matchTokenPrp = x_matchTokenPrp * 100;
nexttile(3)
boxchart(cat_y, x_matchTokenPrp)
title ({'Matched tokens between', 'HMM and RNN'}, 'FontSize',10)
ylabel('Proportion of matched tokens (%)')
meanD_matched = get_mean_based_categor(x_matchTokenPrp, cat_y);
hold on
plot(1:catNum, meanD_matched, 'ko-', 'LineWidth',1)
yticks([10:10:100])
ylim([10 105])

p = kruskalwallis(x_matchTokenPrp, cat_y, 'off'); 
if p > 0.001
    distext = sprintf('\\itp\\rm = %.3f (Kruskal–Wallis test)', p);
else
    distext = sprintf('\\itp\\rm < 0.001 (Kruskal–Wallis test)');
end
text(1, 20, distext, 'FontName','Helvetica', 'FontSize',14)

% Fig.2H
nexttile(4)
plot(1:catNum, meanD_matched, 'o-', 'LineWidth',2, 'Color', color_value(1, :))
ylabel('Matched Proportion (%)')
ylim([50 101])
yyaxis right
plot(1:catNum, meanD_rCER, '*-', 'LineWidth',2, 'Color', color_value(2, :))
hold on
plot(1:catNum, 100 - (meanD_rCER + meanD_matched), '*-', 'LineWidth',2, 'Color', color_value(3, :))
ylabel('CER/Temporal mismatch rate (%)')
ax = gca;
ax.YAxis(2).Color = [0 0 0];
legend({'Matched Proportion', 'RNN CER',  'Temporal Mismatch rate'}, 'Box','off', 'Location', 'bestoutside', 'AutoUpdate','off')
xlim([0 catNum+1])
xticks(1:catNum)
xticklabels(FieldList)
box off
ylim([0 35])

% Fig.2I
% Number of characters typed per minute, estimated from the RNN output tokens
% without considering whether each predicted token was correct
x_NumType = boxchart_data_make("typedNum", FieldList, All_HMM_RNN);
x_SecType = boxchart_data_make("typedSec", FieldList, All_HMM_RNN);
x_ch_per_m = round((x_NumType ./x_SecType)*60); % Number of characters typed per minute, rounded to integers
nexttile(5)
boxchart(cat_y, x_ch_per_m)
title ({'Typing speed estimated ', 'from RNN output'}, 'FontSize',12)
ylabel('Characters per minute (CPM)')
meanD = get_mean_based_categor(x_ch_per_m, cat_y);
hold on
plot(1:catNum, meanD, 'ko-', 'LineWidth',1)
ylim([20 130])
yticks([20:20:120])
p = kruskalwallis(x_ch_per_m, cat_y, 'off'); 
if p > 0.001
    distext = sprintf('\\itp\\rm = %.3f', p);
else
    distext = sprintf('\\itp\\rm < 0.001');
end
text(1, 100, {distext, '(Kruskal–Wallis test)'}, 'FontName','Helvetica', 'FontSize',14)

%% Fig. 2J

figure
tiledlayout(2, 4, 'TileSpacing','compact','Padding','compact')
all_axes = gobjects(2*4);

% Plot within-day changes as the sentence sequence progresses,
% after grouping the data using the specified bin size
bin_num = 10; % Number of consecutive sentences grouped into each bin
[all_axes] = plot_values_withindata_acrossday_by_sentences(x_ch_per_m, catList, cat_y, bin_num, 'Characters per minute (CPM)', FieldList, [20 110]);

all_axes{1}.Position = [ 0.14    0.11    0.05    0.8];
all_axes{2}.Position = [ 0.2    0.11    0.05    0.8];
all_axes{3}.Position = [ 0.26   0.11    0.05    0.8];
all_axes{4}.Position = [ 0.32    0.11    0.05    0.8];
all_axes{5}.Position = [ 0.38    0.11    0.05    0.8];
all_axes{6}.Position = [ 0.44    0.11    0.05    0.8];
all_axes{7}.Position = [ 0.5   0.11    0.05    0.8];



%% Fig.2K and L

FieldList = fieldnames(All_HMM_RNN);
FieldNum = numel(FieldList);


% Proportion of RNN tokens matched to HMM tokens
[x_matchTokenPrp, all_y] = boxchart_data_make("matchTokenPro", FieldList, All_HMM_RNN);
x_matchTokenPrp = x_matchTokenPrp * 100;
cat_y = categorical(all_y, FieldList, 'Ordinal', true);

% Number of characters typed per minute, estimated from RNN output tokens
% without considering whether each predicted token was correct
x_NumType = boxchart_data_make("typedNum", FieldList, All_HMM_RNN);
x_SecType = boxchart_data_make("typedSec", FieldList, All_HMM_RNN);
x_ch_per_m = round((x_NumType ./x_SecType)*60); 

catList = categories(cat_y);
catNum = numel(catList);

figure

subplot(1, 9, 1)
gs = gscatter(x_matchTokenPrp, x_ch_per_m, cat_y, [[0 0.4470 0.7410]; [0.8500 0.3250 0.0980]; [0.8500 0.3250 0.0980]; ...
    [0.9290 0.6940 0.1250]; [0.9290 0.6940 0.1250]; [0.4660 0.6740 0.1880]; [0.4660 0.6740 0.1880]]);
set(gs, 'MarkerSize',15)
xlabel('Proportion of matched tokens (%)')
ylabel('Characters per minute (CPM)')
legend('Box', 'off', 'AutoUpdate','off', 'Location', 'bestoutside')
ax1 = gca;


% Perform linear regression between CPM and accuracy
mdl = fitlm(x_matchTokenPrp, x_ch_per_m);
xFit = linspace(50, 99, 100);
yFit = mdl.Coefficients.Estimate(1) + mdl.Coefficients.Estimate(2) * xFit;
hold on
plot(xFit, yFit, 'r-', 'LineWidth',4)

% Calculate the correlation
[r, p] = corr(x_matchTokenPrp, x_ch_per_m);
if p > 0.001
    corr_disp = sprintf('r = %.2f (\\it{p}\\rm = %.3f)', r, p);
else
    corr_disp = sprintf('r = %.2f (\\it{p}\\rm < 0.001)', r);
end
r2 = mdl.Rsquared.Ordinary;
r2_disp = sprintf('R^2 = %.2f', r2);
text(10, 130, {corr_disp, r2_disp })


% Examine the relationship between accuracy and typing speed within each day
slopeList = nan(catNum,1);
rList = nan(catNum,1);
pList = nan(catNum,1);

for i = 1:catNum
    currentCat = catList{i};
    ind = cat_y == currentCat;

    matD = x_matchTokenPrp(ind);
    cpmD = x_ch_per_m(ind);

    % Pearson correlation coefficient
    [r, p] = corr(matD, cpmD, 'Type', 'Pearson');
    rList(i) = r;
    pList(i) = p;

    % Linear regression: y = a*x + b
    mdl = fitlm(matD, cpmD);
    slopeList(i) = mdl.Coefficients.Estimate(2);

    % Plot the data and regression line
    subplot(1, 9, i+1)
    scatter(matD, cpmD, 20, 'filled', 'MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.6)
    hold on
    xFit = linspace(min(matD), max(matD), 100);
    yFit = mdl.Coefficients.Estimate(1) + mdl.Coefficients.Estimate(2)*xFit;
    plot(xFit, yFit, 'r', 'LineWidth',1.5)
    title(sprintf('%s', currentCat))
    disp = sprintf('r = %.2f', r);
    if p > 0.001
        disr = sprintf('(\\it{p}\\rm = %.3f)', p);
    else
        disr = sprintf('(\\it{p}\\rm <0.001)');
    end
    r2_disp = sprintf('slope = %.2f', slopeList(i));

    text(5, 130, {disp, disr, r2_disp}, 'FontSize',10)

    if i == 4
        xlabel('Proportion of matched tokens (%)')
    end

    if i == 1
        ylabel('Characters per minute (CPM)')
    end

    if i ~= 1
        ax = gca;
        ax.YAxis.Visible = 'off';
    end
    grid on
    pbaspect([1, 3, 1])

    xlim([0 100])
    ylim([20 145])

    axAll{i} = gca;
end

%% Internal function

function [total_rest_FR] = my_rest_state_FR_get(cueNum, segD, sigma_bin, bin_sec)
    total_rest_FR = [];
    
    for ci = 1:cueNum

        clippedData = segD.TXrest{ci};
        if isempty(clippedData)
            continue
        elseif size(clippedData, 1) < 10
            continue
        end
    
        clippedFR = nan(size(clippedData));
        chNum = size(clippedFR, 2);
        for chi = 1:chNum
            txData = clippedData(:, chi);
            FR_s = my_gaussKernel_convert(sigma_bin, bin_sec, txData, true);
            clippedFR(:, chi) = FR_s;
        end
        
        total_rest_FR = [total_rest_FR; clippedFR];
    
    end
end

function fr_hz = my_gaussKernel_convert(sigma_bin, bin_sec, sigD, to_hz)
    

    
    if nargin < 4
        to_hz = true;
    end


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
    fr_hz = tmp(startIdx:endIdx);

    if to_hz
        fr_hz = fr_hz / bin_sec;
    end

end

function [sent_idx, tokens] = true_sentence_convert_idx_includingBlank(trueLabel, KeyboardLabel)


    blankToken = "Blank";  
    bk_idx = find(KeyboardLabel == blankToken);

    txt = char(upper(trueLabel));
    raw_tokens = strings(0);

    % ==== Step 1. tokenize====
    for i = 1:length(txt)
        ch = txt(i);
        if ch== ' '
            raw_tokens(end+1) = "Sp";
        else
            raw_tokens(end+1) = string(ch);
        end
    end

    % ==== 2) KeyboardLabelに無いトークンは無視 ====
    valid_tokens = strings(0);  % 有効なトークンだけ残す

    for i = 1:numel(raw_tokens)
        t = raw_tokens(i);
        if any(KeyboardLabel == t)
            valid_tokens(end+1) = t;
        else
            fprintf('Ignore: Token not found in the template "%s" (Location %d)\n', t, i);
        end
    end

    % ==== 3) 先頭/間/末尾に Blank を挿入 ====
    tokens = strings(0);
    sent_idx = [];

    % 先頭にBlank
    tokens(end+1) = blankToken;
    sent_idx(end+1) = bk_idx;

    % 各トークンの前後に Blank を挿入（= 前のBlankはすでに入っているので、トークン→Blankの順）
    for i = 1:numel(valid_tokens)
        t = valid_tokens(i);
        k = find(KeyboardLabel == t, 1);  % 必ず見つかるはず
        tokens(end+1)   = t;
        sent_idx(end+1) = k;

        % 後ろにBlank
        tokens(end+1)   = blankToken;
        sent_idx(end+1) = bk_idx;
    end

end

function [FR] = my_ncTX_FR_convert(ncTX, sigma_bin, bin_sec)

    FR = nan(size(ncTX));
    chNum = size(ncTX, 2);
    for chi = 1:chNum
        clippedD = ncTX(:, chi);
        FR_s = my_gaussKernel_convert(sigma_bin, bin_sec, clippedD, true);
        FR(:, chi) = FR_s;
    end

end

function [dw_zFR] = down_sample_FR(zFR, downRate)
    [dataNum, chNum] = size(zFR);
    dw_dataNum = floor(dataNum/downRate);
    dw_zFR = nan(dw_dataNum, chNum);
    
    for i = 1:dw_dataNum
        s_ind = (i-1)*downRate+1;
        e_ind = i*downRate;
        clippedData = zFR(s_ind:e_ind, :);
        dw_zFR(i, :) = mean(clippedData); % 平均化してdown sample
    end

end

function [adj_Template] = adjust_length_template(templates50ms, sent_idx, T, w_blank, min_states)

   
    K = numel(sent_idx);
    Tk = zeros(K,1);
    isBlankTok = false(K,1);
    for k = 1:K
        idx = sent_idx(k);
        Tk(k) = size(templates50ms{idx},1);
        isBlankTok(k) = (idx==1); 
    end


    L_mu = sum(Tk);

    if L_mu/T < 1.2
        adj_Template = templates50ms;
        disp('The length of the template was not adjusted.')
        return;
    end


    if L_mu <= 0 
        adj_Template = templates50ms; 
        return; 
    end
    

    g = T / L_mu;
    
    adj_Template = templates50ms;
    for k = 1:K
        idx = sent_idx(k);
        Tin = size(templates50ms{idx},1);
        if isBlankTok(k)
            gk = max((g * w_blank), min_states/Tin);
        else
            gk = max(g, min_states/Tin);
        end
        Lout = max(min_states, round(Tin * gk));
        adj_Template{idx} = stretch_template_linear(templates50ms{idx}, Lout);
    end

    disp('The length of the template was adjusted.')
end

function [pi0, A, Mu, slices, metaD] = build_hmm_from_templates(templates50ms, sent_idx)
    
    % Mu: Mean firing-rate vectors derived from the templates (emissions)
    % A: State transition matrix
    
    % Construct a left-to-right HMM for each token,
    % with the number of states determined by the number of template rows
    J = numel(sent_idx); % Number of tokens
    C = size(templates50ms{1},2);

    % Determine the number of transition states for each token
    Tk = zeros(J,1);
    for j=1:J
        Tk(j)=size(templates50ms{sent_idx(j)},1);
    end
    S  = sum(Tk);

    % Check whether the first and last tokens are Blank (optional)
    if sent_idx(1) ~= 1 || sent_idx(end) ~= 1
        warning('The first and last elements of sent_idx should be Blank (=1).');
    end
    
    % Initialize metadata and HMM parameters
    Mu     = zeros(S,C);
    labels = strings(S,1);
    gidx   = zeros(S,1);
    A      = zeros(S,S);
    slices = cell(J,1);
    
    % Loop over tokens
    offset = 0;
    for j = 1:J
        idx = sent_idx(j);
        Tj  = Tk(j); % Number of states for the current token
        si  = (offset+1):(offset+Tj); % Global state indices
        Mu(si,:) = templates50ms{idx};     % Emission mean
        labels(si) = "g"+ idx + "#" + string(1:Tj);
        gidx(si)   = idx;
        if idx == 1 % Blank token
            slices{j} = [];
        else
            slices{j} = si;
        end
        % slices{j} = si;
    
        isBlank = (idx == 1);  % Flag indicating whether the current token is Blank
        
        % State transitions: stay / next / skip one state
        if isBlank % ---- Blank block ----
            for x = 1:Tj
                s = si(x);
                if x < Tj -2
                    A(s, s)   = 0.2;
                    A(s, s+1) = 0.6; 
                    A(s, s+2) = 0.2; 
                elseif x == Tj -1
                    A(s, s) = 0.2;
                    A(s, s+1) = 0.8; % Transition to the final state
                else % End of the block
                    if j < J
                        % Stay with probability 0.5 or transition to the
                        % first state of the next token with probability 0.5
                        A(s,s)        = 0.5;
                        A(s, s+1)     = 0.5;   % First state of the next token
                    else % Final Blank token
                        A(s, s) = 1.0;
                    end

                end
            end

        else % ---- Token block ----

            for x = 1:Tj
                s = si(x);
                if x <= Tj-2 % Intermediate states within the current token
                    A(s,s)     = 0.2; % Stay in the current state
                    A(s,s+1)   = 0.6; % Transition to the next state
                    A(s,s+2)   = 0.2; % Skip one state

                elseif x == Tj-1
                    A(s,s)     = 0.2;
                    A(s,s+1)   = 0.8;

                else
                    if j < J-1 % Not the final non-Blank token
                        A(s,s)    = 0.2;
                        A(s, s+1) = 0.8;   % Transition to the next Blank
                    else % Final state of the last non-Blank token
                        A(s,s)    = 0.8;
                        A(s, s+1) = 0.2;   % Transition to the final Blank
                    end
                end

            end

        end

        offset = si(end);
    end
    
    % Row normalization: ensure that transition probabilities
    % from each state sum to 1
    rowsum = sum(A,2);
    A = A./max(rowsum,eps);
    
    % Initial state distribution:
    % force the sequence to start from the first state
    pi0 = zeros(S,1);
    pi0(1)=1;
    
    metaD.labels = labels;
    metaD.gidx = gidx;
end

function logB = log_emission_gauss(obs, Mu, sigma2)
    % obs: [T x C] observations (50-ms bins)
    % Mu : [S x C] mean vector for each state
    % sigma2: Scalar variance (I*sigma^2). Because the input signals are
    % already z-scored, equal variance across all channels is assumed,
    % and sigma2 = 1 can therefore be used.
    
    [~, C] = size(obs);
    
    % Precompute the constant term of the log probability density
    % for the multivariate Gaussian distribution
    const = -0.5*C*log(2*pi*sigma2);
    
    % Vectorized computation of the isotropic Gaussian log-likelihood
    % between each observation and each state
    % log N(y | mu, sigma^2 I) = const - (1/(2*sigma^2)) * ||y-mu||^2
    % ||y-mu||^2 = ||y||^2 + ||mu||^2 - 2*mu(T)*y
    % This corresponds to the squared Euclidean distance between
    % the observed vector y and the state mean vector mu
    obs2 = sum(obs.^2, 2)';             % [1 x T]、squared norm at each time point, summed across channels
    Mu2  = sum(Mu.^2, 2);               % [S x 1]、squared norm for each state, summed across channels
    cross = Mu * obs';                  % [S x T]
     % Mu2 + obs2 is implicitly expanded to [S x T].
    % Vectorization avoids nested loops over states, time points, and channels,
    % thereby improving computational efficiency
    logB = const - (1/(2*sigma2)) * (Mu2 + obs2 - 2*cross);

    % Transpose to [T x S] for the Viterbi search
    logB = logB.';
end

function logB_masked = apply_time_window_constraint(logB, slices, frac)
    % logB: [T x S]
    % slices: {J x 1}, state ranges corresponding to each token j
    %         (output of build_hmm_from_templates)
    % T: Total number of frames

    if nargin < 3
        frac = 0.3;  % Set the temporal window width to 30% of the total duration
    end

    T = size(logB, 1);

    % Separate non-empty slices (typically non-Blank tokens)
    % from empty slices (Blank tokens)
    nonempty_mask = ~cellfun('isempty', slices);
    nonempty_idx  = find(nonempty_mask);

    J = numel(nonempty_idx);      % Number of non-Blank tokens in the sentence

    win = frac * T;          % Half-width of the temporal window (±30% of T)

    for k = 1:J
        j = nonempty_idx(k);
        tc = (k / J) * T;   % Center time for the current token
        tmin = max(1, round(tc - win));
        tmax = min(T, round(tc + win));

        % Define the allowed temporal window
        allowed = false(T,1);
        allowed(tmin:tmax) = true;

        % Get the state indices corresponding to the current token
        sj = slices{j};

        % Set log-likelihoods outside the allowed window to -Inf
        logB(~allowed, sj) = -Inf;
    end

    logB_masked = logB;
end

function [z, delta, psi] = viterbi_log(pi0, A, logB)
    % Viterbi decoding for an HMM in log space.
    % Using log probabilities converts probability multiplication into
    % addition and improves numerical stability.
    % Although path probabilities decrease as the sequence becomes longer,
    % the Viterbi algorithm selects the path with the highest relative probability.
    %
    % pi0  : [S x 1] Initial state probabilities
    % A    : [S x S] State transition probabilities (linear scale)
    % logB : [T x S] Log emission likelihood for each time point t and state s
    %
    % Outputs:
    % z     : [T x 1] Most likely state sequence (1..S, uint32)
    % delta : [T x S] Maximum log-probability scores
    % psi   : [T x S] Backpointers for reconstructing the optimal state sequence


    [T, S] = size(logB);

    % Convert transition and initial-state probabilities to log space.
    % Zero probabilities remain -Inf to preserve prohibited transitions.
    logA   = -Inf(S,S);
    nzA    = A > 0;
    logA(nzA) = log(A(nzA));

    logPi0 = -Inf(S,1);
    nzP    = pi0 > 0;
    logPi0(nzP) = log(pi0(nzP));

    % Dynamic programming tables (time-major representation)
    delta = -inf(T, S);              % Maximum log-probability of reaching state s at time t
    psi = zeros(T, S, 'uint32');     % Backpointer to the optimal preceding state

    % Initialization: t = 1
    delta(1, :) = (logPi0.').*1 + logB(1, :);

    % Recursion: t = 2,...,T
    for t = 2:T
        % scores(from,to) = delta(t-1,from) + logA(from,to)
        % Dimensions: [S x S] = [S x 1] + [S x S]
        % Rows of logA correspond to from-states and columns to to-states
        scores = logA + delta(t-1, :).';  % [S x S]; addition is used in log space

        % For each to-state (column), obtain the maximum score and
        % the corresponding optimal from-state (row)
        [best, arg] = max(scores, [], 1);               % best: [1 x S], arg: [1 x S]
        delta(t, :) = best + logB(t, :);                % Add emission log-likelihood at time t
        psi(t, :)   = uint32(arg);                      % Store the optimal preceding state
    end

    % Termination: start backtracking from the highest-scoring state at t = T
    [~, zT] = max(delta(T, :));
    z = zeros(T, 1, 'uint32');
    z(T) = uint32(zT);

    % Backtrack from t = T to t = 1 to reconstruct the optimal path
    for t = T:-1:2
        z(t-1) = psi(t, z(t));   % s_{t-1} = psi(t, s_t)
    end
end


% Constrain the final state to the last state of the final Blank token.
% See Methods, p. 20 of the Handwriting paper (Willett F.R. et al. Nature 2021).
function logB_masked = apply_terminal_constraint(logB, metaD)
    % logB: [T x S] Log emission likelihoods
    % metaD: Metadata containing the token index associated with each HMM state

    T = size(logB, 1);

    % Identify the final state of the last Blank token
    blank_states = find(metaD.gidx(:) == 1);   % All states belonging to Blank tokens
    final_state = blank_states(end);           % Final state of the last Blank token
    
    % % At the final time point, prohibit all states except final_state
    logB(T, setdiff(1:size(logB,2), final_state)) = -Inf; 

    logB_masked = logB;
end

% Convert the Viterbi state path into a segment structure to refine
% the estimated onset and offset times.
% Blank periods are excluded from the output.
function segments = z_path_to_segments(z_path, slices, tokens)
    % z_path: [T x 1] decoded state sequence
    % slices: {J x 1}, state-index range corresponding to the j-th token
    % tokens: Token labels corresponding to positions in the sequence
    % Output:
    % segments(k) = struct('TypeK','j','t0','t1','len')

    T = numel(z_path);

    % Blank tokens are represented by empty entries ([]) in slices.
    % Determine the maximum state index using the non-empty slices
    % and the decoded state path.
    max_from_slices = 0;
    if ~isempty(slices)
        nonempty = slices(~cellfun('isempty', slices));
        if ~isempty(nonempty)
            max_from_slices = max(cellfun(@max, nonempty));
        end
    end
    S = max([max_from_slices; double(max(z_path))]);

    % Map each HMM state to its token position j within the sequence
    state2char = zeros(S,1,'uint32');
    J = numel(slices);
    for j = 1:J
        sj = slices{j};
        if ~isempty(sj)
            state2char(sj) = uint32(j);
        end
    end

    % Convert the decoded state sequence into token-position indices
    char_of_t = state2char(z_path);  % [T x 1]
    segments = struct('TypeK', {}, 'j',{},'t0',{},'t1',{},'len',{});

    run_st = 1; 
    k = 0;
    for t = 2:T+1
        if t==T+1 || char_of_t(t) ~= char_of_t(run_st)
            j_curr = double(char_of_t(run_st));  % 0 = Blank; 1..J = non-Blank token positions
            if j_curr == 0 
                run_st = t; % For Blank periods, update the start index and continue
                continue;
            end
            k = k+1;
            segments(k).j   = j_curr;
            segments(k).t0  = run_st;
            segments(k).t1  = t-1;
            segments(k).len = segments(k).t1 - segments(k).t0 + 1;

            % Label used for display
            segments(k).TypeK = string(tokens{j_curr});
            run_st = t;
        end
    end

    % Handle the unlikely case in which the entire decoded path is Blank
    if isempty(segments) && sum(z_path~=1) < 1
        k = k + 1;
        segments(k).TypeK = "Blank";
        segments(k).j   = 1;
        segments(k).t0  = 1;
        segments(k).t1  = t-1;
        segments(k).len = segments(k).t1 - segments(k).t0 + 1;
    end

end


% Refine the onset and offset times by stretching the existing template
% and shifting it in time to find the position with the highest correlation
% between the template and the observed neural activity.
function segments_ref = refine_segments_by_grid(obs, templates50ms, sent_idx, segments, dt, opt, KeyboardLabel)
    % obs: [T x C] observed z-scored firing rates in 50-ms bins
    % templates50ms: Cell array containing the template for each class [Ti x C]
    % sent_idx: [1 x J] token sequence for the sentence
    %           (each token corresponds to a template ID)
    % segments: Segment array derived from the Viterbi path (j, t0, t1, len)
    % dt: Duration of one bin in seconds
    % opt: Structure containing grid-search parameters
    % KeyboardLabel: Labels corresponding to template IDs


    segments_ref = struct();
    [T, chNum] = size(obs);

    maxShiftBins = round(opt.maxShiftSec / dt);
    stepBins     = max(1, round(opt.stepSec / dt));
    timeShiftSeries = -maxShiftBins:stepBins:maxShiftBins;
    timeShiftNum = numel(timeShiftSeries);

    % Loop over tokens
    K = numel(segments);
    for k = 1:K
        j_tok   = segments(k).j;                 % Position within the sentence (1..J)
        cls_id  = sent_idx(j_tok);               % Template ID
        currentLabel = KeyboardLabel{cls_id};    % Corresponding typing token
        temp0   = templates50ms{cls_id};         % [Tin x chNum]
        Tin     = size(temp0,1);                 % Original template length in bins

        % Define boundaries to prevent overlap with neighboring segments
        prev_end  = 0;  
        if k>1
            prev_end  = segments_ref(k-1).t1; % Use the refined endpoint of the preceding segment
        end

        next_start= T+1;
        if k<K
            next_start= segments(k+1).t0;    
        end

        % Get the initial timing information for the current segment
        t0_vit = segments(k).t0;  % Viterbi-derived onset bin
        best_r = -Inf;
        best_t0 = segments(k).t0; % Initial value for the best onset bin
        best_t1 = segments(k).t1; % Initial value for the best offset bin
        best_stretch = NaN;
        best_time_shift = NaN;
        
        % Initialize a matrix to store correlation values from the grid search
        stretchNum = numel(opt.stretchGrid);        
        GridRch = nan(stretchNum, timeShiftNum);

        % Grid searchを行う
        % 伸縮させる程度
        for sf = 1:stretchNum
            currentStretch = opt.stretchGrid(sf);
            L = max(1, round(Tin * currentStretch));   % Stretched template length in bins
            
            % Linearly stretch/compress the template to length L
            templ = stretch_template_linear(temp0, L);  % [L x chNum]

            % Search across temporal shifts
            for sft = 1:timeShiftNum
                current_shift_time = timeShiftSeries(sft);
                t0 = t0_vit + current_shift_time; % Shift the onset time
                t1 = t0 + L - 1;                  % Offset of the stretched template

                % Enforce recording boundaries and prevent overlap with
                % neighboring segments
                if t0 < 1 || t1 > T
                    continue
                end

                if t0 <= prev_end || t1 >= next_start
                    continue
                end

                % Extract the observed activity corresponding to the
                % candidate template duration and temporal position
                seg = obs(t0:t1, :);   % [L x C]

                % Compute Pearson correlation for each channel and then
                % average across channels while ignoring NaN values
                % ---- Vectorized column-wise Pearson correlation ----

                % Center each channel
                seg_c   = seg   - mean(seg,   1, 'omitnan');
                templ_c = templ - mean(templ, 1, 'omitnan');
                
                % Compute channel-wise norms
                seg_n   = sqrt(sum(seg_c.^2,   1, 'omitnan'));
                templ_n = sqrt(sum(templ_c.^2, 1, 'omitnan'));
                denom   = seg_n .* templ_n;

                % Compute correlations between corresponding channels
                rch = sum(seg_c .* templ_c, 1, 'omitnan') ./ denom;
                % ---------------------------------------------



                rmean = mean(rch, 'omitnan');
                GridRch(sf, sft) = rmean;

                 % Update the best-fitting parameters
                if rmean > best_r
                    best_r = rmean;
                    best_t0 = t0;
                    best_t1 = t1;
                    best_stretch = currentStretch;
                    best_time_shift = current_shift_time;
                end
            end
        end

        % Store the refined segment information
        segments_ref(k).TypeK  = currentLabel;
        segments_ref(k).TokenNum  = cls_id;
        segments_ref(k).j      = j_tok;
        segments_ref(k).t0     = best_t0;
        segments_ref(k).t1     = best_t1;
        segments_ref(k).len    = best_t1 - best_t0 + 1;
        segments_ref(k).best_r = best_r;
        segments_ref(k).best_stretch    = best_stretch;
        segments_ref(k).best_time_shift = best_time_shift; % Temporal shift in bins
        segments_ref(k).GridR           = GridRch;
        segments_ref(k).StretchSeries   = opt.stretchGrid;
        segments_ref(k).TimeShiftSeries = timeShiftSeries*dt; % Convert temporal shifts back to seconds

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

function [TBL] = my_segments_refined_converter(segments_refined, dt)
    K  = numel(segments_refined);
    label_out = strings(K,1);
    t_start_s = zeros(K,1);
    t_end_s   = zeros(K,1);
    
    for k = 1:K
        j_in_sent    = segments_refined(k).j;                % 文中の位置
        label_out(k) = segments_refined(k).TypeK;   % 実ラベル
        t_start_s(k) = (segments_refined(k).t0 - 1) * dt; % 秒
        t_end_s(k)   = (segments_refined(k).t1)     * dt;
    end
    
    TBL = table(label_out, t_start_s, t_end_s, (t_end_s - t_start_s), ...
                'VariableNames', {'Label','Start_s','End_s','Dur_s'});
end

function [decodedS] = make_decoded_copy_typing(TBL)
    decodedS = "";
    for i = 1:height(TBL)
        currentS = string(TBL.Label(i));
        if currentS == "Blank"
            continue
        elseif currentS == "Sp"
            currentS = " ";
        end
        decodedS = decodedS + currentS;
    end

end

function new_tokens = remove_blank_from_tokens(tokens)
    indBlank = (tokens== "Blank");
    new_tokens = tokens(~indBlank);
end

function [editDist, CER] = char_error_rate(str_ref, str_out)

    n = length(str_ref);
    m = length(str_out);
    D = zeros(n+1, m+1);
    
    D(:,1) = 0:n;
    D(1,:) = 0:m;
    
    for i = 2:n+1
        for j = 2:m+1
            cost = str_ref(i-1) ~= str_out(j-1);
            D(i,j) = min([
                D(i-1,j) + 1;       
                D(i,j-1) + 1;       
                D(i-1,j-1) + cost;  
            ]);
        end
    end
    
    editDist = D(end,end);
    CER = editDist / n;
end

function [timeData] = my_rms_FR_plot(zFR, bin_sec_zFR)

    all_zFR = sqrt(mean(zFR.^2, 2));


    N = size(zFR,1);
    timeData = (0:N-1) * bin_sec_zFR;          
    

    plot(timeData, all_zFR, 'LineWidth', 0.8)
    hold on
    xlabel('Time (s)')
    ylabel({'Population',  'RMS firing rate (z)'})
    box off
    ax = gca;
    ax.TickLength = [0 0];

end

function [yl] = token_output_plot(ax, plot_proportion, TBL, timeData, color_value)

    hold(ax, 'on')
    yl = ax.YLim;
    ypos = yl(1) + plot_proportion * (yl(2) - yl(1));
    
    for ti = 1:height(TBL)

        currentStartTime = TBL.Start_s(ti);
        [~, sInd] = min(abs(timeData - currentStartTime));
        s_time = timeData(sInd);


        currentEndTime = TBL.End_s(ti);
        [~, eInd] = min(abs(timeData - currentEndTime));
        e_time = timeData(eInd);
        

        fill([s_time, e_time, e_time, s_time], [yl(1), yl(1), yl(2), yl(2)], color_value, 'EdgeColor','none', 'FaceAlpha', 0.1)


        medianTime = mean([s_time, e_time]);
        text(ax, medianTime, ypos, TBL.Label(ti), 'HorizontalAlignment','center', 'FontSize',10, 'FontName', 'Helvetica')

        if s_time - e_time == 0
            xline(ax, s_time, 'Color', color_value, 'LineWidth',0.5)
        end

    end

end

function RNN_outputTBL = rnn_output_convert_table(current_logit_output, current_logit_output_time)


    rnn_output = ["_", "Sp", ",", "?", ".", ...
                  "A", "B", "C", "D", "E", "F", "G", ...
                  "H", "I", "J", "K", "L", "M", "N", ...
                  "O", "P", "Q", "R", "S", "T", "U", ...
                  "V", "W", "X", "Y", "Z"];

    blank_id  = 1;
    

    ind_zero = current_logit_output(:, 1) == 0;
    logits = current_logit_output(~ind_zero, 1:31); 
    logit_output_time = current_logit_output_time(~ind_zero);
        
    % ---- (1) Softmax with numerical stability ----
    mx   = max(logits, [], 2);
    ex   = exp(logits - mx);           % T x C
    post = ex ./ sum(ex, 2);           
    
    % ---- (2) Argmax path ----
    [~, path] = max(post, [], 2);   % T x 1, 1..C
    
    % ---- (3) Run-length encoding (values, lengths, starts) ----
    [values, ~, starts, ends] = rle1d(path);
    

    blank_ind = values == blank_id;

    token_values = values(~blank_ind);
    token_starts = starts(~blank_ind);
    token_ends = ends(~blank_ind);
    
    RNN_output_token = strings(0);
    RNN_output_stime = [];
    RNN_output_etime = [];
    for i = 1:numel(token_values)
        RNN_output_token(end+1) = rnn_output(token_values(i));
        RNN_output_stime(end+1) = logit_output_time(token_starts(i));
        RNN_output_etime(end+1) = logit_output_time(token_ends(i));
    end
    
    
    RNN_outputTBL = table(RNN_output_token', RNN_output_stime', RNN_output_etime', (RNN_output_etime' - RNN_output_stime'), ...
                'VariableNames', {'Label','Start_s','End_s','Dur_s'});

end

function [vals, lens, starts, ends] = rle1d(x)

    if isempty(x)
        vals=[]; 
        lens=[]; 
        starts=[]; 
        return; 
    end

    dx = [true; diff(x)~=0]; 
    starts = find(dx); 
    vals = x(starts); 
    ends = [starts(2:end)-1; numel(x)]; 
    lens = ends - starts + 1;
end

function [decodedList] = ngram_output_plot(segD, currentTranscript, ax, portion1, portion2, FontSizeVal)

    yl = ax.YLim;

    currentDecoder_output = segD.ngram_decoder_partial_output{currentTranscript};
    currentDecoderTime = segD.Decoder_output_after_go_cue_sec{currentTranscript};
    decoderNum = numel(currentDecoder_output);
    
    partialOutput = "";
    decoded_PartialList = {};
    decoded_Partial_time = [];
    for ti = 1:decoderNum
        
        if ischar(currentDecoder_output{ti}) || isstring(currentDecoder_output{ti})
            clippedCur = string(currentDecoder_output{ti});
            
            if ~strcmp(clippedCur, partialOutput)
                partialOutput = clippedCur;
                decoded_PartialList{end+1} = partialOutput;
                decoded_Partial_time(end+1) = currentDecoderTime(ti);
            end
        elseif ti == decoderNum 
            decoded_PartialList{end+1} = string(segD.ngram_decoder_final_output{currentTranscript}(ti));
            decoded_Partial_time(end+1) = currentDecoderTime(ti);
        end
    
    
    end
    
    
    decodedList = table(decoded_PartialList(:), decoded_Partial_time(:), 'VariableNames', {'Output', 'time'});
    ypos1 = yl(1) + portion1 * (yl(2) - yl(1));
    ypos2 = yl(1) + portion2 * (yl(2) - yl(1));
    for i = 1:height(decodedList)
        currentS = decodedList.Output{i};
        currentTime = decodedList.time(i);
        xline(ax, currentTime, 'k-', 'LineWidth',1)
        if mod(i, 2) == 0
            text(ax, currentTime, ypos2, currentS, 'FontSize',FontSizeVal, 'FontName', 'Helvetica')
        else
            text(ax, currentTime, ypos1, currentS, 'FontSize',FontSizeVal, 'FontName', 'Helvetica')
        end
    end

end

function [TBL, RNN_outputTBL, tokens] = tbl_and_rnn_tbl_make(segD, currentTranscript, sigma_bin, bin_sec, aveFR_session, stdFR_session, refined_template, KeyboardLabel)
    
    trueLabel = segD.transcriptions{currentTranscript};
    
    %%%%%%%%%%%%%%% Step 1. Tokenize the sentence and assign template indices
    [sent_idx, tokens] = true_sentence_convert_idx_includingBlank(trueLabel, KeyboardLabel);
    
    %%%%%%%%%%%%%%% Step 2. Extract ncTX during typing, convert it to FR,
    % and z-score it using the mean and standard deviation of the Rest-state FR
    ncTX = segD.TXgo{currentTranscript};
    FR = my_ncTX_FR_convert(ncTX, sigma_bin, bin_sec);
    zFR = (FR - aveFR_session)./stdFR_session;
    
    %%%%%%%%%%%%%%% Step 3. Downsample the z-scored FR from 10-ms to 50-ms bins
    % Specify the downsampling factor.
    % For example, use downRate = 5 to reduce the sampling rate by a factor of 5
    downRate = 5;
    dw_zFR = down_sample_FR(zFR, downRate);
    T = size(dw_zFR, 1);
    
    %%%%%%%%%%%%%%% Step 4. Select the templates used for HMM alignment
    % used_Template = originalTemplate; % Use the Original Templates generated from the Finger Sweep
    used_Template = refined_template; % 30 class HMMでrefineさせたtemplateを使用する場合
    
     % Adjust the template lengths only when Mu is too long relative to the
    % available neural data. The adjustment is determined automatically
    % within the function.
    adj_Template = adjust_length_template(used_Template, sent_idx, T, 0.2, 2);
    
    %%%%%%%%%%%%%%% Step 5. Construct the HMM parameters
    % Generate the mean firing-rate vectors (Mu) from the templates and the
    % state-transition matrix (A) from sent_idx
    [pi0, A, Mu, slices, metaD] = build_hmm_from_templates(adj_Template, sent_idx);
    
    %%%%%%%%%%%%%%% Step 6. Calculate the multivariate Gaussian log-likelihood
    % (logB) from the observed neural activity and the mean template vector
    % (Mu) for each state
    % logB is a [time x state] matrix of log probability densities,
    % corresponding to the log emission likelihoods
    sigma2 = 1;   
    logB = log_emission_gauss(dw_zFR, Mu, sigma2);
    
    %%%%%%%%%%%%%%% Step 7. Apply temporal constraints and perform Viterbi decoding
    %%%%%%% Temporal-window constraint:
    % Center the j-th token around (j/M)*T and allow a window of +/-0.3T
    logB = apply_time_window_constraint(logB, slices, 0.3); 
    %%%%%%% Terminal-state constraint:
    % Force the sequence to terminate at the final state of the last Blank token
    logB = apply_terminal_constraint(logB, metaD);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Decode the most likely state sequence
    [z_path, delta, psi] = viterbi_log(pi0, A, logB);   
    
     %%%%%%%%%%%%%%% Step 8. Local refinement of token onset and offset times
    segments = z_path_to_segments(z_path, slices, tokens);
    
    dt = 0.05;  

    refineOpt = struct( ...
        'maxShiftSec', 0.5, ...   % ±0.5 s
        'stepSec',     0.05, ...  % 0.05 s 刻み
        'stretchGrid', 0.4:0.0786:1.5 ...  % 0.4 ～ 1.5
    );

    segments_refined = refine_segments_by_grid(dw_zFR, adj_Template, sent_idx, segments, dt, refineOpt, KeyboardLabel);
    
    %%%%%%%%%%%%%%% Step 9. Convert the refined results to a table
    TBL = my_segments_refined_converter(segments_refined, dt);
    
    %%%%%%%%%%%%%%% Step 10. Organize the RNN output
    current_logit_output = segD.decoder_logit_output{currentTranscript};
    current_logit_output_time = segD.Decoder_output_after_go_cue_sec{currentTranscript};
    
    RNN_outputTBL = rnn_output_convert_table(current_logit_output, current_logit_output_time);


end

function [TP] = compare_tbl_rnntbla(T, R, leadMin, leadMax)
    % leadMin = 0; % Minimum required lead time of the HMM output relative to the RNN output (s)
    % leadMax = 1.5; % Exclude HMM outputs that occur too early relative to the RNN output
    
    % --- Greedy matching in temporal order between HMM and RNN outputs
    %     with matching token labels ---
    i=1; j=1;
    keep    = false(height(T),1);
    leadV   = nan(height(T),1);
    iouV    = nan(height(T),1);
    drV     = nan(height(T),1);
    Rpair_s = nan(height(T),1);
    Rpair_e = nan(height(T),1);
    
    while i<=height(T) && j<=height(R)
        Li = T.Label(i); 
        Lj = R.Label(j);
        if Li == Lj
            % HMM onset and offset times
            Hs=T.Start_s(i); 
            He=T.End_s(i);
            
            % RNN onset and offset times
            Rs=R.Start_s(j);
            Re=R.End_s(j);
    
            lead = Rs - Hs;                              % Positive values indicate that HMM precedes RNN
            inter = max(0, min(He,Re)-max(Hs,Rs));       % Duration of temporal overlap between HMM and RNN intervals
            uni   = max(He,Re)-min(Hs,Rs);               % Duration of the union of HMM and RNN intervals
            IoU   = inter / max(uni, eps);               % Intersection over Union (IoU) of the two intervals
            durH  = max(He-Hs, eps);
            durR  = max(Re-Rs, eps);
            dr    = durH / durR;                         % Ratio of HMM duration to RNN duration
    
            ok = (lead >= leadMin) && (lead <= leadMax);
    
            if ok
                keep(i)   = true;
                leadV(i)  = lead;
                iouV(i)   = IoU;
                drV(i)    = dr;
                Rpair_s(i)= Rs; 
                Rpair_e(i)= Re;
            end

            % Advance both indices after matching tokens with the same label
            i=i+1; j=j+1;
    
        elseif T.Start_s(i) <= R.Start_s(j)
            i=i+1;
        else % Advance to the next RNN output when the current RNN output precedes the HMM output
            j=j+1;
        end
    end
    
    % --- Create the output table using the HMM output as the reference ---
    TP = table(T.Label(keep), T.Start_s(keep), T.End_s(keep), ...
               Rpair_s(keep), Rpair_e(keep), ...
               leadV(keep), iouV(keep), drV(keep), ...
               'VariableNames', {'Label','HMM_Start','HMM_End','RNN_Start','RNN_End','Lead_s','IoU','DurRatio'});

end

function [all_x, all_y] = boxchart_data_make(SelectFieldName, FieldList, All_HMM_RNN)

    all_x = [];
    all_y = [];

    FieldNum = numel(FieldList);

    for i = 1:FieldNum
        selectFieName = FieldList{i};
        curD = All_HMM_RNN.(selectFieName).Results.(SelectFieldName);
        curL = repmat(string(FieldList{i}), numel(curD), 1);
    
        all_x = [all_x; curD'];
        all_y = [all_y; curL];
    
    end

end

function [meanD] = get_mean_based_categor(x_ED, cat_y)

    cat_type = categories(cat_y);
    
    meanD = [];
    for i = 1:numel(cat_type)
        currentCat = cat_type{i};
        ind = (cat_y == currentCat);
        selectx = x_ED(ind);
        meanD(end+1) = mean(selectx);
    end

end

function [all_axes] = plot_values_withindata_acrossday_by_sentences(x_values, catList, cat_y, bin_num, ylabelname, FieldList, ylim_value)

    catNum = numel(catList);
    slopeList = nan(catNum,1);
    
    all_axes = {};
    for i = 1:catNum
        
        currentCat = catList{i};
        ind = cat_y == currentCat;
        
        currentD = x_values(ind);
        sessionX = 1:numel(currentD);
        
        % y = a*x + b
        mdl = fitlm(sessionX, currentD);
        slopeList(i) = mdl.Coefficients.Estimate(2);
        
    
        subplot(1, 8, i)
        

        [mean_num, ci_data] = get_mean_ci_based_group(currentD, bin_num);
        % 95% CI
        x_plotD = bin_num/2:bin_num:bin_num*numel(mean_num);
        fill([x_plotD, flip(x_plotD)], [ci_data(1, :), flip(ci_data(2, :))], [0 0 1], 'EdgeColor','none', 'FaceAlpha', 0.1)
        hold on

        plot(x_plotD, mean_num, 'b-', 'LineWidth', 1)
        
        yFit = mdl.Coefficients.Estimate(1) + mdl.Coefficients.Estimate(2)* sessionX;
        plot(sessionX, yFit, 'r', 'LineWidth',2)
        title(sprintf('%s', currentCat))
    
        ylim(ylim_value)
    
        if i == 4
            xlabel('Sentence number within session')
        end
        
        if i == 1
            ylabel(ylabelname)
        end
        
        if i ~= 1
            ax = gca;
            ax.YAxis.Visible = 'off';
        end
        
        grid on
        box off
        pbaspect([1 3 1])
    
        all_axes{i} = gca;
    
    end

end

function [mean_num, ci_data] = get_mean_ci_based_group(currentD, bin_num)

    datasize = numel(currentD);
    groupNum = floor(datasize/bin_num);
    ci_data = nan(2, groupNum);
    mean_num = nan(1, groupNum);
    
    for gi = 1:groupNum
        inds = (gi-1)*bin_num + 1;
        inde = gi * bin_num;
    
        clippedData = currentD(inds:inde);
        [h, ~, cidata] = normfit(clippedData);
    
        mean_num(gi) = h;
        ci_data(:, gi) = cidata;
    end

end