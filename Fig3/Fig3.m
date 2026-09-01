%% Setup
% See README.md for installation instructions and software requirements.
clear;
clc;
close all;

%%%%%%%% Note:  Select the project directory as the current working directory before running this script.
projectDir = string(pwd);

% Load the finger sweep dataset used for Figure 1.
dataFile = fullfile(projectDir, '..', ...
    'Fig1', 'data', 't18FingerSweepData.mat');
segD = load(dataFile);

% Generate finger movement labels.
[OriginalLabel, convertedLabels, myOrder, myOrder_short] = my_label_converter();

%% Fig.3A
% Temporal dynamics of class-selective channels

% Prepare 4D data (Task x Trial x Data x Channel)
TX_segmentedD = {segD.zTXpre, segD.zTXgo};
PW_segmentedD = {segD.zPWpre, segD.zPWgo};
TXorig_segmentedD = {segD.TXpre, segD.TXgo}; 

% Minimum number of data points required before and after the Go cue.
minDelayDur = 150; 
minGoDur = 240;

% Define analysis time windows
time_window_data = 100; % Number of data points in each analysis window
sliding_data = 10;      % Step size of the sliding analysis window
bin_sec = 0.01;         % Duration of one data point (s); 0.01 = 10 ms
S_time = minDelayDur * -bin_sec;
E_time = minGoDur*bin_sec;

% Generate analysis time windows.
[time_char, time_window_list, total_analized_bin_num] = create_AnaT_data(S_time, E_time, time_window_data, bin_sec, sliding_data);

% Construct 4D data arrays
[inputFeaturesTX, TX_DeGo, ErrorTrial_tx, Transcriptions] = Four_D_data_make(TX_segmentedD, minDelayDur, minGoDur, segD, myOrder, convertedLabels, OriginalLabel);
[inputFeaturesPW, PW_DeGo, ErrorTrial_pw] = Four_D_data_make(PW_segmentedD, minDelayDur, minGoDur, segD, myOrder, convertedLabels, OriginalLabel);

% Construct the corresponding 4D array from raw ncTX for firing-rate calculation.
[inputFeaturesTXorig, TXorig_DeGo, ErrorTrial_txorig] = Four_D_data_make(TXorig_segmentedD, minDelayDur, minGoDur, segD, myOrder, convertedLabels, OriginalLabel);

% Calculate firing rates from ncTX
sigma_bin = 3; % Standard deviation of the Gaussian kernel (bins)
% Calculate firing rates from the segmented raw ncTX data.
inputFeaturesFR = my_FR_make(sigma_bin, inputFeaturesTXorig, bin_sec);

%%% Identify class-selective channels using the Kruskal-Wallis test %%%

% Significance threshold used for FDR correction.
kw_pvalue2use = 0.05;

time_window_listNum = size(time_window_list, 2);

%%%%%%%%%%%%%%%% z-scored ncTX
fprintf('Analyzing z-scored ncTX...\n')
% Identify class-selective channels using the Kruskal-Wallis test followed by FDR correction.
[Results_KW_TX, total_good_ch_num_TX] = my_kw_test(inputFeaturesTX, time_window_listNum, time_window_list, time_char, segD, kw_pvalue2use);

%%%%%%%%%%%%%%%% z-scored power
fprintf('Analyzing z-scored power...\n')
[Results_KW_PW, total_good_ch_num_PW] = my_kw_test(inputFeaturesPW, time_window_listNum, time_window_list, time_char, segD, kw_pvalue2use);

%%%%%%%%%%%%%%%% Firing rate
fprintf('Analyzing firing rate...\n')
[Results_KW_FR, total_good_ch_num_FR] = my_kw_test(inputFeaturesFR, time_window_listNum, time_window_list, time_char, segD, kw_pvalue2use);

% Plot temporal dynamics of class-selective channels
time_vals = str2double(time_char);

figure;
co = colororder("gem");
plot(time_vals, total_good_ch_num_TX, '-o', 'LineWidth', 1, 'Color', co(1, :));
hold on;
plot(time_vals, total_good_ch_num_PW, '-o', 'LineWidth', 1, 'Color', co(2, :));
plot(time_vals, total_good_ch_num_FR, '-o', 'LineWidth', 1, 'Color', co(3, :));
xlabel('Time (s) from Go (t=0)'); 
xticks(-1:2); xlim([-1.1, 2.0])
ylabel('Number of class-selective channels');
title('Temporal dynamics of class-selective channels');
L = legend({'z-ncTX', 'z-PW', 'FR'}, 'Location', 'NorthWest', 'FontName','Helvetica', 'FontSize',6);
L.Box = 'off';
L.AutoUpdate = 'off';
AX = gca;
AX.LineWidth = 1;
yticks(300:20:360), ylim([295 365])
xline(0, '-', 'LineWidth',0.5, 'Color', [0.5 0.5 0.5])
box off


%% Fig.3B

% Array combinations to analyze.
ArrayList = ["All", "r6d", "l6d"];

% Subarray combinations to analyze.
% 1 = Subarray-1, 2 = Subarray-2, 3 = both subarrays.
Subarraylist = 3;
% Structure for storing SVM results.
SVMresults = struct();

for arraylisti = 1:numel(ArrayList)
    % Select the current array.
    CurrentArray = ArrayList(arraylisti);
    for subarrayli = 1:numel(Subarraylist)
        if CurrentArray == "All" & Subarraylist(subarrayli) == 1
            continue
        elseif CurrentArray == "All" & Subarraylist(subarrayli) == 2
            continue
        end

        CurrentSubA = Subarraylist(subarrayli); 
        % Features used for SVM decoding.
        usedfeature = "TX_PW";
        
        % Array names and corresponding channel indices.
        Array_allocation = ["v6d", "d6d", "r6d", "All", "l6d"];
        Array_chNum = {1:128, 129:256, 257:384, 1:384, 1:256};
                
        TimeWinSz = numel(time_char);
        time_vals = str2double(time_char);
        
        [~, trialNum, dataNum, ChNum] = size(inputFeaturesTX);
        
        
        % Concatenate the 4D ncTX and power feature arrays.
        inputFeaturesTX_PW = cat_inputFeatures(inputFeaturesTX, inputFeaturesPW);
        % inputFeaturesTX_PW = cat_inputFeatures(inputFeaturesFR,inputFeaturesPW); % For FR
        
        if usedfeature == "TX"
            inputFeatures = inputFeaturesTX;
        elseif usedfeature == "PW"
            inputFeatures = inputFeaturesPW;
        elseif usedfeature == "TX_PW"
            inputFeatures = inputFeaturesTX_PW;
        end
        
        % Generate a descriptive name for the current analysis.
        TitleName = CurrentArray+ "(" + string(CurrentSubA) + ") " + usedfeature + ...
            " Input " + string(time_window_data) + "data " + string(bin_sec) + "bin";
        disp([char(TitleName), ' start....'])
        
        % Create class labels.
        ydata = create_ydata(myOrder, trialNum);
        
        total_gnd = {};
        total_y_pred = {};
        results_acc = nan(TimeWinSz, 1);
        
        for timei = 1:TimeWinSz
            % Identify significant channels for the current time window,
            % taking into account the selected array, subarray, and feature type. 
            used_ch_ind = get_timeWbased_sigChind(string(time_char{timei}), usedfeature, time_char, Results_KW_TX, Results_KW_PW, CurrentArray, CurrentSubA, Array_chNum, Array_allocation, ChNum);
            
            % Select the current analysis time window.
            used_timewindow = time_window_list{timei};
            
            % Extract data used for SVM decoding.
            currentAllData = inputFeatures(:, :, used_timewindow, used_ch_ind);
        
            % Stratified 10-fold cross-validation.
            cv = cvpartition(ydata, 'KFold', 10, 'Stratify', true);
        
            % Perform SVM decoding using the predefined n-fold
            % cross-validation. Within each fold, training and test data
            % are z-scored using the mean and standard deviation calculated
            % from the training data only.
            [y_gnd_allFolds, y_pred_allFolds, mean_acc_allFolds] = my_nfold_cv_SVM(cv, currentAllData, ydata);
            total_gnd{timei} = y_gnd_allFolds;
            total_y_pred{timei} = y_pred_allFolds;
            acc = mean(y_gnd_allFolds == y_pred_allFolds);
            results_acc(timei, 1) = acc;
        
            fprintf([time_char{timei}, 's \n'])
            fprintf('10Fold CV SVM Accuracy: %.2f%%\n\n', acc * 100);
        end
        
        figure
        plot(time_vals, results_acc *100, '-o', 'LineWidth', 2);
        hold on;
        xlabel('Time (s)'); xticks(time_vals)
        ylabel('10fCV SVM Accuracy (%)');
        title(TitleName);
        xline(0, 'r--')
        
        filedname = char(CurrentArray + string(CurrentSubA));
        SVMresults.(filedname).total_gnd = total_gnd;
        SVMresults.(filedname).total_y_pred = total_y_pred;
        SVMresults.(filedname).results_acc = results_acc;

        drawnow
    end

end


%%%%%%%%%%%%%%% Save the spatiotemporal cluster analysis results
% Create the output directory if it does not exist
outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'SVMResults.mat'), ...
    'SVMresults', 'time_vals', '-v7.3');

%%%%%%%%%%%%%%% Visualize temporal dynamics across arrays

% Get field names from the SVM results structure.
FilednameList = fieldnames(SVMresults);

% Extract SVM decoding accuracy for each array.
Plot_all_acc = nan(numel(time_vals), numel(FilednameList));
for i = 1:numel(FilednameList)
    Plot_all_acc(:,i) = SVMresults.(FilednameList{i}).results_acc;
end
Plot_all_acc = Plot_all_acc*100;

co = colororder("gem");

% Plot temporal changes in SVM decoding accuracy for each array.
% Note: The correspondence between columns and arrays is hard-coded.
LegendName = {"All arrays", "Right 6d arrays", "Left 6d arrays"};
figure
plot(time_vals, Plot_all_acc(:, 1), 'o-', 'LineWidth',1, 'Color', co(1, :))
hold on
plot(time_vals, Plot_all_acc(:, 2), 'o-', 'LineWidth',1, 'Color', co(6, :))
plot(time_vals, Plot_all_acc(:, 3), 'o-', 'LineWidth',1, 'Color', co(5, :))

box off
L = legend(LegendName, 'Box','off', 'AutoUpdate','off', 'Location','best');
xlabel('Time (s)')
ylabel('10-fold CV SVM accuracy (%)')
xline(0, '-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5])
title({'Temporal dynamics of SVM', 'decoding accuracy (10-fold CV)'})

xticks([-1:2]), xlim([-1.1 2.1])
yticks([30:30:90]), ylim([28 95])

%% Fig.3C
% Multi-finger movement classification using a multilayer perceptron (MLP)

% Prepare input data for MLP classification

% Define array names and corresponding channel indices.
Array_allocation = ["v6d", "d6d", "r6d", "l6d" "All"];
Array_chNum = {1:128, 129:256, 257:384, 1:256, 1:384};

inputFeaturesTX_PW = cat_inputFeatures(inputFeaturesTX, inputFeaturesPW);

inputFeature = "TX_PW";
usedArray = "All";
ana_time = "0.70";

% Extract data from the specified array and analysis time window.
[usedChInd, anaD, t_range] = data_vec_input_make(inputFeature, usedArray, ana_time, Array_allocation, Array_chNum, inputFeaturesTX, inputFeaturesPW, inputFeaturesTX_PW, time_window_list, time_char);

% Define preprocessing and cross-validation parameters

% Apply L2 normalization to each input feature vector before MLP training.
% Note: L2 normalization is used here for the MLP input but is generally
% not recommended for convolutional neural network (CNN) inputs.
l2norm = true;

% Enable data augmentation.
data_augmentation = true;

% kfold CVを行う
% With 5-fold CV, 80% of the data are assigned to the training set in
% each fold. The remaining 20% are equally divided between validation
% and test sets (10% each).
k = 5; 

%%% Prepare data for MLP input
% Prepare input features and class labels for MLP classification.
% Class labels are reordered according to myOrder, and samples are shuffled.
[inputFeatures, transcriptions, labelslist, inputFeatures_orig, transcriptions_orig, labelslist_orig] = created_data_for_inputMLP(anaD, myOrder, t_range, usedChInd);

%%% Select class-selective channels using the Kruskal-Wallis test
% Significance threshold for FDR correction.
alpha_for_FDR = 0.05;

% Identify class-selective channels using the Kruskal-Wallis test
% followed by FDR correction.
sigChIdx = kw_test_with_FDR(inputFeatures_orig, transcriptions_orig, alpha_for_FDR);
fprintf('ChNum with KW-test corrected by FDR: %d\n\n', sum(sigChIdx))

%%% Generate train, validation, and test splits
% Generate cross-validation indices.
% For 5-fold CV, 80% of the data are used for training in each fold,
% while the remaining 20% are equally divided into validation (10%)
% and test (10%) sets.

idxCombination = my_train_val_test_idx_make_kfold(transcriptions, k);


InputSize = sum(sigChIdx);
outputSize = numel(countcats(categorical(transcriptions)));

layers = [
    % Input layer
    featureInputLayer(InputSize)
    
    % Hidden layer
    fullyConnectedLayer(165)
    leakyReluLayer(0.01) % leaky ReLUを使用する
    dropoutLayer(0.4) % Dropoutは0.4
    
    % Output layer
    fullyConnectedLayer(outputSize)
    softmaxLayer
    classificationLayer
];

[Accuracies, All_y, All_pred, macro_F1] = my_MLP_train_and_test(k, inputFeatures, transcriptions, idxCombination, data_augmentation, l2norm, layers, sigChIdx, true, myOrder);
acc_val = sprintf('%.2f', mean(All_y == All_pred)*100);
f1_val = sprintf('%.2f', macro_F1);

Title1 = "(" + string(acc_val) + "% accuracy)";
disptitle = {"Confusion matrix of multiple-finger movement classification using MLP", Title1};  
title(disptitle)

%% Fig.3D
% Visualization of neural representations in a two-dimensional latent space
% using an autoencoder

Array_allocation = ["All"];
Array_chNum = {1:384};

inputFeature = "TX_PW";

Results_latent = struct();

time_num = numel(time_char);

for arrayi = 1:numel(Array_allocation)
    usedArray = Array_allocation(arrayi);
    disp(usedArray + " start...")
    Results_latent.(usedArray).latent = cell(time_num, 1);
    Results_latent.(usedArray).time = nan(time_num, 1);
    Results_latent.(usedArray).transcriptions = cell(time_num, 1);
    Results_latent.(usedArray).labelslist = cell(time_num, 1);
    Results_latent.(usedArray).inputFeature = inputFeature;

    % Analyze the time window centered at 0.7 s after the Go cue.
    for timei = 18 
        ana_time = string(time_char{timei});
        disp(ana_time + "s")
        
        [usedChInd, anaD, t_range] = data_vec_input_make(inputFeature, usedArray, ana_time, Array_allocation, Array_chNum, inputFeaturesTX, inputFeaturesPW, inputFeaturesTX_PW, time_window_list, time_char);
        
        % Apply L2 normalization to input feature vectors.
        l2norm = true;

        % Enable data augmentation.
        data_augmentation = true;

        % Perform k-fold cross-validation.
        % With k = 5, 80% of the data are used for training in each fold.
        k = 5; 
        
        % Prepare data for network input.
        % Class labels are reordered according to myOrder, and samples
        % are shuffled.
        [inputFeatures, transcriptions, labelslist, inputFeatures_orig, transcriptions_orig, labelslist_orig] = created_data_for_inputMLP(anaD, myOrder, t_range, usedChInd);
        
        %%%%%%%%%%%%%%%%% Select class-selective channels using the Kruskal-Wallis test
        % Significance threshold for FDR correction.
        alpha_for_FDR = 0.05;

        % Identify class-selective channels using the Kruskal-Wallis test
        % followed by FDR correction. All samples are used for this
        % channel-selection step.
        sigChIdx = kw_test_with_FDR(inputFeatures_orig, transcriptions_orig, alpha_for_FDR);
        fprintf('ChNum with KW-test corrected by FDR: %d\n\n', sum(sigChIdx))
        
        % Generate train, validation, and test splits.
        % With 5-fold CV, 80% of the data are used for training.
        % The remaining 20% are equally divided into validation (10%)
        % and test (10%) sets.
        idxCombination = my_train_val_test_idx_make_kfold(transcriptions, k);
        
        
        %%%%%%%%% Step.1 Construct the autoencoder
        
        InputSize = sum(sigChIdx);
        classSize = numel(countcats(categorical(transcriptions)));
        LayersNum = [48, 16];
        layers = layerGraph();
        
        %%%% Encoder
        % InputLayer
        layers = addLayers(layers, featureInputLayer(InputSize, 'Name', 'input'));
        
        % First encoder layer
        layers = addLayers(layers, fullyConnectedLayer(LayersNum(1), 'Name', 'encoder_fc1'));
        layers = addLayers(layers, eluLayer('Name', 'encoder_elu1'));
        layers = addLayers(layers, dropoutLayer(0.3, 'Name', 'encoder_drop1'));
        
        % Second encoder layer
        layers = addLayers(layers, fullyConnectedLayer(LayersNum(2), 'Name', 'encoder_fc2'));
        layers = addLayers(layers, eluLayer('Name', 'encoder_elu2'));
        layers = addLayers(layers, dropoutLayer(0.3, 'Name', 'encoder_drop2'));
        
        % Two-dimensional latent space
        layers = addLayers(layers, fullyConnectedLayer(2, 'Name', 'latent'));
        
        %%%% Decoder
        % First decoder layer
        layers = addLayers(layers, fullyConnectedLayer(LayersNum(2), 'Name', 'decoder_fc1'));
        layers = addLayers(layers, eluLayer('Name', 'decoder_elu1'));
        
        % Second decoder layer
        layers = addLayers(layers, fullyConnectedLayer(LayersNum(1), 'Name', 'decoder_fc2'));
        layers = addLayers(layers, eluLayer('Name', 'decoder_elu2'));
        
        % Decoder output
        layers = addLayers(layers, fullyConnectedLayer(InputSize, 'Name', 'decoder_output'));
        
        % Classifier head
        layers = addLayers(layers, fullyConnectedLayer(classSize, 'Name', 'class_output'));
        layers = addLayers(layers, softmaxLayer('Name', 'softmax'));
        
        % Connect network layers.
        layers = connectLayers(layers, 'input', 'encoder_fc1');
        layers = connectLayers(layers, 'encoder_fc1', 'encoder_elu1');
        layers = connectLayers(layers, 'encoder_elu1', 'encoder_drop1');
        layers = connectLayers(layers, 'encoder_drop1', 'encoder_fc2');
        layers = connectLayers(layers, 'encoder_fc2', 'encoder_elu2');
        layers = connectLayers(layers, 'encoder_elu2', 'encoder_drop2');
        layers = connectLayers(layers, 'encoder_drop2', 'latent');
        layers = connectLayers(layers, 'latent', 'class_output');
        layers = connectLayers(layers, 'class_output', 'softmax');
        layers = connectLayers(layers, 'latent', 'decoder_fc1');
        layers = connectLayers(layers, 'decoder_fc1', 'decoder_elu1');
        layers = connectLayers(layers, 'decoder_elu1', 'decoder_fc2');
        layers = connectLayers(layers, 'decoder_fc2', 'decoder_elu2');
        layers = connectLayers(layers, 'decoder_elu2', 'decoder_output');
        
        
        % Convert the layer graph to a dlnetwork object.
        net = dlnetwork(layers);
        
        
        %%%%%%%%% Step.2 Train the network using k-fold cross-validation
        
        % training parameters
        numEpochs = 100;
        miniBatchSize = 32;
        initialLearnRate = 1e-3;
        decayRate = 0.5;

        % Reduce the learning rate by half every 30 epochs.
        decayEpochs = 30;  

        % optimizer
        trailingAvg = [];
        trailingAvgSq = [];

        % Weight assigned to the reconstruction (MSE) loss when combining
        % reconstruction and classification losses.
        % The classification-loss weight is (1 - loss_alpha).
        loss_alpha = 0.5;
        
        iteralNum = 1;
        epoch_sum = 0;
        bestNet = net;

        for ki = 1:k
            net = bestNet;
            for val_test = 1:2

                % Extract training, validation, and test data.
                [Xtrain, Ytrain] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{1, ki});
                if val_test == 1
                    [Xval, Yval] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{2, ki});
                    [Xtest, Ytest] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{3, ki});
                else
                    [Xval, Yval] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{3, ki});
                    [Xtest, Ytest] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{2, ki});
                end
        
                % Apply data augmentation to the training data.
                if data_augmentation
                    M = 5;
                    [Xtrain, Ytrain] = create_augmentation(M, Xtrain, Ytrain);
                end
        
                % Calculate the mean and standard deviation using only
                % the training data.
                mu = mean(Xtrain);
                sigma = std(Xtrain);

                % Avoid division by zero when the standard deviation is zero.
                sigma(sigma==0) = 1; 
        
                % Apply z-scoring and, when enabled, L2 normalization.
                [Xtrain_z] = my_zscore_l2norm(Xtrain, mu, sigma, l2norm); 
                [Xval_z] = my_zscore_l2norm(Xval, mu, sigma, l2norm);
                [Xtest_z] = my_zscore_l2norm(Xtest, mu, sigma, l2norm);
        
                % Number of iterations between validation evaluations.
                ValidationFrequencyNum = ceil(size(Xtrain_z, 1)/miniBatchSize);
        
                % Train the network
        
                Best_loss = inf;
                stop_count = 0;
                for epoch = 1:numEpochs
        
                    % Train the network using mini-batches.
                    for i = 1:ValidationFrequencyNum
                        if i == ValidationFrequencyNum
                            SelectIdx = (i-1)*miniBatchSize + 1:size(Xtrain_z, 1);
                        else
                            SelectIdx = (i-1)*miniBatchSize + 1:i*miniBatchSize;
                        end
        
                        % Select channels identified by the Kruskal-Wallis test.
                        dlX = dlarray(Xtrain_z(SelectIdx, sigChIdx)); 

                        % Reconstruction target.
                        dlT_recon = dlX;  

                        % Classification target.
                        dlT_class = Ytrain(SelectIdx);

                        % Convert class labels to one-hot vectors.
                        dlT_class_onehot = dlarray(onehotencode(dlT_class, 2, 'ClassNames', 1:classSize));
        
                        % Calculate the combined loss and gradients using
                        % automatic differentiation.
                        [loss, state, gradients] = dlfeval(@modelLossFunction, net, dlX, dlT_recon, dlT_class_onehot, loss_alpha);
                        net.State = state;
        
                        % Update network parameters using Adam.
                        learnRate = initialLearnRate * decayRate ^ floor(epoch_sum  / decayEpochs);
                        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, trailingAvg, trailingAvgSq, iteralNum, learnRate);
                        iteralNum = iteralNum + 1;
                                                                                    
        
                    end
        
                    epoch_sum = epoch_sum + 1;
        
                    % Evaluate the network using the validation data.
                    val_dlX = dlarray(Xval_z(:, sigChIdx));
                    val_dlT_class_onehot = dlarray(onehotencode(Yval, 2, 'ClassNames', 1:classSize));
                    [val_dlY_recon, val_dlY_class, ~] = forward(net, val_dlX, Outputs=["decoder_output", "softmax"], InputDataFormats="BC");
                    valLoss = loss_alpha * mse(val_dlY_recon, val_dlX, 'DataFormat', "BC") + ...
                        (1-loss_alpha) * crossentropy(val_dlY_class, val_dlT_class_onehot, 'DataFormat', "BC");
                    fprintf('Val loss: %.2f\n', valLoss);
                    
                    if valLoss < Best_loss
                        Best_loss = valLoss;
                        stop_count = 0;
                        bestNet = net;
                    else
                        stop_count = stop_count + 1;
                    end

                    disp(stop_count)

                    if stop_count > 5
                        fprintf('Roop %d/%d, Break \n', (ki-1)*2+val_test, k*2);
                        break
                    end
        
                end
        
            end
        
        end
        
        
        %%%%%%%%% Step.3 Extract the two-dimensional latent representation
        
        dlXall = dlarray(inputFeatures(:, sigChIdx));
        dlLatent = predict(bestNet, dlXall, Outputs="latent", InputDataFormats="BC");
        latent = extractdata(dlLatent);
    
        Results_latent.(usedArray).latent{timei} = latent;
        Results_latent.(usedArray).time(timei) = double(ana_time);
        Results_latent.(usedArray).transcriptions{timei} = transcriptions;
        Results_latent.(usedArray).labelslist{timei} = labelslist;
    
    end

end

%%% Reproduce the latent-space visualization shown in Fig. 3D

% Because autoencoder training is stochastic, the two-dimensional latent
% representation can vary across training runs. To reproduce the exact
% visualization shown in Fig. 3D, load the saved latent-space results
% provided in the data directory.

dataFile = fullfile(projectDir, 'data', 'LatentSpace_Autoencoder_result.mat');
load(dataFile)

%%% Visualize the two-dimensional latent space
color_value = colororder("gem12");

% Array to visualize.
DispArray = "All";

% Draw ellipses around individual movement classes.
circleOn = true;

% Labels used to assign colors according to finger identity.
LabelG1 = ["R Thumb", "R Index", "R Middle", "R Ring", "R Pinky", "L Thumb", "L Index", "L Middle", "L Ring", "L Pinky", "None"];

% Labels used to assign marker shapes according to movement direction.
LabelG2 = ["Up", "Down", "In", "None"];
MarkerLib = ["^", "v", "o", "square"];

timed = Results_latent.(DispArray).time;

F = figure;

nTimes = numel(timed);


for timei = 18
    
    disp('timei'); disp('Start...')
    Ldata = Results_latent.(DispArray).latent{timei};
    labelslist = Results_latent.(DispArray).labelslist{timei};
    transcriptions = Results_latent.(DispArray).transcriptions{timei};

    
    split_label = cellfun(@split, labelslist, 'UniformOutput', false);
    % Remove movement-direction labels (Up, Down, and In) to obtain
    % finger-identity labels.
    cat_label = cellfun(@make_cat_string, split_label, 'UniformOutput', false);
    cat_label = [cat_label{:}];

    % Extract movement-direction labels.
    GesOnly = cellfun(@make_cat_gesture, split_label, 'UniformOutput', false);
    GesOnly = [GesOnly{:}];


    
    for labeli = 1:numel(LabelG1)
        currentL = LabelG1(labeli);
        ind_label = contains(cat_label, currentL);
        for gesi = 1:numel(LabelG2)
            currentG = LabelG2(gesi);
            
            if currentL == "None" && currentG ~= "None"
                continue
            end

            ind_ges = contains(GesOnly, currentG);
            BothInd = ind_label & ind_ges;
            
           
            % Plot latent-space representations.
            % Colors indicate finger identity and marker shapes indicate
            % movement direction.
            scatter(Ldata(BothInd, 1), Ldata(BothInd, 2), 25, color_value(labeli, :),  'filled', MarkerLib(gesi));  % 'transcriptions'が色
            hold on

        end

    end

    
    if circleOn
        scale = 2;

        for taski = 1:numel(myOrder)
            currentL = split(myOrder(taski));
            % Calculate ellipse coordinates for the current movement class.
            [ellipse, ~] = create_circle(Ldata(transcriptions==taski, :), scale);
            if numel(currentL) > 1 
                new_l = currentL(1) + " " + currentL(2); 
            else
                new_l = myOrder(taski);
            end
            Ind = find(contains(LabelG1, new_l)); 
            plot(ellipse(:,1), ellipse(:,2), '-', 'LineWidth', 1, 'Color', color_value(Ind, :)); 
        end
    end
    


    % Create legends

    hold on
    for i = 1:numel(LabelG1)
        h_color(i) = scatter(nan, nan, 50, color_value(i,:), 'filled');
    end
    % Create marker legend for movement directions.
    for j = 1:numel(LabelG2)
        h_marker(j) = scatter(nan,nan,50,'k',MarkerLib(j));
    end
    L = legend([h_color h_marker], [LabelG1 LabelG2], 'Location','eastoutside', 'Box', 'off');
    L.FontSize = 10;


    titleName = DispArray + " array " + string(timed(timei)) + "s";
    title("2D latent representation: " + titleName)
    xlabel('Latent Dimension 1');
    ylabel('Latent Dimension 2');
    disp('End.')

end

% Apply consistent font formatting to all axes.
set(findall(F,'Type','axes'), 'FontName','Helvetica');

%% Fig.3E
% Pairwise Mahalanobis distances between movement classes

% Define arrays and channel indices
Array_allocation = ["v6d", "d6d", "r6d", "l6d", "All"];
Array_chNum = {1:128, 129:256, 257:384, 1:256, 1:384};

num_task = numel(myOrder_short);
xlabelData = {};
for i = 1:num_task
    xlabelData{end+1} = char(myOrder_short(i));
end


MahalanobisR = struct();


inputFeature = "TX_PW";


for arrayi = 1:numel(Array_allocation)
    usedArray = Array_allocation(arrayi);
    
    MahalanobisR.(usedArray) = [];
    


    for  timei = 1:numel(time_char) 
        ana_time = string(time_char{timei});

        % Prepare input variables for data_vec_input_make.
        [usedChInd, anaD, t_range] = data_vec_input_make(inputFeature, usedArray, ana_time, Array_allocation, Array_chNum, inputFeaturesTX, inputFeaturesPW, inputFeaturesTX_PW, time_window_list, time_char);
        
        % Convert the data into a 3D array (Task x Trial x Channel)
        % required for Mahalanobis-distance calculation.
        % Data within the selected time window are averaged here.
        data_vec = my_data_vec_create(anaD, usedChInd, t_range);
        
        % Calculate pairwise Mahalanobis distances between movement classes.
        % Input data dimensions: Task x Trial x Channel.
        distance_matrix = my_Mahalanobis(data_vec);

        % Normalize Mahalanobis distances by the square root of the
        % number of channels included in the analysis.
        distance_matrix_N = distance_matrix./sqrt(sum(usedChInd));
        
        % Convert the symmetric distance matrices to vectors containing
        % the unique pairwise distances.
        oneVecOrig = squareform(distance_matrix);
        normalizedOneVec = squareform(distance_matrix_N);
       
        % Visualize distance matrices at 0.7 s after the Go cue
        if timei == 18 
            if usedArray == "r6d" || usedArray == "l6d"
                figure
                imagesc(distance_matrix_N);
                C = colorbar;
                C.FontSize = 10; C.FontName = 'Helvetica';
                C.Label.String = 'normalized Mahalanobis distance';
                pbaspect([ 1 1 1 ])
                
                xticks(1:1:num_task)
                xticklabels(xlabelData)
                yticks(1:1:num_task)
                yticklabels(xlabelData)
                
                AX=gca; AX.FontName = 'Helvetica'; AX.FontSize = 6;
                xlabel('Task', 'FontName','Helvetica', 'FontSize', 10);
                ylabel('Task', 'FontName','Helvetica', 'FontSize', 10);
                title_Name = inputFeature + " " + usedArray + " array " + ana_time + "(s)";
                safe_title = strrep(title_Name, '_', '\_'); 
                title(safe_title, 'FontName','Helvetica', 'FontSize', 12);
            end
        end
        

        MahalanobisR.(usedArray).distance_matrix{timei} = distance_matrix;
        MahalanobisR.(usedArray).distance_matrix_N{timei} = distance_matrix_N;

        [muHat,~,muCI,~] = normfit(oneVecOrig);
        MahalanobisR.(usedArray).mean_ci(timei, 1) = muHat;
        MahalanobisR.(usedArray).mean_ci(timei, 2:3) = muCI';
        [muHat,~,muCI,~] = normfit(normalizedOneVec);
        MahalanobisR.(usedArray).mean_ci_N(timei, 1) = muHat;
        MahalanobisR.(usedArray).mean_ci_N(timei, 2:3) = muCI';    
    end
    
    MahalanobisR.(usedArray).timeval = double(string(time_char));

end

%%%%%%%%%%%%%%% Save the spatiotemporal cluster analysis results
% Create the output directory if it does not exist
outputDir = fullfile(projectDir, 'results');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'MahalanobisResults.mat'), ...
    'MahalanobisR', '-v7.3');

%% Fig.3F
% Hierarchical clustering of movement representations based on
% normalized Mahalanobis distances

% Define movement groups based on labels
Right_F = ["R Thumb Up",  "R Thumb Down",  "R Thumb In", "R Index Up",  "R Index Down",  "R Index In", "R Middle Up", "R Middle Down", "R Middle In", ...
    "R Ring Up",   "R Ring Down",   "R Ring In", "R Pinky Up",  "R Pinky Down",  "R Pinky In"];
Left_F = ["L Thumb Up",  "L Thumb Down",  "L Thumb In", "L Index Up",  "L Index Down",  "L Index In", "L Middle Up", "L Middle Down", "L Middle In", ...
    "L Ring Up",   "L Ring Down",   "L Ring In", "L Pinky Up",  "L Pinky Down",  "L Pinky In"];

Up_F = ["R Thumb Up","R Index Up","R Middle Up","R Ring Up","R Pinky Up","L Thumb Up","L Index Up","L Middle Up","L Ring Up","L Pinky Up"];
Down_F = ["R Thumb Down","R Index Down","R Middle Down","R Ring Down","R Pinky Down","L Thumb Down","L Index Down","L Middle Down","L Ring Down","L Pinky Down"];
In_F = ["R Thumb In","R Index In","R Middle In","R Ring In","R Pinky In","L Thumb In","L Index In","L Middle In","L Ring In","L Pinky In"];

%%% Right 6d: hierarchical clustering by hand
% Define label groups used for dendrogram coloring.
dendroLabelL = {Right_F, Left_F}; 

F = figure;
usedArray = "r6d";
timeD = MahalanobisR.(usedArray).timeval;

% Visualize the time window centered at 0.7 s after the Go cue.
for timei = 18

    % Retrieve the normalized Mahalanobis-distance matrix.
    distance_matrix = MahalanobisR.(usedArray).distance_matrix_N{timei};

    % Plot hierarchical clustering dendrogram.
    my_dendrogram_plot(distance_matrix, myOrder, dendroLabelL, color_value, 8, 45);

    titleName = usedArray + " " + string(timeD(timei)) + "s";
    title(titleName)
    ylabel('Normalized Mahalanobis distance', 'FontSize',10)
    pbaspect([2 1.2 1])
end

set(findall(F,'Type','axes'), 'FontName','Helvetica');

%%% Left 6d: hierarchical clustering by movement direction
% Define label groups used for dendrogram coloring.
dendroLabelL = {Up_F, Down_F, In_F}; 

F = figure;
usedArray = "l6d";
timeD = MahalanobisR.(usedArray).timeval;

for timei = 18

    distance_matrix = MahalanobisR.(usedArray).distance_matrix_N{timei};

    my_dendrogram_plot(distance_matrix, myOrder, dendroLabelL, color_value, 8, 45);

    titleName = usedArray + " " + string(timeD(timei)) + "s";
    title(titleName)
    ylabel('Normalized Mahalanobis distance', 'FontSize',10)
    pbaspect([2 1.2 1])
end

set(findall(F,'Type','axes'), 'FontName','Helvetica');

%% Fig.3G
% Visualizing Changes in Mahalanobis Distance Over Time

FieldNameList = {"All", "r6d", "l6d"};

figure
for i = 1:numel(FieldNameList)
    xdata = MahalanobisR.(FieldNameList{i}).timeval;
    ydata = MahalanobisR.(FieldNameList{i}).mean_ci_N(:, 1);
    plot(xdata, ydata, '-', 'Color', color_value(i, :), 'LineWidth',2), hold on
end

L = legend(FieldNameList);
L.Box = 'off'; L.AutoUpdate = 'off';
L.Location = 'best';

for i = 1:numel(FieldNameList)
    cidata = MahalanobisR.(FieldNameList{i}).mean_ci_N(:, 2:3);
    fill([xdata, flip(xdata)], [cidata(:, 1)', flip(cidata(:, 2))'], color_value(i,:), ...
             'FaceAlpha', 0.2, 'EdgeColor', 'none')
end
box off

xlabel('Time (s)')
ylabel('Normalized Mahalanobis distance')
xline(0, 'r--', 'LineWidth',1)
title('Temporal dynamics of Mahalanobis distance')

%% Fig.3H
% Relationship between normalized Mahalanobis distance
% and SVM decoding accuracy

% Load SVM decoding results
dataFile = fullfile(projectDir, 'data', 'SVMresults.mat');
load(dataFile);

mahaTime = MahalanobisR.All.timeval;
accTime = time_vals;


% [~, bmin] = min(abs(accTime - min(mahaTime))); [~, bmax] = min(abs(accTime - max(mahaTime)));
% timeInd = bmin:bmax;

% Time indices used for the current analysis.
timeInd = 6:26;

accdata = [];
mahadata = [];
SVMarray =  ["v6d3", "d6d3", "r6d3", "All3"];
Mahaarray = ["v6d", "d6d", "r6d", "All"];
labeld = [];
for arrayi = 1:4
    if arrayi == 1
        accdata = SVMresults.(SVMarray(arrayi)).results_acc(timeInd);
        mahadata = MahalanobisR.(Mahaarray(arrayi)).mean_ci_N(timeInd, 1);
        dataNum = numel(accdata);
        labeld = repelem(Mahaarray(arrayi), dataNum, 1);
    else
        accdata = cat(1, accdata, SVMresults.(SVMarray(arrayi)).results_acc(timeInd));
        mahadata = cat(1, mahadata, MahalanobisR.(Mahaarray(arrayi)).mean_ci_N(timeInd, 1));
        dataNum = numel(SVMresults.(SVMarray(arrayi)).results_acc(timeInd));
        labeld = cat(1, labeld, repelem(Mahaarray(arrayi), dataNum, 1));
    end

end
labeld = categorical(labeld);

% Convert decoding accuracy from proportion to percentage.
acc_pct = accdata * 100;

%%% Plot relationship between decoding accuracy and Mahalanobis distance
figure; hold on

gscatter(acc_pct, mahadata, labeld, color_value(1:4, :), 'o', 10, 'filled');
hold on
xlabel('10-fold CV SVM accuracy (%)');
ylabel('Normalized Mahalanobis distance');
title({'Relationship between Mahalanobis', 'distance and decoding accuracy'});
L = legend;
L.Box = 'off'; L.AutoUpdate = 'off'; L.Location = 'northwest';


% Pearson correlation
[r, p] = corr(acc_pct, mahadata, 'Type', 'Pearson');
if p < 0.05 && p > 0.001
    Corr_s = sprintf('Correlation r = %.2f (p = %.3g)', r, p);
elseif p < 0.001
    Corr_s = sprintf('Correlation r = %.2f (p < 0.001)', r);
end

% Linear regression
mdl = fitlm(acc_pct, mahadata);

% Regression coefficients for y = b0 + b1*x.
b0 = mdl.Coefficients.Estimate(1);
b1 = mdl.Coefficients.Estimate(2);

% Coefficient of determination.
R2 = mdl.Rsquared.Ordinary;
r2_s = sprintf('R^2 = %.2f\n', R2);

xfit = linspace(min(acc_pct), max(acc_pct), 100);
yfit = b0 + b1*xfit;
mdl_s = sprintf('y = %.2fx  %.2f', b1, b0);
plot(xfit, yfit, 'r-', 'LineWidth', 1.5);

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

function [Results_KW_PW, total_good_ch_num] = my_kw_test(data, time_window_listNum, time_window_list, time_char, segD, kw_pvalue2use)
    Results_KW_PW = {};
    total_good_ch_num =[];

    for tw = 1:time_window_listNum
        time_window = time_window_list{1, tw};
        kw_result = kw_analyze_channel(data, time_window, segD.Actual_usedChannel);

        pvals = kw_result.("p from KW");
        [cri_p, sig_idx] = fdr_bh(pvals, kw_pvalue2use);
        ind_goodch_num = sum(sig_idx);
        kw_result.FDR = sig_idx;
        
        fprintf([time_char{1, tw},': Number of significant channels（adj-p < %.2f (FDR)）: %d\n'], kw_pvalue2use, ind_goodch_num);
        Results_KW_PW{end+1} = kw_result;
        total_good_ch_num(end+1) = ind_goodch_num;
    end

end


function [kw_result] = kw_analyze_channel(data, time_window, Actual_usedChannel)
    % data [Task × Trial × Time × Ch]
    [taskNum, trialNum, ~, chNum] = size(data);
    
    KW_pvals = nan(chNum, 1);
    H_vals = nan(chNum, 1);
    H_corr_vals = nan(chNum, 1);
    pH_vals = nan(chNum, 1);
    
    for ch = 1:chNum
        % 1Ch: [Task × Trial × Time]
        oneCh_data = squeeze(data(:,:,:,ch));  % -> Task × Trial × Time
    
        % Average: [Task × Trial]
        avg_data = nan(taskNum, trialNum);
        for i = 1:taskNum
            selectD = squeeze(oneCh_data(i, :, time_window)); % Trial * Time
            avg_data(i, :) = (mean(selectD, 2))';
        end
    
        for i = 1:taskNum
            selectD = avg_data(i, :)';
            selectY = repmat(i, size(selectD, 1), 1);
            if i == 1
                X = selectD;
                G = selectY;
            else
                X = cat(1, X, selectD);
                G = cat(1, G, selectY);
            end
        end
    

        valid_idx = ~isnan(X);
        X = X(valid_idx);
        G = G(valid_idx);
    
        % Kruskal-Wallis test
        [p, ~, stats] = kruskalwallis(X, G, 'off'); 
        [H, H_corrected, pH] = kw_h_calculate(stats);
        KW_pvals(ch) = p;
        H_vals(ch) = H;
        H_corr_vals(ch) = H_corrected;
        pH_vals(ch) = pH;
    end
    
    
    kw_result = table((1:chNum)', Actual_usedChannel', ...
         H_vals,  H_corr_vals, pH_vals, KW_pvals, ...
        'VariableNames',{'ChNum', 'ActualChNum', 'H', 'Corrected H', 'p from H', 'p from KW'});
end

function [H, H_corrected, p] = kw_h_calculate(stats)
    N = sum(stats.n);                    
    meanRanks = stats.meanranks;
    ni = stats.n;
    
    H = (12 / (N * (N + 1))) * sum(ni .* meanRanks.^2) - 3 * (N + 1);
    

    if isfield(stats, 'sumt') && stats.sumt > 0
        tieCorrection = 1 - stats.sumt / (N * (N^2 - 1));
        H_corrected = H / tieCorrection;
    else
        H_corrected = H;
    end
    
    k = length(stats.n); 
    
    p = 1 - chi2cdf(H_corrected, k - 1);

end

% False Discovery Rate for solving multiple comparisons
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

function [newfourDdata] = cat_inputFeatures(data1, data2)

    if ~ isequal(size(data1), size(data2))
        error('The data structures of the entered data are different..')
    end
    [taskNum, trialNum, dataNum, chNum] = size(data1);
    
    newfourDdata = nan(taskNum, trialNum, dataNum, chNum*2);
    
    for taski = 1:taskNum
        for triali = 1:trialNum
            clipdata1 = squeeze(data1(taski, triali, :, :));
            clipdata2 = squeeze(data2(taski, triali, :, :));
            catD = cat(2, clipdata1, clipdata2);
            newfourDdata(taski, triali, :, :) = catD;
        end
    
    end

end

function [used_ch_ind] = get_timeWbased_sigChind(TargetTimeWind, usedfeature, time_char, Results_KW_TX, Results_KW_PW, CurrentArray, CurrentSubA, Array_chNum, Array_allocation, ChNum)
    TimeWindNum = find(string(time_char) == TargetTimeWind);

    ch_ind_tx = Results_KW_TX{TimeWindNum}.FDR;
    ch_ind_pw = Results_KW_PW{TimeWindNum}.FDR;
    

    array_ch_ind = array_ch_ind_make(CurrentArray, CurrentSubA, Array_chNum, Array_allocation, ChNum);

    ch_ind_tx_array = ch_ind_tx & array_ch_ind;
    ch_ind_pw_array = ch_ind_pw & array_ch_ind;
    
    if usedfeature == "TX"
        used_ch_ind = ch_ind_tx_array;
    elseif usedfeature == "PW"
        used_ch_ind = ch_ind_pw_array;
    elseif usedfeature == "TX_PW"
        used_ch_ind = [ch_ind_tx_array; ch_ind_pw_array];
    end

end

function [y_gnd_allFolds, y_pred_allFolds, mean_acc_allFolds] = my_nfold_cv_SVM(cv, currentAllData, ydata)

    % 結果を格納
    y_gnd_allFolds = [];
    y_pred_allFolds = [];
    mean_acc_allFolds = [];

    [taskNum, trialNum, dataNum, chNum] = size(currentAllData);
    
    %%%%%%%%%%%%%%% Step0. Convert to Task * Trial * data * Chを (Task * Trial) * data * Ch
    newAllignedDat = nan(taskNum*trialNum, dataNum, chNum);
    k = 1;
    for taski = 1:taskNum
        for triali = 1:trialNum
            clippedData = squeeze(currentAllData(taski, triali, :, :));
            newAllignedDat(k, :, :) = clippedData;
            k = k + 1;
        end
    end
    

    % % Progress bar
    % h = waitbar(0, sprintf('Running CV 0/%d...', cv.NumTestSets));
    % c = onCleanup(@() safeCloseWaitbar(h));  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 10Fold loop
    for cvi = 1:cv.NumTestSets
        %%%%%%%%%%%%%%% Step1. Train, Test-Ind
        train_idx = training(cv, cvi);
        test_idx = test(cv, cvi);
        
        %%%%%%%%%%%%%%% Step2. calculate mn and sd for z-scoring using Train data

        data3D_train = newAllignedDat(train_idx, :, :);

        mu_std_data = [];
        for triali = 1:sum(train_idx)
            if isempty(mu_std_data)
                mu_std_data = squeeze(data3D_train(triali, :, :));
            else
                mu_std_data = cat(1, mu_std_data, squeeze(data3D_train(triali, :, :)));
            end
        end

        mn = mean(mu_std_data, 1);
        sd = std(mu_std_data, 0, 1);

        %%%%%%%%%%%%%%% Step3. z-scoring
        xdata = nan(taskNum*trialNum, chNum);
        k = 1;
        for taski = 1:taskNum
            for triali = 1:trialNum
                clippeddata = squeeze(currentAllData(taski, triali, :, :));
                z_clippeddata = (clippeddata - mn)./(sd + 1e-8); 
                xdata(k, :) = mean(z_clippeddata, 1); 
                k = k + 1;
            end
        end
    
        %%%%%%%%%%%%%%% Step4. z-scoring for Train data
        X_train = xdata(train_idx,:);
        y_train = ydata(train_idx);
        X_test  = xdata(test_idx,:);
        y_test  = ydata(test_idx);
    
        % train sugin SVM
        t = templateSVM('KernelFunction', 'linear');
        % t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto'); 
        SVMModel = fitcecoc(X_train, y_train, 'Learners', t); 

        pred_val = predict(SVMModel, X_test);
        
        % Results
        y_gnd_allFolds = [y_gnd_allFolds; y_test];
        y_pred_allFolds = [y_pred_allFolds; pred_val];
        mean_acc_allFolds = [mean_acc_allFolds; mean(pred_val == y_test)];

        % ---- Update progress ----
        % waitbar(cvi / cv.NumTestSets, h, ...
        %     sprintf('Running CV %d/%d...', cvi, cv.NumTestSets));
    
    end

    function safeCloseWaitbar(h)
        if isvalid(h); close(h); end
    end


end

function [ydata] = create_ydata(myOrder, trialNum)
    ydata = [];
    for i = 1:numel(myOrder)
        if isempty(ydata)
            ydata = repelem(myOrder(i), trialNum, 1);
        else
            ydata = cat(1, ydata, repelem(myOrder(i), trialNum, 1));
        end
    end
    ydata = categorical(ydata);

end

function [array_ch_ind] = array_ch_ind_make(CurrentArray, CurrentSubA, Array_chNum, Array_allocation, ChNum)

    selectChNum = Array_chNum{Array_allocation == CurrentArray};

    % Subarra index

    subarray1_ind = 1:64; %  subarray1
    subarray2_ind = 65:128; % subarray2
    
    if CurrentSubA == 1 % sub1
        SubInd = subarray1_ind;
    elseif CurrentSubA == 2 % sub2
        SubInd = subarray2_ind;
    elseif CurrentSubA == 3 % sub1+sub2
        SubInd = 1:128;
    else
        error('Error.')
    end
    
    if CurrentArray == "All"
        usedChNum = selectChNum;
    else
        usedChNum = selectChNum(SubInd);
    end
    
    array_ch_ind = zeros(ChNum, 1);
    array_ch_ind(usedChNum) = 1;
    array_ch_ind = logical(array_ch_ind);

end

function [gnd_new] = convert_myOrder(myOrder, myOrder_short, gnd)
    catNum = numel(countcats(gnd));
    gnd_new = cell(numel(gnd), 1);
    for cati = 1:catNum
        ind_gnd = (gnd==myOrder(cati));
        gnd_new(ind_gnd) = {myOrder_short(cati)};
    end

    gnd_new = categorical(string(gnd_new));

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

function [usedChInd, anaD, t_range] = data_vec_input_make(inputFeature, usedArray, ana_time, Array_allocation, Array_chNum, inputFeaturesTX, inputFeaturesPW, inputFeaturesTX_PW, time_window_list, time_char)

    usedChInd = zeros(max([Array_chNum{:}]), 1);
    usedChInd(Array_chNum{Array_allocation==usedArray}) = 1;
    usedChInd = logical(usedChInd);
    
    if inputFeature == "TX"
        anaD = inputFeaturesTX;
    elseif inputFeature == "PW"
        anaD = inputFeaturesPW;
    elseif inputFeature == "TX_PW"
        anaD = inputFeaturesTX_PW;
        usedChInd = cat(1, usedChInd, usedChInd);
    end
    
    t_range = time_window_list{time_char == ana_time};
end

function [inputFeatures, transcriptions, labelslist, inputFeatures_orig, transcriptions_orig, labelslist_orig] = created_data_for_inputMLP(usedInputData, myOrder, SelectedTimeIdx, usedChInd)

    [taskNum, trialNum, ~, ~] = size(usedInputData);

    inputFeatures_orig = [];
    transcriptions_orig = [];
    labelslist_orig = {};

    for taski = 1:taskNum
        currentLabel = myOrder{taski};
        for triali = 1:trialNum
            currentData = squeeze(usedInputData(taski, triali, SelectedTimeIdx, usedChInd));

            inputFeatures_orig = [inputFeatures_orig;mean(currentData)];
            transcriptions_orig = [transcriptions_orig; taski]; 
            if isempty(labelslist_orig)
                labelslist_orig{1} = currentLabel;
            else
                labelslist_orig = cat(1, labelslist_orig, currentLabel);
            end
        end

    end

    num_samples = size(inputFeatures_orig, 1);
    % rng(42);  % Any fixed value (to ensure reproducibility)
    randIdx = randperm(num_samples);

    inputFeatures = inputFeatures_orig(randIdx, :);
    transcriptions = transcriptions_orig(randIdx);
    labelslist = labelslist_orig(randIdx);

end

function [sig_idx] = kw_test_with_FDR(inputFeatures_orig, transcriptions_orig, alpha_for_FDR)

   
    [~, ChNum] = size(inputFeatures_orig);
    P_results = [];

    for chi = 1:ChNum
        p = kruskalwallis(inputFeatures_orig(:, chi), transcriptions_orig, 'off');
        P_results = [P_results; p];
    end
    
    % FDR correction
    [crit_p, sig_idx] = fdr_bh(P_results, alpha_for_FDR);

end

function [idxCombination] = my_train_val_test_idx_make_kfold(transcriptions, k)

    idxCombination = {};
    cv1 = cvpartition(transcriptions, "KFold", k, 'Stratify', true);
    for itei = 1:cv1.NumTestSets
        idxTrain = find(training(cv1, itei));     
        idxTemporal = find(test(cv1, itei));       
        
        cv2 = cvpartition(transcriptions(idxTemporal), 'Holdout', 0.5, 'Stratify', true);
        idxVal = idxTemporal(training(cv2));
        idxTest = idxTemporal(test(cv2));
        idxCombination{1, itei} = idxTrain;
        idxCombination{2, itei} = idxVal;
        idxCombination{3, itei} = idxTest;
    end


end

function [Accuracies, All_y, All_pred, macro_F1] = my_MLP_train_and_test(k, inputFeatures, transcriptions, idxCombination, data_augmentation, l2norm, layers, sigChIdx, figureOn, myOrder)

    All_y = [];
    All_pred = [];
    Accuracies = [];

    for ki = 1:k
    
        for val_test = 1:2
            
            [Xtrain, Ytrain] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{1, ki});
            if val_test == 1
                [Xval, Yval] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{2, ki});
                [Xtest, Ytest] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{3, ki});
            else
                [Xval, Yval] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{3, ki});
                [Xtest, Ytest] = create_inputfeatures_transcript(inputFeatures, transcriptions, idxCombination{2, ki});
            end
    
            % Augmentation
            if data_augmentation
                M = 5; 
                [Xtrain, Ytrain] = create_augmentation(M, Xtrain, Ytrain);
            end
    
            
            mu = mean(Xtrain);
            sigma = std(Xtrain);
            sigma(sigma==0) = 1; 
    
            
            [Xtrain_z] = my_zscore_l2norm(Xtrain, mu, sigma, l2norm); 
            [Xval_z] = my_zscore_l2norm(Xval, mu, sigma, l2norm);
            [Xtest_z] = my_zscore_l2norm(Xtest, mu, sigma, l2norm);
    
            ValidationFrequencyNum = ceil(size(Xtrain_z, 1)/32); 
    
   

            options = trainingOptions('adam', ...
                'InitialLearnRate', 1e-3, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropFactor', 0.5, ...
                'LearnRateDropPeriod', 10, ...
                'L2Regularization', 1e-4, ...
                'MaxEpochs', 1000, ...
                'MiniBatchSize', 32, ...
                'Shuffle', 'every-epoch', ...
                'ValidationData', {Xval_z(:, sigChIdx), categorical(Yval)}, ...
                'ValidationPatience', 5, ...
                'ValidationFrequency', ValidationFrequencyNum, ...
                'ExecutionEnvironment','auto', ...
                'Verbose',false, ...
                'Plots','none');
    
            % Train
            net = trainNetwork(Xtrain_z(:, sigChIdx), categorical(Ytrain), layers, options);
    
            % Test
            Ypred = classify(net, Xtest_z(:, sigChIdx));
            accuracy = mean(Ypred == categorical(Ytest));
            Accuracies = [Accuracies;accuracy];
            fprintf('Test accuracy: %.2f%%\n', accuracy * 100);
    
            All_y = [All_y;categorical(Ytest)];
            All_pred = [All_pred;Ypred];
        end
    
    end
  
    Final_accuracy = mean(All_y == All_pred);
    fprintf('Total Test accuracy: %.2f%%\n', Final_accuracy * 100);

    % F1 score
    macro_F1 = f1_cal(All_y, All_pred);

    fprintf('F1 score: %.2f\n', macro_F1);

    if figureOn
        figure, confusionchart(All_y, All_pred, "ColumnDisplayLabels",myOrder, 'RowDisplayLabels',myOrder)
    end

end


function [output_x, output_y] = create_augmentation(M, Xtrain, Ytrain)

    
    TaskNum = numel(countcats(categorical(Ytrain)));
    AugData_All = {};
    AugLabelResults = {};
    for Taski = 1:TaskNum
    

        selectData = Xtrain(Ytrain==Taski, :);
        [sampN, ChNum] = size(selectData);
        X_aug = zeros(sampN*M, ChNum);
        

        mu_temp = mean(selectData);
        Sigma_temp = cov(selectData);
        epsilon = 1e-6;                       
        Sigma_temp = Sigma_temp + epsilon * eye(ChNum);     
        

        L = chol(Sigma_temp, 'lower');             % ChNum x ChNum
        
        for i = 1:sampN
            for m = 1:M
                    
                    Y = randn(1, ChNum);
            
                    
                    C = 1e-6 * randn(1, ChNum);   
            
                    
                    x_aug = Y * L' + mu_temp + C;
            
                    
                    X_aug((i-1)*M + m, :) = x_aug;
            end
        end
        
        
        AugDataResults = cat(1, selectData, X_aug);
        AuglabelsData = repmat(Taski, size(AugDataResults, 1), 1);
    
        AugData_All{end+1} = AugDataResults;
        AugLabelResults{end+1} = AuglabelsData;
    
    end
    
    
    aug_xdata = cat(1, AugData_All{:});
    aug_ydata = cat(1, AugLabelResults{:});
    
    randIdx = randperm(numel(aug_ydata));

    output_x = aug_xdata(randIdx, :);
    output_y = aug_ydata(randIdx);

end

function [XTrain_z] = my_zscore_l2norm(Xtrain, mu, sigma, l2norm)

    XTrain_z = (Xtrain - mu)./sigma;

    if l2norm
        norms = vecnorm(XTrain_z, 2, 2);
        norms(norms == 0) = 1; 
        XTrain_z = XTrain_z ./norms;
    end
end

function [clipped_input, clipped_transcription] = create_inputfeatures_transcript(inputFeatures, transcriptions, idx)
    clipped_input = inputFeatures(idx, :);
    clipped_transcription = transcriptions(idx);
end

function [macro_F1] = f1_cal(All_y, All_pred)

    % F1 score
    labels = unique(All_y); 
    numClasses = numel(labels);
    F1_per_class = zeros(numClasses, 1);
    
    for i = 1:numClasses
        class = labels(i);
        tp = sum((All_pred == class) & (All_y == class));
        fp = sum((All_pred == class) & (All_y ~= class));
        fn = sum((All_pred ~= class) & (All_y == class));
        
        precision = tp / (tp + fp + eps);
        recall    = tp / (tp + fn + eps);
        F1_per_class(i) = 2 * (precision * recall) / (precision + recall + eps);
    end

    macro_F1 = mean(F1_per_class);

end


function [loss, state, gradients] = modelLossFunction(net, dlX, dlT_recon, dlT_class_onehot, alpha)

    % Forward pass
    [dlY_recon, dlY_class, state] = forward(net, dlX, Outputs=["decoder_output", "softmax"], InputDataFormats="BC");
    % calculate Loss
    lossMSE = mse(dlY_recon, dlT_recon, 'DataFormat', "BC");
    lossCE = crossentropy(dlY_class, dlT_class_onehot, 'DataFormat', "BC");


    loss = alpha * lossMSE + (1 - alpha) * lossCE;

    gradients = dlgradient(loss, net.Learnables);
end

function [output] = make_cat_string(x)
    if numel(x) > 1
        output = x{1} + " " + x{2};
    else
        output = x{:};
    end

    output = string(output);

end

function [output] = make_cat_gesture(x)
    if numel(x) > 1
        output = x{3};
    else
        output = x{:};
    end

    output = string(output);

end

function [ellipse, mu] = create_circle(selectedPoints, scale)

    covMatrix = cov(selectedPoints);
    mu = mean(selectedPoints);

    [V, D] = eig(covMatrix);  % V: eigenvectors, D: eigenvalues (diag)
    

    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];  
    
    ellipse = (V * sqrt(D) * scale * circle)' + mu;

end

function [data_vec] = my_data_vec_create(anaD, usedChInd, t_range)  

    [num_task, num_trial, ~, ~] = size(anaD);
    usedChNum = sum(usedChInd);

    data_vec = zeros(num_task, num_trial, usedChNum); 

    for taski = 1:num_task
        for triali = 1:num_trial
            temp = squeeze(anaD(taski, triali, t_range, usedChInd));
            avg_temp = mean(temp, 1); 
            data_vec(taski, triali, :) = avg_temp;
        end
    end

end

function distance_matrix = my_Mahalanobis(data_vec)
    [num_task, ~, ChNum] = size(data_vec);
    
    means = zeros(num_task, ChNum);
    covs = cell(num_task, 1);
    
    for task = 1:num_task
        X = squeeze(data_vec(task, :, :));  
        means(task, :) = mean(X, 1);
        covs{task} = cov(X); 
    end
    
    distance_matrix = zeros(num_task, num_task);
    for i = 1:num_task
        mu_i = means(i, :)';
        for j = 1:num_task
            mu_j = means(j, :)';
            cov_avg = 0.5 * (covs{i} + covs{j}) + 1e-3 * eye(ChNum);  
            d = sqrt((mu_i - mu_j)' * (cov_avg \ (mu_i - mu_j)));
            distance_matrix(i, j) = d;
        end
    end
end

function my_dendrogram_plot(distance_matrix, myOrder, dendroLabelL, Color_val, Fsize, RotaionAngel)

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