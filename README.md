<h1>Atrial Fibrillation Screening System with Specificity Improvement by Applying Deep Learning Model</h>

<h2>Sequence-to-sequence classification by using TCN</h2>
A TCN (Temporal Convolutional Network) model was used to predict 2 class of ECG signals, Non-AF and AF.
For the TCN model, there are the original code example from MATLAB help center.
Please visit this <a href = "https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-classification-using-1-d-convolutions.html;jsessionid=b8db79fa013c93b2af62544f8454">link</a>.

<h3>ECG data set description</h3> 
:card_index_dividers:
<p>The training set and testing set for 30-second ECG segments were collected in mat files.
For example, in this folder, Train/2cats_N2PA/…, there are</p>

:white_small_square:_ArrhythmiasTrain_30sec_2cats_N2PA_CutAF_900seg_NAME.mat_

:white_small_square:_ArrhythmiasTrain_30sec_2cats_N2PA_CutAF_900seg.mat_

The 2 files are consisting of NSR (300 segments), AF (900 segments), PAC (300 segments)  and PVC (300 segments) arrhythmias. The PAC and PVC segments are not contaminated with AF signals. Thus, these 2 files are called CutAF.

:white_small_square:  _ArrhythmiasTrain_30sec_2cats_N2PA_nonCutAF_900seg_NAME.mat_

:white_small_square:  _ArrhythmiasTrain_30sec_2cats_N2PA_nonCutAF_900seg.mat_

And these 2 files are like the 2 file above, however, AF signals are including in the PAC and PVC segments. That is why they are called nonCutAF.

Now, if we want to access the training ECG signals without AF information, we can see there are 2 files with similar file name.

:label:_ArrhythmiasTrain_30sec_2cats_N2PA_CutAF_900seg_NAME.mat_

:anatomical_heart:_ArrhythmiasTrain_30sec_2cats_N2PA_CutAF_900seg.mat_

The first one is collecting file details of samples used in the training set such as file name, file address, number of data points, etc.
And the second one is containing ECG signal segments and label data in XTrain and YTrain variables, respectively.

<h3>Here is an example code for accessing the data set.</h3>

%% Load 30-sec ECG segment data set
% Specify the working directory

`cd ~/Desktop/data/Train/2cats_N2PA`

% Import ECG data — There are 2 components i.e. XTrain and YTrain variables

<img width="555" alt="Screen Shot 2565-09-02 at 14 53 20" src="https://user-images.githubusercontent.com/79197378/188087190-d0fdfcf3-c997-48b9-abe9-f3ab24d30bf2.png">

`ECGdat = load("ArrhythmiasTrain_30sec_2cats_N2PA_CutAF_900seg.mat");`

% Training set and testing set

<img width="310" alt="Screen Shot 2565-09-02 at 14 53 56" src="https://user-images.githubusercontent.com/79197378/188087373-9fbe04c4-8813-44e7-927e-e3c6a9412586.png">

`XTrain = ECGdat.XTrain;`

`YTrain = ECGdat.YTrain;`

% Save data as a csv file

`writecell(XTrain, "ECG_signal.csv")`

`writecell(YTrain, "ECG_label.csv")`

<img width="180" alt="Screen Shot 2565-09-02 at 14 57 31" src="https://user-images.githubusercontent.com/79197378/188087635-a10a0677-e0cd-41b3-9fe3-6b38cedd4b73.png">

For signal file (ECG_signal.csv), rows are the 30-sec ECG segments (1-by-3840 ECG signal points). 
And each point was labeled as a type of cardiac arrhythmias (Non-AF and AF).
Now, we can use the data for not only MATLAB but other platforms also.
