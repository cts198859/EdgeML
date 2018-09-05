# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''
 Functions to check sanity of input arguments
 for the example script.
'''
import argparse
import datetime
import os
import numpy as np
import pandas as pd

FEATURE_COLS = ['BITRATE_KBPS', 'BUFFER_SEC', 'DLTIME0_SEC', 'DLTIME1_SEC', 'DLTIME2_SEC',
                'DLTIME3_SEC', 'DLTIME4_SEC', 'DLTIME5_SEC', 'DLTIME6_SEC', 'DLTIME7_SEC',
                'THPT0_MBPS', 'THPT1_MBPS', 'THPT2_MBPS', 'THPT3_MBPS', 'THPT4_MBPS',
                'THPT5_MBPS', 'THPT6_MBPS', 'THPT7_MBPS', 'THP_REGIME_IND']
TARGET_COL = 'ACTION'


def checkIntPos(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def checkIntNneg(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid non-neg int value" % value)
    return ivalue


def checkFloatNneg(value):
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid non-neg float value" % value)
    return fvalue


def checkFloatPos(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive float value" % value)
    return fvalue


def getArgs():
    '''
    Function to parse arguments for Bonsai Algorithm
    '''
    parser = argparse.ArgumentParser(
        description='HyperParams for Bonsai Algorithm')
    parser.add_argument('-dir', '--data-dir',
                        default='/home/tchu/bonsai_train',
                        help='Data directory containing' +
                        'train.npy and test.npy')
    parser.add_argument('-d', '--depth', type=checkIntNneg, default=3,
                        help='Depth of Bonsai Tree ' +
                        '(default: 3 try: [0, 1, 3])')
    parser.add_argument('-s', '--sigma', type=float, default=1.0,
                        help='Parameter for sigmoid sharpness ' +
                        '(default: 1.0 try: [3.0, 0.05, 0.1]')
    parser.add_argument('-e', '--epochs', type=checkIntPos, default=42,
                        help='Total Epochs (default: 42 try:[100, 150, 60])')
    parser.add_argument('-b', '--batch-size', type=checkIntPos,
                        help='Batch Size to be used ' +
                        '(default: max(100, sqrt(train_samples)))')
    parser.add_argument('-lr', '--learning-rate', type=checkFloatPos,
                        default=0.01, help='Initial Learning rate for ' +
                        'Adam Optimizer (default: 0.01)')

    parser.add_argument('-rW', type=float, default=0.0001,
                        help='Regularizer for predictor parameter W  ' +
                        '(default: 0.0001 try: [0.01, 0.001, 0.00001])')
    parser.add_argument('-rV', type=float, default=0.0001,
                        help='Regularizer for predictor parameter V  ' +
                        '(default: 0.0001 try: [0.01, 0.001, 0.00001])')
    parser.add_argument('-rT', type=float, default=0.0001,
                        help='Regularizer for branching parameter Theta  ' +
                        '(default: 0.0001 try: [0.01, 0.001, 0.00001])')

    parser.add_argument('-sW', type=checkFloatPos, default=1.0,
                        help='Sparsity for predictor parameter W  ' +
                        '(default: For Binary classification 1.0 else 0.2 ' +
                        'try: [0.1, 0.3, 0.5])')
    parser.add_argument('-sV', type=checkFloatPos, default=1.0,
                        help='Sparsity for predictor parameter V  ' +
                        '(default: For Binary classification 1.0 else 0.2 ' +
                        'try: [0.1, 0.3, 0.5])')
    parser.add_argument('-sT', type=checkFloatPos, default=1.0,
                        help='Sparsity for branching parameter Theta  ' +
                        '(default: For Binary classification 1.0 else 0.2 ' +
                        'try: [0.1, 0.3, 0.5])')
    parser.add_argument('-oF', '--output-file', default=None,
                        help='Output file for dumping the program output, ' +
                        '(default: stdout)')

    return parser.parse_args()


def createTimeStampDir(dataDir):
    '''
    Creates a Directory with timestamp as it's name
    '''
    if os.path.isdir(dataDir + '/TFBonsaiResults') is False:
        try:
            os.mkdir(dataDir + '/TFBonsaiResults')
        except OSError:
            print("Creation of the directory %s failed" %
                  dataDir + '/TFBonsaiResults')

    currDir = 'TFBonsaiResults/' + datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%y")
    if os.path.isdir(dataDir + '/' + currDir) is False:
        try:
            os.mkdir(dataDir + '/' + currDir)
        except OSError:
            print("Creation of the directory %s failed" %
                  dataDir + '/' + currDir)
        else:
            return (dataDir + '/' + currDir)
    return None


def oneHot(Y, numClasses):
    lab = Y.astype('uint8')
    lab = np.array(lab) - min(lab)

    lab_ = np.zeros((len(Y), numClasses))
    lab_[np.arange(len(Y)), lab] = 1
    if (numClasses == 2):
        return np.reshape(lab, [-1, 1])
    else:
        return lab_


def preProcessData(dataDir):
    '''
    Function to pre-process input data
    Expects a .npy file of form [lbl feats] for each datapoint
    Outputs a train and test set datapoints appended with 1 for Bias induction
    dataDimension, numClasses are inferred directly
    '''
    train_df = pd.read_csv(dataDir + '/train_data/tv_live_abr_train.csv')
    test_df = pd.read_csv(dataDir + '/train_data/tv_live_abr_test.csv')
    train_df = tempProcess(train_df)
    test_df = tempProcess(test_df)
    train_df['THP_REGIME_IND'] = 0
    train_df.loc[train_df.THP_REGIME == 'hi', 'THP_REGIME_IND'] = 1
    test_df['THP_REGIME_IND'] = 0
    test_df.loc[test_df.THP_REGIME == 'hi', 'THP_REGIME_IND'] = 1
    Xtrain = train_df[FEATURE_COLS].values
    Ytrain_ = train_df[TARGET_COL].values
    Xtest = test_df[FEATURE_COLS].values
    Ytest_ = test_df[TARGET_COL].values
    dataDimension = int(Xtrain.shape[1])
    numClasses = np.max(Ytrain_) - np.min(Ytrain_) + 1
    numClasses = int(max(numClasses, max(Ytest_) - min(Ytest_) + 1))

    # Mean Var Normalisation
    mean = np.mean(Xtrain, axis=0)
    std = np.std(Xtrain, axis=0)
    std[std[:] < 0.000001] = 1
    Xtrain = (Xtrain - mean) / std
    Xtest = (Xtest - mean) / std
    # End Mean Var normalisation
    Ytrain = oneHot(Ytrain_, numClasses)
    Ytest = oneHot(Ytest_, numClasses)
    trainBias = np.ones([Xtrain.shape[0], 1])
    Xtrain = np.append(Xtrain, trainBias, axis=1)
    testBias = np.ones([Xtest.shape[0], 1])
    Xtest = np.append(Xtest, testBias, axis=1)

    return dataDimension + 1, numClasses, Xtrain, Ytrain, Xtest, Ytest


def tempProcess(df):
    new_df_ls = []
    for session in df.SESSION.unique():
        cur_df = df[df.SESSION == session]
        cur_df['ACTION'] = cur_df.ACTION.shift(-1)
        new_df_ls.append(cur_df.iloc[:-1])
    return pd.concat(new_df_ls).reset_index(drop=True)


def dumpCommand(list, currDir):
    '''
    Dumps the current command to a file for further use
    '''
    commandFile = open(currDir + '/command.txt', 'w')
    command = "python3"

    command = command + " " + ' '.join(list)
    commandFile.write(command)

    commandFile.flush()
    commandFile.close()
