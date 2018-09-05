import helpermethods
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../../')

from edgeml.trainer.bonsaiTrainer import BonsaiTrainer
from edgeml.graph.bonsai import Bonsai


def main():
    # Fixing seeds for reproducibility
    tf.set_random_seed(42)
    np.random.seed(42)

    # Hyper Param pre-processing
    args = helpermethods.getArgs()

    sigma = args.sigma
    depth = args.depth

    regT = args.rT
    regW = args.rW
    regV = args.rV

    totalEpochs = args.epochs

    learningRate = args.learning_rate

    dataDir = args.data_dir

    outFile = args.output_file

    (dataDimension, numClasses,
        Xtrain, Ytrain, Xtest, Ytest) = helpermethods.preProcessData(dataDir)

    sparW = args.sW
    sparV = args.sV
    sparT = args.sT

    if args.batch_size is None:
        batchSize = np.maximum(100, int(np.ceil(np.sqrt(Ytrain.shape[0]))))
    else:
        batchSize = args.batch_size

    useMCHLoss = True

    if numClasses == 2:
        numClasses = 1

    X = tf.placeholder("float32", [None, dataDimension])
    Y = tf.placeholder("float32", [None, numClasses])

    currDir = helpermethods.createTimeStampDir(dataDir)

    helpermethods.dumpCommand(sys.argv, currDir)

    # numClasses = 1 for binary case
    bonsaiObj = Bonsai(numClasses, dataDimension, depth, sigma)

    bonsaiTrainer = BonsaiTrainer(bonsaiObj,
                                  regW, regT, regV,
                                  sparW, sparT, sparV,
                                  learningRate, X, Y, useMCHLoss, outFile)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    bonsaiTrainer.train(batchSize, totalEpochs, sess,
                        Xtrain, Xtest, Ytrain, Ytest, dataDir, currDir)


if __name__ == '__main__':
    main()

# For the following command:
# Data - Curet
# python2 bonsai_example.py -dir ./curet/ -d 2 -p 22 -rW 0.00001 -rV 0.00001 -rT 0.000001 -sW 0.5 -sV 0.5 -sT 1 -e 300 -s 0.1 -b 20
# Final Output - useMCHLoss = True
# Maximum Test accuracy at compressed model size(including early stopping): 0.93727726 at Epoch: 297
# Final Test Accuracy: 0.9337135

# Non-Zeros: 24231.0 Model Size: 115.65625 KB hasSparse: True

# Data - usps2
# python2 bonsai_example.py -dir /mnt/c/Users/t-vekusu/Downloads/datasets/usps-binary/ -d 2 -p 22 -rW 0.00001  -rV 0.00001 -rT 0.000001  -sW 0.5 -sV 0.5 -sT 1 -e 300 -s 0.1 -b 20
# Maximum Test accuracy at compressed model size(including early stopping): 0.9521674 at Epoch: 246
# Final Test Accuracy: 0.94170403

# Non-Zeros: 2636.0 Model Size: 19.1328125 KB hasSparse: True
