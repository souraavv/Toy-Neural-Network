#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#include "utility.h"
#include "readcfile.h"
#include "mlpclassifier.h"


int main(int argc, char* argv[]) {
    
    if (argc < 10) {
        printf ("Given proper arguements:\n");
        printf ("int noOfLayers                  :: argv[1]\nint noOfFeatures                :: argv[2]\nint noOfIteration               :: argv[3]\ndouble learningRate             :: argv[4]\nint* neuronsInLayer             :: argv[5] (a list)\nint* layersActivationFunc       :: argv[6] (a list)\nint noOfRows                    :: argv[7]\nint partition                   :: argv[8]\nfile name                       :: argv[9]\nflag (stochastic/batch/mini)    :: argv[10]\nnumber of batch mini-batch      :: argv[11]");
        exit(0);
    }

    struct NeuralNetwork* newNet = (struct NeuralNetwork*) malloc (sizeof(struct NeuralNetwork));
    newNet->noOfLayers = atoi(argv[1]);
    newNet->noOfHiddenLayer = newNet->noOfLayers - 2;
    newNet->noOfFeatures = atoi(argv[2]);
    newNet->noOfIteration = atoi(argv[3]);
    newNet->learningRate = atof(argv[4]);
    newNet->neuronsInLayer = (int*) malloc(sizeof(int) * newNet->noOfLayers);
    char* curCnt = strtok(argv[5], ",");
    newNet->neuronsInLayer[0] = newNet->noOfFeatures;
    for (int layer = 1; layer < newNet->noOfLayers; ++layer) {
        newNet->neuronsInLayer[layer] = atoi(curCnt);
        curCnt = strtok(NULL, ",");
    }
    newNet->layersActivationFunc = (int*) malloc(sizeof(int) * newNet->noOfLayers);
    char* curAct = strtok(argv[6], ",");
    for (int layer = 1; layer < newNet->noOfLayers; ++layer) {
        newNet->layersActivationFunc[layer] = (layer + 1 == newNet->noOfLayers) ? giveActFuncLastLayer(curAct) : giveActFunc(curAct);
        curAct = strtok(NULL, ",");
    }

    newNet->deltaValues = (double**)malloc(sizeof(double*) * newNet->noOfLayers);
    for (int layer = 0; layer < newNet->noOfLayers; ++layer) {
        newNet->deltaValues[layer] = (double*) malloc (sizeof(double) * (newNet->neuronsInLayer[layer] + 1)); // 1 based
    }

    newNet->actValue = (double**)malloc(sizeof(double*) * newNet->noOfLayers);
    for (int layer = 0; layer < newNet->noOfLayers; ++layer) {
        newNet->actValue[layer] = (double*) malloc (sizeof(double) * (newNet->neuronsInLayer[layer] + 1)); // 1 to n neurons
    }

    newNet->zValue = (double**)malloc(sizeof(double*) * newNet->noOfLayers);
    for (int layer = 0; layer < newNet->noOfLayers; ++layer) {
        newNet->zValue[layer] = (double*) malloc (sizeof(double) * (newNet->neuronsInLayer[layer] + 1));
    }

    newNet->bias = (double**)malloc(sizeof(double*) * (newNet->noOfLayers - 1));
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        newNet->bias[layer] = (double*) malloc (sizeof(double) * (newNet->neuronsInLayer[layer + 1] + 1)); // 1 based indexing
    }

    newNet->weight = (double***)malloc(sizeof(double**) * (newNet->noOfLayers - 1));
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        newNet->weight[layer] = (double**)malloc(sizeof(double*) * (newNet->neuronsInLayer[layer] + 1)); // 1 based indexing
    }

    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        for (int cur_neuron = 1; cur_neuron <= newNet->neuronsInLayer[layer]; ++cur_neuron) {
            newNet->weight[layer][cur_neuron] = (double*) malloc (sizeof(double) * (newNet->neuronsInLayer[layer + 1] + 1)); // 1 based indexing
        }
    }
    newNet->actualOutput = (double*) malloc (sizeof(double) * (newNet->neuronsInLayer[newNet->noOfLayers - 1] + 1));
    for (int i = 0; i < (newNet->neuronsInLayer[newNet->noOfLayers - 1] + 1); ++i) {
        newNet->actualOutput[i] = 0.0;
    }
    newNet->noOfRows = atoi(argv[7]);
    newNet->partition = atoi(argv[8]);
    int trainRows = (newNet->partition * newNet->noOfRows) / 100; 
    int testRows = newNet->noOfRows - trainRows;
    double l = 0.0001, r = 0.001;
    double diff = r - l;
    
    srand(time(NULL));
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        for (int cur_neuron = 1; cur_neuron <= newNet->neuronsInLayer[layer]; ++cur_neuron) {
            for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[layer + 1]; ++next_neuron) {
                double rand_w = l + (diff * (double)rand())/(double)RAND_MAX;
                newNet->weight[layer][cur_neuron][next_neuron] = rand_w;
            }
        }
    }
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[layer + 1]; ++next_neuron) {
            newNet->bias[layer][next_neuron] = 0.0;
        }
    }

    double** inputData;
    double** outputData;
    inputData = (double**) malloc (sizeof(double*) * (newNet->noOfRows + 1));
    outputData = (double**) malloc (sizeof(double*) * (newNet->noOfRows + 1));
    int outputCol = newNet->neuronsInLayer[newNet->noOfLayers - 1] + 1;
    
    for (int row = 0; row <= newNet->noOfRows; ++row) {
        outputData[row] = (double*) malloc (sizeof(double) * outputCol);
        inputData[row] = (double*) malloc (sizeof(double) * (newNet->noOfFeatures + 1));
    }
    char* fileName = argv[9];
    char path []= "./src/";
    strcat(path, fileName);
    readDataFile(newNet, path, inputData, outputData, newNet->noOfFeatures, newNet->noOfRows);

    int flag = atoi(argv[10]);
    if (flag == 1) {
        training (newNet, inputData, outputData, trainRows);

    }
    else if (flag == 2) {
        training_batch(newNet, inputData, outputData, trainRows, 1);
    }
    else {
        int numberOfBatch = atoi(argv[11]);
        training_batch(newNet, inputData, outputData, trainRows, numberOfBatch);
    }
    testing (newNet, inputData, outputData, testRows, trainRows + 1);
    return 0;
}
