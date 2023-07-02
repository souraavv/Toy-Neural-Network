#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#include "utility.h"
#include "mlpregression.h"

double*** meanWeight;
double** meanBias;

FILE* ptrS, *ptrB, *ptrMB;

void reset_mean_weight_and_bias(struct NeuralNetwork* newNet) {
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        for (int neuron = 1; neuron <= newNet->neuronsInLayer[layer + 1]; ++neuron) {
            meanBias[layer][neuron] = 0.0;
        }
    }
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        for (int cur_neuron = 1; cur_neuron <= newNet->neuronsInLayer[layer]; ++cur_neuron) {
            for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[layer + 1]; ++next_neuron) {
                meanWeight[layer][cur_neuron][next_neuron] = 0.0;
            }
        }
    }
}
void init_mean_weight_and_bias(struct NeuralNetwork* newNet) {
    meanBias = (double**)malloc(sizeof(double*) * (newNet->noOfLayers - 1));
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        meanBias[layer] = (double*) malloc (sizeof(double) * (newNet->neuronsInLayer[layer + 1] + 1)); // 1 based indexing
    }
    meanWeight = (double***)malloc(sizeof(double**) * (newNet->noOfLayers - 1));
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        meanWeight[layer] = (double**)malloc(sizeof(double*) * (newNet->neuronsInLayer[layer] + 1)); // 1 based indexing
        
    }
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        for (int cur_neuron = 1; cur_neuron <= newNet->neuronsInLayer[layer]; ++cur_neuron) {
            meanWeight[layer][cur_neuron] = (double*) malloc (sizeof(double) * (newNet->neuronsInLayer[layer + 1] + 1)); // 1 based indexing
        }
    }
    reset_mean_weight_and_bias(newNet);
}

void update_weight_and_bias(struct NeuralNetwork* newNet, int batchSize) {
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        for (int cur_neuron = 1; cur_neuron <= newNet->neuronsInLayer[layer]; ++cur_neuron) {
            for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[layer + 1]; ++next_neuron) {
                meanWeight[layer][cur_neuron][next_neuron] /= (double)batchSize;
                newNet->weight[layer][cur_neuron][next_neuron] -= newNet->learningRate * meanWeight[layer][cur_neuron][next_neuron];
            }
        }
    }
    for (int layer = 0; layer < newNet->noOfLayers - 1; ++layer) {
        for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[layer + 1]; ++next_neuron) {
            meanBias[layer][next_neuron] /= (double) batchSize;
            newNet->bias[layer][next_neuron] -= newNet->learningRate * meanBias[layer][next_neuron];
        }
    }
}

void compute_mean_weight_and_bias (struct NeuralNetwork* newNet) {
    int lastLayer = newNet->noOfLayers - 1;
    int lastLayerSize = newNet->neuronsInLayer[lastLayer];
    int outputLayerActFunc = newNet->layersActivationFunc[lastLayer];
    for (int neuron = 1; neuron <= lastLayerSize; ++neuron) {
        if (outputLayerActFunc == 1) { 
            newNet->deltaValues[lastLayer][neuron] = (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]);    
        }
        else { 
            newNet->deltaValues[lastLayer][neuron] = (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]); // dcBydz
        }
    }
    for (int curLayer = lastLayer - 1; curLayer > 0; --curLayer) {
        for (int neuron = 1; neuron <= newNet->neuronsInLayer[curLayer]; ++neuron) {
            double dcByda = 0.0; 
            int nextLayer = curLayer + 1;
            for (int neuron_next = 1; neuron_next <= newNet->neuronsInLayer[nextLayer]; neuron_next++) {
                dcByda += newNet->weight[curLayer][neuron][neuron_next] * newNet->deltaValues[nextLayer][neuron_next];
            }
            double daBydz = 0.0;
            int actFunc = newNet->layersActivationFunc[curLayer];
            if (actFunc == 1) {
                daBydz = derivSigmoid(newNet->zValue[curLayer][neuron]);
            } 
            else if (actFunc == 2) {
                daBydz = derivRelu(newNet->zValue[curLayer][neuron]);
            }
            else if (actFunc == 3) {
                daBydz = derivTanh(newNet->zValue[curLayer][neuron]);
            }
            newNet->deltaValues[curLayer][neuron] = dcByda * daBydz; // dcBydz;
        }
    }
    for (int layer = 0; layer <= newNet->noOfLayers - 2; layer++) {
        int nextLayer = layer + 1;
        for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[nextLayer]; ++next_neuron) {
            meanBias[layer][next_neuron] += newNet->deltaValues[nextLayer][next_neuron]; 
        }
        for (int cur_neuron = 1; cur_neuron <= newNet->neuronsInLayer[layer]; ++cur_neuron) {
            for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[nextLayer]; ++next_neuron) {
                double dcBydw = (newNet->deltaValues[nextLayer][next_neuron] * newNet->actValue[layer][cur_neuron]); 
                meanWeight[layer][cur_neuron][next_neuron] += dcBydw; 
            }
        }
    }
}

void forwardPropagation_regression (struct NeuralNetwork* newNet) {
    int totalLayers = newNet->noOfLayers;
    int finalLayer = totalLayers - 1;

    for (int curLayer = 1; curLayer < totalLayers; ++curLayer) {
        int prevLayer = curLayer - 1;
        int prevLayerSize =  newNet->neuronsInLayer[prevLayer];
        int curLayerSize = newNet->neuronsInLayer[curLayer];
        matrixMult(newNet->actValue[prevLayer], newNet->bias[prevLayer], newNet->weight[prevLayer], newNet->zValue[curLayer], prevLayerSize, curLayerSize);
        
        int actFunc = newNet->layersActivationFunc[curLayer];
        if (curLayer != finalLayer) {
            if (actFunc == 1) {
                computeSigmoid(newNet->actValue[curLayer], newNet->zValue[curLayer], curLayerSize);
            }
            else if(actFunc == 2) {
                computeRelu(newNet->actValue[curLayer], newNet->zValue[curLayer], curLayerSize);
            }
            else if (actFunc == 3) {
                computeTanh(newNet->actValue[curLayer], newNet->zValue[curLayer], curLayerSize);
            }
            else {
                printf ("error : Not defined activation Function\n");
                exit(0);
            }
        }   
        else if (curLayer == finalLayer){
            for (int neuron = 1; neuron <= newNet->neuronsInLayer[curLayer]; ++neuron) {
                newNet->actValue[curLayer][neuron] = newNet->zValue[curLayer][neuron];
            }
        }
    }
}

void backwardPropagation (struct NeuralNetwork* newNet) {
    int lastLayer = newNet->noOfLayers - 1; 
    int lastLayerSize = newNet->neuronsInLayer[lastLayer];
    int outputLayerActFunc = newNet->layersActivationFunc[lastLayer];
    for (int neuron = 1; neuron <= lastLayerSize; ++neuron) {
        if (outputLayerActFunc == 1) { 
            newNet->deltaValues[lastLayer][neuron] = (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]);
        }
        else {  
            newNet->deltaValues[lastLayer][neuron] = (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]); // dcBydz
        }
    }
    for (int curLayer = lastLayer - 1; curLayer > 0; --curLayer) {
        for (int neuron = 1; neuron <= newNet->neuronsInLayer[curLayer]; ++neuron) {
            double dcByda = 0.0; 
            int nextLayer = curLayer + 1;
            for (int neuron_next = 1; neuron_next <= newNet->neuronsInLayer[nextLayer]; neuron_next++) {
                dcByda += newNet->weight[curLayer][neuron][neuron_next] * newNet->deltaValues[nextLayer][neuron_next];
            }
            double daBydz = 0.0;
            int actFunc = newNet->layersActivationFunc[curLayer];
            if (actFunc == 1) {
                daBydz = derivSigmoid(newNet->zValue[curLayer][neuron]);
            } 
            else if (actFunc == 2) {
                daBydz = derivRelu(newNet->zValue[curLayer][neuron]);
            }
            else if (actFunc == 3) {
                daBydz = derivTanh(newNet->zValue[curLayer][neuron]);
            }
            newNet->deltaValues[curLayer][neuron] = dcByda * daBydz; // dcBydz;
        }
    }
    for (int layer = 0; layer <= newNet->noOfLayers - 2; layer++) {
        int nextLayer = layer + 1;
        for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[nextLayer]; ++next_neuron) {
            newNet->bias[layer][next_neuron] -= newNet->learningRate * newNet->deltaValues[nextLayer][next_neuron]; 
        }
        for (int cur_neuron = 1; cur_neuron <= newNet->neuronsInLayer[layer]; ++cur_neuron) {
            for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[nextLayer]; ++next_neuron) {
                double dcBydw = (newNet->deltaValues[nextLayer][next_neuron] * newNet->actValue[layer][cur_neuron]); 
                newNet->weight[layer][cur_neuron][next_neuron] -= newNet->learningRate * dcBydw; 
            }
        }
    }
}

void training_batch(struct NeuralNetwork* newNet, double** inputData, double** outputData, int trainRows, int numberOfBatch) {
    if (numberOfBatch == 1)
        ptrB = fopen ("r_batch.csv", "w");
    else
        ptrMB = fopen ("r_minibatch.csv", "w");

    int batchSize = (trainRows + numberOfBatch - 1) / numberOfBatch;
    init_mean_weight_and_bias(newNet);
    for (int itr = 1; itr <= newNet->noOfIteration; ++itr) {
        double loss = 0.0;
        for (int batch_id = 0; batch_id < numberOfBatch; ++batch_id) {
            int start = (batch_id * batchSize) + 1;
            int end = min(trainRows, start + batchSize - 1); 
            reset_mean_weight_and_bias(newNet);
            for (int row = start; row <= end; ++row) {
                newNet->actValue[0] = inputData[row];
                newNet->actualOutput = outputData[row];
                forwardPropagation_regression(newNet);
                loss += computeLoss(newNet);
                compute_mean_weight_and_bias(newNet);
            }
            update_weight_and_bias(newNet, batchSize);
        }
        double compLoss = loss / (double)trainRows;
        if(numberOfBatch == 1) {
            fprintf (ptrB, "%d,%f\n", itr, compLoss);
            print_loss("Training",loss, trainRows);
        }
        else {
            fprintf (ptrMB, "%d,%f\n", itr, compLoss);
            print_loss("Training",loss, trainRows);
        }
    }
    if (numberOfBatch == 1) fclose(ptrB);
    else fclose(ptrMB);
}

void training (struct NeuralNetwork* newNet, double** inputData, double** outputData, int trainRows) {
    ptrS = fopen ("r_stoc.csv", "w");
    for (int itr = 1; itr <= newNet->noOfIteration; ++itr) {
        double loss = 0.0;
        for (int row = 1; row <= trainRows; ++row) {
            newNet->actValue[0] = inputData[row]; 
            newNet->actualOutput = outputData[row]; 
            forwardPropagation_regression(newNet);
            loss += computeLoss(newNet);
            backwardPropagation(newNet);
        }
        double compLoss = loss / (double)trainRows;
        fprintf (ptrS, "%d,%f\n", itr, compLoss);
        print_loss("Regression Training For stochastic ", loss, trainRows);
    }
    fclose(ptrS);
}

void testing (struct NeuralNetwork* newNet, double** inputData, double** outputData, int testRows, int start) {
    double loss = 0.0;
    for (int row = start; row < start + testRows - 1; ++row) {
        newNet->actValue[0] = inputData[row]; 
        newNet->actualOutput = outputData[row]; 
        forwardPropagation_regression(newNet);
        loss += computeLoss(newNet);
    }
    print_loss("Regression Testing", loss, testRows);
}

