#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#include "mlpclassifier.h"
#include "utility.h"

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
            double zLastLayer = newNet->zValue[lastLayer][neuron];
            newNet->deltaValues[lastLayer][neuron] = (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]) * derivSigmoid(zLastLayer);    
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

void forwardPropagation (struct NeuralNetwork* newNet) {
    int totalLayers = newNet->noOfLayers;
    int finalLayer = totalLayers - 1;

    for (int curLayer = 1; curLayer < totalLayers; ++curLayer) {
        int prevLayer = curLayer - 1;
        int prevLayerSize =  newNet->neuronsInLayer[prevLayer];
        int curLayerSize = newNet->neuronsInLayer[curLayer];
        // this computes z{cur_layer} = Weight{prev_layer} * activation{previous_layer} + bais{previous_layer}
        matrixMult(newNet->actValue[prevLayer], newNet->bias[prevLayer], newNet->weight[prevLayer], newNet->zValue[curLayer], prevLayerSize, curLayerSize);
        // get the activation function of this layer.
        int actFunc = newNet->layersActivationFunc[curLayer];
        // if this layers is not the final layer
        if (curLayer != finalLayer) {
            // take the z-values and compute the a value for this layer.
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
                printf ("error : Activation Function is not defined\n");
                exit(0);
            }
        }   
        else {
            // for final layers we are using softmax of sigmoid
            if (actFunc == 1) {
                computeSigmoid(newNet->actValue[curLayer], newNet->zValue[curLayer], curLayerSize);
            }       
            else if (actFunc == 2){
                computeSoftMax(newNet->actValue[curLayer], newNet->zValue[curLayer], curLayerSize);
            }
            else {
                printf ("error : Activation Function is not defined\n");
                exit(0);
            }
        }
    }
}

void backwardPropagation (struct NeuralNetwork* newNet) {
    // start from the last layer.
    int lastLayer = newNet->noOfLayers - 1; 
    int lastLayerSize = newNet->neuronsInLayer[lastLayer];
    // get the output layer activation function
    int outputLayerActFunc = newNet->layersActivationFunc[lastLayer];
    // computation of dc/dz for last layer. 
    for (int neuron = 1; neuron <= lastLayerSize; ++neuron) {
        if (outputLayerActFunc == 1) { 
            double zLastLayer = newNet->zValue[lastLayer][neuron];
            newNet->deltaValues[lastLayer][neuron] = (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]) * derivSigmoid(zLastLayer);    
        }
        else {  
            newNet->deltaValues[lastLayer][neuron] = (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]); // dcBydz
        }
    }
    // now start updating all the previous layer, starting from the second last.
    for (int curLayer = lastLayer - 1; curLayer > 0; --curLayer) {
        for (int neuron = 1; neuron <= newNet->neuronsInLayer[curLayer]; ++neuron) {
            double dcByda = 0.0; 
            int nextLayer = curLayer + 1;
            // compute dc/da
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
    for (int cur_layer = 0; cur_layer <= newNet->noOfLayers - 2; cur_layer++) {
        int nextLayer = cur_layer + 1;
        // for bias neuron (have edge from all the neurons in next cur_layers)
        // delta values have two faces : dc/dw and dc/db depending on which node we are
        // look at documentation : b(j)[L] , here L = cur_layer and j is from the next layer
        for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[nextLayer]; ++next_neuron) {
            newNet->bias[cur_layer][next_neuron] -= newNet->learningRate * newNet->deltaValues[nextLayer][next_neuron]; 
        }
        // for all other neurons (we have an edge from all the neurons in next layers)
        for (int cur_neuron = 1; cur_neuron <= newNet->neuronsInLayer[cur_layer]; ++cur_neuron) {
            for (int next_neuron = 1; next_neuron <= newNet->neuronsInLayer[nextLayer]; ++next_neuron) {
                // dcBydw = dcBydz * dzBydw
                double dcBydw = (newNet->deltaValues[nextLayer][next_neuron] * newNet->actValue[cur_layer][cur_neuron]); 
                newNet->weight[cur_layer][cur_neuron][next_neuron] -= newNet->learningRate * dcBydw; 
            }
        }
    }
}

void training_batch(struct NeuralNetwork* newNet, double** inputData, double** outputData, int trainRows, int numberOfBatch) {
    if (numberOfBatch == 1)
        ptrB = fopen ("c_batch.csv", "w");
    else
        ptrMB = fopen ("c_minibatch.csv", "w");

    int batchSize = (trainRows + numberOfBatch - 1) / numberOfBatch;
    init_mean_weight_and_bias(newNet);
    for (int itr = 1; itr <= newNet->noOfIteration; ++itr) {
        double loss = 0.0;
        int right = 0;
        for (int batch_id = 0; batch_id < numberOfBatch; ++batch_id) {
            int start = (batch_id * batchSize) + 1;
            int end = min(trainRows, start + batchSize - 1); 
            reset_mean_weight_and_bias(newNet);
            for (int row = start; row <= end; ++row) {
                newNet->actValue[0] = inputData[row];
                newNet->actualOutput = outputData[row];
                forwardPropagation(newNet);
                loss += computeLoss(newNet);
                right += (newNet->actualOutput[giveOutputIdx(newNet)] == 1.0);
                compute_mean_weight_and_bias(newNet);
            }
            update_weight_and_bias(newNet, batchSize);
        }
        double compLoss = loss / (double)trainRows;
        if(numberOfBatch == 1) {
            fprintf (ptrB, "%d,%f\n", itr, compLoss);
            print_accuracy_and_loss("Training for Batch Gradient Descent", right, loss, trainRows);
        }
        else {
            fprintf (ptrMB, "%d,%f\n", itr, compLoss);
            print_accuracy_and_loss("Training for Mini-Batch Gradient Descent", right, loss, trainRows);
        }
    }
    if (numberOfBatch == 1) fclose(ptrB);
    else fclose(ptrMB);
}

void training (struct NeuralNetwork* newNet, double** inputData, double** outputData, int trainRows) {
    ptrS = fopen ("c_stoc.csv", "w");
    for (int itr = 1; itr <= newNet->noOfIteration; ++itr) {
        double loss = 0.0;
        int right = 0;
        for (int row = 1; row <= trainRows; ++row) {
            newNet->actValue[0] = inputData[row]; 
            newNet->actualOutput = outputData[row]; 
            forwardPropagation(newNet);
            loss += computeLoss(newNet);
            backwardPropagation(newNet);
            int idx = giveOutputIdx(newNet);
            right += (newNet->actualOutput[idx] == 1.0);
        }
        double compLoss = loss / (double)trainRows;
        fprintf (ptrS, "%d,%f\n", itr, compLoss);
        print_accuracy_and_loss("Training For stochastic ", right, loss, trainRows);
    }
    fclose(ptrS);
}

void testing (struct NeuralNetwork* newNet, double** inputData, double** outputData, int testRows, int start) {
    double loss = 0.0;
    int right = 0;
    for (int row = start; row < start + testRows - 1; ++row) {
        newNet->actValue[0] = inputData[row]; 
        newNet->actualOutput = outputData[row]; 
        forwardPropagation(newNet);
        loss += computeLoss(newNet);
        right += (newNet->actualOutput[giveOutputIdx(newNet)] == 1);
    }
    print_accuracy_and_loss("Testing", right, loss, testRows);
}
