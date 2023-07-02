#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#include "utility.h"

void matrixMult (double* activation, double* bias, double** w, double* z, int prevLayerSize, int curLayerSize) {
    for (int neuron_cur_layer = 1; neuron_cur_layer <= curLayerSize; ++neuron_cur_layer) {
        z[neuron_cur_layer] = 1.0 * bias[neuron_cur_layer]; 
        for (int neuron_prev_layer = 1; neuron_prev_layer <= prevLayerSize; ++neuron_prev_layer) {
            z[neuron_cur_layer] += activation[neuron_prev_layer] * w[neuron_prev_layer][neuron_cur_layer];
        }
    }
}

double sigmoid (double val) {
    return (double)1.0 / (double) (1.0 + exp(-val));
}

double relu (double val) {
    return val < 0.0 ? 0.0 : val;
}

double derivTanh (double val) {
    return 1.0 - (tanh(val) * tanh(val));
}

double derivRelu (double val) {
    return val < 0.0 ? 0.0 : 1.0;
}

void computeSigmoid (double* actValue, double* zValue, int n) {
    for (int i = 1; i <= n; ++i) {
        actValue[i] = sigmoid(zValue[i]);
    }
}

void computeRelu (double* actValue, double* zValue, int n) {
    for (int i = 1; i <= n; ++i) {
        actValue[i] = relu(zValue[i]);
    }
}

void computeTanh (double* actValue, double* zValue, int n) {
    for (int i = 1; i <= n; ++i) {
        actValue[i] = tanh(zValue[i]);
    }
}

void computeSoftMax (double* actValue, double* zValue, int n) {
    double total = 0.0;
    for (int i = 1; i <= n; ++i) {
        total += exp(zValue[i]);
    }
    for (int i = 1; i <= n; ++i) {
        actValue[i] = (double)(exp(zValue[i])) / (double)total;
    }
}


void print_accuracy_and_loss (char* s, int right, double loss, int trainRows) {
    double accuracy = (double) right / (double) trainRows;
    loss /= (double) trainRows;
    printf ("\n--------------[ %s ]---------\n", s);
    printf ("[Accuracy : %.10f], [Loss : %.10f]\n", accuracy, loss);
}

void print_loss (char* s, double loss, int totalRows) {
    loss /= (double) totalRows;
    printf ("\n--------- %s ---------\n", s);
    printf ("[ loss : %.10f]\n", loss);
}

double derivSigmoid (double val) {
    double ans = sigmoid(val) * (1.0 - sigmoid(val));
    return ans;
}

int giveActFunc(char* curAct) {
    return (strcmp(curAct, "sigmoid") == 0 ? 1 : (strcmp(curAct, "relu") == 0 ? 2 : (strcmp(curAct, "tanh") == 0 ? 3 : 4)));
}

int giveActFuncLastLayer(char* curAct) {
    return (strcmp(curAct, "mse") == 0 ? 1 : (strcmp(curAct, "ce") == 0 ? 2 : 3));
}

int generateRandom(int L, int R) {
    srand(time(NULL));
    return (L + (rand() * (R - L))/ RAND_MAX);
}

void swap(int* a, int *b) {
    int temp = *b;
    *b = *a;
    *a = temp;  
}

void generatePermuatation (int* perm, int n) {
    for (int cur_idx = 1; cur_idx < n; ++cur_idx) {
        int new_idx = generateRandom(cur_idx, n);
        swap(&perm[new_idx], &perm[cur_idx]);
    }
}

double computeLoss (struct NeuralNetwork* newNet) {
    int lastLayer = newNet->noOfLayers - 1;
    double ans = 0.0;
    int actValueLastLayer = newNet->layersActivationFunc[lastLayer];
    for (int neuron = 1; neuron <= newNet->neuronsInLayer[lastLayer]; ++neuron) {
        if (actValueLastLayer == 1) {
            ans += 0.5 * (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]) * (newNet->actValue[lastLayer][neuron] - newNet->actualOutput[neuron]);
        }
        else {
            ans -= newNet->actualOutput[neuron] * (log(newNet->actValue[lastLayer][neuron]));
        }
    }
    return ans;
}

int giveOutputIdx(struct NeuralNetwork* newNet) {
    double max = LONG_MIN;
    int whichNeuron = -1;
    int lastLayer = newNet->noOfLayers - 1;
    for (int neuron = 1; neuron <= newNet->neuronsInLayer[lastLayer]; ++neuron) {
        if (max < newNet->actValue[lastLayer][neuron]) {
            max = newNet->actValue[lastLayer][neuron];
            whichNeuron = neuron;
        }
    }
    return whichNeuron;
}

int min (int a, int b) {
    return a < b ? a : b;
}

