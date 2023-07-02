#ifndef _MlpClassifier_h_
#define _MlpClassifier_h_

#include "utility.h"

void reset_mean_weight_and_bias(struct NeuralNetwork* newNet) ;
void init_mean_weight_and_bias(struct NeuralNetwork* newNet);
void update_weight_and_bias(struct NeuralNetwork* newNet, int batchSize) ;
void compute_mean_weight_and_bias (struct NeuralNetwork* newNet) ;

void forwardPropagation (struct NeuralNetwork* newNet);
void backwardPropagation (struct NeuralNetwork* newNet);

void training (struct NeuralNetwork* newNet, double** inputData, double** outputData, int trainRows);
void training_batch(struct NeuralNetwork* newNet, double** inputData, double** outputData, int trainRows, int numberOfBatch);
void testing (struct NeuralNetwork* newNet, double** inputData, double** outputData, int testRows, int start);

#endif