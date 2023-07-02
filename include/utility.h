#ifndef _Utility_h_
#define _Utility_h_

struct NeuralNetwork{
    int noOfLayers; 
    int noOfHiddenLayer; 
    int noOfFeatures; 
    int noOfIteration; 
    int noOfRows; 
    int partition; 
    double learningRate; 
    int* neuronsInLayer; 
    int* layersActivationFunc; 
    double* actualOutput; 
    double* testInput; 
    double** deltaValues;
    double** actValue; 
    double** zValue; 
    double** bias; 
    double*** weight; 
};

double sigmoid (double val);
void computeSigmoid (double* actValue, double* zValue, int n) ;
double derivSigmoid (double val) ;
double relu (double val);
void computeRelu (double* actValue, double* zValue, int n);
double derivRelu (double val);
void computeTanh (double* actValue, double* zValue, int n);
double derivTanh (double val);
void computeSoftMax (double* actValue, double* zValue, int n) ;
void matrixMult (double* activation, double* bias, double** w, double* z, int prevLayerSize, int curLayerSize);
void print_accuracy_and_loss (char* s, int right, double loss, int trainRows);
void print_loss (char* s, double loss, int totalRows);
int giveActFunc(char* curAct) ;
int giveActFuncLastLayer(char* curAct);
int generateRandom(int L, int R) ;
void generatePermuatation (int* perm, int n) ;
double computeLoss (struct NeuralNetwork* newNet);
int giveOutputIdx(struct NeuralNetwork* newNet);
int min (int a, int b) ;
void swap(int* a, int *b) ;

#endif