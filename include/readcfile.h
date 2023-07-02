#ifndef _ReadCfile_h_
#define _ReadCfile_h_

#include "utility.h"

void normalization (struct NeuralNetwork* newNet,double** inputData, int trainRows, int col);
void readDataFile (struct NeuralNetwork* newNet,char* fileName, double** inputData, double** outputData, int features, int lastRow);

#endif