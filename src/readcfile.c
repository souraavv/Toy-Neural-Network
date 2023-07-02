#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>

#include "utility.h"
#include "readcfile.h"

const int MaxRow = 1e5;

void normalization (struct NeuralNetwork* newNet,double** inputData, int trainRows, int col) {
    double* X = (double*) malloc ((col + 1) * sizeof(double));
    double* sigma = (double*) malloc ((col + 1) * sizeof(double));
    double* XTest = (double*) malloc ((col + 1) * sizeof(double));
    double* sigmaTest = (double*) malloc ((col + 1) * sizeof(double));
    double totalRows = newNet->noOfRows;
    int testRows = totalRows - trainRows;
    for (int i = 1; i <= col; ++i) {
        X[i] = sigma[i] = 0.0;
        XTest[i] = sigmaTest[i] = 0.0;
    }
    for (int i = 1; i <= col; ++i) {
        for (int j = 1; j < totalRows; ++j) {
            if (j <= trainRows) 
                X[i] += inputData[j][i];
            else 
                XTest[i] += inputData[j][i];
        }
        X[i] = X[i] / (double) trainRows;
        XTest[i] = XTest[i] / (double) testRows;
    }
    for (int i = 1; i <= col; ++i) {
        for (int j = 1; j < totalRows; ++j) {
            if (j <= trainRows)
                sigma[i] += (inputData[j][i] - X[i]) * (inputData[j][i] - X[i]);
            else
                sigmaTest[i] += (inputData[j][i] - XTest[i]) * (inputData[j][i] - XTest[i]);
        }
        sigma[i] = sqrt (1.0 * sigma[i] / (double)trainRows);
        sigmaTest[i] = sqrt(1.0 * sigmaTest[i] / (double) testRows);
    }
    double threshold = 0.0001;
    for (int i = 1; i <= col; ++i) {
        for (int j = 1; j < totalRows; ++j) {
            if (j <= trainRows) {
                if (sigma[i] > threshold) 
                    inputData[j][i] = (double)(inputData[j][i] - X[i]) / (double)sigma[i];
            }
            else {
                if (sigmaTest[i] > threshold)
                    inputData[j][i] = (double)(inputData[j][i] - XTest[i]) / (double)sigmaTest[i];
            }
        }
    }
}


void readDataFile (struct NeuralNetwork* newNet,char* fileName, double** inputData, double** outputData, int features, int lastRow) {
    FILE* ptr = fopen (fileName, "r");
    char* curLine = (char*) malloc (MaxRow* sizeof(char));
    char* curRowData;
    int* perm = (int*) malloc ((lastRow) * sizeof(int));
    for (int i = 1; i < lastRow; ++i) {
        perm[i] = i;
    }
    generatePermuatation(perm, lastRow);
    for (int row = 0; row < lastRow && fgets(curLine, MaxRow, ptr); ++row) {
        if (row == 0) {
            continue;
        }
        curRowData = strtok(curLine, ",");
        for (int col = 0; col < features; ++col, curRowData = strtok(NULL, ",")) { 
            if (col == 1) { 
                outputData[row][1] = (strcmp(curRowData, "M") == 0 ? 1.0 : 0.0);
                outputData[row][2] = (strcmp(curRowData, "B") == 0 ? 1.0 : 0.0);
            }
            else if (col > 1) { 
                inputData[row][col - 1] = atof(curRowData);
            }
        }
    }
    for (int cur_row = 1; cur_row < lastRow; ++cur_row) {
        int swap_row = perm[cur_row];
        double* tempInput = inputData[cur_row];
        inputData[cur_row] = inputData[swap_row];
        inputData[swap_row] = tempInput;
        double* tempOutput = outputData[cur_row];
        outputData[cur_row] = outputData[swap_row];
        outputData[swap_row] = tempOutput;
    }
    fclose(ptr);
    int trainRows = (newNet->partition * newNet->noOfRows) / 100; 
    normalization (newNet,inputData, trainRows, features);
}
