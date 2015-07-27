#include "src/hiddenLayer.h"
#include "src/outputLayer.h"
#include "src/struct.h"
#include <iostream>
#include <stdio.h>
using namespace std;

const int N = 20005;
double x1[N], x2[N], y[N];
int main() {
    FILE *in = fopen("syn_data.txt", "r");
    int m = 0;
    for (; fscanf(in, "%lf\t%lf\t%lf", &x1[m], &x2[m], &y[m]) != EOF; ++m) {}
    int inputSize = 2, hiddenSize = 3, outputSize = 1;
    HiddenLayer hiddenLayer, hiddenLayer2;
    OutputLayer outputLayer;
    hiddenLayer.init(inputSize, hiddenSize);
    hiddenLayer2.init(hiddenSize, hiddenSize);
    outputLayer.init(hiddenSize, outputSize);
    hiddenLayer2.setInput(hiddenLayer.getOutput());
    outputLayer.setInput(hiddenLayer2.getOutput());
    int passNum = 1000;
    for (int i = 0; i < passNum; ++i) {
        double totalError = 0;
        for (int j = 0; j < m; ++j) {
            Vector input(inputSize);
            Vector trueY(outputSize);
            input[0] = x1[j], input[1] = x2[j], trueY[0] = y[j];
            hiddenLayer.setInput(input);
            hiddenLayer.forward();
            hiddenLayer2.forward();
            outputLayer.forward();
            //cout << trueY[0] << "  " << outputLayer.getOutput()[0] << endl;  
            totalError += 0.5 * (trueY[0] - outputLayer.getOutput()[0]) * (trueY[0] - outputLayer.getOutput()[0]);
            outputLayer.backward(outputLayer.getError(), outputLayer.getWeight(), trueY);
            hiddenLayer2.backward(outputLayer.getError(), outputLayer.getWeight(), trueY);
            hiddenLayer.backward(hiddenLayer2.getError(), hiddenLayer2.getWeight(), trueY);
            outputLayer.updateWeight();
            hiddenLayer2.updateWeight();
            hiddenLayer.updateWeight();
        }
        printf("pass %d: error is %.5lf\n", i, totalError / m);
    }
    fclose(in);
    FILE *out = fopen("syn_data_evl.txt", "w");
    for (int j = 0; j < m; ++j) {
        Vector input(2);
        input[0] = x1[j], input[1] = x2[j];
        hiddenLayer.setInput(input);
        hiddenLayer.forward();
        hiddenLayer2.forward();
        outputLayer.forward();
        fprintf(out, "%.5lf\t%.5lf\t%.5lf\t%.5lf\n", x1[j], x2[j], y[j], outputLayer.getOutput()[0]);
    }
    fclose(out);
    return 0;
}
