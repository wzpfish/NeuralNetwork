#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include "struct.h"
double Tanh(double x) {
    double powerOfX = exp(x);
    double powerOfNX = exp(-x);
    double tanh = (powerOfX - powerOfNX) / (powerOfX + powerOfNX);
    return tanh;
}

void Softmax(Vector &output) {
    double sum_power = 0;
    for (int i = 0; i < output._size; ++i) {
        double powerOfX = exp(output[i]);
        sum_power += powerOfX;
    }
    for (int i = 0; i < output._size; ++i) {
        output[i] = exp(output[i]) / sum_power;
    }
}

#endif
