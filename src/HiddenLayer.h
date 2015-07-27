#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include "LayerBase.h"
#include "InputLayer.h"
#include "activation.h"
#include "config.h"
#include <random>

class HiddenLayer: public LayerBase {
private:
    Vector _input;
    Vector _output;
    Matrix _weight;
    Vector _bias;
    Vector _error;
public:
    
    bool init(int inputSize, int outputSize) {
        _input = Vector(inputSize);
        _output = Vector(outputSize);
        _bias = Vector(outputSize);
        _error = Vector(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            _output[i] = 0;
            _bias[i] = ((double) rand()) / RAND_MAX;
            _error[i] = 0;
        }
        _weight = Matrix(inputSize, outputSize);
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                _weight[i][j] = ((double) rand()) / RAND_MAX;
            }
        }
        return true;   
    }
    
    bool forward() {
        assert(_input._size == _weight._rowSize && _weight._colSize == _output._size);
        multiple(_input, transpose(_weight), _output);
        for (int i = 0; i < _output._size; ++i) {
            _output[i] += _bias[i];
            _output[i] = tanh(_output[i]);
        }
        return true;
    }
    
    bool backward(const Vector &error, const Matrix &weight, const Vector &trueY) {
        multiple(error, weight, _error);
        for (int i = 0; i < _error._size; ++i) {
            // here the activation is tanh
            _error[i] *= (1 - _output[i] * _output[i]);
        }
        return true;
    }

    bool updateWeight() {
        for (int i = 0; i < _weight._rowSize; ++i) {
            for (int j = 0; j < _weight._colSize; ++j) {
                // here error * input isgrad
                _weight[i][j] -= (LEARNING_RATE * (_error[j] * _input[i]));
            }
        }
        for (int i = 0; i < _bias._size; ++i) {
            _bias[i] -= (LEARNING_RATE * (_error[i] * 1));
        }
        return true;
    }
    
    bool setInput(const Vector &input) {
        _input = input;
        return true;
    }

    Vector &getInput() {
        return _input;
    } 
    const Vector &getInput() const {
        return const_cast<HiddenLayer *>(this)->getInput();
    }
    
    Vector &getError() {
        return _error;
    }
    const Vector &getError() const {
        return const_cast<HiddenLayer *>(this)->getError();
    }

    Matrix &getWeight() {
        return _weight;
    }
    const Matrix &getWeight() const {
        return const_cast<HiddenLayer *>(this)->getWeight();
    }

    Vector &getOutput() {
        return _output;
    }
    const Vector &getOutput() const {
        return const_cast<HiddenLayer *>(this)->getOutput();
    }
};

#endif
