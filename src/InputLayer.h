#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "LayerBase.h"

class InputLayer: public LayerBase {
private:
    Vector _input;
public:
    bool init(Vector input, 
            int outputSize,
            int weightRowSize, 
            int weightColSize
            ) 
    {
        _input = input;
        /* for (int i = 0; i < _input._size; ++i) {
            _input[i] = input[i];
        }*/ 
        return true;
    }

    void forward(const Vector &input) {
        return;
    }
    
    void backward(const Vector &error, const Matrix &weight) {
        return;
    }
    Vector getOutput() {
        return _input;
    }

    const Vector getOutput() const {
        return const_cast<InputLayer *>(this)->getOutput();
    }

};

#endif
