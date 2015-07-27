#ifndef LAYERBASE_H
#define LAYERBASE_H


#include "struct.h"

class LayerBase {
public:
    
    // input size and output size of this layer
    virtual bool init(int inputSize, int outputSize) = 0;
    
    virtual bool forward() = 0;
    
    // error backward
    // not all params are used.. trueY is only used in the output layer
    virtual bool backward(const Vector &error, const Matrix &weight, const Vector &trueY) = 0;
};

#endif
