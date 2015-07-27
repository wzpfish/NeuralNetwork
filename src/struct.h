
#ifndef STRUCT_H
#define STRUCT_H

#include <memory>
#include <assert.h>

struct Vector {
    int _size;
    std::shared_ptr<double> _ptr;
    
    Vector(): _size(0) {}
    
    Vector(int size): _size(size), _ptr(std::shared_ptr<double>(new double[size], std::default_delete<double[]>())) {}
    
    Vector &operator=(const Vector &v) {
        _size = v._size;
        _ptr = v._ptr;
        return *this;
    }
    
    double& operator[] (int idx) {
        return _ptr.get()[idx];
    }
    
    const double& operator[] (int idx) const {
        return (*const_cast<Vector *>(this))[idx];
    }
};

struct Matrix {
    int _rowSize, _colSize;
    std::shared_ptr<Vector> _ptr;
    
    Matrix(): _rowSize(0), _colSize(0) {}
    
    Matrix(int rowSize, int colSize): _rowSize(rowSize), _colSize(colSize),
        _ptr(std::shared_ptr<Vector>(new Vector[_rowSize], std::default_delete<Vector[]>())) 
    {
        for (int i = 0; i < _rowSize; ++i) {

            _ptr.get()[i] = Vector(_colSize);
        }
    }
    
    Matrix &operator=(const Matrix &m) {
        _rowSize = m._rowSize;
        _colSize = m._colSize;
        _ptr = m._ptr;
        return *this;
    }

    Vector& operator[] (int r) {
        return _ptr.get()[r];
    }
    
    const Vector& operator[] (int r) const {
        return (*const_cast<Matrix *>(this))[r];
    }
};

Matrix transpose(const Matrix &mat) {
    Matrix trans_mat(mat._colSize, mat._rowSize);
    for (int i = 0; i < trans_mat._rowSize; ++i) {
        for (int j = 0; j < trans_mat._colSize; ++j) {
            trans_mat[i][j] = mat[j][i];
        }
    }
    return trans_mat;
}
void multiple(const Vector &input,
        const Matrix &weight,
        Vector &output
        )
{
    for (int i = 0; i < output._size; ++i) {
        output[i] = 0;
    }
    for (int i = 0; i < input._size; ++i) {
        for (int j = 0; j < weight._colSize; ++j) {
            output[j] += (input[i] * weight[i][j]);
        }
    }
}


#endif
