#include "gtest/gtest.h"
#include "src/HiddenLayer.h"
#include "src/OutputLayer.h"
//#include "src/OutputLayer.h"
#include "src/struct.h"
#include <iostream>
namespace Gtest {

class LayerTest: public ::testing::Test {
    public:
        HiddenLayer hiddenLayer;
        OutputLayer outputLayer;
        virtual void SetUp() {
        }
        virtual void TearDown() {
        }
        virtual void TestBody() {
        }
};

/*
void printMatrix(const Matrix &mat) {
    for (int i = 0; i < mat._rowSize; ++i) {
        for (int j = 0; j < mat._colSize; ++j) {
            std::cout << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
*/

/*
void printVector(const Vector &v) {
    for (int i = 0; i < v._size; ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}
*/

TEST_F(LayerTest, initLayerTest) {
    ASSERT_TRUE(hiddenLayer.init(2, 3));
    ASSERT_TRUE(outputLayer.init(3, 2));
}

TEST_F(LayerTest, forwardTest) {
    ASSERT_TRUE(hiddenLayer.init(2, 3));
    ASSERT_TRUE(outputLayer.init(3, 1));
    outputLayer.setInput(hiddenLayer.getOutput());
    Vector input(2);
    input[0] = 1;
    input[1] = 2;
    hiddenLayer.setInput(input);
    int passNum = 2000;
    for (int i = 0; i < passNum; ++i) {
        ASSERT_TRUE(hiddenLayer.forward());
        ASSERT_TRUE(outputLayer.forward());
        Vector output = outputLayer.getOutput();
        ASSERT_EQ(1, output._size);
        Matrix weight;
        Vector error;
        Vector trueY(1);
        trueY[0] = 5;
        ASSERT_TRUE(outputLayer.backward(output, weight, trueY));
        ASSERT_TRUE(hiddenLayer.backward(outputLayer.getError(), outputLayer.getWeight(), trueY));
        ASSERT_TRUE(outputLayer.updateWeight());
        ASSERT_TRUE(hiddenLayer.updateWeight());
    }
}

} // namespace
