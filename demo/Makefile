CXX=g++
CC=gcc
BIN=./bin

CXXFLAGS=-g -std=c++11
LOCAL_ROOT=../..
THIRD_INCPATH=-I$(LOCAL_ROOT)/NeuralNetwork

.PHONY:clean

mlp.out:  mlp.cc
	$(CXX) mlp.cc $(THIRD_INCPATH)  $(CXXFLAGS) -o mlp.out

clean:
	rm -rf *.out *.a *.o *.dSYM
