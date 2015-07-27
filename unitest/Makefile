CXX=g++
CC=gcc
BIN=./bin

CXXFLAGS=-g -std=c++11  -lgtest_main
LOCAL_ROOT=../..
GTEST_HOME=$(LOCAL_ROOT)/gtest
THIRD_INCPATH=-I$(GTEST_HOME)/include \
	      -I$(LOCAL_ROOT)/NeuralNetwork

.PHONY:clean

test :  main.cc
	mkdir -p $(BIN)
	$(CXX) main.cc $(THIRD_INCPATH) -L$(GTEST_HOME)/lib $(CXXFLAGS) -o $(BIN)/test.out

clean:
	rm -rf $(BIN)/*
