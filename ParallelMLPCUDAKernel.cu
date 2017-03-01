
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <ctime>

////////////////////////////////////////////////////////////////////////
//																	  //
//				Multi-Layer Perceptron w/ Backpropagation	    	  //
//						 Parallel Implementation				      //
//																	  //
//	    Joel Gooch, BSc Computer Science, University of Plymouth	  //
//																	  //
////////////////////////////////////////////////////////////////////////


//global variables
__device__ __constant__ int maxEpochs = 3;
__device__ __constant__ float learningRate = 0.2;
__device__ __constant__ float momentum = 0.5f;

// number of training samples to use for training
int numberOfTrainingSamplesToUse = 1000;
// number of testing samples to use for final test
int numberOfTestingSamplesToUse = 512;
// how many threads to run in parallel with different weights
int numberOfParallelTrainingSessions = 512;

// class for individual data entries
class dataEntry {
public:
	std::vector<float> features;
	float expectedClassification;
	dataEntry(std::vector<float> f, float c) : features(f), expectedClassification(c) {}
};

// class to hold whole data set
class dataSet {
public:
	std::vector<dataEntry> trainingSet;
	std::vector<dataEntry> testingSet;
	std::vector<dataEntry> validationSet;
	dataSet() {}
	void clearDataSet() {
		trainingSet.clear();
		testingSet.clear();
		validationSet.clear();
	}
};

// class used to read data from .txt file
class dataReader {
public:
	dataSet dataSet;
	bool loadDataFile(const char* filename);
	void processLine(std::string line);
private:
	std::vector<dataEntry> data;
	int noOfFeatures;
	int noOfTargets;
};

// function that handles loading data from .txt file
bool dataReader::loadDataFile(const char* filename) {

	// open file to read from
	std::fstream inputFile;
	inputFile.open(filename, std::ios::in);

	if (inputFile.is_open()) {
		std::string line = "";
		//read data from file
		while (!inputFile.eof()) {
			getline(inputFile, line);
			// check line is something other than just a blank new line
			if (line.length() > 2) {
				// check if the second from last character in line is a - sign
				// dynamically calculate number of features accordingly
				if (line.end()[-2] == '-') {
					noOfFeatures = line.size() / 2 - 1;
				}
				else {
					noOfFeatures = line.size() / 2;
				}
				//process individual line
				processLine(line);
			}
		}

		// randomize data order
		std::srand(std::time(0));
		random_shuffle(data.begin(), data.end());

		// customise data index splits from whole data set		
		int trainingDataEndIndex = numberOfTrainingSamplesToUse;
		int testingSetSize = numberOfTestingSamplesToUse;
		//int validationSetSize = 0;													 

		//fill training data set
		for (int i = 0; i < trainingDataEndIndex; i++) {
			dataSet.trainingSet.push_back(data[i]);
		}

		// fill training data set
		for (int i = trainingDataEndIndex; i < trainingDataEndIndex + testingSetSize; i++) {
			dataSet.testingSet.push_back(data[i]);
		}

		// fill validation data set
		//for (int i = trainingDataEndIndex + testingSetSize; i < (int)data.size(); i++) {
		//	dataSet.validationSet.push_back(data[i]);
		//}
		printf("Success opening input file: %s, reads: %d \n", filename, data.size());

		// close file
		inputFile.close();
		return true;
	}
	else {
		printf("Error opening input file: %s \n", filename);
		return false;
	}
}

// function to process individual line from .txt file
void dataReader::processLine(std::string line) {
	// initialise new data entry variables
	std::vector<float> features;
	float expectedClassification = 0;

	// store inputs
	char* cstr = new char[line.size() + 1];
	char* t;
	strcpy_s(cstr, line.size() + 1, line.c_str());

	// tokenise 
	int i = 0;
	char* nextToken = NULL;
	t = strtok_s(cstr, ",", &nextToken);

	while (t != NULL && i < noOfFeatures + 1) {
		// allocate memory for new value
		float *value = (float*)malloc(sizeof(float));
		// convert string to float
		*value = std::stof(t);
		// add value to features or classification output
		if (i < noOfFeatures) {
			features.push_back(*value);
		}
		else {
			expectedClassification = *value;
		}
		// move to next token
		t = strtok_s(NULL, ",", &nextToken);
		i++;
	}

	// add to data set
	data.push_back(dataEntry(features, expectedClassification));
}


// function that runs all data entries in parallel on the GPU
__global__ void feedForwardKernel(float* data, int noOfEntries, int noOfInputNodes, int noOfHiddenNodes, 
	int noOfOutputNodes, float* d_inputLayerWeights, float* d_hiddenLayerWeights, float* d_hiddenNodeOutputs, float* d_outputNodeOutputs) {
	// calculate global thread index
	int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;

	// ensure only required threads run
	if (globalIdX < noOfEntries) {
		// calculate weighted summation of input > hidden layer
		for (int feature = 0; feature < noOfInputNodes; feature++) {
			for (int hiddenNode = 0; hiddenNode < noOfHiddenNodes; hiddenNode++) {
				d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] += data[(globalIdX * noOfInputNodes) + feature] *
					d_inputLayerWeights[(feature * noOfHiddenNodes) + hiddenNode];
			}
		}
		__syncthreads();

		// calculate bias contributions for input > hidden layer
		for (int hiddenNode = 0; hiddenNode < noOfHiddenNodes; hiddenNode++) {
			d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] += 1 * d_inputLayerWeights[(noOfInputNodes * noOfHiddenNodes) + hiddenNode];
		}
		__syncthreads();

		// apply hyperbolic tangent activation function
		for (int i = 0; i < noOfHiddenNodes; i++) {
			d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + i] = tanh(d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + i]);
		}
		__syncthreads();

		// calculate weighted summation of hidden > output layer
		for (int hiddenNode = 0; hiddenNode < noOfHiddenNodes; hiddenNode++) {
			for (int outputNode = 0; outputNode < noOfOutputNodes; outputNode++) {
				d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + outputNode] += d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] *
					d_hiddenLayerWeights[(hiddenNode * noOfOutputNodes) + outputNode];
			}
		}
		__syncthreads();

		// calculate bias contributions for hidden > output layer
		for (int outputNode = 0; outputNode < noOfOutputNodes; outputNode++) {
			d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + outputNode] += 1 * d_hiddenLayerWeights[(noOfHiddenNodes * noOfOutputNodes) + outputNode];
		}
		__syncthreads();

		// apply hyperbolic tangent activation function
		for (int i = 0; i < noOfOutputNodes; i++) {
			d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + i] = tanh(d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + i]);
		}	
	}
}

// function that returns derivative hyperbolic tangent value
__device__ float derivativeAcivationFunction(float val) {
	return 1 - (val * val);
}

// function that calculates delta values of all training examples in parallel on the GPU.
// IMPORTANT - THIS FUNCTION IS NOT IN USE IN THIS PROJECT AND IS A PREVIOUS ATTEMPT. IS INCLUDED FOR COMPLETENESS FOR FUTURE UPDATES TO THIS PROJECT.
__global__ void calculateDeltasKernel(float* d_outputNodeOutputs, float* d_hiddenNodeOutputs, float* d_trainingExpectedClassifications, float* d_outputLayerDeltas,
	float* d_hiddenLayerDeltas, float* d_hiddenLayerWeights, int noOfHiddenNodes, int noOfOutputNodes, int noOfEntries) {
	// calculate global thread index
	int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalIdX < noOfEntries) {

		// calculate output layer deltas
		for (int i = 0; i < noOfOutputNodes; i++) {
			d_outputLayerDeltas[globalIdX * noOfOutputNodes + i] = d_trainingExpectedClassifications[globalIdX * noOfOutputNodes + i] - 
				d_outputNodeOutputs[globalIdX * noOfOutputNodes + i];
			d_outputLayerDeltas[globalIdX * noOfOutputNodes + i] *= derivativeAcivationFunction(d_outputNodeOutputs[globalIdX * noOfOutputNodes + i]);
		}

		// calculate hidden layer deltas
		// for each node in hidden layer
		for (int i = 0; i < noOfHiddenNodes + 1; i++) {
			float deltaOfWeights = 0.0f;
			// sum deltas of nodes in outputlayer
			for (int j = 0; j < noOfOutputNodes; j++) {
				deltaOfWeights += d_hiddenLayerWeights[i * noOfOutputNodes + j] * d_outputLayerDeltas[globalIdX * noOfOutputNodes + j];
			}

			// if the current node is the bias node
			if (i == noOfHiddenNodes) {
				// calculate delta using default value of 1 for bias node
				d_hiddenLayerDeltas[globalIdX * (noOfHiddenNodes + 1) + i] = deltaOfWeights * derivativeAcivationFunction(1);
			}
			else {
				// calculate delta using actual hidden node output value
				d_hiddenLayerDeltas[globalIdX * (noOfHiddenNodes + 1) + i] = deltaOfWeights * 
					derivativeAcivationFunction(d_hiddenNodeOutputs[globalIdX * noOfHiddenNodes + i]);
			}
		}
	}
}

// function that performs forward pass on the GPU, using different set of weights on each thread
__device__ void forwardPass(int globalIdX, int entry, float* data, int noOfEntries, int noOfInputNodes, int noOfHiddenNodes, 
	int noOfOutputNodes, float* d_inputLayerWeights, int numOfInputWeightsPerSession, float* d_hiddenLayerWeights, int numOfHiddenWeightsPerSession, 
	float* d_hiddenNodeOutputs, float* d_outputNodeOutputs) {
	// calculate weighted summation of input > hidden layer
	for (int feature = 0; feature < noOfInputNodes; feature++) {
		for (int hiddenNode = 0; hiddenNode < noOfHiddenNodes; hiddenNode++) {
			d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] += data[(entry * noOfInputNodes) + feature] *
				d_inputLayerWeights[(globalIdX * numOfInputWeightsPerSession) + (feature * noOfHiddenNodes) + hiddenNode];
		}
	}
	__syncthreads();

	// calculate bias contributions for input > hidden layer
	for (int hiddenNode = 0; hiddenNode < noOfHiddenNodes; hiddenNode++) {
		d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] += 1 * d_inputLayerWeights[(globalIdX * numOfInputWeightsPerSession) + 
			(noOfInputNodes * noOfHiddenNodes) + hiddenNode];
	}
	__syncthreads();

	// apply hyperbolic tangent activation function
	for (int hiddenNode = 0; hiddenNode < noOfHiddenNodes; hiddenNode++) {
		d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] = tanh(d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode]);
	}
	__syncthreads();

	// calculate weighted summation of hidden > output layer
	for (int hiddenNode = 0; hiddenNode < noOfHiddenNodes; hiddenNode++) {
		for (int outputNode = 0; outputNode < noOfOutputNodes; outputNode++) {
			d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + outputNode] += d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] *
				d_hiddenLayerWeights[(globalIdX * numOfHiddenWeightsPerSession) + (hiddenNode * noOfOutputNodes) + outputNode];
		}
	}
	__syncthreads();

	// calculate bias contributions for hidden > output layer
	for (int outputNode = 0; outputNode < noOfOutputNodes; outputNode++) {
		d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + outputNode] += 1 * d_hiddenLayerWeights[(globalIdX * numOfHiddenWeightsPerSession) +
			(noOfHiddenNodes * noOfOutputNodes) + outputNode];
	}
	__syncthreads();

	// apply hyperbolic tangent activation function
	for (int outputNode = 0; outputNode < noOfOutputNodes; outputNode++) {
		d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + outputNode] = tanh(d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + outputNode]);
	}
	__syncthreads();
}

// function that calculates delta values on the GPU, using different set of weights on each thread
__device__ void calculateDeltas(int globalIdX, int entry, float* expectedClassifications, int noOfHiddenNodes, int noOfOutputNodes, 
	float* d_hiddenLayerDeltas, float* d_outputLayerDeltas, float* d_hiddenLayerWeights, float* d_hiddenNodeOutputs, float* d_outputNodeOutputs) {
	// calculate output layer deltas
	for (int i = 0; i < noOfOutputNodes; i++) {
		d_outputLayerDeltas[globalIdX * noOfOutputNodes + i] = expectedClassifications[entry] - d_outputNodeOutputs[globalIdX];
		d_outputLayerDeltas[globalIdX * noOfOutputNodes + i] *= derivativeAcivationFunction(d_outputNodeOutputs[(globalIdX * noOfOutputNodes) + i]);
	}

	__syncthreads();

	float deltaOfWeights = 0.0f;

	// calculate hidden layer deltas
	for (int i = 0; i < noOfHiddenNodes + 1; i++) {
		 deltaOfWeights = 0.0f;

		// sum deltas of nodes in output layer
		for (int j = 0; j < noOfOutputNodes; j++) {
			deltaOfWeights += d_hiddenLayerWeights[(globalIdX * noOfHiddenNodes) + i] * d_outputLayerDeltas[(globalIdX * noOfOutputNodes) + j];
		}

		// if the current node is the bias node
		if (i == noOfHiddenNodes) {
			// calculate deltas using default value of 1 for bias node
			d_hiddenLayerDeltas[(globalIdX * (noOfHiddenNodes + 1)) + i] = deltaOfWeights * derivativeAcivationFunction(1);
		}
		else {
			// calculate deltas using hidden node output value
			d_hiddenLayerDeltas[(globalIdX * (noOfHiddenNodes + 1)) + i] = deltaOfWeights * derivativeAcivationFunction(d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + i]);
		}
	}

	__syncthreads();
}

// function that updates weights on the network on the GPU, using different set of deltas to update a different set of weights on each thread
__device__ void updateWeights(int globalIdX, float* data, int noOfInputNodes, int noOfHiddenNodes, int noOfOutputNodes, float* d_hiddenLayerDeltas, float* d_outputLayerDeltas, 
	float* d_inputLayerWeights, float* d_inputLayerDeltaWeights, int numOfInputWeightsPerSession, float* d_hiddenLayerWeights, float* d_hiddenLayerDeltaWeights, 
	int numOfHiddenWeightsPerSession, float* d_hiddenNodeOutputs) {
	// update hidden > output layer weights
	for (int hiddenNode = 0; hiddenNode < (noOfHiddenNodes + 1); hiddenNode++) {
		for (int i = 0; i < noOfOutputNodes; i++) {
			// if the current node is the bias node
			if (hiddenNode == noOfHiddenNodes) { // run the weight changes using a default value of 1 for the bias hidden node output
				// update current hidden node weight value
				d_hiddenLayerWeights[(globalIdX * numOfHiddenWeightsPerSession) + hiddenNode] += learningRate * 1 * d_outputLayerDeltas[(globalIdX * noOfOutputNodes) + i] + 
					(momentum * d_hiddenLayerDeltaWeights[(globalIdX * numOfHiddenWeightsPerSession) + hiddenNode]);
				// store delta value for hidden node weight for next weight change
				d_hiddenLayerDeltaWeights[(globalIdX * numOfHiddenWeightsPerSession) + hiddenNode] = learningRate * 1 * d_outputLayerDeltas[(globalIdX * noOfOutputNodes) + i] +
					(momentum * d_hiddenLayerDeltaWeights[(globalIdX * numOfHiddenWeightsPerSession) + hiddenNode]);
			}
			else { // run the weight changes using the actual hidden node value
				// update current hidden node weight value
				d_hiddenLayerWeights[(globalIdX * numOfHiddenWeightsPerSession) + hiddenNode] += learningRate * d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] *
					d_outputLayerDeltas[(globalIdX * noOfOutputNodes) + i] + (momentum * d_hiddenLayerDeltaWeights[(globalIdX * numOfHiddenWeightsPerSession) + hiddenNode]);
				// store delta value for hidden node weight for next weight change
				d_hiddenLayerDeltaWeights[(globalIdX * numOfHiddenWeightsPerSession) + hiddenNode] = learningRate *  d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + hiddenNode] *
					d_outputLayerDeltas[(globalIdX * noOfOutputNodes) + i] + (momentum * d_hiddenLayerDeltaWeights[(globalIdX * numOfHiddenWeightsPerSession) + hiddenNode]);
			}
		}
	}

	__syncthreads();

	// update input > hidden layer weights
	for (int inputNode = 0; inputNode < (noOfInputNodes + 1); inputNode++) {
		for (int i = 0; i < noOfHiddenNodes; i++) {
			// if the current node is the bias node
			if (inputNode == noOfInputNodes) { // run the weight changes using a default value of 1 for the bias input node output
				// update current input node weight value
				d_inputLayerWeights[(globalIdX * numOfInputWeightsPerSession) + (inputNode * noOfHiddenNodes) + i] += learningRate * 1 * 
					d_hiddenLayerDeltas[globalIdX * noOfHiddenNodes + i] + (momentum * d_inputLayerDeltaWeights[(globalIdX * numOfInputWeightsPerSession) + 
						(inputNode * noOfHiddenNodes) + i]);
				// store delta value for input node weight for next weight change
				d_inputLayerDeltaWeights[(globalIdX * numOfInputWeightsPerSession) + (inputNode * noOfHiddenNodes) + i] = learningRate * 1 * 
					d_hiddenLayerDeltas[(globalIdX * noOfHiddenNodes) + i] + (momentum * d_inputLayerDeltaWeights[(globalIdX * numOfInputWeightsPerSession) + 
						(inputNode * noOfHiddenNodes) + i]);
			}
			else { // run the weight changes using the actual input node value for the bias input node output
				// update current weight value 
				d_inputLayerWeights[(globalIdX * numOfInputWeightsPerSession) + (inputNode * noOfHiddenNodes) + i] += learningRate * data[(globalIdX * 
					noOfInputNodes) + inputNode] * d_hiddenLayerDeltas[(globalIdX * noOfHiddenNodes) + i] + (momentum * 
						d_inputLayerDeltaWeights[(globalIdX * numOfInputWeightsPerSession) + (inputNode * noOfHiddenNodes) + i]);
				// store delta value for input node weight for next weight change
				d_inputLayerDeltaWeights[(globalIdX * numOfInputWeightsPerSession) + (inputNode * noOfHiddenNodes) + i] = learningRate * 
					data[(globalIdX * noOfInputNodes) + inputNode] * d_hiddenLayerDeltas[(globalIdX * noOfHiddenNodes) + i] + (momentum * 
						d_inputLayerDeltaWeights[(globalIdX * numOfInputWeightsPerSession) + (inputNode * noOfHiddenNodes) + i]);
			}
		}
	}
	__syncthreads();
}

// function that performs MLP training session parallelism on the GPU, each thread runs the same training data, using a different set of randomly initialized weights
__global__ void TrainingSessionParallelism(float* data, float* expectedClassifications, int noOfEntries, int numOfTrainingSessions, int noOfInputNodes, int noOfHiddenNodes, 
	int noOfOutputNodes, float* d_hiddenLayerDeltas, float* d_outputLayerDeltas, float* d_inputLayerWeights, float* d_inputLayerDeltaWeights, int numOfInputWeightsPerSession,
	float* d_hiddenLayerWeights, float* d_hiddenLayerDeltaWeights, int numOfHiddenWeightsPerSession, float* d_hiddenNodeOutputs, float* d_outputNodeOutputs, 
	float* d_totalMSEPerSession) {
	// calculate global thread index
	int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalIdX < numOfTrainingSessions) {

		// Training the Network
		// for a user defined number of epochs
		for (int i = 0; i < maxEpochs; i++) {
			// run every example in the training set
			for (int entry = 0; entry < noOfEntries; entry++) {
				// perform a forward pass of the training data through the network
				forwardPass(globalIdX, entry, data, noOfEntries, noOfInputNodes, noOfHiddenNodes, noOfOutputNodes, d_inputLayerWeights,
					numOfInputWeightsPerSession, d_hiddenLayerWeights, numOfHiddenWeightsPerSession, d_hiddenNodeOutputs, d_outputNodeOutputs);

				__syncthreads();

				// calculate network delta values from previous feedforward
				calculateDeltas(globalIdX, entry, expectedClassifications, noOfHiddenNodes, noOfOutputNodes, d_hiddenLayerDeltas, d_outputLayerDeltas, 
					d_hiddenLayerWeights, d_hiddenNodeOutputs, d_outputNodeOutputs);

				__syncthreads();

				// update weights of the network using newly acquired delta values
				updateWeights(globalIdX, data, noOfInputNodes, noOfHiddenNodes, noOfOutputNodes, d_hiddenLayerDeltas, d_outputLayerDeltas, d_inputLayerWeights, 
					d_inputLayerDeltaWeights, numOfInputWeightsPerSession, d_hiddenLayerWeights, d_hiddenLayerDeltaWeights, numOfHiddenWeightsPerSession, 
					d_hiddenNodeOutputs);

				__syncthreads();

				// clear all hidden node outputs for next training entry
				for (int i = 0; i < noOfHiddenNodes; i++) {
					d_hiddenNodeOutputs[globalIdX * noOfHiddenNodes + i] = 0;
				}
				// clear output node output for next training entry
				d_outputNodeOutputs[globalIdX] = 0;

				__syncthreads();
			}
		}

		__syncthreads();

		// once training is complete do a final forward pass of the training data
		for (int entry = 0; entry < noOfEntries; entry++) {
			// perform a forward pass of current training example through the network
			forwardPass(globalIdX, entry, data, noOfEntries, noOfInputNodes, noOfHiddenNodes, noOfOutputNodes, d_inputLayerWeights, 
				numOfInputWeightsPerSession, d_hiddenLayerWeights, numOfHiddenWeightsPerSession, d_hiddenNodeOutputs, d_outputNodeOutputs);
			// summate total network error across all training examples
			d_totalMSEPerSession[globalIdX] += std::pow(std::abs(expectedClassifications[entry] - d_outputNodeOutputs[globalIdX * noOfOutputNodes]), 2);

			// clear all hidden node outputs for next training entry
			for (int i = 0; i < noOfHiddenNodes; i++) {
				d_hiddenNodeOutputs[(globalIdX * noOfHiddenNodes) + i] = 0;
			}
			// clear output node output for next training entry
			d_outputNodeOutputs[globalIdX] = 0;
		}

		// divide total MSE by number of training examples for final MSE value
		d_totalMSEPerSession[globalIdX] /= noOfEntries;
	}
}


// function that converts data set from data structure to an array for use on GPU
float* convertDataSetForCUDA(float* dataArrayForCUDA, std::vector<dataEntry> dataSet, int noOfDataSamples, int noOfFeatures) {
	// cycles all training examples and extracts each feature value
	for (int i = 0; i < noOfDataSamples; i++) {
		for (int j = 0; j < noOfFeatures; j++) {
			dataArrayForCUDA[i * noOfFeatures + j] = dataSet[i].features[j];
		}
	}
	return dataArrayForCUDA;
}

// function that converts expected classifications of data set from data structure to an array for use on GPU
float* convertExpectedClassificationsForCUDA(float* expectedClassificationsArrayForCUDA, std::vector<dataEntry> dataSet, int noOfDataSamples) {
	// cycles all training examples and extracts expected classification value
	for (int i = 0; i < noOfDataSamples; i++) {
		expectedClassificationsArrayForCUDA[i] = dataSet[i].expectedClassification;
	}
	return expectedClassificationsArrayForCUDA;
}

// function used to generate random weight values and fill array
void generateRandomWeights(float* weightsArrayToFill, int numOfWeightsReq) {
	// random number generator
	std::random_device seed;
	std::mt19937 rng(seed());
	// define each random value should be between -0.5 & 0.5
	std::uniform_real_distribution<float> dist(-0.5, 0.5);

	// fill each value in array with random number
	for (int i = 0; i < numOfWeightsReq; i++) {
		weightsArrayToFill[i] = dist(rng);
	}
}

// general function for printing arrays
void printArray(float* targetArray, int sizeOfArray) {
	printf("\n");
	for (int i = 0; i < sizeOfArray; i++) {
		printf("%d = %f, \n", i, targetArray[i]);
	}
	printf("\n");
}

// function to calculate overall MSE of expected and actual outputs
float calculateOverallMSE(float* actualOutputs, float* expectedOutputs, int noOfEntries) {
	float totalError = 0.0f;
	float error = 0.0f;
	for (int i = 0; i < noOfEntries; i++) {
		//error = std::abs(expectedOutputs[i] - actualOutputs[i]);
		totalError += std::pow(std::abs(expectedOutputs[i] - actualOutputs[i]), 2);
		//printf("expected %f - actual %f = %f\n", expectedOutputs[i], actualOutputs[i], error);
	}
	totalError /= noOfEntries;
	return totalError;
}

int main() {

	// instantiate data reader class to read file
	dataReader d;
	// clock variables for recording performance
	clock_t beginClock;
	clock_t endClock;
	// variable to record milli seconds elapsed
	float ms = 0.0f;
	// variable to store details if GPU run was successful
	cudaError_t cudaError;

	// variable to store number of GPUs present in machine
	int deviceCount;
	// obtain number of GPUs present in machine
	cudaGetDeviceCount(&deviceCount);

	// variable to store which GPU index has the highest compute capability
	int indexOfBestGPU = -1;
	// variable to store value of highest compute capability across all GPUs
	int highestComputeCapability = 0;
	// variable to store number of multi processors, the best GPU has
	int numberOfStreamingMultiProcessors = 0;
	// cycle all GPUs available in machine
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, i);
		printf("Device %d \n", i);
		printf("Name: %s, Compute Capability, %d \n", properties.name, properties.major);
		printf("Number of Streaming Multi Processors: %d \n", properties.multiProcessorCount);
		// if this GPU has a higher compute capability 
		if (properties.major > highestComputeCapability) {
			// set index of best GPU
			indexOfBestGPU = i;
			// set new highest compute capability 
			highestComputeCapability = properties.major;
			// set number of streaming multi processors to number present in best GPU
			numberOfStreamingMultiProcessors = properties.multiProcessorCount;
		}
		printf("Warp Size: %d \n", properties.warpSize);
		printf("Threads Per Block: %d \n", properties.maxThreadsPerBlock);
		printf("Max Block Size: [%d, %d, %d] \n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
		printf("Max Grid Size: [%d, %d, %d] \n \n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);
	}

	// set device to run program as the index of best GPU
	cudaSetDevice(indexOfBestGPU);
	printf("Program Running using GPU index %d\n\n", indexOfBestGPU);

	// start timing for performance testing
	beginClock = clock();

	// load data from specified .txt file
	d.loadDataFile("MushroomDataSetNormalisedTestMin - Copy.txt");

	// vectors to store training, testing and validation sets of data 
	std::vector<dataEntry> trainingSet = d.dataSet.trainingSet;
	std::vector<dataEntry> testingSet = d.dataSet.testingSet;
	std::vector<dataEntry> validationSet = d.dataSet.validationSet;

	// variables for general structre of network
	int noOfTrainingSamples = trainingSet.size();
	int noOfTestingSamples = testingSet.size();
	int noOfFeatures = d.dataSet.trainingSet[0].features.size();
	int noOfClassifications = 1;

	// define network topology
	int inputNodes = noOfFeatures;
	int hiddenNodes = 10;
	int outputNodes = noOfClassifications;

	printf("Training Set: %d entries \n", trainingSet.size());
	printf("Testing Set: %d entries \n", testingSet.size());
	printf("Number of Training Sessions: %d\n", numberOfParallelTrainingSessions);

	// calculate block and grid dimensions to run Training Session Parallelism on GPU
	int trainingSessionParallelismThreadsPerBlock = ceil(numberOfParallelTrainingSessions / numberOfStreamingMultiProcessors);
	int trainingSessionParallelismNoOfBlocks = numberOfStreamingMultiProcessors;
	int trainingSessionParallelismNumOfThreads = trainingSessionParallelismThreadsPerBlock * trainingSessionParallelismNoOfBlocks;

	// allocate memory for training data on host
	int trainingDataSize = noOfFeatures * noOfTrainingSamples;
	float* h_trainingData = (float*)malloc(trainingDataSize * sizeof(float));
	// convert training data from class structure to array for use in CUDA kernel
	h_trainingData = convertDataSetForCUDA(h_trainingData, trainingSet, noOfTrainingSamples, noOfFeatures);
	// allocate memory for training data on device
	float* d_trainingData;
	cudaMalloc(&d_trainingData, trainingDataSize * sizeof(float));
	// copy training data in host array to device array
	cudaMemcpy(d_trainingData, h_trainingData, trainingDataSize * sizeof(float), cudaMemcpyHostToDevice);

	// allocate memory for training set expected classifications on host
	float* h_trainingExpectedClassifications = (float*)malloc(noOfTrainingSamples * sizeof(float));
	convertExpectedClassificationsForCUDA(h_trainingExpectedClassifications, trainingSet, noOfTrainingSamples);
	// allocate memory for training set expected classifications on device
	float* d_trainingExpectedClassifications;
	cudaMalloc(&d_trainingExpectedClassifications, noOfTrainingSamples * sizeof(float));
	// copy training set expected classifications in host array to device array
	cudaMemcpy(d_trainingExpectedClassifications, h_trainingExpectedClassifications, noOfTrainingSamples * sizeof(float), cudaMemcpyHostToDevice);

	// allocate memory for hidden node outputs for all threads on host
	int trainingHiddenNodeOutputsSize = trainingSessionParallelismNumOfThreads * hiddenNodes * noOfTrainingSamples;
	float* h_trainingHiddenNodeOutputs = (float*)malloc(trainingHiddenNodeOutputsSize * sizeof(float));
	// allocate memory for hidden node outputs for all threads on device
	float* d_trainingHiddenNodeOutputs;
	cudaMalloc(&d_trainingHiddenNodeOutputs, trainingHiddenNodeOutputsSize * sizeof(float));

	// allocate memory for output node outputs for all threads on host
	int trainingOutputNodeOutputsSize = trainingSessionParallelismNumOfThreads * noOfTrainingSamples;
	float* h_trainingOutputNodeOutputs = (float*)malloc(trainingOutputNodeOutputsSize * sizeof(float));
	// allocate memory for output node outputs for all threads on device
	float* d_trainingOutputNodeOutputs;
	cudaMalloc(&d_trainingOutputNodeOutputs, trainingOutputNodeOutputsSize * sizeof(float));

	// define total number of input layer weights required across all threads
	int numOfInputWeightsTotal = trainingSessionParallelismNumOfThreads * (noOfFeatures + 1) * hiddenNodes;
	// define total number of input layer weights required per thread
	int numOfInputWeightsPerSession = (noOfFeatures + 1) * hiddenNodes;
	// allocate memory for all input layer weights across all threads on host
	float* h_inputLayerWeights = (float*)malloc(numOfInputWeightsTotal * sizeof(float));
	// fill input layer weights with random values between -0.5 & 0.5
	generateRandomWeights(h_inputLayerWeights, numOfInputWeightsTotal);
	// allocate array holding all input layer weights across all threads on device
	float* d_inputLayerWeights;
	cudaMalloc(&d_inputLayerWeights, numOfInputWeightsTotal * sizeof(float));
	// copy input layer weights for all threads in host array to device array
	cudaMemcpy(d_inputLayerWeights, h_inputLayerWeights, numOfInputWeightsTotal * sizeof(float), cudaMemcpyHostToDevice);

	// define total number of hidden layer weights required across all threads
	int numOfHiddenWeightsTotal = trainingSessionParallelismNumOfThreads * (hiddenNodes + 1) * noOfClassifications;
	// define total number of hidden layer weights required per thread
	int numOfHiddenWeightsPerSession = (hiddenNodes + 1) * noOfClassifications;
	// allocate memory for all hidden layer weights across all threads on host
	float* h_hiddenLayerWeights = (float*)malloc(numOfHiddenWeightsTotal * sizeof(float));
	// fill hidden layer weights with random values between -0.5 & 0.5
	generateRandomWeights(h_hiddenLayerWeights, numOfHiddenWeightsTotal);
	// allocate memory for all hidden layer weights across all threads on device
	float* d_hiddenLayerWeights;
	cudaMalloc(&d_hiddenLayerWeights, numOfHiddenWeightsTotal * sizeof(float));
	// copy hidden layer weights for all threads in host array to device array
	cudaMemcpy(d_hiddenLayerWeights, h_hiddenLayerWeights, numOfHiddenWeightsTotal * sizeof(float), cudaMemcpyHostToDevice);

	// allocate memory for input layer delta weight values for all threads on host
	float* h_inputLayerDeltaWeights = (float*)malloc(numOfInputWeightsTotal * sizeof(float));
	// allocate memory for input layer delta weight values for all threads on device
	float* d_inputLayerDeltaWeights;
	cudaMalloc(&d_inputLayerDeltaWeights, numOfInputWeightsTotal * sizeof(float));

	// allocate memory for hidden layer delta weight values for all threads on host
	float* h_hiddenLayerDeltaWeights = (float*)malloc(numOfHiddenWeightsTotal * sizeof(float));
	// allocate memory for hidden layer delta weight values for all threads on device
	float* d_hiddenLayerDeltaWeights;
	cudaMalloc(&d_hiddenLayerDeltaWeights, numOfHiddenWeightsTotal * sizeof(float));

	// allocate memory for hidden layer delta node values for all threads on host
	float* d_hiddenLayerDeltas;
	cudaMalloc(&d_hiddenLayerDeltas, noOfTrainingSamples * (hiddenNodes + 1) * sizeof(float));

	// allocate memory for output layer delta node values for all threads on host
	float* d_outputLayerDeltas;
	cudaMalloc(&d_outputLayerDeltas, noOfTrainingSamples * noOfClassifications * sizeof(float));

	// allocate memory for total MSE values for all threads on host
	float* h_totalMSEPerSession = (float*)malloc(trainingSessionParallelismNumOfThreads * sizeof(float));
	// allocate memory for total MSE values for all threads on device
	float* d_totalMSEPerSession;
	cudaMalloc(&d_totalMSEPerSession, trainingSessionParallelismNumOfThreads * sizeof(float));


	// Run Training Session parallelism on GPU, each thread runs a different set of weights
	TrainingSessionParallelism << <trainingSessionParallelismNoOfBlocks, trainingSessionParallelismThreadsPerBlock >> > (d_trainingData, 
		d_trainingExpectedClassifications, noOfTrainingSamples, trainingSessionParallelismNumOfThreads, noOfFeatures, hiddenNodes, outputNodes, 
		d_hiddenLayerDeltas, d_outputLayerDeltas, d_inputLayerWeights, d_inputLayerDeltaWeights, numOfInputWeightsPerSession, d_hiddenLayerWeights, 
		d_hiddenLayerDeltaWeights, numOfHiddenWeightsPerSession, d_trainingHiddenNodeOutputs, d_trainingOutputNodeOutputs, d_totalMSEPerSession);

	// error validation for GPU run
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		printf("Error running kernel: %s\n", cudaGetErrorString(cudaError));
		std::getchar();
		return 1;
	}

	printf("\nTraining Session Paralellism Kernel Finished\n");

	// copy MSE values across all threads from device array to host array
	cudaMemcpy(h_totalMSEPerSession, d_totalMSEPerSession, trainingSessionParallelismNumOfThreads * sizeof(float), cudaMemcpyDeviceToHost);

	// copy modified weights across all threads back into host arrays
	cudaMemcpy(h_inputLayerWeights, d_inputLayerWeights, numOfInputWeightsTotal * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_hiddenLayerWeights, d_hiddenLayerWeights, numOfHiddenWeightsTotal * sizeof(float), cudaMemcpyDeviceToHost);

	// variables to store index and value of lowest MSE across threads
	float lowestMSE = 1.0f;
	int indexOfLowestMSE = 0;
	// cycle all MSE values recorded across threads
	for (int i = 0; i < trainingSessionParallelismNumOfThreads; i++) {
		// if current threads MSE is lower than current lowest
		if (h_totalMSEPerSession[i] < lowestMSE) {
			// set new lowest MSE value and index
			lowestMSE = h_totalMSEPerSession[i];
			indexOfLowestMSE = i;
		}
	}

	printf("\nLowest Training MSE = %f at thread index %d \n", lowestMSE, indexOfLowestMSE);

	// calculate start indexes of winning weights
	int winningInputWeightStartIndex = indexOfLowestMSE * numOfInputWeightsPerSession;
	int winningHiddenWeightStartIndex = indexOfLowestMSE * numOfHiddenWeightsPerSession;

	// allocate memory for all winning input layer weights on host
	float* h_finalInputLayerWeights = (float*)malloc(numOfInputWeightsPerSession * sizeof(float));
	// allocate memory for all winning input layer weights on device
	float* d_finalInputLayerWeights;
	cudaMalloc(&d_finalInputLayerWeights, numOfInputWeightsPerSession * sizeof(float));

	// allocate memory for all winning hidden layer weights on host
	float* h_finalHiddenLayerWeights = (float*)malloc(numOfHiddenWeightsPerSession * sizeof(float));
	// allocate memory for all winning hidden layer weights on device
	float* d_finalHiddenLayerWeights;
	cudaMalloc(&d_finalHiddenLayerWeights, numOfHiddenWeightsPerSession * sizeof(float));

	// extract winning set of input layer weights from array containing input weights for all threads
	int start = 0;
	for (int i = winningInputWeightStartIndex; i < winningInputWeightStartIndex + numOfInputWeightsPerSession; i++) {
		h_finalInputLayerWeights[start] = h_inputLayerWeights[i];
		start++;
	}

	// extract winning set of hidden layer weights from array containing hidden weights for all threads
	start = 0;
	for (int i = winningHiddenWeightStartIndex; i < winningHiddenWeightStartIndex + numOfHiddenWeightsPerSession; i++) {
		h_finalHiddenLayerWeights[start] = h_hiddenLayerWeights[i];
		start++;
	}
	
	// copy winning set of input layer weights to device array
	cudaMemcpy(d_finalInputLayerWeights, h_finalInputLayerWeights, numOfInputWeightsPerSession * sizeof(float), cudaMemcpyHostToDevice);

	// copy winning set of hidden layer weights to device array
	cudaMemcpy(d_finalHiddenLayerWeights, h_finalHiddenLayerWeights, numOfHiddenWeightsPerSession * sizeof(float), cudaMemcpyHostToDevice);

	// define block and grid dimensions to run Training Session Parallelism on GPU
	int testingFeedForwardThreadsPerBlock = ceil(noOfTestingSamples / numberOfStreamingMultiProcessors);
	int testingFeedForwardNoOfBlocks = numberOfStreamingMultiProcessors;
	int testingFeedForwardNumOfThreads = testingFeedForwardThreadsPerBlock * testingFeedForwardNoOfBlocks;

	
	// allocate memory for testing data on host
	int testingDataSize = noOfFeatures * noOfTestingSamples;
	float* h_testingData = (float*)malloc(testingDataSize * sizeof(float));
	// convert testing data from class structure to array for use in CUDA kernel
	h_testingData = convertDataSetForCUDA(h_testingData, testingSet, noOfTestingSamples, noOfFeatures);
	// allocate memory for testing data on device
	float* d_testingData;
	cudaMalloc(&d_testingData, testingDataSize * sizeof(float));
	// copy testing data in host array to device array
	cudaMemcpy(d_testingData, h_testingData, testingDataSize * sizeof(float), cudaMemcpyHostToDevice);

	// allocate memory for testing set expected classifications on host
	float* h_testingExpectedClassifications = (float*)malloc(noOfTestingSamples * sizeof(float));
	convertExpectedClassificationsForCUDA(h_testingExpectedClassifications, testingSet, noOfTestingSamples);
	// allocate memory for testing set expected classifications on device
	float* d_testingExpectedClassifications;
	cudaMalloc(&d_testingExpectedClassifications, noOfTestingSamples * sizeof(float));
	// copy testing set expected classifications in host array to device array
	cudaMemcpy(d_testingExpectedClassifications, h_testingExpectedClassifications, noOfTestingSamples * sizeof(float), cudaMemcpyHostToDevice);

	// allocate memory for hidden node outputs for all threads on host
	int testingHiddenNodeOutputsSize = testingFeedForwardNumOfThreads * hiddenNodes * noOfTestingSamples;
	float* h_testingHiddenNodeOutputs = (float*)malloc(testingHiddenNodeOutputsSize * sizeof(float));
	// allocate memory for hidden node outputs for all threads on device
	float* d_testingHiddenNodeOutputs;
	cudaMalloc(&d_testingHiddenNodeOutputs, testingHiddenNodeOutputsSize * sizeof(float));

	// allocate memory for output node outputs for all threads on host
	int testingOutputNodeOutputsSize = testingFeedForwardNumOfThreads * noOfTestingSamples;
	float* h_testingOutputNodeOutputs = (float*)malloc(testingOutputNodeOutputsSize * sizeof(float));
	// allocate memory for output node outputs for all threads on device
	float* d_testingOutputNodeOutputs;
	cudaMalloc(&d_testingOutputNodeOutputs, testingOutputNodeOutputsSize * sizeof(float));


	// perform forward pass on GPU, running all testing set examples in parallel
	feedForwardKernel << <testingFeedForwardNoOfBlocks, testingFeedForwardThreadsPerBlock >> > (d_testingData, noOfTestingSamples, 
		noOfFeatures, hiddenNodes, outputNodes, d_finalInputLayerWeights, d_finalHiddenLayerWeights, d_testingHiddenNodeOutputs, 
		d_testingOutputNodeOutputs);
		
	// error validation for GPU run
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		printf("Error running kernel: %s\n", cudaGetErrorString(cudaError));
		std::getchar();
		return 1;
	}
	
	printf("\nTesting Set Feed Forward Kernel Finished\n");
	
	// copy arrays from device to host memory
	cudaMemcpy(h_testingOutputNodeOutputs, d_testingOutputNodeOutputs, noOfTestingSamples * sizeof(float), cudaMemcpyDeviceToHost);

	// calculate overall MSE for testing set forward pass results
	float MSE = calculateOverallMSE(h_testingOutputNodeOutputs, h_testingExpectedClassifications, noOfTestingSamples);
	printf("\nMSE of Testing set = %f \n\n", MSE);

	// free memory allocated on host
	free(h_trainingData);
	h_trainingData = NULL;
	free(h_trainingExpectedClassifications);
	h_trainingExpectedClassifications = NULL;
	free(h_trainingHiddenNodeOutputs);
	h_trainingHiddenNodeOutputs = NULL;
	free(h_trainingOutputNodeOutputs);
	h_trainingOutputNodeOutputs = NULL;
	free(h_inputLayerWeights);
	h_inputLayerWeights = NULL;
	free(h_hiddenLayerWeights);
	h_hiddenLayerWeights = NULL;
	free(h_inputLayerDeltaWeights);
	h_inputLayerDeltaWeights = NULL;
	free(h_hiddenLayerDeltaWeights);
	h_hiddenLayerDeltaWeights = NULL;
	free(h_totalMSEPerSession);
	h_totalMSEPerSession = NULL;
	free(h_testingData);
	h_testingData = NULL;
	free(h_testingExpectedClassifications);
	h_testingExpectedClassifications = NULL;
	free(h_testingHiddenNodeOutputs);
	h_testingHiddenNodeOutputs = NULL;
	free(h_testingOutputNodeOutputs);
	h_testingOutputNodeOutputs = NULL;
	free(h_finalInputLayerWeights);
	h_finalInputLayerWeights = NULL;
	free(h_finalHiddenLayerWeights);
	h_finalHiddenLayerWeights = NULL;

	// free memory allocated on device 
	cudaFree(d_trainingData);
	d_trainingData = NULL;
	cudaFree(d_trainingExpectedClassifications);
	d_trainingExpectedClassifications = NULL;
	cudaFree(d_trainingHiddenNodeOutputs);
	d_trainingHiddenNodeOutputs = NULL;
	cudaFree(d_trainingOutputNodeOutputs);
	d_trainingOutputNodeOutputs = NULL;
	cudaFree(d_inputLayerWeights);
	d_inputLayerWeights = NULL;
	cudaFree(d_hiddenLayerWeights);
	d_hiddenLayerWeights = NULL;
	cudaFree(d_inputLayerDeltaWeights);
	d_inputLayerDeltaWeights = NULL;
	cudaFree(d_hiddenLayerDeltaWeights);
	d_hiddenLayerDeltaWeights = NULL;
	cudaFree(d_outputLayerDeltas);
	d_outputLayerDeltas = NULL;
	cudaFree(d_hiddenLayerDeltas);
	d_hiddenLayerDeltas = NULL;
	cudaFree(d_totalMSEPerSession);
	d_totalMSEPerSession = NULL;
	cudaFree(d_testingData);
	d_testingData = NULL;
	cudaFree(d_testingExpectedClassifications);
	d_testingExpectedClassifications = NULL;
	cudaFree(d_testingHiddenNodeOutputs);
	d_testingHiddenNodeOutputs = NULL;
	cudaFree(d_testingOutputNodeOutputs);
	d_testingOutputNodeOutputs = NULL;
	cudaFree(d_finalInputLayerWeights);
	d_finalInputLayerWeights = NULL;
	cudaFree(d_finalHiddenLayerWeights);
	d_finalHiddenLayerWeights = NULL;

	endClock = clock();
	ms = 1000.0f * (endClock - beginClock) / CLOCKS_PER_SEC;
	printf("Program took %fms \n \n", ms);

	// stop program from closing
	std::getchar();

}