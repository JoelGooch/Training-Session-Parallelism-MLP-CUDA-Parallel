#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <ctime>

////////////////////////////////////////////////////////////////////////
//																	  //
//				Multi-Layer Perceptron w/ Backpropagation	    	  //
//						  Serial Implementation				 	      //
//																	  //
//	    Joel Gooch, BSc Computer Science, University of Plymouth	  //
//																	  //
////////////////////////////////////////////////////////////////////////

//global variables
int maxEpochs = 3;
float learningRate = 0.2f;
float momentum = 0.5f;

// number of training samples to use for training
int numberOfTrainingSamplesToUse = 1000;
// number of testing samples to use for final test
int numberOfTestingSamplesToUse = 512;
// how many threads to run in parallel with different weights
int numberOfTrainingSessions = 512;

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
	int noOfDataEntries = 0;

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
				noOfDataEntries++;
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
		printf("success opening input file: %s, reads: %d \n", filename, data.size());

		// close file
		inputFile.close();
		return true;
	}
	else {
		printf("error opening input file: %s \n", filename);
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

struct Connection {
	float weight;
	float deltaWeight;
};

class Node;

typedef std::vector<Node> Layer;

class Node {
public: // public constructor  functions
	Node(int numOutputs, int myIndex);
	void setOutputVal(float val) { nodeOutputVal = val; }
	float getOutputVal(void) const { return nodeOutputVal; }
	void calculateWeightedSum(const Layer &prevLayer);
	void activationFunction();
	float derivativeActivationFunction();
	void calcOutputGradients(float expectedClassification);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	std::vector<Connection> nodeOutputWeights;
private: // private variables
	float sumDeltaOfWeights(const Layer &nextLayer) const;
	float nodeOutputVal;
	int nodeIndex;
	float gradient;
};

// Node constructor
Node::Node(int numOutputs, int myIndex) {
	nodeIndex = myIndex;
	for (int connection = 0; connection < numOutputs; connection++) {
		// add new connection from node
		nodeOutputWeights.push_back(Connection());
		// generate random number between -0.5 and 0.5
		std::random_device seed;
		std::mt19937 rng(seed());
		std::uniform_real_distribution<float> dist(-0.5, 0.5);
		// set new connection weight to random value
		nodeOutputWeights.back().weight = dist(rng);
	}
}

// function used to update weights after gradients have been calculated
void Node::updateInputWeights(Layer &prevLayer) {
	for (int node = 0; node < prevLayer.size(); node++) {
		// select current node for readability 
		Node &prevNode = prevLayer[node];
		float oldWeight = prevNode.nodeOutputWeights[nodeIndex].weight;
		float oldDeltaWeight = prevNode.nodeOutputWeights[nodeIndex].deltaWeight;
		float newDeltaWeight = learningRate * prevNode.nodeOutputVal * gradient + (momentum * oldDeltaWeight);
		// store value of delta weight for next weight change
		prevNode.nodeOutputWeights[nodeIndex].deltaWeight = newDeltaWeight;
		// modify weight
		prevNode.nodeOutputWeights[nodeIndex].weight += newDeltaWeight;
	}
}

// weighted sum calculation function for hidden-output layer calculation
void Node::calculateWeightedSum(const Layer &prevLayer) {
	float totalSum = 0.0f;

	// for all nodes in layer
	for (unsigned node = 0; node < prevLayer.size(); node++) {
		// calculate weighted sums
		totalSum += prevLayer[node].nodeOutputVal * prevLayer[node].nodeOutputWeights[nodeIndex].weight;
	}
	// set node output value to new value
	nodeOutputVal = totalSum;
}

// sum delta weight values for entire network layer
float Node::sumDeltaOfWeights(const Layer &nextLayer) const {
	float sum = 0.0f;
	// cycle all nodes in layer
	for (int node = 0; node < nextLayer.size() - 1; node++) {
		// summate weight * graident values
		sum += nodeOutputWeights[node].weight * nextLayer[node].gradient;
	}
	return sum;
}

// calculate gradient of node in hidden layer
void Node::calcHiddenGradients(const Layer &nextLayer) {
	// calculate delta weights for next layer
	float deltaOfWeights = sumDeltaOfWeights(nextLayer);
	// set gradient of current node
	gradient = deltaOfWeights * Node::derivativeActivationFunction();
}

// calculate gradient of node in output layer
void Node::calcOutputGradients(float expectedClassification) {
	// calculate error value of node
	float delta = expectedClassification - nodeOutputVal;
	// set gradient of current node
	gradient = delta * Node::derivativeActivationFunction();
}

// hyperbolic tangent activation function
void Node::activationFunction() {
	// hyperbolic tangent activation function
	nodeOutputVal = tanh(nodeOutputVal);
}

// derivative hyperbolic tangent activation function for calculating gradient
float Node::derivativeActivationFunction() {
	// hyperbolic tan derivative activation function
	return 1 - (nodeOutputVal * nodeOutputVal);
}

class Network {
public:
	Network(const std::vector <int> topology);
	void feedForward(std::vector<float> &inputVals);
	void backPropagation(std::vector<dataEntry> &trainingData);
	std::vector<Layer> networkLayers;
};

// Network constructor
Network::Network(const std::vector <int> topology) {
	int numLayers = topology.size();
	for (int layerNum = 0; layerNum < numLayers; layerNum++) {
		// create new layer
		networkLayers.push_back(Layer());

		// if layernum is output layer then number of outputs is 0, 
		// otherwise number of outputs is number of nodes in next layer
		int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		for (int node = 0; node < topology[layerNum]; node++) {
			networkLayers.back().push_back(Node(numOutputs, node));
		}

		// add bias node
		networkLayers[layerNum].push_back(Node(numOutputs, topology[layerNum]));
		networkLayers[layerNum].back().setOutputVal(1);
	}
}

// function that feeds input values through network and calculates output
void Network::feedForward(std::vector<float> &inputVals) {

	// initialise network with values
	for (unsigned i = 0; i < inputVals.size(); i++) {
		networkLayers[0][i].setOutputVal(inputVals[i]);
	}

	Layer &inputLayer = networkLayers[0];
	// cycle nodes in input layer to calculate output of hidden layer
	for (unsigned node = 0; node < networkLayers[1].size() - 1; node++) {
		networkLayers[1][node].calculateWeightedSum(inputLayer);
		networkLayers[1][node].activationFunction();
	}

	Layer &hiddenLayer = networkLayers[1];
	// cycle nodes in hidden layer to calculate output of output layer
	for (unsigned node = 0; node < networkLayers[2].size() - 1; node++) {
		networkLayers[2][node].calculateWeightedSum(hiddenLayer);
		networkLayers[2][node].activationFunction();
	}
}

// performs back propagation
void Network::backPropagation(std::vector<dataEntry> &trainingData) {
	// value of current epoch of training
	int currEpoch = 0; 

	// run for user defined number of epochs
	while (currEpoch < maxEpochs) {
		// cycle every entry in training data
		for (int entry = 0; entry < trainingData.size(); entry++) {
			dataEntry trainingEntry = trainingData[entry];

			// perform forward pass of data through network
			feedForward(trainingEntry.features);

			Layer &outputLayer = networkLayers.back();
			// cycle all nodes in output layer
			for (int node = 0; node < outputLayer.size() - 1; node++) {
				outputLayer[node].calcOutputGradients(trainingData[entry].expectedClassification);
			}

			// calculate hidden layer gradients
			for (int layerNum = networkLayers.size() - 2; layerNum > 0; layerNum--) {
				Layer &hiddenLayer = networkLayers[layerNum];
				Layer &nextLayer = networkLayers[layerNum + 1];
				// cycle all nodes in hidden layer
				for (int node = 0; node < hiddenLayer.size(); node++) {
					hiddenLayer[node].calcHiddenGradients(nextLayer);
				}
			}

			// update weights input and all hidden layers
			for (int layerNum = networkLayers.size() - 1; layerNum > 0; layerNum--) {
				Layer &currLayer = networkLayers[layerNum];
				Layer &prevLayer = networkLayers[layerNum - 1];
				for (int node = 0; node < currLayer.size() - 1; node++) {
					currLayer[node].updateInputWeights(prevLayer);
				}
			}
		}
		// move onto next epoch
		currEpoch++;
	}
}

// function to set all weights in network to newly initialised random numbers
void randomizeNetworkWeights(Network &myNetwork) {
	// generate random number between -0.5 and 0.5
	std::random_device seed;
	std::mt19937 rng(seed());
	std::uniform_real_distribution<float> dist(-0.5, 0.5);
	// cycle all nodes within network and change its weight to new random value
	for (int layer = 0; layer < myNetwork.networkLayers.size(); layer++) {
		for (int node = 0; node < myNetwork.networkLayers[layer].size(); node++) {
			for (int weight = 0; weight < myNetwork.networkLayers[layer][node].nodeOutputWeights.size(); weight++) {
				myNetwork.networkLayers[layer][node].nodeOutputWeights[weight].weight = dist(rng);
				myNetwork.networkLayers[layer][node].nodeOutputWeights[weight].deltaWeight = 0;
			}
		}
	}
}

int main() {

	// instantiate data reader class to read file
	dataReader d;
	// clock variables for recording performance
	clock_t beginClock;
	clock_t endClock;
	// variable to record milli seconds elapsed
	float ms = 0.0f;
	float totalError = 0.0f;

	// start timing for performance testing
	beginClock = clock();

	// load data from .txt file
	d.loadDataFile("MushroomDataSetNormalisedTestMin - Copy.txt");

	// vectors to store training, testing and validation sets of data 
	std::vector<dataEntry> trainingSet = d.dataSet.trainingSet;
	std::vector<dataEntry> testingSet = d.dataSet.testingSet;
	std::vector<dataEntry> validationSet = d.dataSet.validationSet;

	int noOfFeatures = d.dataSet.trainingSet[0].features.size();
	int noOfClassifications = 1;

	// define network topology
	std::vector<int> topology;
	int inputNodes = noOfFeatures;
	int hiddenNodes = 10;
	int outputNodes = 1;
	topology.push_back(inputNodes);
	topology.push_back(hiddenNodes);
	topology.push_back(outputNodes);

	// create network
	Network myNetwork(topology);

	printf("Training Set: %d entries \n", trainingSet.size());
	printf("Testing Set: %d entries \n", testingSet.size());
	printf("Number of Training Sessions: %d\n", numberOfTrainingSessions);

	// variable to store index of lowest MSE training session result
	int lowestMSEIndex = -1;
	// variable to store value of lowest MSE training session result
	float lowestMSE = 2.0f;

	// vectors to store winning input and hidden layer weights
	std::vector<float> inputLayerWinningWeights;
	std::vector<float> hiddenLayerWinningWeights;

	for (int trainingSession = 0; trainingSession < numberOfTrainingSessions; trainingSession++) {
		// randomize all network weights for new training session run
		randomizeNetworkWeights(myNetwork);

		// train the network using backpropagation
		myNetwork.backPropagation(trainingSet);

		// cycle all training examples
		for (unsigned i = 0; i < trainingSet.size(); i++) {
			// set pointer to next training entry
			dataEntry &data = trainingSet[i];

			// feed training example through network
			myNetwork.feedForward(data.features);

			// calculating total MSE error
			totalError += std::pow(std::abs(data.expectedClassification - myNetwork.networkLayers[2][0].getOutputVal()), 2);
		}

		// calculate MSE accross all training data
		totalError /= trainingSet.size();
		//printf("MSE on training session %d error across all training examples: %f \n \n", trainingSession, totalError);

		if (totalError < lowestMSE) {
			// clear winning weights arrays for new weights
			inputLayerWinningWeights.clear();
			hiddenLayerWinningWeights.clear();

			// set new lowest MSE value and index
			lowestMSE = totalError;
			lowestMSEIndex = trainingSession;

			// store values of winning input layer weights
			for (int node = 0; node < myNetwork.networkLayers[0].size(); node++) {
				for (int weight = 0; weight < myNetwork.networkLayers[0][node].nodeOutputWeights.size(); weight++) {
					inputLayerWinningWeights.push_back(myNetwork.networkLayers[0][node].nodeOutputWeights[weight].weight);
				}
			}

			// store values of winning hidden layer weights
			for (int node = 0; node < myNetwork.networkLayers[1].size(); node++) {
				for (int weight = 0; weight < myNetwork.networkLayers[1][node].nodeOutputWeights.size(); weight++) {
					hiddenLayerWinningWeights.push_back(myNetwork.networkLayers[1][node].nodeOutputWeights[weight].weight);
				}
			}
		}
		// reset total error value for next training session
		totalError = 0.0;
	}

	printf("Lowest Training MSE = %f in Training Session %d \n \n", lowestMSE, lowestMSEIndex);

	// variable used to cycle index of winning input weights vector in loop
	int vectorStartIndex = 0;

	// cycle input layer and set current network weights to winning set
	for (int node = 0; node < myNetwork.networkLayers[0].size(); node++) {
		for (int weight = 0; weight < myNetwork.networkLayers[0][node].nodeOutputWeights.size(); weight++) {
			myNetwork.networkLayers[0][node].nodeOutputWeights[weight].weight = inputLayerWinningWeights[vectorStartIndex];
			vectorStartIndex++;
		}
	}
	// reset variable for cycling winning hidden weights vector
	vectorStartIndex = 0;

	// cycle hidden layer and set current network weights to winning set
	for (int node = 0; node < myNetwork.networkLayers[1].size(); node++) {
		for (int weight = 0; weight < myNetwork.networkLayers[1][node].nodeOutputWeights.size(); weight++) {
			myNetwork.networkLayers[1][node].nodeOutputWeights[weight].weight = hiddenLayerWinningWeights[vectorStartIndex];
			vectorStartIndex++;
		}
	}

	// run testing data through network
	printf("Testing Set: %d entries \n", testingSet.size());
	for (unsigned i = 0; i < testingSet.size(); i++) {
		dataEntry &data = testingSet[i];
		// feed testing data entry forward through network
		myNetwork.feedForward(data.features);
		// sum total MSE error of network
		totalError += std::pow(std::abs(data.expectedClassification - myNetwork.networkLayers[2][0].getOutputVal()), 2);

	}
	// divide total MSE error by number of entries for final value
	totalError /= testingSet.size();
	printf("MSE error across all test examples: %f \n \n", totalError);

	// record time taken to run program
	endClock = clock();
	ms = 1000.0f * (endClock - beginClock) / CLOCKS_PER_SEC;
	printf("Program took %fms \n \n", ms);

	// stop program from closing
	std::getchar();

}