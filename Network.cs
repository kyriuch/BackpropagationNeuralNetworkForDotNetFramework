
using System.Collections.Generic;

namespace BackpropagationNeuralNetwork
{
	internal class Layers
	{
		private Layer inputLayer;
		private Layer hiddenLayer;
		private Layer outputLayer;

		internal Layers(NetworkConfiguration networkConfiguration)
		{
			inputLayer = new Layer(networkConfiguration.InputLayerNeurons);
			hiddenLayer = new Layer(networkConfiguration.HiddenLayerNeurons, inputLayer);
			outputLayer = new Layer(networkConfiguration.OutputLayerNeurons, hiddenLayer);
		}

		internal Layer getInputLayer() => inputLayer;
		internal Layer getHiddenLayer() => hiddenLayer;
		internal Layer getOutputLayer() => outputLayer;
	}

	internal class NetworkData
	{
		public List<List<double>> Inputs;
		public List<List<double>> ExpectedOutputs;
		public Dictionary<int, List<double>> ActualOutputs;
	}

	internal struct NetworkConfiguration
	{
		public int InputLayerNeurons;
		public int HiddenLayerNeurons;
		public int OutputLayerNeurons;
		public int MaxEras;
		public double MinError;
		public double LearningRate;
		public double Momentum;
	}

    public class Network
    {
		private Layers layers;
		private NetworkData networkData;
		private NetworkConfiguration networkConfiguration;

		internal Network(NetworkConfiguration networkConfiguration)
		{
			initializeNetwork(networkConfiguration);
		}

		private void initializeNetwork(NetworkConfiguration networkConfiguration)
		{
			layers = new Layers(networkConfiguration);
			this.networkConfiguration = networkConfiguration;
		}
    }
}
