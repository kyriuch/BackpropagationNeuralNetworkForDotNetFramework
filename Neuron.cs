﻿
using System;
using System.Collections.Generic;
using System.Linq;

namespace BackpropagationNeuralNetwork
{
	internal static class IdHelper
	{
		private static int id = 0;

		public static void GetNextId(ref int value)
		{
			value = id++;
		}
	}

	internal static class RandomHelper
	{
		private static Random random = new Random();

		public static double GetRandomMinusOneToOne()
		{
			return random.NextDouble() * 2 - 1;
		}
	}

	internal struct NeuronData
	{
		public int Id;
		public double Output;
		public double BiasWeight;
		public double BiasDiff;
		public double SignalError;
	}

	internal sealed class Neuron
	{
		private NeuronData neuronData;

		private Dictionary<int, Neuron> leftSideNeurons;
		private Dictionary<int, double> weights;
		private Dictionary<int, double> weightsDifferences;

		internal Neuron() // input layer neuron constructor
		{
			initializeNeuronData(true);
			
		}

		internal Neuron(List<Neuron> attachedNeurons) // hidden/output layer neuron constructor
		{
			initializeNeuronData(false);
			initializeDictionaries(attachedNeurons);
		}

		internal int getNeuronId()
		{
			return neuronData.Id;
		}

		internal void activate()
		{
			neuronData.Output = 1d / (1d + Math.Exp(neuronData.Output));
		}

		internal void sumWeights()
		{
			neuronData.Output = weights.Select(pair => pair.Value * leftSideNeurons[pair.Key].neuronData.Output).Sum() + neuronData.BiasWeight;
		}

		private void initializeNeuronData(bool isInputLayerNeuron)
		{
			neuronData = new NeuronData();
			IdHelper.GetNextId(ref neuronData.Id);

			if(!isInputLayerNeuron)
			{
				neuronData.BiasWeight = RandomHelper.GetRandomMinusOneToOne();
				neuronData.BiasDiff = 0d;
			}
		}

		private void initializeDictionaries(List<Neuron> attachedNeurons)
		{
			leftSideNeurons = new Dictionary<int, Neuron>(attachedNeurons.Count);
			weights = new Dictionary<int, double>(attachedNeurons.Count);
			weightsDifferences = new Dictionary<int, double>(attachedNeurons.Count);

			foreach(Neuron neuron in attachedNeurons)
			{
				leftSideNeurons.Add(neuron.getNeuronId(), neuron);
				weights.Add(neuron.getNeuronId(), RandomHelper.GetRandomMinusOneToOne());
				weightsDifferences.Add(neuron.getNeuronId(), 0d);
			}
		}
	}
}