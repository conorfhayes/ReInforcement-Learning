import java.util.ArrayList;
import java.util.Random;


public class NeuralNetwork {
	
	public int inputSize = 0;
	public int hiddenSize1 = 0;
	public int hiddenSize2 = 0;
	public int outputSize = 0;
	public int state;
	public double learningRate = 0.0;
	//public ArrayList<Double> bias_1 = new ArrayList<Double>();
	//public ArrayList<Double> bias_2 = new ArrayList<Double>();
	
	public NeuralNetwork(double learningRate, int state, int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
		
		{
			this.state = state;
			this.learningRate = learningRate;
			this.inputSize = inputSize;
			this.hiddenSize1 = hiddenSize1;
			this.hiddenSize2 = hiddenSize2;
			this.outputSize = outputSize;
	        		
		}
	
	public ArrayList<ArrayList<Double>> bias_1 = new ArrayList<ArrayList<Double>>();
	public ArrayList<ArrayList<Double>> bias_2 = new ArrayList<ArrayList<Double>>();
	public ArrayList<ArrayList<Double>> hidden1_weights = new ArrayList<ArrayList<Double>>();
	public ArrayList<ArrayList<Double>> hidden2_weights = new ArrayList<ArrayList<Double>>();
	public ArrayList<ArrayList<Double>> hidden3_weights = new ArrayList<ArrayList<Double>>();
	
	public void initialiseWeightsAndBias()
	
	{
		this.bias_1 = RandomUniform(-0.5, 0.5, 1, hiddenSize1);
		this.bias_2 = RandomUniform(-0.5, 0.5, 1, hiddenSize2);
		
		this.hidden1_weights = RandomUniform(-1.0, 1.0, inputSize, hiddenSize1);
		this.hidden2_weights = RandomUniform(-1.0, 1.0, hiddenSize1, hiddenSize2);
		this.hidden3_weights = RandomUniform(-1.0, 1.0, hiddenSize2, outputSize);
	}
	
	public ArrayList<ArrayList<Double>> RandomUniform (Double number1, Double number2, int size1, int size2)
	
	{
		ArrayList<ArrayList<Double>> arrayFill = new ArrayList<ArrayList<Double>>();
		
		for (int x = 0; x < size1; x ++)
		{			
			for (int i = 0; i < size2; i ++ )				
			{	
				Random r = new Random();
				Double fillValue = r.nextDouble() * (number2 - number1) + number1;
				arrayFill.get(x).add(i, fillValue);
			}
			
		}
		
		return arrayFill;
	}
	
	public ArrayList<ArrayList<Double>> getBias(int layer)
	{
		if (layer == 1)
		{
			return this.bias_1;
		}
		
		else if (layer == 2)
		{
			return this.bias_2;
		}
		
		System.out.println("Error :: Cannot Return Correct Bias, Check Layer Input");
		return this.bias_1;
	}
	
	public void setBias(ArrayList<ArrayList<Double>> bias, int layer)
	{
		if (layer == 1)
		{
			this.bias_1 = bias;
			return;
		}
		
		else if (layer == 2)
		{
			this.bias_2 = bias;
			return;
		}		
		
		System.out.println("Error :: Cannot Set Bias, Check Layer Input");
		
	}
	
	public ArrayList<ArrayList<Double>> getWeights(int layer)
	{
		if (layer == 1)
		{
			return this.hidden1_weights;
		}
		
		else if (layer == 2)
		{
			return this.hidden2_weights;
		}
		
		else if (layer == 3)
		{
			return this.hidden3_weights;
		}
		
		System.out.println("Error :: Cannot Return Correct Weights, Check Layer Input");
		return this.hidden3_weights;
	}
	
	public void setWeights(ArrayList<ArrayList<Double>> weights, int layer)
	{
		if (layer == 1)
		{
			this.hidden1_weights = weights;
			return;
		}
		
		else if (layer == 2)
		{
			this.hidden2_weights = weights;
			return;
		}
		
		else if (layer == 3)
		{
			this.hidden3_weights = weights;
			return;
		}
		
		System.out.println("Error :: Cannot Set Weights, Check Layer Input");
		
		
	}
	
	public ArrayList<ArrayList<Double>> transpose(ArrayList<ArrayList<Double>> matrixIn)
	
	{
		ArrayList<ArrayList<Double>>matrixOut = new ArrayList<ArrayList<Double>>();
	    if (!matrixIn.isEmpty()) {
	        int noOfElementsInList = matrixIn.get(0).size();
	        for (int i = 0; i < noOfElementsInList; i++) {
	            ArrayList<Double> col = new ArrayList<Double>();
	            for (ArrayList<Double> row : matrixIn) {
	                col.add(row.get(i));
	            }
	            matrixOut.add(col);
	        }
	    }

	    return matrixOut;
	}
	
	public ArrayList<ArrayList<Double>> dotProduct (ArrayList<ArrayList<Double>> matrix1, ArrayList<ArrayList<Double>> matrix2)
		
		{
		
			ArrayList<ArrayList<Double>> arrayFill = new ArrayList<>(matrix1.size());
			
			for (int i = 0; i < matrix2.get(0).size(); i++)
			{
				arrayFill.add(new ArrayList());
			}
			
			
			for (int row = 0; row < arrayFill.size(); row ++)
				
			{			
				for (int col = 0; col < arrayFill.get(0).size(); col ++ )		
					
				{	
					double cell = 0;
					
					for (int j = 0; j < matrix2.size(); j++)
					
					{
						cell += matrix1.get(row).get(j) * matrix2.get(j).get(col);
						
					}
					
					arrayFill.get(row).set(col, cell);
					
				}
			}
			
			return arrayFill;
		}
	
	public int getColumnSize(ArrayList<ArrayList<Double>> matrix1)
	{	
		
		int col = matrix1.get(0).size();
		return col;
	}
	
	public int getRowSize(ArrayList<ArrayList<Double>> matrix1)
	{	
		
		int row = matrix1.size();
		return row;
	}
	
		

	public ArrayList<ArrayList<Double>> addMatrix (ArrayList<ArrayList<Double>> matrix1, ArrayList<ArrayList<Double>> matrix2)
	
	{	
		ArrayList<ArrayList<Double>> arrayFill = new ArrayList<ArrayList<Double>>();
		
		for (int i = 0; i < matrix1.size(); i++)
		{
			for (int j = 0; j < matrix2.get(0).size(); j ++)
			{
				arrayFill.get(i).set(j, matrix1.get(i).get(j) + matrix2.get(0).get(j));
			}
		}	
		
		return arrayFill;
	}
	
	public ArrayList<ArrayList<Double>> subtractMatrix (ArrayList<ArrayList<Double>> matrix1, ArrayList<ArrayList<Double>> matrix2)
	
	{	
		ArrayList<ArrayList<Double>> arrayFill = new ArrayList<ArrayList<Double>>();
		
		for (int i = 0; i < matrix1.size(); i++)
		{
			for (int j = 0; j < matrix2.get(0).size(); j ++)
			{
				arrayFill.get(i).set(j, matrix1.get(i).get(j) - matrix2.get(0).get(j));
			}
		}	
		
		return arrayFill;
	}
	
	public ArrayList<ArrayList<Double>> sigmoid(ArrayList<ArrayList<Double>> hidden)
	{
		for (int i = 0; i < hidden.size(); i++)
		{
		    for (int j = 0; j < hidden.get(i).size(); j++)
		    {
		    	hidden.get(i).set(j, 1.0/(1.0+(Math.exp(-j))) );
		    } 
		}
		
		return hidden;
	}
	
	public ArrayList<ArrayList<Double>> sigmoid_derivative(ArrayList<ArrayList<Double>> hidden)
	{
		for (int i = 0; i < hidden.size(); i++)
		{
		    for (int j = 0; j < hidden.get(i).size(); j++)
		    {
		    	hidden.get(i).set(j, 1.0/(1.0+(Math.exp(-j))) );
		    } 
		}
		
		return hidden;
	}
	
	
	public ArrayList<ArrayList<Double>> feedForwardStep (int layer, ArrayList<ArrayList<Double>> nodes)
	
	{
		ArrayList<ArrayList<Double>> bias = getBias(layer);
		ArrayList<ArrayList<Double>> weights = getWeights(layer);
		ArrayList<ArrayList<Double>> hidden_ = dotProduct(nodes, weights);
		ArrayList<ArrayList<Double>> hidden =  addMatrix(hidden_, bias);
		//ArrayList<ArrayList<Double>> z_hidden = sigmoid(hidden_add);
		
		return hidden;
	}
	
	public ArrayList<ArrayList<Double>> backPropagationStep (String networkType, int layer, ArrayList<ArrayList<Double>> input1, 
			ArrayList<ArrayList<Double>> input2, ArrayList<ArrayList<Double>> input3)
	
	{
		ArrayList<ArrayList<Double>> bias = getBias(layer);
		ArrayList<ArrayList<Double>> weights = getWeights(layer);
		ArrayList<ArrayList<Double>> error = new ArrayList<ArrayList<Double>>();
		
		if (layer == 3)
		{
			error = subtractMatrix(input1, input2);
		}
		
		else
		{
			error = dotProduct(input1, transpose(weights));
		}
		
		ArrayList<ArrayList<Double>> prediction = sigmoid_derivative(input2);
		ArrayList<ArrayList<Double>> delta = dotProduct(error, prediction);
		ArrayList<ArrayList<Double>> cost = dotProduct(transpose(input3), prediction);
		
		ArrayList<ArrayList<Double>> learningRateMatrix = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> biasMatrix = new ArrayList<ArrayList<Double>>();
		
		
		for (int i = 0; i < prediction.size(); i ++)
		{
			for (int j = 0; j < prediction.get(i).size(); j ++)
			{
				learningRateMatrix.get(i).set(j, this.learningRate);
			}
		}
		
		setWeights(subtractMatrix(weights, dotProduct(learningRateMatrix, prediction)), layer);
		
		int biasSum = 0;
		
		for (int i = 0; i < cost.size(); i ++)
		{
			for (int j = 0; j < cost.get(i).size(); j ++)
			{
				biasSum += cost.get(i).get(j);
			}
		}
		
		double biasValue = biasSum * this.learningRate;
		for (int i = 0; i < bias.size(); i ++)
		{
			for (int j = 0; j < bias.get(i).size(); j ++)
			{
				biasMatrix.get(i).set(j, biasValue);
			}
		}
		
		
		setBias(subtractMatrix(bias, biasMatrix), layer);
		
		return delta;
		
		
		
		
	
	}
	
}
