package RL_DEED;
import java.util.ArrayList;
import java.util.Random;


public class NeuralNetwork {
	
	public int inputSize = 0;
	public int hiddenSize1 = 0;
	public int hiddenSize2 = 0;
	public int outputSize = 0;
	public int state = 0;
	public double learningRate = 0.0;
	public double[][] bias_1 = new double[0][0];
	public double[][] bias_2 = new double[0][0];
	public double[][] bias_3 = new double[0][0];
	public double[][] hidden1_weights = new double[0][0];
	public double[][] hidden2_weights = new double[0][0];
	public double[][] output_weights = new double[0][0];
	
	public NeuralNetwork(double learningRate, int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
		
		{
			this.learningRate = learningRate;
			this.inputSize = inputSize;
			this.hiddenSize1 = hiddenSize1;
			this.hiddenSize2 = hiddenSize2;
			this.outputSize = outputSize;
			initialiseWeightsAndBias();
	        		
		}
	
	public void setState(int state)
	{
		this.state = state;
	}
	
	public void initialiseWeightsAndBias()
	
	{
		this.bias_1 = RandomUniform(-0.5, 0.5, 1, hiddenSize1);
		this.bias_2 = RandomUniform(-0.5, 0.5, 1, hiddenSize2);
		this.bias_3 = RandomUniform(-0.5, 0.5, 1, outputSize);
		
		this.hidden1_weights = RandomUniform(-1.0, 1.0, inputSize, hiddenSize1);
		this.hidden2_weights = RandomUniform(-1.0, 1.0, hiddenSize1, hiddenSize2);
		this.output_weights = RandomUniform(-1.0, 1.0, hiddenSize2, outputSize);
	}
	
	public double[][] RandomUniform (Double number1, Double number2, int size1, int size2)
	
	{
		double [][] arrayFill = new double[size1][size2];	
		
		for (int x = 0; x < size1; x ++)
		{
			for (int i = 0; i < size2; i ++ )				
			{	
				Random r = new Random();
				Double fillValue = r.nextDouble() * (number2 - number1) + number1;
				arrayFill[x][i] = fillValue;
			}
			
		}
		
		return arrayFill;
	}
	
	public double[][] getBias(int layer)
	{
		if (layer == 1)
		{
			return this.bias_1;
		}
		
		else if (layer == 2)
		{
			return this.bias_2;
		}
		
		else if (layer == 3)
		{
			return this.bias_3;
		}
		
		System.out.println("Error :: Cannot Return Correct Bias, Check Layer Input");
		return this.bias_1;
	}
	
	public void setBias(double[][] bias, int layer)
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
	
	public double[][] getWeights(int layer)
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
			return this.output_weights;
		}
		
		System.out.println("Error :: Cannot Return Correct Weights, Check Layer Input");
		return this.output_weights;
	}
	
	public void setWeights(double[][] weights, int layer)
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
			this.output_weights = weights;
			return;
		}
		
		System.out.println("Error :: Cannot Set Weights, Check Layer Input");
		
		
	}
	
	public static double[][] transpose(double[][] a) {        
		int m = a.length;
        int n = a[0].length;
        
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                b[j][i] = a[i][j];
            }
        }
        return b;
    }
	
	public static double[][] add(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }
	
	public static double[][] subtract(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i][j] = a[i][j] - b[i][j];
            }
        }
        return c;
    }
	
	public static double[][] dot(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n1; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }
	
	public static double[][] multiply(double[][] x, double[][] a) {
        int m = a.length;
        int n = a[0].length;

        if (x.length != m || x[0].length != n) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        double[][] y = new double[m][n];
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                y[j][i] = a[j][i] * x[j][i];
            }
        }
        return y;
    }

	
	public ArrayList<ArrayList<Double>> dotProduct (ArrayList<ArrayList<Double>> matrix1, ArrayList<ArrayList<Double>> matrix2)
		
		{
		
			ArrayList<ArrayList<Double>> arrayFill = new ArrayList<ArrayList<Double>>(matrix1.size());
			
			for (int i = 0; i < matrix1.size(); i ++)
			{
				arrayFill.add(new ArrayList());
				
				for (int j = 0; j < matrix2.get(0).size(); j ++)
				{
					arrayFill.get(i).add(0.0);
				}

				
			}			
			
			double cell = 0;
			
			for (int i = 0; i < matrix1.size(); i ++)	
				
			{	
				for (int j = 0; j < matrix2.get(0).size(); j ++)
					
				{	
					//System.out.println("Array Fill at 0 " + arrayFill.get(0).get(1));
					for (int k = 0; k < matrix1.get(0).size(); k ++ )		
						
					{								
						cell += matrix1.get(i).get(k) * matrix2.get(k).get(j);
						arrayFill.get(i).set(j, cell);						
					}						
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
		ArrayList<ArrayList<Double>> arrayFill = new ArrayList<ArrayList<Double>>(matrix1.size());
		
		for (int i = 0; i < matrix1.size(); i ++)
		{
			arrayFill.add(new ArrayList());
			
			for (int j = 0; j < matrix2.get(0).size(); j ++)
			{
				arrayFill.get(i).add(0.0);
			}			
		}
		
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
	
	public static double[][] sigmoid(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] z = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                z[i][j] = (1.0 / (1 + Math.exp(-a[i][j])));
            }
        }
        return z;
    }
	
	public static double[][] sigmoid_derivative(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] z = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                z[i][j] = (1.0 / (1 + Math.exp(-a[i][j])));
            }
        }
        return z;
    }
	
	
	
	public double[][] feedForwardStep (int layer, double[][] nodes)
	
	{
		double[][] bias = getBias(layer);
		double[][] weights = getWeights(layer);
		//System.out.print(weights.get(0).size());
		double[][] hidden_ = dot(nodes, weights);
		double[][] hidden =  add(hidden_, bias);
		//ArrayList<ArrayList<Double>> z_hidden = sigmoid(hidden_add);
		
		return hidden;
	}
	
	public double[][] backPropagationStep (String networkType, int layer, double[][] input1, 
			double[][] input2, double[][] input3)
	
	{
		double[][] bias = getBias(layer);
		double[][] weights = getWeights(layer);
		double[][] error = new double[0][0];
		
		if (layer == 3)
		{
			error = subtract(input1, input2);
		}
		
		else
		{
			error = dot(input1, transpose(weights));
		}
		
		double[][] prediction = sigmoid_derivative(input2);
		double[][] delta = dot(error, prediction);
		double[][] cost = dot(transpose(input3), prediction);
		
		double[][] learningRateMatrix = new double[0][0];
		double[][] biasMatrix = new double[0][0];
		
		if (networkType == "Expierience Replay")
		{
		for (int i = 0; i < prediction.length; i ++)
		{
			for (int j = 0; j < prediction[i].length; j ++)
			{
				learningRateMatrix[i][j] = this.learningRate;
			}
		}
		
		setWeights(subtract(weights, dot(learningRateMatrix, prediction)), layer);
		
		int biasSum = 0;
		
		for (int i = 0; i < cost.length; i ++)
		{
			for (int j = 0; j < cost[i].length; j ++)
			{
				biasSum += cost[i][j];
			}
		}
		
		double biasValue = biasSum * this.learningRate;
		for (int i = 0; i < bias.length; i ++)
		{
			for (int j = 0; j < bias[i].length; j ++)
			{
				biasMatrix[i][j] = biasValue;
			}
		}
		
		
		setBias(subtract(bias, biasMatrix), layer);
		}
		
		return delta;		
	}
	

	
	
}
