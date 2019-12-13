

import java.util.ArrayList;
import java.util.Arrays;
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
	public String networkName = "";
	
	public NeuralNetwork(double learningRate, int inputSize, int hiddenSize1, int hiddenSize2, int outputSize, String networkName)
		
		{
		
			this.learningRate = learningRate;
			this.inputSize = inputSize;
			this.hiddenSize1 = hiddenSize1;
			this.hiddenSize2 = hiddenSize2;
			this.outputSize = outputSize;
			this.networkName = networkName;
			initialiseWeightsAndBias();
			
			
			
	        		
		}
	
	private static Random random;
    private static long seed;
    
    static {
        seed = System.currentTimeMillis();
        random = new Random(seed);
    }
	
	public void setState(int state)
	{
		this.state = state;
	}
	
	 public static void setSeed(long s) {
	        seed = s;
	        random = new Random(seed);
	    }
	
	public static double uniform() {
        return random.nextDouble();
    }
	
	public static int uniform(int n) {
	        if (n <= 0) {
	            throw new IllegalArgumentException("argument must be positive: " + n);
	        }
	        return random.nextInt(n);
	  }
	
	
	public static long uniform(long n) {
        if (n <= 0L) {
            throw new IllegalArgumentException("argument must be positive: " + n);
        }

        long r = random.nextLong();
        long m = n - 1;

        // power of two
        if ((n & m) == 0L) {
            return r & m;
        }

        // reject over-represented candidates
        long u = r >>> 1;
        while (u + m - (r = u % n) < 0L) {
            u = random.nextLong() >>> 1;
        }
        return r;
    }

	
	public static int uniform(int a, int b) {
        if ((b <= a) || ((long) b - a >= Integer.MAX_VALUE)) {
            throw new IllegalArgumentException("invalid range: [" + a + ", " + b + ")");
        }
        return a + uniform(b - a);
    }

    /**
     * Returns a random real number uniformly in [a, b).
     *
     * @param a the left endpoint
     * @param b the right endpoint
     * @return a random real number uniformly in [a, b)
     * @throws IllegalArgumentException unless {@code a < b}
     */
    public static double uniform(double a, double b) {
        if (!(a < b)) {
            throw new IllegalArgumentException("invalid range: [" + a + ", " + b + ")");
        }
        return a + uniform() * (b - a);
    }
	
	public static double[][] random(int m, int n) {
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = uniform(0.0, 1.0);
            }
        }
        return a;
    }
	
	 public static double[][] power(double[][] x, int a) {
	        int m = x.length;
	        int n = x[0].length;

	        double[][] y = new double[m][n];
	        for (int i = 0; i < m; i++) {
	            for (int j = 0; j < n; j++) {
	                y[i][j] = Math.pow(x[i][j], a);
	            }
	        }
	        return y;
	    }
	 
	 public static double[][] subtract(double a, double[][] b) {
	        int m = b.length;
	        int n = b[0].length;
	        double[][] c = new double[m][n];
	        for (int i = 0; i < m; i++) {
	            for (int j = 0; j < n; j++) {
	                c[i][j] = a - b[i][j];
	            }
	        }
	        return c;
	    }
	 
	 public static double[][] divide(double[][] x, int a) {
	        int m = x.length;
	        int n = x[0].length;

	        double[][] z = new double[m][n];

	        for (int i = 0; i < m; i++) {
	            for (int j = 0; j < n; j++) {
	                z[i][j] = (x[i][j] / a);
	            }
	        }
	        return z;
	    }
	
	public void initialiseWeightsAndBias()
	
	{
		this.bias_1 = RandomUniform(-0.5, 0.5, 1, hiddenSize1);
		this.bias_2 = RandomUniform(-0.5, 0.5, 1, hiddenSize2);
		this.bias_3 = RandomUniform(-0.5, 0.5, 1, outputSize);
		
		this.hidden1_weights = RandomUniform(-1.0, 1.0, inputSize, hiddenSize1);
		this.hidden2_weights = RandomUniform(-1.0, 1.0, hiddenSize1, hiddenSize2);
		this.output_weights = RandomUniform(-1.0, 1.0, hiddenSize2, outputSize);
		
		this.hidden1_weights = random(inputSize, hiddenSize1);
		this.hidden2_weights = random(hiddenSize1, hiddenSize2);
		this.output_weights = random(hiddenSize2, outputSize);
		
		this.bias_1 = random(1, hiddenSize1);
		this.bias_2 = random(1, hiddenSize2);
		this.bias_3 = random(1, outputSize);
		
		
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
	
	public static double[][] multiply(double x, double[][] a) {
        int m = a.length;
        int n = a[0].length;

        double[][] y = new double[m][n];
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                y[j][i] = a[j][i] * x;
            }
        }
        return y;
    }
	
	public double[][] updateWeights(int layer, double learningRate, double[][] matrix)
	
	{
		double[][] weights = getWeights(layer);
		double[][] learningRateMatrix = new double[1][1];
		learningRateMatrix[0][0] = learningRate;
		
		double[][] weights_ = multiply(learningRate, matrix);
		double[][] weights__ = subtract(weights, weights_);
		//System.out.println(Arrays.deepToString(weights));
		//double[][] test = getWeights(layer) = weights;
		setWeights(weights__, layer);
		
		return weights__;
	}
	
	public void updateBias(int layer, double learningRate, double[][] matrix)
	
	{
		double[][] bias = getBias(layer);
		double[][] learningRateMatrix = new double[1][1];
		learningRateMatrix[0][0] = learningRate;
		
		double[][] bias_ = multiply(learningRate, matrix);
		bias = subtract(bias, bias_);
		setBias(bias, layer);
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
		
		else if (layer == 3)
		{
			this.bias_3 = bias;
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
		
		//System.out.println("Error :: Cannot Return Correct Weights, Check Layer Input");
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
			
			//System.out.println("output_weights: " + Arrays.deepToString(this.output_weights));
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
        //System.out.println(m);
        //System.out.println(n);
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
        double[][] z_ = new double[m][n];
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                z_[i][j] = ( 1 - (1.0 / (1 + Math.exp(-a[i][j]))));
            }
        }
        
        double[][] out = multiply(z, z_);
        
        return out;
    }
	
	public static double[][] RELU(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] z = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                z[i][j] = Math.max(0.0, a[i][j]);
            }
        }
        return z;
    }
	
	public static double[][] RELU_derivative(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] z = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
            	if (a[i][j] <= 0.0)
            	{
            		z[i][j] = 0.001;
            	}
            	if (a[i][j] > 0.0)
            	{
            		z[i][j] = 1.0;
            	}
                
            }
        }
        return z;
    }
	
	
	
	
	public double[][] feedForwardStep (double[][] bias, double[][] weights, double[][] nodes)
	
	{
		
		double[][] hidden_ = dot(nodes, weights);
		double[][] hidden =  add(hidden_, bias);
		//ArrayList<ArrayList<Double>> z_hidden = sigmoid(hidden_add);
		
		return hidden;
	}
	

	

	

	
	
}
