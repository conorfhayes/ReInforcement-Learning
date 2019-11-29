import java.util.ArrayList;
import java.util.Random;


public class NeuralNetwork {
	
	public int inputSize = 0;
	public int hiddenSize1 = 0;
	public int hiddenSize2 = 0;
	public int outputSize = 0;
	public int state;
	//public ArrayList<Double> bias_1 = new ArrayList<Double>();
	//public ArrayList<Double> bias_2 = new ArrayList<Double>();
	
	public NeuralNetwork(int state, int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
		
		{
			this.state = state;
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
	
public ArrayList<ArrayList<Double>> dotProduct (Double number1, Double number2, int size1, int size2)
	
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
	
	public ArrayList<ArrayList<Double>> feedForward (int inputSize, int hiddenSize1, int hiddenSize2)
	
	{
		
	}
	
	

	
}
