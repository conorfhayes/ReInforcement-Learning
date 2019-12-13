/*
 * Sample implementation of a Gridworld environment, using a Q-learning agent
 * Dr Patrick Mannion, Galway-Mayo Institute of Technology, March 2017
 */



import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


public class Environment {

	// environment parameters
	private int numActions = 4;
	private String[] actionLabels = new String[] {"North","East","South","West"};
	private static int xDimension = 10;
	private static int yDimension = 10;
	private int numEpisodes = 50000;
	private int maxTimesteps = 100;
	private boolean debug = false; // set to true to enable console output for debugging
	private boolean goalReached = false;
	private int[] goalLocationXY = new int[] {6,6};
	private int[] agentStartXY = new int[] {2,2};
	private float goalReward = 1.0f;
	private float stepPenalty = -1.0f;
	
	// Agent parameters
	private Agent agent;
	private int[] currentAgentCoords = new int[]{-1,-1}; // agent's current x,y position (state vector)
	private int[] previousAgentCoords  = new int[]{-1,-1};; // agent's previous x,y position (state vector)
	private float alpha = 0.10f;
	private boolean alphaDecays = false;
	private float alphaDecayRate = 0.9f;
	private float gamma = 1.0f;
	private float epsilon = 1.0f;
	private boolean epsilonDecays = true;
	private float epsilonDecayRate = 0.99f;
	public int miniBatchSize = 64;
	public double learningRate = 0.001;
	
	// data logging
	ArrayList<Integer> movesToGoal = new ArrayList<Integer>();
	
	public Environment() {
		
	}
	
	public void setupAgent() {
		NeuralNetwork targetNetwork = new NeuralNetwork(0.0001, 1, 3, 3, 4, "Target Network");
		NeuralNetwork policyNetwork = new NeuralNetwork(0.0001, 1, 3, 3, 4, "Policy Network");
		ExpierienceReplay expierienceReplay = new ExpierienceReplay();
		agent = new Agent(getNumStates(),numActions,alpha,gamma,epsilon, targetNetwork, policyNetwork, expierienceReplay);
		if(debug) {
			agent.enableDebugging();
		}
	}
	
	public static double sum(double[][] array) {
	    double sum = 0;
	    for (double value : array[0]) {
	        sum += value;
	    }
	    return sum;
	}
	
	public static double getMaxValue(double[][] numbers) {
        double maxValue = numbers[0][0];
        int maxValueIndex = 0;
        for (int j = 0; j < numbers.length; j++) {
            for (int i = 0; i < numbers[j].length; i++) {
                if (numbers[j][i] > maxValue) {
                    maxValue = numbers[j][i];
                    
                }
            }
        }
        return maxValue;        
	}
	
	public double[][] fill(int dim1, int dim2, double value)
	{
		double[][] array = new double [dim1][dim2];
		for (int j = 0; j < dim1; j ++)
		{
		for (int i = 0; i < dim2; i ++)
		{
			array[j][i] = value;
		}
		}
		
		return array;
	}
	
	
	public static int getMaxValueIndex(double[][] numbers) {
        double maxValue = numbers[0][0];
        int maxValueIndex = 0;
        for (int j = 0; j < numbers.length; j++) {
            for (int i = 0; i < numbers[j].length; i++) {
                if (numbers[j][i] > maxValue) {
                    maxValue = numbers[j][i];
                    maxValueIndex = i;
                }
            }
        }
        return maxValueIndex;        
	}
	

	
	public void doExperiment() {
		setupAgent();
		for(int e=0;e<numEpisodes;e++) {
			if(debug) {
				System.out.println("\nEnvironment: *************** Episode " + e + " starting ***************");
			}
			doEpisode();
			agent.expierienceReplay.clense();
			System.out.println(agent.expierienceReplay.actionHolder.size());
			
			if (e > 1)
			{
				int[] randomSample = new int[this.miniBatchSize];
				
				for (int i = 0; i < this.miniBatchSize; i ++ )
				{
					
					Random r = new Random();
					int randomNumber =  r.nextInt(((agent.expierienceReplay.actionHolder.size() - 1) - 0) + 1) + 0;
					randomSample[i] = randomNumber;
					
				}		
				
				for (int i = 0; i < this.miniBatchSize; i ++ )
				{
					int index = randomSample[i];
					double erState = agent.expierienceReplay.stateHolder.get(index);						
					int erAction = agent.expierienceReplay.actionHolder.get(index);
					double erReward = agent.expierienceReplay.rewardHolder.get(index);
					double erNextState = agent.expierienceReplay.nextStateHolder.get(index);					
					
					double[][] myDoubleArray =  new double[1][1];
					myDoubleArray[0][0] = (double) erState;
					
					double maxValue = 0;
					int maxIndex = 0;	
					
					
					
					
					// Expierience Replay FeedFordward 1
					
					double[][] StateMTRX = new double[1][1];
					StateMTRX[0][0] = erState;
					//StateMTRX[0][1] = erState;
					//StateMTRX[0][2] = erState;
					//StateMTRX[0][3] = erState;
					
					//StateMTRX = NeuralNetwork.transpose(StateMTRX);
					
					
					double[][] NextStateMTRX = new double[1][1];
					NextStateMTRX[0][0] = erNextState;
					
					double[][] pn_weights1 = agent.policyNetwork.getWeights(1);
					double[][] pn_bias1 = agent.policyNetwork.getBias(1);
					//System.out.println(Arrays.deepToString(pn_weights1));
					//System.out.println(Arrays.deepToString(StateMTRX));
					//System.out.println(Arrays.deepToString(pn_bias1));
					double[][] pn_hidden1 = agent.policyNetwork.feedForwardStep(pn_bias1,pn_weights1, StateMTRX);
					double[][] pn_z_hidden1 = agent.policyNetwork.RELU(pn_hidden1);
										
					double[][] pn_weights2 = agent.policyNetwork.getWeights(2);
					double[][] pn_bias2 = agent.policyNetwork.getBias(2);					
					double[][] pn_hidden2 = agent.policyNetwork.feedForwardStep(pn_bias2,pn_weights2, pn_z_hidden1);
					double[][] pn_z_hidden2 = agent.policyNetwork.RELU(pn_hidden2);
					
					double[][] pn_weights3 = agent.policyNetwork.getWeights(3);
					double[][] pn_bias3 = agent.policyNetwork.getBias(3);					
					double[][] pn_output = agent.policyNetwork.feedForwardStep(pn_bias3,pn_weights3, pn_z_hidden2);
					
					//System.out.println(Arrays.deepToString(pn_output));
					//System.out.println(erState);
					double[][] pn_z_output = agent.policyNetwork.RELU(pn_output);				
					
					double pn_maxQTarget = getMaxValue(pn_output);	
					int pn_maxQTargetIndex = getMaxValueIndex(pn_output);
					
					// Expierience Replay FeedFordward 1			
					
					double[][] tn_weights1 = agent.targetNetwork.getWeights(1);
					double[][] tn_bias1 = agent.targetNetwork.getBias(1);
					//System.out.println(Arrays.deepToString(pn_weights1));
					//System.out.println(Arrays.deepToString(StateMTRX));
					//System.out.println(Arrays.deepToString(pn_bias1));
					double[][] tn_hidden1 = agent.targetNetwork.feedForwardStep(tn_bias1,tn_weights1, NextStateMTRX);
					double[][] tn_z_hidden1 = agent.targetNetwork.RELU(tn_hidden1);
										
					double[][] tn_weights2 = agent.targetNetwork.getWeights(2);
					double[][] tn_bias2 = agent.targetNetwork.getBias(2);					
					double[][] tn_hidden2 = agent.targetNetwork.feedForwardStep(tn_bias2,tn_weights2, tn_z_hidden1);
					double[][] tn_z_hidden2 = agent.targetNetwork.RELU(tn_hidden2);
					
					double[][] tn_weights3 = agent.targetNetwork.getWeights(3);
					double[][] tn_bias3 = agent.targetNetwork.getBias(3);					
					double[][] tn_output = agent.targetNetwork.feedForwardStep(tn_bias3,tn_weights3, tn_z_hidden2);
					double[][] tn_z_output = agent.targetNetwork.sigmoid(tn_output);
					//double[][] tn_z_output = agent.targetNetwork.sigmoid(tn_output);
					
					double maxQTarget = getMaxValue(tn_output);	
					int maxQTargetIndex = getMaxValueIndex(tn_output);
					
					//System.out.println(pn_maxQTargetIndex);
					//System.out.println("Target :: "  + Arrays.deepToString(tn_output));
					//System.out.println("Predicted :: "  + Arrays.deepToString(pn_output));
					//System.out.println(Arrays.deepToString(pn_output));
					//System.out.println("Action :: " + erAction);
					//System.out.println("Predicted [Action]:: "  + pn_output[0][erAction]);
					//System.out.println("Target [Action]:: "  + tn_output[0][erAction]);
					//System.out.println("Max :: " + maxQTarget);
					
					double targets = erReward + (this.gamma * maxQTarget);
					
					
					double[][] target_fs= tn_output.clone();
					for (int x = 0; x < 4; x ++)
					{
						target_fs[0][x] = 0.0;
					}
					
					double value = targets - pn_output[0][erAction];
					//System.out.println("Value [Action]:: "  + value);
					//System.out.println("Max Target Index :: " + maxQTargetIndex);
					//System.out.println("erAction :: " + erAction);
					//System.out.println("erAction :: " + erAction);
					double check = pn_output[0][erAction];
					
					/*
					if (value > 1.0)
					{
						value = 1.0;
					}
					
					if (value < -1.0)
					{
						value = -1.0;
					}
					*/
						
					double norm_value = (value - (-5)) / (5 - (-5));
					target_fs[0][erAction] = norm_value;
					double error = norm_value;
					
					
					//target_f[0][erAction] = target;
					//System.out.println("Error ::" + error);
					//System.out.println("Agent :: " + agent.getAgentID() +  " Action ::" + erAction);
					//System.out.println("Check ::" + check);
					//System.out.println("Value ::" + value);
					//System.out.println("Value ::" + value);
					
					// BackPropagation
					double[][] error_out = target_fs;
					//double[][] error_out = agent.policyNetwork.subtract(target_f, tn_output);
					//System.out.println("Error out :: " + Arrays.deepToString(error_out));
					double[][] linear_diff = new double[1][4];
					for (int x = 0; x < 4; x ++)
					{
						linear_diff[0][x] = 1.0;
					}
					
					double[][] prediction_out = agent.policyNetwork.RELU_derivative(pn_output);
					double[][] out_error = error_out;
					//System.out.println("Prediction Out :: " + Arrays.deepToString(prediction_out));
					double[][] delta_out = agent.policyNetwork.multiply(out_error, prediction_out);
					//System.out.println("Delta Out :: " + Arrays.deepToString(delta_out));
					double[][] cost_out = agent.policyNetwork.dot(agent.policyNetwork.transpose(pn_z_output), delta_out);
					//System.out.println("Cost Out :: " + Arrays.deepToString(cost_out));
					
					double[][] error_hidden2 = agent.policyNetwork.dot(delta_out, agent.policyNetwork.transpose(pn_weights3));
					double[][] prediction_hidden2 = agent.policyNetwork.RELU_derivative(pn_hidden2);
					double[][] delta_hidden2 = agent.policyNetwork.multiply(error_hidden2, prediction_hidden2);
					double[][] cost_hidden2 = agent.policyNetwork.dot(agent.policyNetwork.transpose(pn_z_hidden1), delta_hidden2);
					
					double[][] error_hidden1 = agent.policyNetwork.dot(delta_hidden2, agent.policyNetwork.transpose(pn_weights2));
					double[][] prediction_hidden1 = agent.policyNetwork.RELU_derivative(pn_hidden1);
					double[][] delta_hidden1 = agent.policyNetwork.multiply(error_hidden1, prediction_hidden1);
					double[][] cost_hidden1 = agent.policyNetwork.dot(StateMTRX, delta_hidden1);
					
					//double[][] dZ3 = target_fs;
		            //double[][] dW3 = agent.policyNetwork.divide(agent.policyNetwork.dot(dZ3, agent.policyNetwork.transpose(pn_z_hidden2)), 100);
		            //double[][] db3 = agent.policyNetwork.divide(dZ3, 100);
		            
		            //double[][] dZ2 = agent.policyNetwork.multiply(agent.policyNetwork.dot(agent.policyNetwork.transpose(pn_weights3), dZ3)
		            //		, agent.policyNetwork.subtract(1.0, agent.policyNetwork.power(pn_z_hidden2, 2)));;
		            //double[][] dW2 = agent.policyNetwork.divide(agent.policyNetwork.dot(dZ2, agent.policyNetwork.transpose(X)), 5);
		            //double[][] db2 = agent.policyNetwork.divide(dZ2, 5);
					//System.out.println("StateMTRX :: " + Arrays.deepToString(StateMTRX));
		            
		            
					
					
					
					double[][] learningRateMTRX = new double[1][1];
					learningRateMTRX[0][0] = this.learningRate;
					
					double [][] learningRateMTRX3 = fill(cost_hidden1[0].length,cost_hidden1.length, this.learningRate);
					double [][] learningRateMTRX2 = fill(cost_hidden2[0].length,cost_hidden2.length, this.learningRate);
					double [][] learningRateMTRX1 = fill(cost_out[0].length,cost_out.length, this.learningRate);
					
					double [][] update_w3 = agent.policyNetwork.multiply(this.learningRate, cost_out);
					double [][] w3 = agent.policyNetwork.subtract(pn_weights3, update_w3);
					agent.policyNetwork.setWeights(w3, 3);
					
					double [][] update_w2 = agent.policyNetwork.multiply(this.learningRate, cost_hidden2);
					double [][] w2 = agent.policyNetwork.subtract(pn_weights2, update_w2);
					agent.policyNetwork.setWeights(w2, 2);
					
					double [][] update_w1 = agent.policyNetwork.multiply(this.learningRate, cost_hidden1);
					double [][] w1 = agent.policyNetwork.subtract(pn_weights1, update_w1);
					agent.policyNetwork.setWeights(w1, 1);
					
					double [][] bias3 = agent.policyNetwork.getBias(3);
					double cost_out_bias3 = sum(cost_out) * this.learningRate;
					double [][] bias_updater3 = fill(1,4, cost_out_bias3);
					double [][] bias_updater3_ = agent.policyNetwork.subtract(bias3, bias_updater3);
					agent.policyNetwork.setBias(bias_updater3_, 3);
					
					double [][] bias2 = agent.policyNetwork.getBias(2);
					double cost_out_bias2 = sum(cost_hidden2) * this.learningRate;
					double [][] bias_updater2 = fill(1,3, cost_out_bias2);
					double [][] bias_updater2_ = agent.policyNetwork.subtract(bias2, bias_updater2);
					agent.policyNetwork.setBias(bias_updater2_, 2);
					
					double [][] bias1 = agent.policyNetwork.getBias(1);
					double cost_out_bias1 = sum(cost_hidden1) * this.learningRate;
					double [][] bias_updater1 = fill(1,3, cost_out_bias1);
					double [][] bias_updater1_ = agent.policyNetwork.subtract(bias1, bias_updater1);
					agent.policyNetwork.setBias(bias_updater1_, 1);
					
					//agent.policyNetwork.updateWeights(3, this.learningRate, cost_out);
					//agent.policyNetwork.updateWeights(2, this.learningRate, cost_hidden2);
					//agent.policyNetwork.updateWeights(1, this.learningRate, cost_hidden1);			
					
					pn_weights1 = agent.policyNetwork.getWeights(1);
					pn_weights2 = agent.policyNetwork.getWeights(2);
					pn_weights3 = agent.policyNetwork.getWeights(3);
					
					
					//System.out.println("Check NN Test" + Arrays.deepToString(test));
					//System.out.println("Check NN Update Test" + Arrays.deepToString(updatetest));
					//System.out.println("Check NN Error Out " + Arrays.deepToString(error_out));
					//System.out.println("Check NN Prediction Out " + Arrays.deepToString(prediction_out));
					//System.out.println("Check NN Delta Out: " + Arrays.deepToString(delta_out));
					//System.out.println("Check NN Cost Out: " + Arrays.deepToString(cost_out));
					
					
					/*		 
					//System.out.println("Check NN Prediction Hidden Out: " + Arrays.deepToString(prediction_out));
					System.out.println("Check NN Error Hidden out : " + Arrays.deepToString(error_out));
					System.out.println("Check NN Delta Hidden out: " + Arrays.deepToString(delta_out));
					System.out.println("Check NN Cost Hidden out : " + Arrays.deepToString(cost_out));
					System.out.println("Check Weights out: " + Arrays.deepToString(pn_weights1));
					
					System.out.println("Check NN Prediction Hidden 2: " + Arrays.deepToString(prediction_hidden2));
					System.out.println("Check NN Error Hidden 2 : " + Arrays.deepToString(error_hidden2));
					System.out.println("Check NN Delta Hidden 2: " + Arrays.deepToString(delta_hidden2));
					System.out.println("Check NN Cost Hidden 2 : " + Arrays.deepToString(cost_hidden2));
					System.out.println("Check Weights 2: " + Arrays.deepToString(pn_weights2));
					
					System.out.println("Check NN Prediction Hidden 1: " + Arrays.deepToString(prediction_hidden1));
					System.out.println("Check NN Error Hidden 1 : " + Arrays.deepToString(error_hidden1));
					System.out.println("Check NN Delta Hidden 1: " + Arrays.deepToString(delta_hidden1));
					System.out.println("Check NN Cost Hidden 1 : " + Arrays.deepToString(cost_hidden1));
					System.out.println("Check Weights 1: " + Arrays.deepToString(pn_weights1));
					
					 
					System.out.println("Check Weights 1: " + Arrays.deepToString(pn_weights1));
					System.out.println("Check Weights 2: " + Arrays.deepToString(pn_weights2));
					System.out.println("Check Weights 3: " + Arrays.deepToString(pn_weights3));
					*/
					
					System.out.println("Check Weights 1: " + Arrays.deepToString(pn_weights1));
					//System.out.println("Check Weights 2: " + Arrays.deepToString(pn_weights2));
					//System.out.println("Check Weights 3: " + Arrays.deepToString(pn_weights3));
					}	
					
					//System.out.println();
					//System.out.println("Check NN Test" + Arrays.deepToString(cost_out));

					
					
					//System.out.println(Arrays.deepToString(weights3));
					
					if (agent.counter == 100 || e < 5)
					{
						
						//System.out.println(" **************** Updating Weights ******************");
						//update_w3 = agent.targetNetwork.dot(learningRateMTRX1, cost_out);					
						//w3 = agent.targetNetwork.subtract(pn_weights3, update_w3);
						//agent.targetNetwork.setWeights(w3, 3);
						
						//update_w2 = agent.targetNetwork.dot(learningRateMTRX2, cost_hidden2);
						//w2 = agent.targetNetwork.subtract(pn_weights2, update_w2);
						//agent.targetNetwork.setWeights(w2, 2);
						
						//update_w1 = agent.targetNetwork.dot(learningRateMTRX3, cost_hidden1);
						//w1 = agent.targetNetwork.subtract(pn_weights1, update_w1);
						//agent.targetNetwork.setWeights(w1, 1);
						
						agent.targetNetwork.bias_1 = agent.policyNetwork.bias_1;
						agent.targetNetwork.bias_2 = agent.policyNetwork.bias_2;
						agent.targetNetwork.bias_3 = agent.policyNetwork.bias_3;
						
						//System.out.println("Target Weights Check :: " + Arrays.deepToString(agent.targetNetwork.bias_1));
						//System.out.println("Policy Weights Check :: " + Arrays.deepToString(agent.policyNetwork.bias_1));
						
						agent.targetNetwork.hidden1_weights = agent.policyNetwork.hidden1_weights;
						agent.targetNetwork.hidden2_weights = agent.policyNetwork.hidden2_weights;
						agent.targetNetwork.output_weights = agent.policyNetwork.output_weights;
						
						//agent.targetNetwork.setWeights(agent.policyNetwork.getWeights(3), 3);
						//agent.targetNetwork.setWeights(agent.policyNetwork.getWeights(2), 2);
						//agent.targetNetwork.setWeights(agent.policyNetwork.getWeights(1), 1);
						
						//agent.targetNetwork.setBias(agent.policyNetwork.getBias(3), 3);
						//agent.targetNetwork.setBias(agent.policyNetwork.getBias(2), 2);
						//agent.targetNetwork.setBias(agent.policyNetwork.getBias(1), 1);
						agent.counter = 0;
						
						
					}
					
				}
				
			}
		}
				
		
	
	public static int getMaxValueIndexSelection(double[][] numbers) {
        double maxValue = numbers[0][0];
        //System.out.println("Min Allowed: " +  minAllowed);
        //System.out.println("Max Allowed: " +  maxAllowed);
        int maxValueIndex = 0;
        for (int j = 0; j < numbers.length; j++) {            
            	
            if (numbers[0][j] > maxValue) {
                maxValue = numbers[0][j];
                maxValueIndex = j;               
            	
            }
        }
        return maxValueIndex;        
	}	
	
	public void doEpisode() {
		int stepsTaken = 0;	// performance metric, see slide # 
		currentAgentCoords[0] = agentStartXY[0]; // reset agent position
		currentAgentCoords[1] = agentStartXY[1];
		goalReached = false;
		
		for(int t=0; t<maxTimesteps;t++) {
			if(!goalReached) {
				if(debug) {
					System.out.println("\nEnvironment: *************** Timestep " + t + " starting ***************");
				}
				doTimestep();
				stepsTaken++;
			}
			else {
				break;
			}
		}
		
		// wrap up episode
		decayAlpha();
		decayEpsilon();
		movesToGoal.add(stepsTaken);
	}
	
	public void doTimestep() {
		int currentStateNo = Utilities.getStateNoFromXY(currentAgentCoords, new int[] {xDimension,yDimension});
		//int selectedAction = agent.selectAction(currentStateNo);
		
		//System.out.println((double) currentStateNo/100);
		double[][] myDoubleArray =  new double[1][1];
		myDoubleArray[0][0] = (double) currentStateNo/1000;
		
		double[][] stateMatrix = new double[1][1];
		stateMatrix[0][0] = (double) currentStateNo/1000;
		//DataSet state = currentStateNo/100;
		// FeedForward
		//agent.policyNetwork.setState(currentState);
		
		double[][] weights1 = agent.policyNetwork.getWeights(1);
		double[][] bias1 = agent.policyNetwork.getBias(1);
		
		double[][] hidden1 = agent.policyNetwork.feedForwardStep(bias1,weights1, stateMatrix);
		double[][] z_hidden1 = agent.policyNetwork.RELU(hidden1);
							
		double[][] weights2 = agent.policyNetwork.getWeights(2);
		double[][] bias2 = agent.policyNetwork.getBias(2);					
		double[][] hidden2 = agent.policyNetwork.feedForwardStep(bias2,weights2, z_hidden1);
		double[][] z_hidden2 = agent.policyNetwork.RELU(hidden2);
		
		double[][] weights3 = agent.policyNetwork.getWeights(3);
		double[][] bias3 = agent.policyNetwork.getBias(3);					
		double[][] output = agent.policyNetwork.feedForwardStep(bias3,weights3, z_hidden2);
		double[][] z_output = agent.policyNetwork.sigmoid(output);				
		
		double pn_maxQTarget = getMaxValue(output);	
		int pn_maxQTargetIndex = getMaxValueIndex(output);
		
		//agent.expierienceReplay.addExpierience(action, state, reward, nextState);
		//System.out.println(Arrays.deepToString(output));
		double randomValue = Math.random();
		int selectedAction = getMaxValueIndexSelection(output);

		double maxValue = 0;
		int maxIndex = 0;
		
		if (Math.random() < epsilon)
		{
			selectedAction = maxIndex;
		}
		else
			
		{	Random r = new Random();
			selectedAction = r.nextInt((3 - 0) + 1) + 0;			
		}
		previousAgentCoords = currentAgentCoords;
		currentAgentCoords = getNextStateXY(previousAgentCoords, selectedAction);		
		float reward = calculateReward(previousAgentCoords, selectedAction, currentAgentCoords);
		int nextStateNo = Utilities.getStateNoFromXY(currentAgentCoords, new int[] {xDimension,yDimension});
		
		System.out.println("Reward :: " + reward);
		System.out.println("Action :: " + selectedAction);
		
		agent.expierienceReplay.addExpierience(selectedAction, (double) currentStateNo/1000, reward, (double) nextStateNo/1000);
		
		
		
		
		
		agent.updateQValue(currentStateNo, selectedAction, nextStateNo, reward);	
		if(debug) {
			System.out.println("Environment: previousState [" + previousAgentCoords[0] + "," + previousAgentCoords[1] + "]; selected move " + actionLabels[selectedAction] + "; currentState [" + currentAgentCoords[0] + "," + currentAgentCoords[1] + "];");
		}
	}
	
	// Implementation of the Environment Reward Function (see slides #13, #21)
	public float calculateReward(int[] previousAgentCoords, int selectedAction, int[] currentAgentCoords) {
		float reward = 0.0f;
		// check if the goal state has been reached
		if(currentAgentCoords[0] == goalLocationXY[0] && currentAgentCoords[1] == goalLocationXY[1]) {
			reward = goalReward;
			goalReached = true;
		}
		else {
			reward = stepPenalty;
		}
		return reward;
	}
	
	// Models environment transitions (i.e. returns the next state s', given the current state and selected action (see slide #19)
	public int[] getNextStateXY(int[] currentStateXY, int action) {
		// work out the agent's next position, x=0 y=0 is at the bottom left corner of the grid
		// actions which would move the agent off the grid will leave its position unchanged		
		int[] nextStateXY = new int[] {-1,-1};
		
		if(action == 0) { // move north
			if(currentStateXY[1] < yDimension-1) { // ensure agent is not at northmost row
				nextStateXY = new int[] {currentStateXY[0],currentStateXY[1]+1};
			}
			else { // keep agent at current position if this action would move it off the grid
				nextStateXY = new int[] {currentStateXY[0],currentStateXY[1]};
			}
		}
		else if(action == 1) { // move east
			if(currentStateXY[0] < xDimension-1) { // ensure agent is not at eastmost column
				nextStateXY = new int[] {currentStateXY[0]+1,currentStateXY[1]};
			}
			else { // keep agent at current position if this action would move it off the grid
				nextStateXY = new int[] {currentStateXY[0],currentStateXY[1]};
			}
		}
		else if(action == 2) { // move south
			if(currentStateXY[1] > 0) {  // ensure agent is not at southmost row
				nextStateXY = new int[] {currentStateXY[0],currentStateXY[1]-1};
			}
			else { // keep agent at current position if this action would move it off the grid
				nextStateXY = new int[] {currentStateXY[0],currentStateXY[1]};
			}
		}
		else if(action == 3) { // move west
			if(currentStateXY[0] > 0) { // ensure agent is not at westmost column
				nextStateXY = new int[] {currentStateXY[0]-1,currentStateXY[1]};
			}
			else { // keep agent at current position if this action would move it off the grid
				nextStateXY = new int[] {currentStateXY[0],currentStateXY[1]};
			}
		}
				
		return nextStateXY;
	}
		
	public int getNumStates() {
		return xDimension * yDimension;
	}
	
	public void enableDebugging() {
		this.debug = true;
	}
	
	public void disableDebugging() {
		this.debug = false;
		agent.disableDebugging();
	}
	
	public void decayAlpha() {
		if(alphaDecays) {
			alpha = alpha*alphaDecayRate;
			agent.setAlpha(alpha);
		}
	}
	
	public void decayEpsilon() {
		if(epsilonDecays) {
			epsilon = epsilon*epsilonDecayRate;
			agent.setEpsilon(epsilon);
		}
	}
	
	public ArrayList<Integer> getMovesToGoal() {
		return movesToGoal;
	}
	
	public float [][] getQTable() {
		return agent.copyQTable();
	}
	
	public static int getXDimension() {
		return xDimension;
	}
	
	public static int getYDimension() {
		return yDimension;
	}
}
