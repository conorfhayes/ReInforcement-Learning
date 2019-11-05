import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;


public class TLO_Agent {
	
			
	public Double U1[] = {150.0, 470.0, 786.7988, 38.5397, 0.1524, 450.0, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80.0, 80.0};
	public Double U2[] = {135.0, 470.0, 451.3251, 46.1591, 0.1058, 600.0, 0.036, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80.0, 80.0};
	public Double U3[] = {73.0, 340.0, 1049.9977, 40.3965, 0.0280, 320.0, 0.028, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 80.0, 80.0};
	public Double U4[] = {60.0, 300.0, 1243.5311, 38.3055, 0.0354, 260.0, 0.052, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 50.0, 50.0};
	public Double U5[] = {73.0, 243.0, 1658.5696, 36.3278, 0.0211, 280.0, 0.063, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50.0, 50.0};
	public Double U6[] = {57.0, 160.0, 1356.6592, 38.2704, 0.0179, 310.0, 0.048, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50.0, 50.0};
	public Double U7[] = {20.0, 130.0, 1450.7045, 36.5104, 0.0121, 300.0, 0.086, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30.0, 30.0};
	public Double U8[] = {47.0, 120.0, 1450.7045, 36.5104, 0.0121, 340.0, 0.082, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30.0, 30.0};
	public Double U9[] = {20.0, 80.0, 1455.6056, 39.5804, 0.1090, 270.0, 0.098, 350.0056, -3.9524, 0.0465, 0.5475, 0.0234, 30.0, 30.0};
	public Double U10[] = {10.0, 55.0, 1469.4026, 40.5407, 0.1295, 380.0, 0.094, 360.0012, -3.9864, 0.0470, 0.5475, 0.0234, 30.0, 30.0};
	
	public ArrayList<ArrayList<Double> > UHolder = new ArrayList<ArrayList<Double> >();
	ArrayList<Integer> deltaDemands = new ArrayList<Integer>();
	
	
	public double PDM_hold[] = {1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
            						   1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184};
	
	public ArrayList<Double> minPower = new ArrayList<Double>();
	public int id;
	public int numStates;
	public int numActions = 101;
	public int maxNumActions;
	
	public float[][][] qTable;
	public int[] selectedActions;
	public int[] previousActions;
	
	public int[] genRanges = new int[10];
	public int[] genOffsets = new int[10];
	
	public int minDeltaDemand = Integer.MAX_VALUE;
	public int maxDeltaDemand = Integer.MIN_VALUE;
	public int demandRange = 0;
	public int demandOffset = 0;
	public int powerGranularityMARL = 1;
	public int percentageIncrement = 1;
	//public int[] powerArray;
	public int currentState;
	public int action;
	public float maxQ;
	public int[] stateVector;
	public int[] P1M_array_D;
	public double power = 0;
	public ArrayList<Double> PNM;
	public double previousAgentCost = 0;
	public double previousAgentEmissions = 0;
	public double previousAgentPower = 0;
	public double P1M_Minus_D = 0;
	public double P1M_Minus = 0;
	public ArrayList<Double> action_holder = new ArrayList<Double>();
	public ArrayList<Double> powerArray = new ArrayList<Double>();
	public ArrayList<Integer> action_ = new ArrayList<Integer>();
	//public TLO_QHashMapStorage qValuesHashMap;
	public int numPercentSteps = 100 / percentageIncrement + 1;
	public double epsilon = 0.0;
	public double gamma = 1;
	public double alpha = 0.1;
	public int numObjectives = 3;
	
	TLO_Environment newEnvironment = new TLO_Environment();
	
	
	public TLO_Agent(double alpha, double gamma, double epsilon, int id)
	
	{
		ArrayList<ArrayList<Double>> UHolder_ = new ArrayList<ArrayList<Double>>();
		this.id = id;
		//this.alpha = alpha;
        //this.gamma = gamma;
        //this.epsilon = epsilon; 
        setU(UHolder_);
        calculateNumStatesActionsMARL(UHolder_);
        initialiseQvalues((this.demandRange*numPercentSteps)+1000, numPercentSteps + 1, this.numObjectives);
        //System.out.println("Gen Ranges: " + this.genRanges[id - 2]);
        //qValuesHashMap = new TLO_QHashMapStorage(this.demandRange*numPercentSteps, numPercentSteps+1, this.numObjectives);
        maxNumActions = 300;
        		
	}
	
	public void setU(ArrayList<ArrayList<Double>> x)
	{
		x.add(new ArrayList<Double>(Arrays.asList(this.U2)));	
		x.add(new ArrayList<Double>(Arrays.asList(this.U3)));	
		x.add(new ArrayList<Double>(Arrays.asList(this.U4)));	
		x.add(new ArrayList<Double>(Arrays.asList(this.U5)));	
		x.add(new ArrayList<Double>(Arrays.asList(this.U6)));
		x.add(new ArrayList<Double>(Arrays.asList(this.U7)));
		x.add(new ArrayList<Double>(Arrays.asList(this.U8)));
		x.add(new ArrayList<Double>(Arrays.asList(this.U9)));
		x.add(new ArrayList<Double>(Arrays.asList(this.U10)));	
		
		this.UHolder = x;
	}
	
	public double getAgentPower()
	{
		return this.power;
	}
	
	public void setAgentPower(double power)
	{
		this.power = power;
	}
	private float[][][] initialiseQvalues(int numStates, int numActions, int numObjectives) 
	{	//System.out.println("Q Table numState" + numStates);
		qTable = new float[numStates][numActions][numObjectives];

		for(int i=0;i<numStates;i++) 
		{
			for(int j=0; j<numActions;j++) 
			{
				for (int k = 0; k < numObjectives; k++)
				{
					qTable[i][j][k] = 0;
				}
			}
		}
		return qTable;
	}
	
	public int getMaxValuedAction(int state, int objective) 
	{
		ArrayList<Double> action_tracker = new ArrayList<Double>();
		int action = 0;
		int maxActionIndex = -1;
		
        while (action < this.numActions)
        {
        	//System.out.println(action);
            double valueQ = qTable[state][action][objective];
            action_tracker.add(valueQ);
            
            action = action + 1;
        }       
        
    	maxActionIndex = action_tracker.indexOf(Collections.max(action_tracker));       
        return maxActionIndex;        
	}
	
	public int getStateMARL(int hour, TLO_Agent agent, double power_) {
		int state = 0;
		if (hour > 0 && hour < 24)
		{
		//System.out.println(hour - 1);
		int deltaDemand = deltaDemands.get(hour) + demandOffset; // add demand offset so state no is always >= 0
		int power = (int) (power_ - genOffsets[agent.getAgentID() - 1])*numPercentSteps / (percentageIncrement * (genRanges[agent.getAgentID() - 1] - 1));
		//System.out.println("State Power: " + power);
		state = getStateNoFromXY(new int[]{deltaDemand,power}, new int[]{demandRange,numPercentSteps+1});
		}
		//action = (currentPositions.get(agent).intValue()-genOffsets_MARL[agent+1])*numPercentSteps/(percentageIncrement*(genRanges_MARL[agent+1]-1));
		//System.out.println("Get State MARL: " + state);
	

		return state;
	}	

	
	public static int getStateNoFromXY(int[] state, int[] basesForStateNo) {
		int numStates = basesForStateNo[0] * basesForStateNo[1];
		//System.out.println("0: " + basesForStateNo[0]);
		//System.out.println("1 " + basesForStateNo[1]);
		int stateNo = 0;
		for (int i = 0;i < state.length; i++) {
			stateNo = stateNo * basesForStateNo[i] + state[i];
		}
		//check state is allowed
		//System.out.println("State Number " + stateNo);
		
		return stateNo;
	}
	
	
	
	public void calculateNumStatesActionsMARL(ArrayList<ArrayList<Double>> UHolder ) { 
		// calculate changes in demand
		deltaDemands.add(0); // no change in demand for first hour


		for (int i = 1; i < PDM_hold.length; i++) 
		{
			deltaDemands.add((int) (PDM_hold[i]  - PDM_hold[i - 1]));
			
		}

		for (int i = 1; i < deltaDemands.size(); i++) 
		{
			if (deltaDemands.get(i) < minDeltaDemand) 
			{
				minDeltaDemand = deltaDemands.get(i);
			}
			
			if (deltaDemands.get(i) > maxDeltaDemand) 
			{
				maxDeltaDemand = deltaDemands.get(i);
			}
		}	


		demandRange = 1 + maxDeltaDemand + Math.abs(minDeltaDemand);
		demandOffset = Math.abs(minDeltaDemand);
		
		//System.out.println("Power Granduality:" + powerGranularityMARL);

		genRanges[0] = (this.U1[1].intValue() - this.U1[0].intValue()) / powerGranularityMARL + 1;
		genOffsets[0] = this.U1[0].intValue();
		
		for (int i = 0 ;i < UHolder.size(); i++) 
		{
			//System.out.println("Generator Max Power: " + UHolder.get(i).get(1).intValue());
			genRanges[i+1] = (UHolder.get(i).get(1).intValue() - UHolder.get(i).get(0).intValue()) / powerGranularityMARL + 1;
			genOffsets[i+1] = UHolder.get(i).get(0).intValue();
		}
	}
	
	public double getMaxQValue(int state, int objective) 
	{
		int maxIndex = getMaxValuedAction(state, objective);
		return qTable[state][maxIndex][objective];
	}
	
	public ArrayList<Double> getMinPower() 
	{	this.minPower.add(135.0);this.minPower.add(73.0);this.minPower.add(60.0);this.minPower.add(73.0);
		this.minPower.add(157.0);this.minPower.add(20.0);this.minPower.add(47.0);this.minPower.add(20.0);
		this.minPower.add(10.0);
		return this.minPower;
	}
	
	public void updateQTable(int previousState, int selectedAction, int currentState, double reward, TLO_Agent agent, int objective) 
	{
		double oldQ = qTable[previousState][selectedAction][objective];
		//System.out.println("Old Q: " + oldQ);
		double maxQ = getMaxQValue(currentState, objective);
		double newQ = oldQ + alpha * (reward + gamma * maxQ - oldQ);
		qTable[previousState][selectedAction][objective] = (float) newQ;
	}
	
	public void setQtable(float[][][] values, int o) {
		for(int s=0;s<numStates;s++) {
			for(int a=0;a<numActions;a++) {
				qTable[s][a][o] = values[s][a][o]; 
			}
		}
	}
	
	public void saveCurrentState(int currentState) 
	{
        this.currentState = currentState;      		
	}
	
	public double getPreviousAgentCost()
	{
		return this.previousAgentCost;
	}
	
	public void setPreviousAgentCost(double cost)
	{
		this.previousAgentCost = cost;
	}
	
	public double getPreviousAgentPower()
	{
		return this.previousAgentPower;
	}
	
	public void setPreviousAgentPower(double power)
	{
		this.previousAgentPower = power;
	}
	
	public double getPreviousAgentEmissions()
	{
		return this.previousAgentEmissions;
	}
	
	public void setPreviousAgentEmissions(double emissions)
	{
		this.previousAgentEmissions = emissions;
	}
	
	public void setP1M_MinusD(double P1M_MinusD)
	{
		this.P1M_Minus_D = P1M_MinusD;
	}
	
	public double getP1M_Minus()
	{
		return this.P1M_Minus;
	}
	
	public void setP1M_Minus(double P1M_Minus)
	{
		this.P1M_Minus = P1M_Minus;
	}
	
	public double getP1M_MinusD()
	{
		return this.P1M_Minus_D;
	}
        		

    public int getState()
    {
        return this.currentState;
    }
	
	/* public int selectAction(int hour, int state, TLO_Agent agent, double[] thresHolds) 
	{
		int selectedAction = -1;
		double randomValue = Math.random();
		
		if(randomValue < epsilon) 
		{
			selectedAction = selectRandomAction();		
			//System.out.println("Random Action: " + selectedAction);
		}
		else 
		{
			selectedAction = getSelectedAction(hour, state, agent, thresHolds);		
			//System.out.println("Selected Action :" + selectedAction);
		}
		
		return selectedAction;
	}
	*/
	
	public int selectRandomAction() 
	{
		return (int) (Math.random() * numActions);
	}
	
	public void savePnm(ArrayList<Double> PNM)
	{		
		this.PNM = PNM;
	}
	
	public double getAlpha() {
		return this.alpha;
	}
	
	public int getAgentID() {
		return this.id;
	}
	
	public void setAlpha(float alpha) {
		this.alpha = alpha;
	}
	
	public double getGamma() {
		return this.alpha;
	}
	
	public void setGamma(float gamma) {
		this.gamma = gamma;
	}
	
	public double getEpsilon() {
		return this.epsilon;
	}
	
	public void setEpsilon(float epsilon) {
		this.epsilon = epsilon;
	}
	
	public void saveAction(int action) {
		
		this.action = action;
	}
	
	public int getAction() {
		return this.action;
	}
	
	public void saveMaxQ(float maxQ) {
		this.maxQ = maxQ;
	}
	
	public float getMaxQ() {
		return this.maxQ;
	}
	
	public ArrayList<ArrayList<Integer>> create_bins(int lower_bound, int width, int quantity) 
	{
        ArrayList<ArrayList<Integer>> bins = new ArrayList<ArrayList<Integer>>();
        
        int j;
        
        int a = 0;
        int b = 0;
        

        for(j = lower_bound; j < (lower_bound + quantity * width + 1); j = j + width)
        {
        	bins.add(new ArrayList<Integer>());
        	bins.get(b).add(j);
        	bins.get(b).add(j + width);

        	a = a + 1;
        	b = b + 1;        	
        }
        
        return bins;         		
	}
	
	public int find_bin(double value, ArrayList<ArrayList<Integer>> bins) 
	{	
        for (int i = 0; i < bins.size(); i ++ )
        {
        	if (bins.get(i).get(0) <= value && value < bins.get(i).get(1) )
        	{        		
        		return i;    			
        	}            
        }        
        return -1;
        		
	}
	
	public int getNextState(double PDM_delta, double power_, TLO_Agent agent)
	{
		int[] base = {10,10};
		double power = power_;
		double PDM;
		double PDM_;
		double PDM_rescale;
		double power_rescale = 0;
		
		
		
		PDM_rescale = (PDM_delta - (-296)) * (492 - 0) / (196 - (-296)) + 0;
		ArrayList<ArrayList<Integer>> bins_PDM = create_bins(0,20,25);
		
		int id = agent.getAgentID() - 2;
		power_rescale = ((power - UHolder.get(id).get(0)) * (100 - 0)) / (UHolder.get(id).get(1) - UHolder.get(id).get(0)) + 0;		
		
		ArrayList<ArrayList<Integer>> bins_power = create_bins(0,5,20);		
		int bin_index1 = find_bin(PDM_rescale, bins_PDM);
		int bin_index2 = find_bin(power_rescale, bins_power);
		int[] currentState = {bin_index1, bin_index2};
		//int[] currentState = {(int) PDM_rescale, (int) power_rescale};
		
		
		int res = 0;
		int i = 0;
		
		while (i < 2) 
		{
			res = res * base[i] + currentState[i];
			i = i + 1;
		}
		
		int state = res;
		return state;		
	}
	
	
	
	
	
	public void updateQValuesDEED(int previousState, int currentState, int selectedAction, double cost, double emissions
			, double violations) {
		//_agent.updateQValuesDEED(previousState, currentState, action, cost, emissions, violations, (float) reward);
		//float oldQ = getQValue(previousState,selectedAction); 
		double reward = 0;
		double maxQ = 0;
		double newQ = 0;
		for (int i = 0; i < numObjectives; i++)
		{
			if (i == 0)
			{
				reward = violations;
			}
			if (i == 1)
			{
				reward = cost;
			}
			if (i == 2)
			{
				reward = emissions;
			}
			
			float oldQ = qTable[previousState][selectedAction][i];
			maxQ = getMaxQValue(currentState,i);
			//maxQ = 0;
			newQ = oldQ + this.alpha * (reward + this.gamma * maxQ - oldQ);
			qTable[previousState][selectedAction][i] = (float) newQ;
			
		}
		
		
			
		//System.out.println("Old Q: " + oldQ);
		// determine max Q value for current state
		
	}	

	public int selectActionDEED(int stateNo, int minAllowedAction, int maxAllowedAction, double[] thresHolds,
			String rewardType) {
		//epsilon greedy implementation modified for DEED domain
		int selectedAction = -1;
		//System.out.println("Min Allowed Action" + minAllowedAction);
		//System.out.println("Max Allowed Action" + maxAllowedAction);
		boolean random = false;
		float maxQ = -Float.MAX_VALUE;
		int[] doubleValues = new int[maxNumActions + 2];
		int maxDV = 0;
		Double valueQ1 = 0.0;
		Double valueQ2 = 0.0;
		Double valueQ3 = 0.0;
		Integer bestAction_ = 0;
		ArrayList<Integer> action_holder = new ArrayList<Integer>();
		ArrayList<ArrayList<Double>> qVector_holder = new ArrayList<ArrayList<Double>>();
		
		ArrayList<Integer> action_holder_ = new ArrayList<Integer>();
		
		for (int action = minAllowedAction ; action < maxAllowedAction ; action++)
		{
			action_holder_.add(action);
		}
		
		if ( Math.random() < epsilon ) {		
			int actionIndex = (int) (Math.random() * action_holder_.size());
			//return actionIndex;
			return action_holder_.get(actionIndex);
		}
		else 
		{
			for( int action = minAllowedAction ; action < maxAllowedAction ; action ++ ) 
			{
				valueQ1 = (double) qTable[stateNo][action][0];
				valueQ2 = (double) qTable[stateNo][action][1];
				valueQ3 = (double) qTable[stateNo][action][2];
				Double qVector[] = {valueQ1, valueQ2, valueQ3};
				
				qVector_holder.add(new ArrayList<Double>(Arrays.asList(qVector)));
				action_holder.add(action);				
			}
			
			ArrayList<Integer> bestAction = new ArrayList<Integer>();
			ArrayList<Integer> bestActionCheck = new ArrayList<Integer>();
			
			bestAction.add(action_holder.get(0));
			bestActionCheck.add(0);
			
			ArrayList<ArrayList<Double>> actionVector = new ArrayList<ArrayList<Double>>();
			ArrayList<ArrayList<Double>> B = new ArrayList<ArrayList<Double>>();
			B.add(qVector_holder.get(0));
			
			
			int a = 1;
			int compareResult = 0;
			
			while (a < qVector_holder.size()) 
			{
				compareResult = compare(qVector_holder.get(a), B.get(0), thresHolds);
				
				if (compareResult > 0)
				{
					actionVector.clear();
					bestAction.clear();
					bestActionCheck.clear();
					B.clear();
					
					B.add(qVector_holder.get(a));
					bestAction.add(action_holder.get(a));
					bestActionCheck.add(a);		
					actionVector.add(qVector_holder.get(a));
					
					//bestAction_ = bestAction.get(0);
				}
				else if (compareResult == 0)
				{
					bestAction.add(action_holder.get(a));
					bestActionCheck.add(a);						
				}
				
				a = a + 1;
			}
			
			if (bestAction.size() > 1)
			{				
				Random r = new Random();
				int check = r.nextInt(bestAction.size());
				actionVector.clear();
				
				bestAction_ = bestAction.get(check);
				actionVector.add(qVector_holder.get(action_holder.indexOf(bestAction_)));			
			}
			/* if (bestAction.size() > 1)
			{
				double qValue1 = 0;
				double qValue2 = 0;
				double qValue3 = 0;
				double reward = 0;
				
				ArrayList<Double> tieBreaker = new ArrayList<Double>();

				for (int x = 0; x < bestAction.size(); x++)
				{	
					
					qValue1 = qVector_holder.get(x).get(0);
					qValue2 = qVector_holder.get(x).get(1);
					qValue3 = qVector_holder.get(x).get(2);
					reward = (qValue1 * 0.5) + (qValue2 * 0.225) + (qValue3 * 0.275);
					//reward = qValue1 * qValue2 * qValue3;
					
					tieBreaker.add(reward);
					int maxIndex = tieBreaker.indexOf(Collections.max(tieBreaker));
					bestAction_ = bestAction.get(maxIndex);
					actionVector.add(qVector_holder.get(action_holder.indexOf(bestAction_)));						
				}
				
			} */
			else
			{
				bestAction_ = bestAction.get(0);
				actionVector.clear();
				
				actionVector.add(qVector_holder.get(action_holder.indexOf(bestAction_)));
			}
		
			
			for (int x = 0; x < thresHolds.length; x ++)
			{				
								
				thresHolds[x] = thresHolds[x] - actionVector.get(0).get(x);			
				
			}
		}		
		
		//System.out.println(bestAction_);
		return bestAction_;
	}
		
	public static int compare(ArrayList<Double> a, ArrayList<Double> b, double[] thresHolds )
	{
		double thresHoldA, thresHoldB;
		for (int i = 0; i < thresHolds.length; i ++)
		{
			thresHoldA = Math.min(a.get(i), thresHolds[i]);
			thresHoldB = Math.min(b.get(i), thresHolds[i]);
			
			if (thresHoldA > thresHoldB)
			{
				return 1;
			}
			else if (thresHoldA < thresHoldB)
			{
				return -1;
			}
		}
			
		for (int j = 0; j < thresHolds.length; j ++)
		{
			if (a.get(j) > b.get(j))
			{
				return 1;
			}
			else if (a.get(j) < b.get(j))
			{
				return -1;
			}
		}
		
	return 0;
	
	}
		//return 0;
	
	
	

	/*public float getMaxQ(final int stateNo)	{		
		int winner = 0;
		boolean foundNewWinner = false;
		boolean done = false;

		while(!done)
		{
			foundNewWinner = false;
			for(int i = 0; i < numActions; i++)
			{
				if(i != winner){             // Avoid self-comparison.
					if(getQValue(stateNo,i) > getQValue(stateNo,winner)){
						winner = i;
						foundNewWinner = true;
					}
				}
			}
			if(foundNewWinner == false){
				done = true;
			}
		}
		return getQValue(stateNo,winner);
	}

	public int getNumStates() {
		return numStates;
	} */
	


}
