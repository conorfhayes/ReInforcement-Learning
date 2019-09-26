import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.*;
import java.lang.Math;

public class Environment {
	
	public double[][] B = {
						 {0.000049, 0.000015, 0.000015, 0.000015, 0.000016, 0.000017, 0.000017, 0.000018, 0.000019, 0.000020},
				         {0.000014, 0.000045, 0.000016, 0.000016, 0.000017, 0.000015, 0.000015, 0.000016, 0.000018, 0.000018},
				         {0.000015, 0.000016, 0.000039, 0.000010, 0.000012, 0.000012, 0.000014, 0.000014, 0.000016, 0.000016},
				         {0.000015, 0.000016, 0.000010, 0.000040, 0.000014, 0.000010, 0.000011, 0.000012, 0.000014, 0.000015},
				         {0.000016, 0.000017, 0.000012, 0.000014, 0.000035, 0.000011, 0.000013, 0.000013, 0.000015, 0.000016},
				         {0.000017, 0.000015, 0.000012, 0.000010, 0.000011, 0.000036, 0.000012, 0.000012, 0.000014, 0.000015},
				         {0.000017, 0.000015, 0.000014, 0.000011, 0.000013, 0.000012, 0.000038, 0.000016, 0.000016, 0.000018},
				         {0.000018, 0.000016, 0.000014, 0.000012, 0.000013, 0.000012, 0.000016, 0.000040, 0.000015, 0.000016},
				         {0.000019, 0.000018, 0.000016, 0.000014, 0.000015, 0.000014, 0.000016, 0.000015, 0.000042, 0.000019},
				         {0.000020, 0.000018, 0.000016, 0.000014, 0.000016, 0.000015, 0.000018, 0.000016, 0.000019, 0.000044}
				         };

	
	public Double U1[] = {150.0, 470.0, 786.7988, 38.5397, 0.1524, 450.0, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80.0, 80.0};
	
	public Double initialPNM[] = {135.0, 73.0, 60.0, 73.0, 57.0, 20.0, 47.0, 20.0, 10.0};
	public ArrayList<ArrayList<Double> > UHolder = new ArrayList<ArrayList<Double> >();
	
	
	public double PDM_hold[] = {1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
            						   1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184};
	
	public ArrayList<Agent> _agents_;
	public int numActions = 101;
	public double epsilon = 0.05;
	public double gamma = 0.75;
	public double alpha = 0.1;	
	public ArrayList<Double> P1M_array = new ArrayList<Double>();
	public ArrayList<Double> P1M_T_array = new ArrayList<Double>();
	public int[] PNM;
	public ArrayList<Double> previousPNM = new ArrayList<Double>(Arrays.asList(initialPNM));
	public String scalarization;
	public double P1M_minus;
	public int percentageIncrement = 1; // 1% gen power increment
	public int numPercentSteps = 100/percentageIncrement + 1;
	
	
	public Agent createAgent(int id) 
	{
		Agent agent_ = new Agent(alpha, gamma, epsilon, id);
		return agent_;
	}	
	
	public ArrayList<ArrayList<Double>> getU()
	{		
		return this.UHolder;		
	}
	
	public double getP1M_minus()
	{
		return this.P1M_minus;
	}
	
	public void setP1M_minus(double P1M)
	{
		this.P1M_minus = P1M;
	}
	
	public double[] getPowerDemand(int hour)
	{	
		double[] powerResults = {0,0,0};
		double PDM_;
		double PDM;
		double PDM_delta;
		if (hour - 1 == 0)
		{
			PDM_ = 0;
		}
		else
		{
			PDM_ = this.PDM_hold[hour - 2];
		}
		
		PDM = this.PDM_hold[hour - 1];
		PDM_delta = PDM - PDM_;
		powerResults[0] = PDM; powerResults[1] = PDM_; powerResults[2] = PDM_delta;
		return powerResults;
	}
	
	public ArrayList<Double> getPreviousPNM()
	{
		return this.previousPNM;
	}
	
	public void setPreviousPNM(ArrayList<Double> previousPNM)
	{
		this.previousPNM = previousPNM;
	}
	
	public double getCost(ArrayList<Double> PNM_, int agentID, Agent agent)
	{	
		double cost = 0;
		int id = agentID;
		ArrayList<ArrayList<Double>> UHolder = agent.UHolder;
		//System.out.println(UHolder);
		
		cost = UHolder.get(id).get(2) + (UHolder.get(id).get(3) * (PNM_.get(id))) +  
			   UHolder.get(id).get(4) * (Math.pow(PNM_.get(id), 2)) + Math.abs(UHolder.get(id).get(5) * 
			   Math.sin(UHolder.get(id).get(6) * (UHolder.get(id).get(0) - PNM_.get(id))));
		
		return cost;
	}
	
	public double getEmissions(ArrayList<Double> PNM_, int agentID, Agent agent)
	{	
		double emissions = 0;
		int id = agentID;
		ArrayList<ArrayList<Double>> UHolder = agent.UHolder;
		
		emissions = UHolder.get(id).get(7) + (UHolder.get(id).get(8) * PNM_.get(id)) + (
			    UHolder.get(id).get(9) * Math.pow(PNM_.get(id),2)) + (
				UHolder.get(id).get(10) * Math.exp(UHolder.get(id).get(11) * PNM_.get(id)));
		
		return emissions;
	}
	
	public double getP1MCost(double P1M_)
	{	
		double cost = 0;
		
		cost = this.U1[2] + (this.U1[3] * (P1M_)) + (this.U1[4] * Math.pow(P1M_,2)) + 
				Math.abs(this.U1[5] * Math.sin(this.U1[6] * (this.U1[0] - P1M_)));
		
		//150.0, 470.0, 786.7988, 38.5397, 0.1524, 450.0, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80.0, 80.0
		
		return cost;
	}
	
	public double getP1MEmissions(double P1M_)
	{	
		double emissions = 0;
		//150.0, 470.0, 786.7988, 38.5397, 0.1524, 450.0, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80.0, 80.0
		
		emissions = this.U1[7] + (this.U1[8] * P1M_) + (this.U1[9] * Math.pow(P1M_,2)) + (
				this.U1[10] * (Math.exp(this.U1[11] * P1M_)));
		
		return emissions;
	}
	
	
	
	public double getP1M(ArrayList<Double> PNM, double currentPDM)
	{
		ArrayList<Double> B_array = new ArrayList<Double>();
		ArrayList<Double> C_array = new ArrayList<Double>();
		double holder;
		double cholder;
		
		double sum1 = 0;
		double sum2 = 0;
		double sum3 = 0;
		
		int n = 0;
		
		while(n <= 8)
		{
			sum1 = sum1 + (this.B[0][n + 1] * PNM.get(n));
			n = n + 1;
		}
		
		//double b = (2 * (B_array.stream().mapToDouble(x -> x).sum())) - 1;		
		int nn = 0;
		int jj = 0;	
		
		while(nn <= 8)
		{
			while(jj <= 8)
			{
				//cholder = (PNM.get(nn) * this.B[nn + 1][jj + 1] * PNM.get(nn));
				sum2 =  sum2 + (PNM.get(nn) * this.B[nn + 1][jj + 1] * PNM.get(nn));
				jj = jj + 1;
			}
			sum3 = sum3 + PNM.get(nn);
			nn = nn + 1;
		}
		
		double sumPNM = PNM.stream().mapToDouble(xx -> xx).sum();
		double sumC_array = C_array.stream().mapToDouble(y -> y).sum();
		sum2 = sum2 + currentPDM - sum3;
		//double d = (Math.pow(b, 2)) - (4 * a * c);
		//double totalP1MMinus = ((-b - Math.sqrt(d))/(2 * a));
		//double totalP1MPlus = ((-b + Math.sqrt(d))/(2 * a));
		double a = this.B[0][0];
		double b = sum1;
		double c = sum2;
		double root1 = (-b + Math.sqrt(Math.pow(a, 2) - 4 * a * c)) / (2 * a);
		double root2 = (-b - Math.sqrt(Math.pow(a, 2) - 4 * a * c)) / (2 * a);
		
		System.out.println(a);
		System.out.println(b);
		System.out.println(c);
		double totalP1M = Math.min(root1, root2);
		
		return totalP1M;
	}
	
	public double getPLM(ArrayList<Double> PNM, double[] PDM, double P1M)
	{
		ArrayList<Double> A_array = new ArrayList<Double>();
		ArrayList<Double> B_array = new ArrayList<Double>();
		
		int x = 0;
		int n = 2;
		int j = 2;
		
		while (x < PNM.size())
		{
			double _b_ = this.B[0][x+1] * PNM.get(x);
			B_array.add(_b_);
			x = x + 1;
		}
		
		double B_ = B_array.stream().mapToDouble(a -> a).sum();		
		while (n < PNM.size() + 2)
		{
			while(j < PNM.size() + 2)
			{
				double _a_ = (PNM.get(n-2) * this.B[n-1][j-1] * PNM.get(j-2)) + (2 * P1M * B_) + (this.B[0][0]) * (Math.pow(P1M, 2));
				A_array.add(_a_);
				j = j + 1;
			}
			
			n = n + 1;
		}
		
		double PLM = A_array.stream().mapToDouble(yy -> yy).sum();		
		return PLM;
	}
	
	
	
	public double getPNM(int action, Agent agent)
	{
		double PNM = 0;
		
		int id = agent.getAgentID();
		int id_ = id - 2;
		ArrayList<ArrayList<Double>> UHolder = agent.UHolder;
		
		PNM = UHolder.get(id_).get(0) + (action * ((UHolder.get(id_).get(1) - UHolder.get(id_).get(0))/ 100));	
		
		return PNM;
	}
	
	public double[] calculateGlobalReward(int x, int i, ArrayList<Agent> _agents_, ArrayList<Double> PNM, 
										  double currentState, double P1M, int hour, String scalarisation, double previousPDM )
	{
		ArrayList<Double> costReward = new ArrayList<Double>();
		ArrayList<Double> emissionsReward = new ArrayList<Double>();
		ArrayList<Double> previousPNM = new ArrayList<Double>();
		double P1M_minus;
		double cost;
		double emissions;
		double reward = 0;
		double violationPenalty = 0;
		double overallCostReward = 0;
		double overallEmissionsReward = 0;
		double overallPenalty = 0;
		double[] rewardArray = {0,0,0,0};
		int E = 10;
		
		for (int j = 0; j < _agents_.size(); j++)
		{	
			Agent agent = _agents_.get(j);			
			int id = agent.getAgentID() - 2;
			
			cost = getCost(PNM, id, agent);			
			costReward.add(cost);
			
			emissions = getEmissions(PNM, id, agent);			
			emissions = E * emissions;
			emissionsReward.add(emissions);			
		}	
		
		for (int j = 0; j < _agents_.size(); j++)
			
		{
			Agent agent = _agents_.get(j);
			previousPNM.add(agent.getPreviousAgentPower());				
		}
		
		double P1M_cost;
		P1M_cost = getP1MCost(P1M);		
		costReward.add(P1M_cost);
		
		double P1M_emissions_ = getP1MEmissions(P1M);		
		double P1M_emissions = E * P1M_emissions_;
		emissionsReward.add(P1M_emissions);					
		
		int C = 1000000;
		double h1;
		double h2;

		if (P1M > 470)
		{
			h1 = P1M - 470;
		}
		else if (P1M < 150)
		{
			h1 = 150 - P1M;
		}
		else
		{
			h1 = 0;
		}
		
		P1M_minus = getP1M(previousPNM, previousPDM);
		
		if (P1M - P1M_minus > 80)
		{
			h2 = (P1M - P1M_minus) - 80;
		}
		else if ((P1M - P1M_minus) < (-80))
		{
			h2 = (P1M - P1M_minus) + 80;
		}
		else
		{
			h2 = 0;
		}
		
		if (h1 != 0 && h2 == 0)
		{
			violationPenalty = (Math.abs(h1 + 1) * this.U1[11]) * C;
		}
		else if (h1 == 0 && h2 != 0)
		{
			violationPenalty = (Math.abs(h2 + 1) * this.U1[11]) * C;
		}
		else if (h1 == 0 && h2 == 0)
		{
			violationPenalty = 0;
		}
		else if (h1 != 0 && h2 != 0)
		{
			violationPenalty = (C * (Math.abs(h1 + 1) * this.U1[11])) + (C * ((Math.abs(h2 + 1) * this.U1[11])));
		}
		
		overallCostReward = (costReward.stream().mapToDouble(a -> a).sum());
		overallEmissionsReward = (emissionsReward.stream().mapToDouble(a -> a).sum());
		overallPenalty = (violationPenalty);	
		
		if (scalarisation == "hypervolume")
		{				
			reward = -(overallCostReward + overallEmissionsReward + overallPenalty);
		}
		
		if (scalarisation == "linear")
		{				
			reward = -((overallCostReward * 0.225) + (overallEmissionsReward * 0.275) + (overallPenalty * 0.5));
		}
		
		
		rewardArray[0] = reward; rewardArray[1] = overallCostReward; rewardArray[2] = overallEmissionsReward;
		rewardArray[3] = overallPenalty;
		
		return rewardArray;
	}	
	
	public double[] calculateDifferenceReward(int x, int i, ArrayList<Agent> _agents_, Agent _agent, double currentPDM, 
			ArrayList<Double> PNM,double currentState, double P1M, int hour, String scalarisation, double previousPDM)
	{
		
		ArrayList<Double> costReward = new ArrayList<Double>();
		ArrayList<Double> costReward_D = new ArrayList<Double>();
		ArrayList<Double> emissionsReward = new ArrayList<Double>();
		ArrayList<Double> emissionsReward_D = new ArrayList<Double>();
		ArrayList<Double> previousPNM = new ArrayList<Double>();
		double P1M_minus;
		double cost;
		double emissions;
		double reward = 0;
		double violationPenalty = 0;
		double overallCostReward = 0;
		double overallEmissionsReward = 0;
		double overallPenalty = 0;
		double[] rewardArray = {0,0,0,0};
		int E = 10;
		//ArrayList<Double> previousPNM = new ArrayList<Double>();
		ArrayList<ArrayList<Double>> UHolder = _agent.UHolder;
		
		for (int j = 0; j < _agents_.size(); j++)
		{	
			Agent agent = _agents_.get(j);			
			int id = agent.getAgentID() - 2;
			
			cost = getCost(PNM, id, agent);			
			costReward.add(cost);
			
			emissions = getEmissions(PNM, id, agent);			
			emissions = E * emissions;
			emissionsReward.add(emissions);			
		}	
		
		for (int j = 0; j < _agents_.size(); j++)
			
		{
			Agent agent = _agents_.get(j);
			previousPNM.add(agent.getPreviousAgentPower());				
		}
		
		double P1M_cost;
		P1M_cost = getP1MCost(P1M);		
		costReward.add(P1M_cost);
		
		double P1M_emissions_ = getP1MEmissions(P1M);
		
		double P1M_emissions = E * P1M_emissions_;
		emissionsReward.add(P1M_emissions);
		
		
		int C = 1000000;
		double h1;
		double h2;
		
		if (P1M > 470)
		{
			h1 = P1M - 470;
		}
		else if (P1M < 150)
		{
			h1 = 150 - P1M;
		}
		else
		{
			h1 = 0;
		}
		
		P1M_minus = getP1M(previousPNM, previousPDM);
		
		
		if (P1M - P1M_minus > 80)
		{
			h2 = (P1M - P1M_minus) - 80;
		}
		else if ((P1M - P1M_minus) < (-80))
		{
			h2 = (P1M - P1M_minus) + 80;
		}
		else
		{
			h2 = 0;
		}
		
		
		if (h1 != 0 && h2 == 0)
		{
			violationPenalty = (Math.abs(h1 + 1) * this.U1[11]) * C;
		}
		else if (h1 == 0 && h2 != 0)
		{
			violationPenalty = (Math.abs(h2 + 1) * this.U1[11]) * C;
		}
		else if (h1 == 0 && h2 == 0)
		{
			violationPenalty = 0;
		}
		else if (h1 != 0 && h2 != 0)
		{
			violationPenalty = (C * (Math.abs(h1 + 1) * this.U1[11])) + (C * ((Math.abs(h2 + 1) * this.U1[11])));
		}

		double previousAgentPower = previousPNM.get(_agent.getAgentID() - 2);				
		
		overallCostReward = (costReward.stream().mapToDouble(a -> a).sum());
		overallEmissionsReward = (emissionsReward.stream().mapToDouble(a -> a).sum());
		overallPenalty = (violationPenalty);	
		
		ArrayList<Double> _PNM_ = new ArrayList<Double>(PNM);
		
		_PNM_.set(_agent.getAgentID() - 2, previousAgentPower);
	
		
		for (int j = 0; j < _agents_.size(); j++)
		{
			Agent agent = _agents_.get(j);
			double cost_D;
			double emissions_D;
			int id = agent.getAgentID() - 2;
			
			cost_D = getCost(_PNM_, id, agent);			
			costReward_D.add(cost_D);
			
			emissions_D = getEmissions(_PNM_, id, agent);			
			double emissions_D_ = E * emissions_D;
			emissionsReward_D.add(emissions_D_);	
		
		}		
		
		double P1M_D = getP1M(_PNM_, currentPDM);		
		double P1M_D_cost = getP1MCost(P1M);
		costReward_D.add(P1M_D_cost);
		
		double P1M_D_emissions_ = getP1MEmissions(P1M);		
		double P1M_D_emissions = E * P1M_D_emissions_;
		emissionsReward_D.add(P1M_D_emissions);		
		
		double agent_cost = costReward_D.stream().mapToDouble(h -> h).sum();
		double agent_emissions = emissionsReward_D.stream().mapToDouble(h -> h).sum();
		
		double violationPenalty_D = 0;		
		double h1_D = 0;
		double h2_D = 0;
		
		if (P1M_D > 470)
		{
			h1_D = P1M_D - 470;
		}
		else if (P1M_D < 150)
		{
			h1_D = 150 - P1M_D;
		}
		else
		{
			h1_D = 0;
		}
		
		
		if (P1M_D - P1M > 80)
		{
			h2_D = (P1M_D - P1M) - 80;
		}
		else if ((P1M_D - P1M) < (-80))
		{
			h2_D = (P1M_D - P1M) + 80;
		}
		else
		{
			h2_D = 0;
		}
		

		if (h1_D != 0 && h2_D == 0)
		{
			violationPenalty_D = (Math.abs(h1_D + 1) * this.U1[11]) * C;
		}
		else if (h1_D == 0 && h2_D != 0)
		{
			violationPenalty_D = (Math.abs(h2_D + 1) * this.U1[11]) * C;
		}
		else if (h1_D == 0 && h2_D == 0)
		{
			violationPenalty_D = 0;
		}
		else if (h1_D != 0 && h2_D != 0)
		{
			violationPenalty_D = (1*1000*1000*1 * (Math.abs(h1_D + 1) * this.U1[11])) 
								+ (1*1000*1000*1 * ((Math.abs(h2_D + 1) * this.U1[11])));
		}

		double totalCost = 0;
		double totalEmissions = 0;
		double totalViolationPenalty = 0;
	
		if (scalarisation == "hypervolume")
		{	
			totalCost = costReward.stream().mapToDouble(c -> c).sum() - agent_cost;
			totalEmissions = emissionsReward.stream().mapToDouble(d -> d).sum() - agent_emissions;
			totalViolationPenalty = violationPenalty - violationPenalty_D;
			
			//System.out.println("TC: " + totalCost);
			//System.out.println("TE: " + totalEmissions);
			//System.out.println("TV: " + totalViolationPenalty);
			//System.out.println(" ");
			
			
			reward  = totalCost + totalEmissions + totalViolationPenalty;		
			reward = -reward;
		}
		
		if (scalarisation == "linear")
		{				
			totalCost = (costReward.stream().mapToDouble(a -> a).sum() - agent_cost) * 0.225;
			totalEmissions = (emissionsReward.stream().mapToDouble(a -> a).sum() - agent_emissions) * 0.275;
			totalViolationPenalty = (violationPenalty - violationPenalty_D) * 0.5;
			
			reward = -(totalCost + totalEmissions + totalViolationPenalty);
			
		}
	
		rewardArray[0] = reward; rewardArray[1] = costReward.stream().mapToDouble(c -> c).sum(); 
		rewardArray[2] = emissionsReward.stream().mapToDouble(d -> d).sum();
		rewardArray[3] = violationPenalty;
	
	return rewardArray;
	}	



	
	private double getP1M_minus_D() {
		// TODO Auto-generated method stub
		return 0;
	}

	public double[] timeStep(ArrayList<Agent> _agents_, int j, String rewardType, String scalarization)
	{
		double[] timeStepVector = {0,0,0,0};
		double[] rewardReturnHolder = {0,0,0,0};
		this.scalarization = scalarization;
		int place = j;
		ArrayList<Agent> _agentsList_ = _agents_;
		
		int hour;
		int b = 0;
		ArrayList<Double> rewardTotal = new ArrayList<Double>();
		ArrayList<Double> costTotal = new ArrayList<Double>();
		ArrayList<Double> emissionsTotal = new ArrayList<Double>();
		ArrayList<Double> violationsTotal = new ArrayList<Double>();
		
		
		double cost = 0;
		double emissions = 0;
		double violations = 0;
		double reward = 0;
		
		double currentPDM;
		double previousPDM;
		double PDM_delta;
		double P1M = 0;
		double previousAgentPower = 0;
		double[] powerArray = {0,0,0};
		//Agent agent = null;
		double Pn;
		int action;
		int previousState;
		ArrayList<Double> PNM = new ArrayList<Double>();
		int currentState = -1;
		
		hour = 1;
			
		while (hour < 24)
		{
			b = b + 1;
			PNM.clear();
			
			powerArray = getPowerDemand(hour);
			currentPDM = powerArray[0]; previousPDM = powerArray[1]; PDM_delta = powerArray[2];
			
			
			for (int i = 0; i < _agents_.size(); i++)
			{
				Agent agent_ = _agents_.get(i);
				int id = agent_.getAgentID() - 2;
				
			
				
				previousAgentPower = agent_.getPreviousAgentPower();
				//System.out.println("Previous Agent Power: " + previousAgentPower);
				currentState = agent_.getState();
				//System.out.println(agent_.getAgentID());
				
				int minAllowedAction = 0;
				int maxAllowedAction = 0;
				int minAllowedPosition = 0;
				int maxAllowedPosition = 0;
				
				minAllowedPosition = (int) (previousAgentPower - agent_.UHolder.get(id).get(13));
				maxAllowedPosition = (int) (previousAgentPower + agent_.UHolder.get(id).get(12));
				
				if (minAllowedPosition < agent_.UHolder.get(id).get(0))
				{
					minAllowedPosition = (int) (agent_.UHolder.get(id).get(0) - 0);
				}
				
				if (maxAllowedPosition > agent_.UHolder.get(id).get(1))
				{
					maxAllowedPosition = (int) (agent_.UHolder.get(id).get(1) - 0);
				}
				
				//System.out.print("maxAllowedPosition: " + minAllowedPosition);
				double minPosLessOffset = minAllowedPosition - agent_.genOffsets[id+1];
				double maxPosLessOffset = maxAllowedPosition - agent_.genOffsets[id+1];
				double actionFraction = (double) numPercentSteps / (double) (percentageIncrement * (agent_.genRanges[id+1] - 1));
				Double minAllowedAction_D = minPosLessOffset * actionFraction;
				Double maxAllowedAction_D = maxPosLessOffset * actionFraction;
				minAllowedAction = minAllowedAction_D.intValue();
				maxAllowedAction = maxAllowedAction_D.intValue();			
				
				action = agent_.selectActionDEED(currentState, minAllowedAction, maxAllowedAction+1);
				//action = agent_.selectAction(hour, currentState, agent_);
				
				//Pn = getPNM(action, agent_);				
				Pn = action * percentageIncrement * (agent_.genRanges[id + 1] - 1) / (double) numPercentSteps + agent_.genOffsets[id + 1];
				
				agent_.setAgentPower(Pn);
                PNM.add(Pn);
                agent_.saveAction(action);
			}
			
			for (int x = 0; x < _agents_.size(); x++)
			{				
				Agent agent = _agents_.get(x);
				agent.savePnm(PNM);	
			}
			
			P1M = getP1M(PNM, currentPDM);			
			
			if (rewardType == "Global")
			{
				rewardReturnHolder = calculateGlobalReward(place, b, _agentsList_, PNM, currentState,P1M, hour, this.scalarization, previousPDM);
				reward = rewardReturnHolder[0]; cost = rewardReturnHolder[1]; emissions = rewardReturnHolder[2];
				violations = rewardReturnHolder[3];
				
				emissionsTotal.add(emissions);  rewardTotal.add(reward); violationsTotal.add(violations);
				costTotal.add(cost);
			}
			
			for (int z = 0; z < _agents_.size(); z ++)
			{
				Agent _agent = _agents_.get(z);
				
				previousState = _agent.getState();
				action = _agent.getAction();
				
				if (rewardType == "Difference")
				{
					rewardReturnHolder = calculateDifferenceReward(place, b, _agentsList_, 
							_agent, currentPDM, PNM, currentState,P1M, hour, this.scalarization, previousPDM);
					
					reward = rewardReturnHolder[0]; cost = rewardReturnHolder[1]; emissions = rewardReturnHolder[2];
					violations = rewardReturnHolder[3];					
				}
				
				currentState = _agent.getStateMARL(hour + 1, _agent, _agent.getAgentPower(), action);
				//currentState = _agent.getNextState(PDM_delta, _agent.getPreviousAgentPower(), _agent);
				_agent.saveCurrentState(currentState);
				
				//public int getStateMARL(int hour, Agent agent, double power_)
				
				_agent.setPreviousAgentPower(_agent.getAgentPower());
				_agent.updateQValuesDEED(previousState, currentState, action, (float) reward);				
			}
			
			
			hour = hour + 1;
			
			
			if (rewardType == "Difference")
			{
				emissionsTotal.add(emissions);  rewardTotal.add(reward); violationsTotal.add(violations);
				costTotal.add(cost);				
			}			
					
			}	
				
		double totalCost = costTotal.stream().mapToDouble(ii -> ii).sum();
		double totalEmissions = emissionsTotal.stream().mapToDouble(ik -> ik).sum();
		double totalViolations = violationsTotal.stream().mapToDouble(il -> il).sum();
		double totalReward = rewardTotal.stream().mapToDouble(ix -> ix).sum();
		
		timeStepVector[0] = totalCost; timeStepVector[1] = totalEmissions; timeStepVector[2] = totalReward;
		timeStepVector[3] = totalViolations;
		
		System.out.println("Total Cost: " + totalCost);
		System.out.println("Total Emissions: " + totalEmissions);
		System.out.println("Total Violations: " + totalViolations);
		System.out.println("Total Reward: " + totalReward);
		System.out.println(" ");
		
		return timeStepVector;
		
	}
	

}

