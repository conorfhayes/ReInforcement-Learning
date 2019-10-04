import java.util.ArrayList;

public class TLO_Main {
	
	public static int numEpisodes = 20000;
	public static int numAgents = 9;
	public static double[] timeStepVector;
	public static double totalCost;
	public static double totalEmissions;
	public static double totalViolations;
	
	public static ArrayList<Double> totalCostArray = new ArrayList<Double>();
	public static ArrayList<Double> totalEmissionsArray = new ArrayList<Double>();
	public static ArrayList<Double> totalViolationsArray = new ArrayList<Double>();
	
	public static ArrayList<Double> outputCostArray = new ArrayList<Double>();
	public static ArrayList<Double> outputEmissionsArray = new ArrayList<Double>();
	public static ArrayList<Double> outputViolationsArray = new ArrayList<Double>();
	
	public static ArrayList<ArrayList<Double>> runCostResults = new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> runEmissionsResults = new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> runViolationsResults = new ArrayList<ArrayList<Double>>();
	
	public static void main(String[] args)
	{
		//System.out.println("Hello");
		int inc = 0;
		int starter = 1;
		double[] globalCollector = {0,0,0,0};
		ArrayList<TLO_Agent> _agentsGlobal_ = new ArrayList<TLO_Agent>();
		
		while (inc <= 10)
		{	
			TLO_Environment envGlobal = new TLO_Environment();
			//envGlobal.setU(envGlobal.UHolder);
			
			while (starter <= numAgents)
			{
				TLO_Agent agentGlobal = envGlobal.createAgent(starter + 1);
				agentGlobal.setU(agentGlobal.UHolder);
				_agentsGlobal_.add(agentGlobal);
				starter = starter + 1;
			}
			
			
			
			System.out.println("*************** Run " + inc + " ***************");
			
			int j = 1;
			totalCostArray.clear();
			totalEmissionsArray.clear();
			totalViolationsArray.clear();
			
			while (j <= numEpisodes)
			{
				System.out.println("*** Episode: " + j + " ***");
				timeStepVector = envGlobal.timeStep(_agentsGlobal_, j, "Global", "hypervolume");
				
				totalCost = timeStepVector[0];
				totalEmissions = timeStepVector[1];
				totalViolations = timeStepVector[3];
				
				totalCostArray.add(totalCost); totalEmissionsArray.add(totalEmissions);
				totalViolationsArray.add(totalViolations);
				
				System.out.println("Total Cost: " + totalCost);
				System.out.println("Total Emissions: " + totalEmissions);
				System.out.println("Total Violations: " + totalViolations);				
				System.out.println(" ");
					
				
				j = j + 1;
			}
			
			runCostResults.add(totalCostArray);
			runEmissionsResults.add(totalEmissionsArray);
			runViolationsResults.add(totalViolationsArray);
			
			//System.out.println(runCostResults.get(0));
			
			inc = inc + 1;	
			
		}
			double cost = 0;
			double emissions = 0;
			double violations = 0;
			for (int x = 0; x <= numEpisodes - 1; x ++)
			{
				for (int y = 0; y < inc; y++)
					
				{
					cost = cost + runCostResults.get(y).get(x);
					emissions = emissions + runEmissionsResults.get(y).get(x);
					violations = violations + runViolationsResults.get(y).get(x);
				}
				
				cost = cost / inc; emissions = emissions / inc; violations = violations / inc;
				cost = cost / 1000000 ; emissions = emissions / 10000000; violations = violations / 1000000; 
				outputCostArray.add(cost); outputEmissionsArray.add(emissions); outputViolationsArray.add(violations);
			}
			
			System.out.println("Output Cost Array: " + outputCostArray);
			System.out.println("Output Emissions Array: " + outputEmissionsArray);
			System.out.println("Output Violations Array: " + outputViolationsArray);
				
	}
}
