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
	public static ArrayList<ArrayList<Double>> runCostResults = new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> runEmissionsResults = new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> runViolationsResults = new ArrayList<ArrayList<Double>>();
	
	public static void main(String[] args)
	{
		//System.out.println("Hello");
		int inc = 1;
		int starter = 1;
		double[] globalCollector = {0,0,0,0};
		ArrayList<TLO_Agent> _agentsGlobal_ = new ArrayList<TLO_Agent>();
		
		while (inc <= 1)
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
	}
}