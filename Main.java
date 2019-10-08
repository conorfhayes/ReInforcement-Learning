import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class Main {
	
	public static int numEpisodes = 20000;
	public static int numAgents = 9;
	public static int numRuns = 20;
	
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
	public static String rewardType = new String();
	public static String scalarisation = new String();
	
	public static void main(String[] args) throws IOException
	{
		//System.out.println("Hello");
		int inc = 1;
		int div = numRuns;
		//int starter = 1;
		double[] globalCollector = {0,0,0,0};
		//ArrayList<Agent> _agentsGlobal_ = new ArrayList<Agent>();
		
		while (inc <= numRuns )
		{	
			ArrayList<Agent> _agentsGlobal_ = new ArrayList<Agent>();
			int starter = 1;
			Environment envGlobal = new Environment();
			//envGlobal.setU(envGlobal.UHolder);
			
			while (starter <= numAgents)
			{
				Agent agentGlobal = envGlobal.createAgent(starter + 1);
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
				
				rewardType = "Global";
				scalarisation = "linear";
				timeStepVector = envGlobal.timeStep(_agentsGlobal_, j, rewardType, scalarisation);
				
				totalCost = timeStepVector[0];
				totalEmissions = timeStepVector[1];
				totalViolations = timeStepVector[3];
				
				totalCostArray.add(totalCost); totalEmissionsArray.add(totalEmissions);
				totalViolationsArray.add(totalViolations);
				
				//System.out.println("*** Episode: " + j + " ***");
				//System.out.println("Total Cost: " + totalCost);
				//System.out.println("Total Emissions: " + totalEmissions);
				//System.out.println("Total Violations: " + totalViolations);				
				//System.out.println(" ");
					
				
				j = j + 1;
			}
			
			runCostResults.add(totalCostArray);
			runEmissionsResults.add(totalEmissionsArray);
			runViolationsResults.add(totalViolationsArray);

			
			inc = inc + 1;			
		}		
		
		double cost = 0;
		double emissions = 0;
		double violations = 0;

		for (int x = 0; x < numEpisodes; x ++)
		{
			cost = 0;
			emissions = 0;
			violations = 0;
			
			for (int y = 0; y < numRuns; y++)
				
			{
				cost = cost + runCostResults.get(y).get(x);
				emissions = emissions + runEmissionsResults.get(y).get(x);
				violations = violations + runViolationsResults.get(y).get(x);
			}
			
			cost = cost / div; emissions = emissions / div; violations = violations / div;
			cost = cost / 1000000 ; emissions = emissions / 100000; violations = violations / 1000000; 
			outputCostArray.add(cost); outputEmissionsArray.add(emissions); outputViolationsArray.add(violations);
		}
		

		FileWriter fw = new FileWriter(rewardType + "_" + scalarisation + "_" + "Cost_" + java.time.LocalDate.now() + "_" +  java.time.LocalTime.now());
		PrintWriter pw = new PrintWriter(fw);
		for (int e = 0; e < numEpisodes; e ++)
		{
			pw.println(outputCostArray.get(e).toString());
		}

		pw.flush();			
		pw.close();
		fw.close();
		
		FileWriter fw1 = new FileWriter(rewardType + "_" + scalarisation + "_" + "Emissions_" + java.time.LocalDate.now() + "_" +  java.time.LocalTime.now());
		PrintWriter pw1 = new PrintWriter(fw1);
		for (int e = 0; e < numEpisodes; e ++)
		{
			pw1.println(outputEmissionsArray.get(e).toString());
		}

		pw1.flush();			
		pw1.close();
		fw1.close();
		
		FileWriter fw2 = new FileWriter(rewardType + "_" + scalarisation + "_" + "Violations_" + java.time.LocalDate.now() + "_" +  java.time.LocalTime.now());
		PrintWriter pw2 = new PrintWriter(fw2);
		for (int e = 0; e < numEpisodes; e ++)
		{
			pw2.println(outputEmissionsArray.get(e).toString());
		}

		pw2.flush();			
		pw2.close();
		fw2.close();
	}
}
