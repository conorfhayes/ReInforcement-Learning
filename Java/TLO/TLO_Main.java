import java.util.ArrayList;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.ExecutorService;

import com.smartxls.ChartFormat;
import com.smartxls.ChartShape;
import com.smartxls.RangeStyle;
import com.smartxls.WorkBook;

public class TLO_Main {
	
	public static int numEpisodes = 20000;
	public static int numAgents = 9;
	public static double[] timeStepVector;
	public static double totalCost;
	public static double totalEmissions;
	public static double costThreshold;
	public static double violationsThreshold;
	public static double totalViolations;
	public static int numRuns = 50;
	
	//public static WorkBook output_workbook;
	
	public static ArrayList<Double> FinaltotalCostArray = new ArrayList<Double>();
	public static ArrayList<Double> FinaltotalEmissionsArray = new ArrayList<Double>();
	public static ArrayList<Double> FinaltotalViolationsArray = new ArrayList<Double>();
	
	public static ArrayList<Double> FinaloutputCostArray = new ArrayList<Double>();
	public static ArrayList<Double> FinaloutputEmissionsArray = new ArrayList<Double>();
	public static ArrayList<Double> FinaloutputViolationsArray = new ArrayList<Double>();
	
	public static ArrayList<Double> FinaltotalCostThreshold = new ArrayList<Double>();
	public static ArrayList<Double> FinaltotalViolationsThreshold = new ArrayList<Double>();
	
	public static long startTime;
	public static long stopTime;
	public static long elapsedTime;

	
	public static void main(String[] args) throws Exception
	{
		
		ArrayList<ArrayList<Double>> runCostResults = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> runEmissionsResults = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> runViolationsResults = new ArrayList<ArrayList<Double>>();
		
		ArrayList<ArrayList<Double>> runCostThresholdResults = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> runViolationsThresholdResults = new ArrayList<ArrayList<Double>>();
	
		
		ArrayList<Double> outputCostArray = new ArrayList<Double>();
		ArrayList<Double> outputEmissionsArray = new ArrayList<Double>();
		ArrayList<Double> outputViolationsArray = new ArrayList<Double>();
		ArrayList<Double> outputCostThresholdArray = new ArrayList<Double>();
		ArrayList<Double> outputViolationsThresholdArray = new ArrayList<Double>();
		
		WorkBook output_workbook = new WorkBook();
		int inc = 1;

		double[] globalCollector = {0,0,0,0};
		startTime = System.currentTimeMillis();
		int div = numRuns;
		
		
		while (inc <= numRuns)
		{	
			int starter = 1;
			ArrayList<TLO_Agent> _agentsGlobal_ = new ArrayList<TLO_Agent>();
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
			
			ArrayList<Double> totalCostArray = new ArrayList<Double>();
			ArrayList<Double> totalEmissionsArray = new ArrayList<Double>();
			ArrayList<Double> totalViolationsArray = new ArrayList<Double>();
			
			ArrayList<Double> totalCostThreshold = new ArrayList<Double>();
			ArrayList<Double> totalViolationsThreshold = new ArrayList<Double>();
			
			while (j <= numEpisodes)
			{
				
				timeStepVector = envGlobal.timeStep(_agentsGlobal_, j, "Global", "hypervolume");
				
				totalCost = timeStepVector[0];
				totalEmissions = timeStepVector[1];
				totalViolations = timeStepVector[3];
				violationsThreshold = timeStepVector[4];
				costThreshold = timeStepVector[5];
				
				
				totalCostArray.add(totalCost); totalEmissionsArray.add(totalEmissions);
				totalViolationsArray.add(totalViolations); totalCostThreshold.add(costThreshold);
				totalViolationsThreshold.add(violationsThreshold);
				
				//System.out.println("*** Episode: " + j + " ***");
				//System.out.println("Total Cost: " + totalCost);
				//System.out.println("Total Emissions: " + totalEmissions);
				//System.out.println("Total Violations: " + totalViolations);				
				//System.out.println(" ");
				

				if (j == numEpisodes)
				{
					FinaltotalCostArray.add(totalCost / 1000000);
					FinaltotalEmissionsArray.add(totalEmissions / 100000);
					FinaltotalViolationsArray.add(totalViolations / 1000000);
					FinaltotalViolationsThreshold.add(violationsThreshold / 1000000);
					FinaltotalCostThreshold.add(costThreshold / 1000000);
				}
					
				
				j = j + 1;
			}
			
			runCostResults.add(totalCostArray);
			runEmissionsResults.add(totalEmissionsArray);
			runViolationsResults.add(totalViolationsArray);
			runCostThresholdResults.add(totalCostThreshold);
			runViolationsThresholdResults.add(totalViolationsThreshold);
			
			
			inc = inc + 1;	
			
		}
		
		stopTime = System.currentTimeMillis();
		//System.out.println(stopTime);
		elapsedTime = stopTime - startTime;
		//System.out.println(elapsedTime);
		double cost = 0;
		double emissions = 0;
		double violations = 0;
		double costThreshold = 0;
		double violationsThreshold = 0;
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
				costThreshold = costThreshold + runCostThresholdResults.get(y).get(x);
				violationsThreshold = violationsThreshold + runViolationsThresholdResults.get(y).get(x);
			}
			
			cost = cost / div; emissions = emissions / div; violations = violations / div;
			violationsThreshold = violationsThreshold/div; costThreshold = costThreshold/div;
			cost = cost / 1000000 ; emissions = emissions / 1000000; violations = violations / 1000000; 
			violationsThreshold = violationsThreshold/1000000; costThreshold = costThreshold/1000000;
			outputCostArray.add(cost); outputEmissionsArray.add(emissions); outputViolationsArray.add(violations);
			outputViolationsThresholdArray.add(violationsThreshold); outputCostThresholdArray.add(costThreshold);
			
		}
			
		//System.out.println("Output Cost Array: " + outputCostArray);
		//System.out.println("Output Emissions Array: " + outputEmissionsArray);
		//System.out.println("Output Violations Array: " + outputViolationsArray);
		
		FileWriter fw = new FileWriter("TLO_Cost_" + java.time.LocalDate.now() + "_" +  java.time.LocalTime.now());
		PrintWriter pw = new PrintWriter(fw);
		for (int e = 0; e < numEpisodes; e ++)
		{
			pw.println(outputCostArray.get(e).toString());
		}
		//br.write(outputCostArray.toString());
		pw.flush();			
		pw.close();
		fw.close();
		
		FileWriter fw_comp = new FileWriter("TLO_Computation_" + java.time.LocalDate.now() + "_" +  java.time.LocalTime.now());
		PrintWriter pw_comp = new PrintWriter(fw_comp);		
		pw_comp.println(String.valueOf(elapsedTime));
		
		//br.write(outputCostArray.toString());
		pw_comp.flush();			
		pw_comp.close();
		fw_comp.close();
		
		FileWriter fw1 = new FileWriter("TLO_Emissions_" + java.time.LocalDate.now() + "_" +  java.time.LocalTime.now());
		PrintWriter pw1 = new PrintWriter(fw1);
		for (int e = 0; e < numEpisodes; e ++)
		{
			pw1.println(outputEmissionsArray.get(e).toString());
		}
		//br.write(outputCostArray.toString());
		pw1.flush();			
		pw1.close();
		fw1.close();
		
		FileWriter fw2 = new FileWriter("TLO_Violations_" + java.time.LocalDate.now() + "_" +  java.time.LocalTime.now());
		PrintWriter pw2 = new PrintWriter(fw2);
		for (int e = 0; e < numEpisodes; e ++)
		{
			pw2.println(outputViolationsArray.get(e).toString());
		}
		//br.write(outputCostArray.toString());
		pw2.flush();			
		pw2.close();
		fw2.close();
		
		try
		{
		output_workbook.insertSheets(0, 1);
		
		for (int xx = 0; xx < 2; xx ++)
		{
			String SheetName;
			if (xx == 0)
			{
				SheetName = "Run Results";
			}
			else
			{
				SheetName = "Final Results";
				
			}
			output_workbook.setSheetName(xx, SheetName);
			output_workbook.setSheet(xx);
			
			output_workbook.setText(0,0,"Soln No");
			output_workbook.setText(0,1,"Cost");
			output_workbook.setText(0,2,"Emissions");
			output_workbook.setText(0,3,"Violations");
			output_workbook.setText(0,4,"Violations Threshold");
			output_workbook.setText(0,5,"Cost Threshold");
			
			if (xx == 0)
			{
				int rowCounter = 1;
				for (int point = 0; point < numEpisodes; point++) 
				{
					
					output_workbook.setFormula(rowCounter, 0, "" + rowCounter);
					output_workbook.setFormula(rowCounter, 1, "" + outputCostArray.get(point));
					output_workbook.setFormula(rowCounter, 2, "" + outputEmissionsArray.get(point));
					output_workbook.setFormula(rowCounter, 3, "" + outputViolationsArray.get(point));	
					output_workbook.setFormula(rowCounter, 4, "" + outputViolationsThresholdArray.get(point));
					output_workbook.setFormula(rowCounter, 5, "" + outputCostThresholdArray.get(point));
					rowCounter++;
					
				}
			}
			
			if (xx == 1)
			{
				int rowCounter = 1;
				for (int point = 0; point < numRuns; point++) 
				{
					output_workbook.setFormula(rowCounter, 0, "" + rowCounter);
					output_workbook.setFormula(rowCounter, 1, "" + FinaltotalCostArray.get(point));
					output_workbook.setFormula(rowCounter, 2, "" + FinaltotalEmissionsArray.get(point)/10);
					output_workbook.setFormula(rowCounter, 3, "" + FinaltotalViolationsArray.get(point));				
					rowCounter++;
					
				}
			}
		}	
		
		output_workbook.write("./Output/" + "TLO_Output_" + java.time.LocalDate.now() + "_" +  java.time.LocalTime.now() + ".xls");
	}
		catch (Exception ex)
		{
			ex.printStackTrace();
		}
		
	}
		
}
