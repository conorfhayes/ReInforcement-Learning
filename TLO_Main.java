import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

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
	public static double totalViolations;
	public static int numRuns = 50;
	
	//public static WorkBook output_workbook;
	
	public static ArrayList<Double> FinaltotalCostArray = new ArrayList<Double>();
	public static ArrayList<Double> FinaltotalEmissionsArray = new ArrayList<Double>();
	public static ArrayList<Double> FinaltotalViolationsArray = new ArrayList<Double>();
	
	public static ArrayList<Double> FinaloutputCostArray = new ArrayList<Double>();
	public static ArrayList<Double> FinaloutputEmissionsArray = new ArrayList<Double>();
	public static ArrayList<Double> FinaloutputViolationsArray = new ArrayList<Double>();

	
	public static void main(String[] args) throws Exception
	{
		
		ArrayList<ArrayList<Double>> runCostResults = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> runEmissionsResults = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> runViolationsResults = new ArrayList<ArrayList<Double>>();
	
		
		ArrayList<Double> outputCostArray = new ArrayList<Double>();
		ArrayList<Double> outputEmissionsArray = new ArrayList<Double>();
		ArrayList<Double> outputViolationsArray = new ArrayList<Double>();
		
		WorkBook output_workbook = new WorkBook();
		//System.out.println("Hello");
		int inc = 1;
		//int starter = 1;
		double[] globalCollector = {0,0,0,0};
		
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
			
			while (j <= numEpisodes)
			{
				
				timeStepVector = envGlobal.timeStep(_agentsGlobal_, j, "Global", "hypervolume");
				
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
				

				if (j == numEpisodes)
				{
					FinaltotalCostArray.add(totalCost / 100000);
					FinaltotalEmissionsArray.add(totalEmissions / 10000);
					FinaltotalViolationsArray.add(totalViolations / 100000);
				}
					
				
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
			cost = cost / 1000000 ; emissions = emissions / 1000000; violations = violations / 1000000; 
			outputCostArray.add(cost); outputEmissionsArray.add(emissions); outputViolationsArray.add(violations);
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
			output_workbook.setText(0,1,"Cost (x10^6)");
			output_workbook.setText(0,2,"Emissions (x10^5)");
			output_workbook.setText(0,3,"Violations (x10^6)");
			
			if (xx == 0)
			{
				int rowCounter = 1;
				for (int point = 0; point < numEpisodes; point++) 
				{
					
					output_workbook.setFormula(rowCounter, 0, "" + rowCounter);
					output_workbook.setFormula(rowCounter, 1, "" + outputCostArray.get(point));
					output_workbook.setFormula(rowCounter, 2, "" + outputEmissionsArray.get(point));
					output_workbook.setFormula(rowCounter, 3, "" + outputViolationsArray.get(point));				
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
					output_workbook.setFormula(rowCounter, 2, "" + FinaltotalEmissionsArray.get(point));
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
