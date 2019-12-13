/*
 * Sample implementation of a RL Experiment runner/data logging class
 * Dr Patrick Mannion, Galway-Mayo Institute of Technology, March 2017
 */



import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

public class ExpRunner {

	static int numRuns = 1; // number of statistical runs to be conducted
	
	
	public static void main(String[] args) {
		
		ArrayList<ArrayList<Integer>> results = new ArrayList<ArrayList<Integer>>();
		ArrayList<float[][]> QTables = new ArrayList<float[][]>();
		
		// create a timestamped directory to store results for this experiment
		String experimentName = "Gridworld_"+System.currentTimeMillis();
		new File("output/" + experimentName).mkdirs();
		
		// conduct the specified number of runs
		for(int run=0; run<numRuns; run++) {
			System.out.println("\nExpRunner: *************** Run " + run + " starting ***************");
			
			// initialise the environment and run the experiment
			Environment env = new Environment();
			//env.enableDebugging(); // uncomment this for console output 
			env.doExperiment();
			
			// add number of moves for this run to the results list
			results.add(env.getMovesToGoal());			
			
			// add Q table for this run to the list of Q tables
			QTables.add(env.getQTable());			
		}
		
		// finally, output results table and list of final Q tables to the file system
		Utilities.resultsToCSVFile(results,experimentName);
		Utilities.QTablesToFile(QTables,new int[]{Environment.getXDimension(),Environment.getYDimension()},experimentName);

	}

}
