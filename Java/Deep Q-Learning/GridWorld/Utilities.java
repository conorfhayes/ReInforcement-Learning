/*
 * Utilities class with static methods
 * Dr Patrick Mannion, Galway-Mayo Institute of Technology, March 2017
 */


import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class Utilities {

	// returns a formatted Q values table
	public static String qTableAsString(float[][] qTable, int[] basesForStateNo) {
		DecimalFormat df = new DecimalFormat("0.000");
		String output = "State#	Xcoord	Ycoord	|	North	East	South	West";
		for(int s=0; s<qTable.length;s++) {
			int[] coord = getXYfromStateNo(s, basesForStateNo);
			output += "\n"+s+"		" + coord[0] + "		" + coord[1] + "		|	" + df.format(qTable[s][0]) + "	" + df.format(qTable[s][1]) + "	" + df.format(qTable[s][2]) + "	" + df.format(qTable[s][3]);
		}
		return output;
	}

	public static void qTableToConsole(float[][] qTable, int[] basesForStateNo) {
		System.out.println(qTableAsString(qTable, basesForStateNo));
	}

	public static void qTableToFile(float[][] qTable, int[] basesForStateNo, String fileName) {
		System.out.println(qTableAsString(qTable, basesForStateNo));
	}

	// mixed radix conversion (converts a state vector [x,y] to a single integer)
	public static int getStateNoFromXY(int[] state, int[] basesForStateNo) {
		int numStates = basesForStateNo[0]*basesForStateNo[1];
		int stateNo = 0;
		for (int i=0;i<state.length;i++) {
			stateNo = stateNo * basesForStateNo[i] + state[i];
		}
		//check state is allowed
		if (stateNo >= numStates || stateNo < 0 ) {
			System.out.println("Vector Conversion Error - X: "+state[0]+" Y: "+state[1]);
			System.out.println("Resultant State Number " + stateNo);
			System.out.println("max allowed states: " + numStates);
		}
		return stateNo;
	}

	// mixed radix conversion (converts a state number to a state vector [x,y]
	public static int[] getXYfromStateNo(int stateNo, int[] basesForStateNo) {
		int[] state = new int[2];
		int inputstateNo = stateNo;
		for(int i=state.length-1;i>-1;i--) {
			//System.out.println("inputStateNo: " + inputstateNo + ", basesForStateNo["+i+"]: " + basesForStateNo[i]);
			state[i] = inputstateNo % basesForStateNo[i];
			inputstateNo = inputstateNo / basesForStateNo[i];
		}
		return state;
	}

	// method to output experimental results in comma separated value format
	public static String resultsToCSVStr(ArrayList<ArrayList<Integer>> results) {
		String output = "Episode No.,";
		for(int run=0; run<results.size(); run++) {
			output += "Run" + run + "steps,";
		}
		for(int time=0; time<results.get(0).size(); time++) {
			output += "\n" + time + ",";
			for(int run=0; run<results.size(); run++) {
				output += "" + results.get(run).get(time) + ",";				
			}
		}
		return output;
	}

	// method to output experimental results to a CSV file
	public static void resultsToCSVFile(ArrayList<ArrayList<Integer>> results, String experimentName) { 
		String resultsTable =  Utilities.resultsToCSVStr(results);
		try (PrintStream out = new PrintStream(new FileOutputStream("output/" + experimentName+"/" + experimentName + "_stepsToGoal.csv"))) {
			out.print(resultsTable);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	// method to output final Q tables to file
	public static void QTablesToFile(ArrayList<float[][]> QTables, int[] basesForStateNo, String experimentName) {
		String output = "";
		for(int run=0; run<QTables.size(); run++) {
			output += "*************** Q table for run "+ run + " ***************\n";
			output += qTableAsString(QTables.get(run), basesForStateNo)+"\n\n";
		}
		try (PrintStream out = new PrintStream(new FileOutputStream("output/" + experimentName+"/" + experimentName + "_QTables.txt"))) {
			out.print(output);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


}
