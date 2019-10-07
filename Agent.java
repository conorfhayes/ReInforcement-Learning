import java.util.ArrayList;

public class Main {
	
	public static int numEpisodes = 20000;
	public static int numAgents = 9;
	
	public static void main(String[] args)
	{
		//System.out.println("Hello");
		int inc = 1;
		int starter = 1;
		double[] globalCollector = {0,0,0,0};
		ArrayList<Agent> _agentsGlobal_ = new ArrayList<Agent>();
		
		while (inc <= 1)
		{	
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
			while (j <= numEpisodes)
			{
				System.out.println("*** Episode: " + j + " ***");
				envGlobal.timeStep(_agentsGlobal_, j, "Global", "hypervolume");
				j = j + 1;
			}
			
			inc = inc + 1;			
		}		
	}
}
