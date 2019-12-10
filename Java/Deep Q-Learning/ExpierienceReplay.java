package RL_DEED;

import java.util.ArrayList;

public class ExpierienceReplay {
	
	public int action = 0;
	public int state = 0;
	public double reward = 0.0;
	public int nextState = 0;
	public ArrayList<Integer> actionHolder = new ArrayList<Integer>();
	public ArrayList<Double> stateHolder= new ArrayList<Double>();
	public ArrayList<Double> rewardHolder = new ArrayList<Double>();
	public ArrayList<Double> nextStateHolder = new ArrayList<Double>();
	
	public ExpierienceReplay()
	
	{
		
		this.actionHolder.add(25);
		this.stateHolder.add(0.14);
		this.rewardHolder.add(-0.015);
		this.nextStateHolder.add(0.5465);
		
		this.actionHolder.add(50);
		this.stateHolder.add(0.24);
		this.rewardHolder.add(-0.015);
		this.nextStateHolder.add(0.75);	
		
		this.actionHolder.add(50);
		this.stateHolder.add(0.24);
		this.rewardHolder.add(-0.015);
		this.nextStateHolder.add(0.75);	
		
		this.actionHolder.add(50);
		this.stateHolder.add(0.24);
		this.rewardHolder.add(-0.015);
		this.nextStateHolder.add(0.75);	
		
		
		this.actionHolder.add(50);
		this.stateHolder.add(0.24);
		this.rewardHolder.add(-0.015);
		this.nextStateHolder.add(0.75);	
	}
	
	public void addExpierience(int action, double normal_currentstate, double reward, double normal_nextState)
	
	{
		this.actionHolder.add(action);
		this.stateHolder.add(normal_currentstate);
		this.rewardHolder.add(reward);
		this.nextStateHolder.add(normal_nextState);		
		
	}
	

	
	public int[] getActionExpierience(ArrayList<Integer> batchIndex)
	
	{
		int[] actionGrabber = new int[batchIndex.size()]; 
		for (int i = 0; i < batchIndex.size(); i ++)
		{	
			int j = batchIndex.get(i);
			actionGrabber[i] = this.actionHolder.get(j); 
		}
		
		return actionGrabber;
		
	}
	
	public double[] getRewardExpierience(ArrayList<Integer> batchIndex)
	
	{
		double[] rewardGrabber = new double[batchIndex.size()]; 
		for (int i = 0; i < batchIndex.size(); i ++)
		{
			int j = batchIndex.get(i);
			rewardGrabber[i] = this.rewardHolder.get(j); 
		}
		
		return rewardGrabber;
		
	}
	
	public double[] getStateExpierience(ArrayList<Integer> batchIndex)
	
	{
		double[] stateGrabber = new double[batchIndex.size()]; 
		
		for (int i = 0; i < batchIndex.size(); i ++)
		{
			int j = batchIndex.get(i);
			stateGrabber[i] = this.stateHolder.get(j); 
		}
		
		return stateGrabber;
		
	}
	
	public double[] getNextStateExpierience(ArrayList<Integer> batchIndex)
	
	{
		double[] stateGrabber = new double[batchIndex.size()]; 
		for (int i = 0; i < batchIndex.size(); i ++)
		{
			int j = batchIndex.get(i);
			stateGrabber[i] = this.nextStateHolder.get(j); 
		}
		
		return stateGrabber;
		
	}
	
	
	

}
