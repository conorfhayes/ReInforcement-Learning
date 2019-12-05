package RL_DEED;

import java.util.ArrayList;

public class ExpierienceReplay {
	
	public int action = 0;
	public int state = 0;
	public double reward = 0.0;
	public int nextState = 0;
	public ArrayList<Integer> actionHolder = new ArrayList<Integer>();
	public ArrayList<Integer> stateHolder= new ArrayList<Integer>();
	public ArrayList<Double> rewardHolder = new ArrayList<Double>();
	public ArrayList<Integer> nextStateHolder = new ArrayList<Integer>();
	
	public ExpierienceReplay()
	
	{
		
        		
	}
	
	public void addExpierience(int action, int state, double reward, int nextState)
	
	{
		this.actionHolder.add(action);
		this.stateHolder.add(state);
		this.rewardHolder.add(reward);
		this.nextStateHolder.add(nextState);
	}
	
	public int[] getActionExpierience(ArrayList<Integer> batchIndex)
	
	{
		int[] actionGrabber = new int[batchIndex.size()]; 
		for (int i = 0; i < batchIndex.size(); i ++)
		{
			actionGrabber[i] = this.actionHolder.get(i); 
		}
		
		return actionGrabber;
		
	}
	
	public double[] getRewardExpierience(ArrayList<Integer> batchIndex)
	
	{
		double[] rewardGrabber = new double[batchIndex.size()]; 
		for (int i = 0; i < batchIndex.size(); i ++)
		{
			rewardGrabber[i] = this.rewardHolder.get(i); 
		}
		
		return rewardGrabber;
		
	}
	
	public double[] getStateExpierience(ArrayList<Integer> batchIndex)
	
	{
		double[] stateGrabber = new double[batchIndex.size()]; 
		for (int i = 0; i < batchIndex.size(); i ++)
		{
			stateGrabber[i] = this.stateHolder.get(i); 
		}
		
		return stateGrabber;
		
	}
	
	public double[] getNextStateExpierience(ArrayList<Integer> batchIndex)
	
	{
		double[] stateGrabber = new double[batchIndex.size()]; 
		for (int i = 0; i < batchIndex.size(); i ++)
		{
			stateGrabber[i] = this.nextStateHolder.get(i); 
		}
		
		return stateGrabber;
		
	}
	
	
	

}
