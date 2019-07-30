import numpy as np
from numpy import array
import math
from random import randint
import time, os, fnmatch, shutil
from collections import Counter
import matplotlib.pyplot as plt
from random import seed
import random
from plotnine import *
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

def costGraph(df, numEpisodes):
    x_axis = np.arange(0, numEpisodes, 200)
    print(x_axis)
    df['Counter'] = x_axis
    print(df[0:])
    x = df['Counter']
    df1 = df.melt(id_vars=["Counter"], var_name="Reward", value_name="Value")
    print(df1)

    color_dict = {'Global (λ)': 'red',
                  'TLO Global (λ)': 'green',
                  'TLO Difference (λ)': 'blue',
                  'Local (λ)': 'black',
                  'Difference (λ)': 'orange'}

    costG = (ggplot(df1) +
            aes(x=df1['Counter'], y=df1['Value'], color = df1['Reward']) +
            geom_line(alpha=0.7, size=0.75) +
            scale_color_manual(values=color_dict) +
            #geom_line(aes(x='x', y=df['difference']), alpha=0.5, size=0.5, color= df['difference']) +
            #geom_line(aes(x='x', y=df['local']), alpha=0.5, size=0.5, color= df['local']) +
            #scale_x_continuous(lim = (0, max(x_axis)), breaks= range(0,len(x_axis)+ 5000, 5000)) +
            scale_y_continuous(lim = (2.5, max(df1['Value'])), breaks = np.arange(2.5, max(df1['Value']) + 0.2, 0.2)) +
            ylab(" Cost ($ x 10^6) ") +
            xlab(" Episode ") +
            ggtitle(" ") +
            theme_matplotlib() +
            theme(axis_text_y = element_text(size =6)) +
            theme(axis_text_x=element_text(size=6)) +
            theme(legend_position="top",
            legend_text=element_text(size=8),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank()))

    print(costG)


def emissionsGraph(df, numEpisodes):
    x_axis = np.arange(0, numEpisodes, 200)
    print(x_axis)
    df['Counter'] = x_axis
    print(df[0:])
    x = df['Counter']
    df1 = df.melt(id_vars=["Counter"], var_name="Reward", value_name="Value")
    print(df1)

    color_dict = {'Global (λ)': 'red',
                  'TLO Global (λ)': 'green',
                  'TLO Difference (λ)': 'blue',
                  'Local (λ)': 'black',
                  'Difference (λ)': 'orange'}

    emissionsG = (ggplot(df1) +
            aes(x=df1['Counter'], y=df1['Value'], color = df1['Reward']) +
            geom_line(alpha=0.7, size=0.75) +
            scale_color_manual(values=color_dict) +
            #geom_line(aes(x='x', y=df['difference']), alpha=0.5, size=0.5, color= df['difference']) +
            #geom_line(aes(x='x', y=df['local']), alpha=0.5, size=0.5, color= df['local']) +
            #scale_x_continuous(lim = (0, max(x_axis)), breaks= range(0,len(x_axis)+ 5000, 5000)) +
            #scale_y_continuous(lim = (2.5, max(df1['Value'])), breaks = np.arange(2.5, max(df1['Value']) + 0.2, 0.2)) +
            ylab(" Emissions ($ x 10^6) ") +
            xlab(" Episode ") +
            ggtitle(" ") +
            theme_matplotlib() +
            theme(axis_text_y = element_text(size =6)) +
            theme(axis_text_x=element_text(size=6)) +
            theme(legend_position="top",
            legend_text=element_text(size=8),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank()))

    print(emissionsG)


def violationsGraph(df, numEpisodes):
    x_axis = np.arange(0, numEpisodes, 200)
    print(x_axis)
    df['Counter'] = x_axis
    print(df[0:])
    x = df['Counter']
    df1 = df.melt(id_vars=["Counter"], var_name="Reward", value_name="Value")
    print(df1)

    color_dict = {'Global (λ)': 'red',
                  'TLO Global (λ)': 'green',
                  'TLO Difference (λ)': 'blue',
                  'Local (λ)': 'black',
                  'Difference (λ)': 'orange'}

    violationsG = (ggplot(df1) +
            aes(x=df1['Counter'], y=df1['Value'], color = df1['Reward']) +
            geom_line(alpha=0.7, size=0.75) +
            scale_color_manual(values=color_dict) +
            #geom_line(aes(x='x', y=df['difference']), alpha=0.5, size=0.5, color= df['difference']) +
            #geom_line(aes(x='x', y=df['local']), alpha=0.5, size=0.5, color= df['local']) +
            #scale_x_continuous(lim = (0, max(x_axis)), breaks= range(0,len(x_axis)+ 5000, 5000)) +
            #scale_y_continuous(lim = (2.5, max(df1['Value'])), breaks = np.arange(0, max(df1['Value']) + 0.2, 0.2)) +
            ylab(" Violations (1 x 10^6) ") +
            xlab(" Episode ") +
            ggtitle(" ") +
            theme_matplotlib() +
            theme(axis_text_y = element_text(size =6)) +
            theme(axis_text_x=element_text(size=6)) +
            theme(legend_position="top",
            legend_text=element_text(size=8),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank()))

    print(violationsG)



numEpisodes = 20000


dfCost = pd.read_excel("Graphs\TLO_DEED_Cost_Hypervolume_All_Rewards.xlsx")
dfEmissions = pd.read_excel("Graphs\TLO_DEED_Emissions_Hypervolume_All_Rewards.xlsx")
dfEmissionsNoLocal = pd.read_excel("Graphs\TLO_DEED_Emissions_NoLocal_Hypervolume_All_Rewards.xlsx")
dfViolations = pd.read_excel("Graphs\TLO_DEED_Violations_Hypervolume_All_Rewards.xlsx")
dfViolationsNoLocal = pd.read_excel("Graphs\TLO_DEED_Violations_Hypervolume_NoLocal_All_Rewards.xlsx")

costGraph(dfCost, numEpisodes)
#emissionsGraph(dfEmissions, numEpisodes)
#emissionsGraph(dfEmissionsNoLocal, numEpisodes)
violationsGraph(dfViolations, numEpisodes)
violationsGraph(dfViolationsNoLocal, numEpisodes)
