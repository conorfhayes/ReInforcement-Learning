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


def powerDemandGraph(powerDemand):
    x_axis = np.arange(0, 24, 1)
    df = pd.DataFrame(powerDemand)
    df['powerDemand'] = powerDemand
    df['Demand'] = "Demand"
    print(x_axis)
    df['Hour'] = x_axis
    #print(df[0:])
    #x = df['Counter']
    #df1 = df.melt(id_vars=["Counter"], var_name="Reward", value_name="Value")
    #df = df.melt(id_vars=["Hour"], var_name="Demand", value_name="Value")
    print(df)

    color_dict = {'Demand': 'blue'}

    costG = (ggplot(df) +
            aes(x=df['Hour'], y=df['powerDemand'], color = df['Demand']) +
            geom_line(alpha=0.7, size=0.75) +
            scale_color_manual(values=color_dict) +
            scale_x_continuous(limits=(0,23)) +
            scale_y_continuous(limits=(0, 2500)) +
            ylab(" Power (MW) ") +
            xlab(" Time (Hours)  ") +
            ggtitle(" ") +
            theme_matplotlib() +
            theme(axis_text_y = element_text(size =6)) +
            theme(axis_text_x=element_text(size=6)) +
            theme(legend_position="top",
            legend_text=element_text(size=8),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank()))
    print(costG)

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
                  'Difference (λ)': 'orange',
                  'Global (+)': 'red',
                  'TLO Global (+)': 'green',
                  'TLO Difference (+)': 'blue',
                  'Local (+)': 'black',
                  'Difference (+)': 'orange'
                  }

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
                  'Difference (λ)': 'orange',
                  'Global (+)': 'red',
                  'TLO Global (+)': 'green',
                  'TLO Difference (+)': 'blue',
                  'Local (+)': 'black',
                  'Difference (+)': 'orange'
                  }

    emissionsG = (ggplot(df1) +
            aes(x=df1['Counter'], y=df1['Value'], color = df1['Reward']) +
            geom_line(alpha=0.7, size=0.75) +
            scale_color_manual(values=color_dict) +
            #geom_line(aes(x='x', y=df['difference']), alpha=0.5, size=0.5, color= df['difference']) +
            #geom_line(aes(x='x', y=df['local']), alpha=0.5, size=0.5, color= df['local']) +
            #scale_x_continuous(lim = (0, max(x_axis)), breaks= range(0,len(x_axis)+ 5000, 5000)) +
            #scale_y_continuous(lim = (2.5, max(df1['Value'])), breaks = np.arange(2.5, max(df1['Value']) + 0.2, 0.2)) +
            ylab(" Emissions ($ x 10^5) ") +
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
                  'Difference (λ)': 'orange',
                  'Global (+)': 'red',
                  'TLO Global (+)': 'green',
                  'TLO Difference (+)': 'blue',
                  'Local (+)': 'black',
                  'Difference (+)': 'orange'
                  }

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


df_TLO_Cost1 = pd.read_excel("Results\TLO DEED\DiscountFactor1\TLO_DEED_Problem_AllRewards_Cost_Hypervolume_DiscountFactor1.xlsx")
df_TLO_Cost75 = pd.read_excel("Results\TLO DEED\DiscountFactor0.75\TLO_DEED_Problem_AllRewards_Cost_Hypervolume_DiscountFactor0.75.xlsx")
df_Hypervolume_Cost = pd.read_excel("Results\DEED\Hypervolume\DEED_Problem_AllRewards_Cost_Hypervolume.xlsx")
df_Linear_Cost = pd.read_excel("Results\DEED\Linear\DEED_Problem_AllRewards_Cost_Linear.xlsx")

df_TLO_Violations1 = pd.read_excel("Results\TLO DEED\DiscountFactor1\TLO_DEED_Problem_AllRewards_Violations_Hypervolume_DiscountFactor1.xlsx")
df_TLO_Violations75 = pd.read_excel("Results\TLO DEED\DiscountFactor0.75\TLO_DEED_Problem_AllRewards_Violations_Hypervolume_DiscountFactor0.75.xlsx")
df_Hypervolume_Violations = pd.read_excel("Results\DEED\Hypervolume\DEED_Problem_AllRewards_Violations_Hypervolume.xlsx")
df_Linear_Violations = pd.read_excel("Results\DEED\Linear\DEED_Problem_AllRewards_Violations_Linear.xlsx")

df_TLO_Emissions1 = pd.read_excel("Results\TLO DEED\DiscountFactor1\TLO_DEED_Problem_AllRewards_Emissions_Hypervolume_DiscountFactor1 - Copy.xlsx")

powerDemand = [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
                1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184]

powerSinDemand = [1036, 1776, 2150, 1776, 1036, 1036, 1776, 2150, 1776, 1036, 1036, 1776, 2150, 1776, 1036, 1036, 1776, 2150, 1776, 1036, 1036, 1776, 2150, 1776]

#emissionsGraph(df_TLO_Emissions1, numEpisodes)
powerDemandGraph(powerDemand)
powerDemandGraph(powerSinDemand)


costGraph(df_TLO_Cost1, numEpisodes)
violationsGraph(df_TLO_Violations1, numEpisodes)

costGraph(df_TLO_Cost75, numEpisodes)
violationsGraph(df_TLO_Violations75, numEpisodes)

costGraph(df_Hypervolume_Cost, numEpisodes)
violationsGraph(df_Hypervolume_Violations, numEpisodes)

costGraph(df_Linear_Cost, numEpisodes)
violationsGraph(df_Linear_Violations, numEpisodes)




