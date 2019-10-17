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
import plotnine as p9
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def computeAverage(array):
    newArray = []
    for i in array:
        average = (sum(i)) / 200
        #print("Values: ", i)
        #print("Average: ", average)
        newArray.append(average)

    return newArray

def costGraph(df, numEpisodes):
    x_axis = np.arange(0, numEpisodes, 200)
    print(x_axis)
    df['Counter'] = x_axis
    print(df[0:])
    x = df['Counter']
    df1 = df.melt(id_vars=["Counter"], var_name="Reward", value_name="Value")
    print(df1)

    color_dict = {'Global (+)': 'green',
                  'Global (λ)': 'green',
                  'TLO Global': 'red',
                  'Global': 'blue',
                  'Difference': 'blue',
                  'Difference (+)': 'blue',
                  'Difference (λ)': 'blue',
                  'TLO Cost-Violations': 'black',
                  'TLO Violations-Cost': 'orange',
                  'TLO Violations-Emissions': 'purple'
                  }

    costG = (ggplot(df1) +
            aes(x=df1['Counter'], y=df1['Value'], color = df1['Reward']) +
            geom_line(alpha=0.7, size=0.75) +
            scale_color_manual(values=color_dict) +
            #geom_line(aes(x='x', y=df['difference']), alpha=0.5, size=0.5, color= df['difference']) +
            #geom_line(aes(x='x', y=df['local']), alpha=0.5, size=0.5, color= df['local']) +
            #scale_x_continuous(lim = (0, max(x_axis)), breaks= range(0,len(x_axis)+ 5000, 5000)) +
            scale_y_continuous(limits = (2.5, max(df1['Value']))
                               , breaks = np.arange(2.5, max(df1['Value']) + 0.2, 0.2)
                               ) +
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

    color_dict = {'Global (+)': 'green',
                  'Global (λ)': 'green',
                  'TLO Global': 'red',
                  'Global': 'blue',
                  'Difference': 'blue',
                  'Difference (+)': 'blue',
                  'Difference (λ)': 'blue',
                  'TLO Cost-Violations': 'black',
                  'TLO Violations-Cost': 'orange',
                  'TLO Violations-Emissions': 'purple'
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
            theme(legend_position="bottom",
            legend_text=element_text(size=6),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank()))

    #print(emissionsG)

def violationsGraph(df, numEpisodes):
    x_axis = np.arange(0, numEpisodes, 200)
    print(x_axis)
    df['Counter'] = x_axis
    print(df[0:])
    x = df['Counter']
    df1 = df.melt(id_vars=["Counter"], var_name="Reward", value_name="Value")
    print(df1)

    color_dict = {'Global (+)': 'green',
                  'Global (λ)': 'green',
                  'TLO Global': 'red',
                  'Global': 'blue',
                  'Difference': 'blue',
                  'Difference (+)': 'blue',
                  'Difference (λ)': 'blue',
                  'TLO Cost-Violations': 'black',
                  'TLO Violations-Cost': 'orange',
                  'TLO Violations-Emissions': 'purple'
                  }

    violationsG = (ggplot(df1) +
            aes(x=df1['Counter'], y=df1['Value'], color = df1['Reward']) +
            geom_line(alpha=0.7, size=0.75) +
            scale_color_manual(values=color_dict) +
            #geom_line(aes(x='x', y=df['difference']), alpha=0.5, size=0.5, color= df['difference']) +
            #geom_line(aes(x='x', y=df['local']), alpha=0.5, size=0.5, color= df['local']) +
            #scale_x_continuous(lim = (0, max(x_axis)), breaks= range(0,len(x_axis)+ 5000, 5000)) +
            scale_y_continuous(limits = (1000, max(df1['Value']))) +
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

Difference_hypervolume = pd.read_excel ("~/Desktop/Results/Difference_hypervolume.xls", sheet_name = "Run Results")
Difference_linear = pd.read_excel ("~/Desktop/Results/Difference_linear.xls", sheet_name = "Run Results")
Global_hypervolume = pd.read_excel ("~/Desktop/Results/Global_hypervolume.xls", sheet_name = "Run Results")
Global_linear = pd.read_excel ("~/Desktop/Results/Global_linear.xls", sheet_name = "Run Results")
TLO_Violations_Cost = pd.read_excel ("~/Desktop/Results/TLO_Output_DynamicThresholdV2.xls", sheet_name = "Run Results")
TLO_Violations_Emissions = pd.read_excel ("~/Desktop/Results/TLO_Output_Violations_Emissions.xls", sheet_name = "Run Results")
TLO_Violations_Cost_Violations = pd.read_excel ("~/Desktop/Results/TLO_Output_Cost_Violations.xls", sheet_name = "Run Results")

TLO_Global_CV_Cost_df = TLO_Violations_Cost_Violations[['Cost']]
TLO_Global_VE_Cost_df = TLO_Violations_Emissions[['Cost']]
TLO_Global_Cost_df = TLO_Violations_Cost[['Cost']]
Difference_Cost_Linear_df = Difference_linear[['Cost']]
Difference_Cost_Hypervolume_df = Difference_hypervolume[['Cost']]
Global_Cost_Linear_df = Global_linear[['Cost']]
Global_Cost_Hypervolume_df = Global_linear[['Cost']]


#TLO_Global_Emissions_df = df_Emissions[['TLO_Global']]
#TLO_Global_Emissions_Linear_df = df_Emissions[['TLO_Global (+)']]
#Difference_Emissions_df = df_Emissions[['Difference']]
#Global_Emissions_df = df_Emissions[['Global']]

TLO_Global_CV_Violations_df = TLO_Violations_Cost_Violations[['Violations']]
TLO_Global_VE_Violations_df = TLO_Violations_Emissions[['Violations']]
TLO_Global_Violations_df = TLO_Violations_Cost[['Violations']]
Difference_Violations_Linear_df = Difference_linear[['Violations']]
Difference_Violations_Hypervolume_df = Difference_hypervolume[['Violations']]
Global_Violations_Linear_df = Global_linear[['Violations']]
Global_Violations_Hypervolume_df = Global_linear[['Violations']]


span = 20000 / 200

TLO_Global_VE_Violations_array = []
for x in TLO_Global_VE_Violations_df.values:
    TLO_Global_VE_Violations_array.append(float (x))

TLO_Global_CV_Violations_array = []
for x in TLO_Global_CV_Violations_df.values:
    TLO_Global_CV_Violations_array.append(float (x))

TLO_Global_CV_Cost_array = []
for x in TLO_Global_CV_Cost_df.values:
    TLO_Global_CV_Cost_array.append(float (x))

TLO_Global_VE_Cost_array = []
for x in TLO_Global_VE_Cost_df.values:
    TLO_Global_VE_Cost_array.append(float (x))

TLO_Global_Cost_array = []
for x in TLO_Global_Cost_df.values:
    TLO_Global_Cost_array.append(float (x))

Difference_Cost_Linear_array = []
for x in Difference_Cost_Linear_df.values:
    Difference_Cost_Linear_array.append(float (x))

Difference_Cost_Hypervolume_array = []
for x in Difference_Cost_Hypervolume_df.values:
    Difference_Cost_Hypervolume_array.append(float (x))

Global_Cost_Linear_array = []
for x in Global_Cost_Linear_df.values:
    Global_Cost_Linear_array.append(float(x))

Global_Cost_Hypervolume_array = []
for x in Global_Cost_Hypervolume_df.values:
    Global_Cost_Hypervolume_array.append(float(x))

TLO_Global_Violations_array = []
for x in TLO_Global_Violations_df.values:
    TLO_Global_Violations_array.append(float(x))

Difference_Violations_Linear_array = []
for x in Difference_Violations_Linear_df.values:
    Difference_Violations_Linear_array.append(float(x))

Difference_Violations_Hypervolume_array = []
for x in Difference_Violations_Hypervolume_df.values:
    Difference_Violations_Hypervolume_array.append(float(x))

Global_Violations_Linear_array = []
for x in Global_Violations_Linear_df.values:
    Global_Violations_Linear_array.append(float(x))

Global_Violations_Hypervolume_array = []
for x in Global_Violations_Hypervolume_df.values:
    Global_Violations_Hypervolume_array.append(float(x))

TLO_Global_CV_Cost_Average = list(split(TLO_Global_CV_Cost_array, int(span)))
TLO_Global_VE_Cost_Average = list(split(TLO_Global_VE_Cost_array, int(span)))
TLO_Global_Cost_Average = list(split(TLO_Global_Cost_array, int(span)))
Global_Cost_Linear_df_Average = list(split(Global_Cost_Linear_array, int(span)))
Global_Cost_Hypervolume_df_Average = list(split(Global_Cost_Hypervolume_array, int(span)))
Difference_Cost_Linear_df_Average = list(split(Difference_Cost_Linear_array, int(span)))
Difference_Cost_Hypervolume_df_Average = list(split(Difference_Cost_Hypervolume_array, int(span)))

TLO_Global_CV_Violations_Average = list(split(TLO_Global_CV_Violations_array, int(span)))
TLO_Global_VE_Violations_Average = list(split(TLO_Global_VE_Violations_array, int(span)))
TLO_Global_Violations_Average = list(split(TLO_Global_Violations_array, int(span)))
Global_Violations_Linear_df_Average = list(split(Global_Violations_Linear_array, int(span)))
Global_Violations_Hypervolume_df_Average = list(split(Global_Violations_Hypervolume_array, int(span)))
Difference_Violations_Linear_df_Average = list(split(Difference_Violations_Linear_array, int(span)))
Difference_Violations_Hypervolume_df_Average = list(split(Difference_Violations_Hypervolume_array, int(span)))

TLO_Global_CV_Cost_Average = computeAverage(TLO_Global_CV_Cost_Average)
TLO_Global_VE_Cost_Average = computeAverage(TLO_Global_VE_Cost_Average)
TLO_Global_Cost_Average = computeAverage(TLO_Global_Cost_Average)
Difference_Cost_Linear_df_Average = computeAverage(Difference_Cost_Linear_df_Average)
Difference_Cost_Hypervolume_df_Average = computeAverage(Difference_Cost_Hypervolume_df_Average)
Global_Cost_Linear_df_Average = computeAverage(Global_Cost_Linear_df_Average)
Global_Cost_Hypervolume_df_Average = computeAverage(Global_Cost_Hypervolume_df_Average)

TLO_Global_CV_Violations_Average = computeAverage(TLO_Global_CV_Violations_Average)
TLO_Global_VE_Violations_Average = computeAverage(TLO_Global_VE_Violations_Average)
TLO_Global_Violations_Average = computeAverage(TLO_Global_Violations_Average)
Difference_Violations_Linear_df_Average = computeAverage(Difference_Violations_Linear_df_Average)
Difference_Violations_Hypervolume_df_Average = computeAverage(Difference_Violations_Hypervolume_df_Average)
Global_Violations_Linear_df_Average = computeAverage(Global_Violations_Linear_df_Average)
Global_Violations_Hypervolume_df_Average = computeAverage(Global_Violations_Hypervolume_df_Average)



costDataFrame_Hypervolume = pd.DataFrame({
                              'TLO Global': TLO_Global_Cost_Average,
                              #'Global (+)': Global_Cost_Linear_df_Average,
                              'Global (λ)': Global_Cost_Hypervolume_df_Average,
                              #'Difference (+)': Difference_Cost_Linear_df_Average,
                              'Difference (λ)': Difference_Cost_Hypervolume_df_Average
                            })

costDataFrame_Linear = pd.DataFrame({
                              'TLO Global': TLO_Global_Cost_Average,
                              'Global (+)': Global_Cost_Linear_df_Average,
                              #'Global (λ)': Global_Cost_Hypervolume_df_Average,
                              'Difference (+)': Difference_Cost_Linear_df_Average
                              #'Difference (λ)': Difference_Cost_Hypervolume_df_Average
                            })

violationsDataFrame_Hypervolume = pd.DataFrame({
                              'TLO Global': TLO_Global_Violations_Average,
                              #'Global (+)': Global_Violations_Linear_df_Average,
                              'Global (λ)': Global_Violations_Hypervolume_df_Average,
                              #'Difference (+)': Difference_Violations_Linear_df_Average
                              'Difference (λ)': Difference_Violations_Hypervolume_df_Average
                              })

violationsDataFrame_Linear = pd.DataFrame({
                              'TLO Global': TLO_Global_Violations_Average,
                              'Global (+)': Global_Violations_Linear_df_Average,
                              #'Global (λ)': Global_Violations_Hypervolume_df_Average,
                              'Difference (+)': Difference_Violations_Linear_df_Average
                              #'Difference (λ)': Difference_Violations_Hypervolume_df_Average
                              })

costDataFrame_TLO = pd.DataFrame({
                              'TLO Cost-Violations': TLO_Global_CV_Cost_Average,
                              #'Global (+)': Global_Cost_Linear_df_Average,
                              'TLO Violations-Cost': TLO_Global_Cost_Average,
                              #'Difference (+)': Difference_Cost_Linear_df_Average,
                              'TLO Violations-Emissions':TLO_Global_VE_Cost_Average
                            })

violationsDataFrame_TLO = pd.DataFrame({
                              'TLO Cost-Violations': TLO_Global_CV_Violations_Average,
                              #'Global (+)': Global_Cost_Linear_df_Average,
                              'TLO Violations-Cost': TLO_Global_Violations_Average,
                              #'Difference (+)': Difference_Cost_Linear_df_Average,
                              'TLO Violations-Emissions':TLO_Global_VE_Violations_Average
                            })


#Hypervolume Graphs
violationsGraph(violationsDataFrame_Hypervolume, 20000)
costGraph(costDataFrame_Hypervolume, 20000)

#Linear Graphs
violationsGraph(violationsDataFrame_Linear, 20000)
costGraph(costDataFrame_Linear, 20000)

#Varying Objective Ordering Graphs
violationsGraph(violationsDataFrame_TLO, 20000)
costGraph(costDataFrame_TLO, 20000)
