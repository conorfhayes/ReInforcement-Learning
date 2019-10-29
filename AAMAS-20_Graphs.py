import numpy as np
from mizani.breaks import log_breaks
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
                  'TLO Violations-Emissions': 'purple',
                  'TLO-1': 'blue',
                  'TLO-2': 'yellow',
                  'TLO-3': 'green',
                  'TLO-4': 'black',
                  'TLO-5': 'orange',
                  'TLO-6': 'grey'
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
            legend_text=element_text(size=6),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank())
            + guides(color=guide_legend(nrow=1)))

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
                  'TLO Violations-Emissions': 'purple',
                  'TLO-1': 'blue',
                  'TLO-2': 'yellow',
                  'TLO-3': 'green',
                  'TLO-4': 'black',
                  'TLO-5': 'orange',
                  'TLO-6': 'grey'
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
            legend_text=element_text(size=6),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank())
            + guides(color=guide_legend(nrow=1))
            + scale_y_continuous(trans = 'log10', labels = lambda l: ["10^%d" % math.log(v,10) for v in l] ))

    print(emissionsG)

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
                  'TLO Violations-Emissions': 'purple',
                  'TLO-1': 'blue',
                  'TLO-2': 'yellow',
                  'TLO-3': 'green',
                  'TLO-4': 'black',
                  'TLO-5': 'orange',
                  'TLO-6': 'grey'
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
            legend_text=element_text(size=6),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank())
            + guides(color=guide_legend(nrow=1))
            )

    print(violationsG)

Difference_hypervolume = pd.read_excel ("~/Desktop/Results/Difference_hypervolume.xls", sheet_name = "Run Results")
Difference_linear = pd.read_excel ("~/Desktop/Results/Difference_linear.xls", sheet_name = "Run Results")
Global_hypervolume = pd.read_excel ("~/Desktop/Results/Global_hypervolume.xls", sheet_name = "Run Results")
Global_linear = pd.read_excel ("~/Desktop/Results/Global_linear.xls", sheet_name = "Run Results")
TLO_Violations_Cost = pd.read_excel ("~/Desktop/Results/TLO_Output_DynamicThresholdV2.xls", sheet_name = "Run Results")
TLO_Violations_Emissions = pd.read_excel ("~/Desktop/Results/TLO_Output_Violations_Emissions.xls", sheet_name = "Run Results")
TLO_Violations_Cost_Violations = pd.read_excel ("~/Desktop/Results/TLO_Output_Cost_Violations.xls", sheet_name = "Run Results")

TLO1_df = pd.read_excel ("~/Desktop/Results/TLO_Output_TLO-1.xls", sheet_name = "Run Results")
TLO2_df = pd.read_excel ("~/Desktop/Results/TLO_Output_TLO-2.xls", sheet_name = "Run Results")
TLO3_df = pd.read_excel ("~/Desktop/Results/TLO_Output_TLO-3.xls", sheet_name = "Run Results")
TLO4_df = pd.read_excel ("~/Desktop/Results/TLO_Output_TLO-4.xls", sheet_name = "Run Results")
TLO5_df = pd.read_excel ("~/Desktop/Results/TLO_Output_TLO-5.xls", sheet_name = "Run Results")
TLO6_df = pd.read_excel ("~/Desktop/Results/TLO_Output_TLO-6.xls", sheet_name = "Run Results")

TLO_Global_CV_Cost_df = TLO_Violations_Cost_Violations[['Cost']]
TLO_Global_VE_Cost_df = TLO_Violations_Emissions[['Cost']]
TLO_Global_Cost_df = TLO_Violations_Cost[['Cost']]
Difference_Cost_Linear_df = Difference_linear[['Cost']]
Difference_Cost_Hypervolume_df = Difference_hypervolume[['Cost']]
Global_Cost_Linear_df = Global_linear[['Cost']]
Global_Cost_Hypervolume_df = Global_linear[['Cost']]

TLO1_Cost_df = TLO1_df[['Cost']]
TLO2_Cost_df = TLO2_df[['Cost']]
TLO3_Cost_df = TLO3_df[['Cost']]
TLO4_Cost_df = TLO4_df[['Cost']]
TLO5_Cost_df = TLO5_df[['Cost']]
TLO6_Cost_df = TLO6_df[['Cost']]

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

TLO1_Violations_df = TLO1_df[['Violations']]
TLO2_Violations_df = TLO2_df[['Violations']]
TLO3_Violations_df = TLO3_df[['Violations']]
TLO4_Violations_df = TLO4_df[['Violations']]
TLO5_Violations_df = TLO5_df[['Violations']]
TLO6_Violations_df = TLO6_df[['Violations']]

TLO_Global_CV_Emissions_df = TLO_Violations_Cost_Violations[['Emissions']]
TLO_Global_VE_Emissions_df = TLO_Violations_Emissions[['Emissions']]
TLO_Global_Emissions_df = TLO_Violations_Cost[['Emissions']]
Difference_Emissions_Linear_df = Difference_linear[['Emissions']]
Difference_Emissions_Hypervolume_df = Difference_hypervolume[['Emissions']]
Global_Emissions_Linear_df = Global_linear[['Emissions']]
Global_Emissions_Hypervolume_df = Global_linear[['Emissions']]

TLO1_Emissions_df = TLO1_df[['Emissions']]
TLO2_Emissions_df = TLO2_df[['Emissions']]
TLO3_Emissions_df = TLO3_df[['Emissions']]
TLO4_Emissions_df = TLO4_df[['Emissions']]
TLO5_Emissions_df = TLO5_df[['Emissions']]
TLO6_Emissions_df = TLO6_df[['Emissions']]
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

TLO1_Cost_array = []
for x in TLO1_Cost_df.values:
    TLO1_Cost_array.append(float (x))

TLO2_Cost_array = []
for x in TLO2_Cost_df.values:
    TLO2_Cost_array.append(float (x))

TLO3_Cost_array = []
for x in TLO3_Cost_df.values:
    TLO3_Cost_array.append(float (x))

TLO4_Cost_array = []
for x in TLO4_Cost_df.values:
    TLO4_Cost_array.append(float (x))

TLO5_Cost_array = []
for x in TLO5_Cost_df.values:
    TLO5_Cost_array.append(float (x))

TLO6_Cost_array = []
for x in TLO6_Cost_df.values:
    TLO6_Cost_array.append(float (x))

TLO1_Violations_array = []
for x in TLO1_Violations_df.values:
    TLO1_Violations_array.append(float (x))

TLO2_Violations_array = []
for x in TLO2_Violations_df.values:
    TLO2_Violations_array.append(float (x))

TLO3_Violations_array = []
for x in TLO3_Violations_df.values:
    TLO3_Violations_array.append(float (x))

TLO4_Violations_array = []
for x in TLO4_Violations_df.values:
    TLO4_Violations_array.append(float (x))

TLO5_Violations_array = []
for x in TLO5_Violations_df.values:
    TLO5_Violations_array.append(float (x))

TLO6_Violations_array = []
for x in TLO6_Violations_df.values:
    TLO6_Violations_array.append(float (x))


TLO1_Emissions_array = []
for x in TLO1_Emissions_df.values:
    TLO1_Emissions_array.append(float (x))

TLO2_Emissions_array = []
for x in TLO2_Emissions_df.values:
    TLO2_Emissions_array.append(float (x))

TLO3_Emissions_array = []
for x in TLO3_Emissions_df.values:
    TLO3_Emissions_array.append(float (x))

TLO4_Emissions_array = []
for x in TLO4_Emissions_df.values:
    TLO4_Emissions_array.append(float (x))

TLO5_Emissions_array = []
for x in TLO5_Emissions_df.values:
    TLO5_Emissions_array.append(float (x))

TLO6_Emissions_array = []
for x in TLO6_Emissions_df.values:
    TLO6_Emissions_array.append(float (x))

TLO_Global_VE_Emissions_array = []
for x in TLO_Global_VE_Emissions_df.values:
    TLO_Global_VE_Emissions_array.append(float (x))

TLO_Global_CV_Emissions_array = []
for x in TLO_Global_CV_Emissions_df.values:
    TLO_Global_CV_Emissions_array.append(float (x))

TLO_Global_CV_Emissions_array = []
for x in TLO_Global_CV_Emissions_df.values:
    TLO_Global_CV_Emissions_array.append(float (x))

TLO_Global_VE_Emissions_array = []
for x in TLO_Global_VE_Emissions_df.values:
    TLO_Global_VE_Emissions_array.append(float (x))

TLO_Global_Emissions_array = []
for x in TLO_Global_Emissions_df.values:
    TLO_Global_Emissions_array.append(float (x))

Difference_Emissions_Linear_array = []
for x in Difference_Emissions_Linear_df.values:
    Difference_Emissions_Linear_array.append(float (x))

Difference_Emissions_Hypervolume_array = []
for x in Difference_Emissions_Hypervolume_df.values:
    Difference_Emissions_Hypervolume_array.append(float (x))

Global_Emissions_Linear_array = []
for x in Global_Emissions_Linear_df.values:
    Global_Emissions_Linear_array.append(float (x))

Global_Emissions_Hypervolume_array = []
for x in Global_Emissions_Hypervolume_df.values:
    Global_Emissions_Hypervolume_array.append(float (x))

TLO_Global_CV_Cost_Average = list(split(TLO_Global_CV_Cost_array, int(span)))
TLO_Global_VE_Cost_Average = list(split(TLO_Global_VE_Cost_array, int(span)))
TLO_Global_Cost_Average = list(split(TLO_Global_Cost_array, int(span)))
Global_Cost_Linear_df_Average = list(split(Global_Cost_Linear_array, int(span)))
Global_Cost_Hypervolume_df_Average = list(split(Global_Cost_Hypervolume_array, int(span)))
Difference_Cost_Linear_df_Average = list(split(Difference_Cost_Linear_array, int(span)))
Difference_Cost_Hypervolume_df_Average = list(split(Difference_Cost_Hypervolume_array, int(span)))
TLO1_Cost_df_Average = list(split(TLO1_Cost_array, int(span)))
TLO2_Cost_df_Average = list(split(TLO2_Cost_array, int(span)))
TLO3_Cost_df_Average = list(split(TLO3_Cost_array, int(span)))
TLO4_Cost_df_Average = list(split(TLO4_Cost_array, int(span)))
TLO5_Cost_df_Average = list(split(TLO5_Cost_array, int(span)))
TLO6_Cost_df_Average = list(split(TLO6_Cost_array, int(span)))

TLO_Global_CV_Violations_Average = list(split(TLO_Global_CV_Violations_array, int(span)))
TLO_Global_VE_Violations_Average = list(split(TLO_Global_VE_Violations_array, int(span)))
TLO_Global_Violations_Average = list(split(TLO_Global_Violations_array, int(span)))
Global_Violations_Linear_df_Average = list(split(Global_Violations_Linear_array, int(span)))
Global_Violations_Hypervolume_df_Average = list(split(Global_Violations_Hypervolume_array, int(span)))
Difference_Violations_Linear_df_Average = list(split(Difference_Violations_Linear_array, int(span)))
Difference_Violations_Hypervolume_df_Average = list(split(Difference_Violations_Hypervolume_array, int(span)))
TLO1_Violations_df_Average = list(split(TLO1_Violations_array, int(span)))
TLO2_Violations_df_Average = list(split(TLO2_Violations_array, int(span)))
TLO3_Violations_df_Average = list(split(TLO3_Violations_array, int(span)))
TLO4_Violations_df_Average = list(split(TLO4_Violations_array, int(span)))
TLO5_Violations_df_Average = list(split(TLO5_Violations_array, int(span)))
TLO6_Violations_df_Average = list(split(TLO6_Violations_array, int(span)))

TLO_Global_CV_Emissions_Average = list(split(TLO_Global_CV_Emissions_array, int(span)))
TLO_Global_VE_Emissions_Average = list(split(TLO_Global_VE_Emissions_array, int(span)))
TLO_Global_Emissions_Average = list(split(TLO_Global_Emissions_array, int(span)))
Global_Emissions_Linear_df_Average = list(split(Global_Emissions_Linear_array, int(span)))
Global_Emissions_Hypervolume_df_Average = list(split(Global_Emissions_Hypervolume_array, int(span)))
Difference_Emissions_Linear_df_Average = list(split(Difference_Emissions_Linear_array, int(span)))
Difference_Emissions_Hypervolume_df_Average = list(split(Difference_Emissions_Hypervolume_array, int(span)))
TLO1_Emissions_df_Average = list(split(TLO1_Emissions_array, int(span)))
TLO2_Emissions_df_Average = list(split(TLO2_Emissions_array, int(span)))
TLO3_Emissions_df_Average = list(split(TLO3_Emissions_array, int(span)))
TLO4_Emissions_df_Average = list(split(TLO4_Emissions_array, int(span)))
TLO5_Emissions_df_Average = list(split(TLO5_Emissions_array, int(span)))
TLO6_Emissions_df_Average = list(split(TLO6_Emissions_array, int(span)))

TLO_Global_CV_Cost_Average = computeAverage(TLO_Global_CV_Cost_Average)
TLO_Global_VE_Cost_Average = computeAverage(TLO_Global_VE_Cost_Average)
TLO_Global_Cost_Average = computeAverage(TLO_Global_Cost_Average)
Difference_Cost_Linear_df_Average = computeAverage(Difference_Cost_Linear_df_Average)
Difference_Cost_Hypervolume_df_Average = computeAverage(Difference_Cost_Hypervolume_df_Average)
Global_Cost_Linear_df_Average = computeAverage(Global_Cost_Linear_df_Average)
Global_Cost_Hypervolume_df_Average = computeAverage(Global_Cost_Hypervolume_df_Average)
TLO1_Cost_Average = computeAverage(TLO1_Cost_df_Average)
TLO2_Cost_Average = computeAverage(TLO2_Cost_df_Average)
TLO3_Cost_Average = computeAverage(TLO3_Cost_df_Average)
TLO4_Cost_Average = computeAverage(TLO4_Cost_df_Average)
TLO5_Cost_Average = computeAverage(TLO5_Cost_df_Average)
TLO6_Cost_Average = computeAverage(TLO6_Cost_df_Average)

TLO_Global_CV_Violations_Average = computeAverage(TLO_Global_CV_Violations_Average)
TLO_Global_VE_Violations_Average = computeAverage(TLO_Global_VE_Violations_Average)
TLO_Global_Violations_Average = computeAverage(TLO_Global_Violations_Average)
Difference_Violations_Linear_df_Average = computeAverage(Difference_Violations_Linear_df_Average)
Difference_Violations_Hypervolume_df_Average = computeAverage(Difference_Violations_Hypervolume_df_Average)
Global_Violations_Linear_df_Average = computeAverage(Global_Violations_Linear_df_Average)
Global_Violations_Hypervolume_df_Average = computeAverage(Global_Violations_Hypervolume_df_Average)
TLO1_Violations_Average = computeAverage(TLO1_Violations_df_Average)
TLO2_Violations_Average = computeAverage(TLO2_Violations_df_Average)
TLO3_Violations_Average = computeAverage(TLO3_Violations_df_Average)
TLO4_Violations_Average = computeAverage(TLO4_Violations_df_Average)
TLO5_Violations_Average = computeAverage(TLO5_Violations_df_Average)
TLO6_Violations_Average = computeAverage(TLO6_Violations_df_Average)

TLO_Global_CV_Emissions_Average = computeAverage(TLO_Global_CV_Emissions_Average)
TLO_Global_VE_Emissions_Average = computeAverage(TLO_Global_VE_Emissions_Average)
TLO_Global_Emissions_Average = computeAverage(TLO_Global_Emissions_Average)
Difference_Emissions_Linear_df_Average = computeAverage(Difference_Emissions_Linear_df_Average)
Difference_Emissions_Hypervolume_df_Average = computeAverage(Difference_Emissions_Hypervolume_df_Average)
Global_Emissions_Linear_df_Average = computeAverage(Global_Emissions_Linear_df_Average)
Global_Emissions_Hypervolume_df_Average = computeAverage(Global_Emissions_Hypervolume_df_Average)
TLO1_Emissions_Average = computeAverage(TLO1_Emissions_df_Average)
TLO2_Emissions_Average = computeAverage(TLO2_Emissions_df_Average)
TLO3_Emissions_Average = computeAverage(TLO3_Emissions_df_Average)
TLO4_Emissions_Average = computeAverage(TLO4_Emissions_df_Average)
TLO5_Emissions_Average = computeAverage(TLO5_Emissions_df_Average)
TLO6_Emissions_Average = computeAverage(TLO6_Emissions_df_Average)

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

emissionsDataFrame_Hypervolume = pd.DataFrame({
                              'TLO Global': TLO_Global_Emissions_Average,
                              #'Global (+)': Global_Emissions_Linear_df_Average,
                              'Global (λ)': Global_Emissions_Hypervolume_df_Average,
                              #'Difference (+)': Difference_Emissions_Linear_df_Average,
                              'Difference (λ)': Difference_Emissions_Hypervolume_df_Average
                            })

emissionsDataFrame_Linear = pd.DataFrame({
                              'TLO Global': TLO_Global_Emissions_Average,
                              'Global (+)': Global_Emissions_Linear_df_Average,
                              #'Global (λ)': Global_Emissions_Hypervolume_df_Average,
                              'Difference (+)': Difference_Emissions_Linear_df_Average
                              #'Difference (λ)': Difference_Emissions_Hypervolume_df_Average
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
emissionsDataFrame_TLO = pd.DataFrame({
                              'TLO Cost-Violations': TLO_Global_CV_Emissions_Average,
                              #'Global (+)': Global_Cost_Linear_df_Average,
                              'TLO Violations-Cost': TLO_Global_Emissions_Average,
                              #'Difference (+)': Difference_Cost_Linear_df_Average,
                              'TLO Violations-Emissions':TLO_Global_VE_Violations_Average
                            })

fixedThresholdsCost = pd.DataFrame({
                              'TLO-1': TLO1_Cost_Average,
                              'TLO-2': TLO2_Cost_Average,
                              'TLO-3': TLO3_Cost_Average,
                              'TLO-4': TLO4_Cost_Average,
                              'TLO-5': TLO5_Cost_Average,
                              'TLO-6': TLO6_Cost_Average,
                             'TLO Global': TLO_Global_Cost_Average

                            })

fixedThresholdsViolations = pd.DataFrame({
                             'TLO-1': TLO1_Violations_Average,
                             'TLO-2': TLO2_Violations_Average,
                             'TLO-3': TLO3_Violations_Average,
                             'TLO-4': TLO4_Violations_Average,
                             'TLO-5': TLO5_Violations_Average,
                             'TLO-6': TLO6_Violations_Average,
                             'TLO Global': TLO_Global_Violations_Average
                            })

fixedThresholdsEmissions = pd.DataFrame({
                             'TLO-1': TLO1_Emissions_Average,
                             'TLO-2': TLO2_Emissions_Average,
                             'TLO-3': TLO3_Emissions_Average,
                             'TLO-4': TLO4_Emissions_Average,
                             'TLO-5': TLO5_Emissions_Average,
                             'TLO-6': TLO6_Emissions_Average,
                             'TLO Global': TLO_Global_Emissions_Average
                            })

fixedThresholdsCost_optimal = pd.DataFrame({
                              #'TLO-1': TLO1_Cost_Average,
                              #'TLO-2': TLO2_Cost_Average,
                              #'TLO-3': TLO3_Cost_Average,
                              'TLO-4': TLO4_Cost_Average,
                              'TLO-5': TLO5_Cost_Average,
                              'TLO-6': TLO6_Cost_Average
                            })

fixedThresholdsEmissions_optimal = pd.DataFrame({
                             #'TLO-1': TLO1_Emissions_Average,
                             #'TLO-2': TLO2_Emissions_Average,
                             #'TLO-3': TLO3_Emissions_Average,
                             'TLO-4': TLO4_Emissions_Average,
                             'TLO-5': TLO5_Emissions_Average,
                             'TLO-6': TLO6_Emissions_Average
                            })

fixedThresholdsViolations_optimal = pd.DataFrame({
                             #'TLO-1': TLO1_Violations_Average,
                             #'TLO-2': TLO2_Violations_Average,
                             #'TLO-3': TLO3_Violations_Average,
                             'TLO-4': TLO4_Violations_Average,
                             'TLO-5': TLO5_Violations_Average,
                             'TLO-6': TLO6_Violations_Average
                            })



#Hypervolume Graphs
#violationsGraph(violationsDataFrame_Hypervolume, 20000)
#costGraph(costDataFrame_Hypervolume, 20000)
#emissionsGraph(emissionsDataFrame_Hypervolume, 20000)

#Linear Graphs
#violationsGraph(violationsDataFrame_Linear, 20000)
#costGraph(costDataFrame_Linear, 20000)
#emissionsGraph(emissionsDataFrame_Linear, 20000)

#Varying Objective Ordering Graphs
#violationsGraph(violationsDataFrame_TLO, 20000)
#costGraph(costDataFrame_TLO, 20000)
#emissionsGraph(emissionsDataFrame_TLO, 20000)

#Fixed Thresholds Experimentation Graphs
violationsGraph(fixedThresholdsViolations, 20000)
costGraph(fixedThresholdsCost, 20000)
emissionsGraph(fixedThresholdsEmissions, 20000)

#violationsGraph(fixedThresholdsViolations_optimal, 20000)
#costGraph(fixedThresholdsCost_optimal, 20000)
#emissionsGraph(fixedThresholdsEmissions_optimal, 20000)
