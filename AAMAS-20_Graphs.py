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

    color_dict = {'Global (+)': 'red',
                  'Global (λ)': 'orange',
                  'TLO Global': 'green',
                  'Global': 'blue',
                  'Difference': 'yellow',
                  'Difference (+)': 'brown',
                  'Difference (λ)': 'pink'
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

    color_dict = {'Global (+)': 'red',
                  'Global (λ)': 'orange',
                  'TLO Global': 'green',
                  'Global': 'blue',
                  'Difference': 'yellow',
                  'Difference (+)': 'brown',
                  'Difference (λ)': 'pink'
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

    print(emissionsG)

def violationsGraph(df, numEpisodes):
    x_axis = np.arange(0, numEpisodes, 200)
    print(x_axis)
    df['Counter'] = x_axis
    print(df[0:])
    x = df['Counter']
    df1 = df.melt(id_vars=["Counter"], var_name="Reward", value_name="Value")
    print(df1)

    color_dict = {'Global (+)': 'red',
                  'Global (λ)': 'orange',
                  'TLO Global': 'green',
                  'Global': 'blue',
                  'Difference': 'yellow',
                  'Difference (+)': 'brown',
                  'Difference (λ)': 'pink'
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

df_Cost = pd.read_excel ("~/Desktop/PhD/2019/DEED/TLO Results/Cost_Exp_3.xlsx")
#df_Emissions = pd.read_excel (r'Path where the Excel file is stored\File name.xlsx')
df_Violations = pd.read_excel ('~/Desktop/PhD/2019/DEED/TLO Results/Violations_Exp_3.xlsx')

TLO_Global_Cost_df = df_Cost[['TLO Global']]
Difference_Cost_Linear_df = df_Cost[['Difference (+)']]
Difference_Cost_Hypervolume_df = df_Cost[['Difference (λ)']]
Difference_Cost_df = df_Cost[['Difference']]
Global_Cost_Linear_df = df_Cost[['Global (+)']]
Global_Cost_Hypervolume_df = df_Cost[['Global (λ)']]
Global_Cost_df = df_Cost[['Global']]


#TLO_Global_Emissions_df = df_Emissions[['TLO_Global']]
#TLO_Global_Emissions_Linear_df = df_Emissions[['TLO_Global (+)']]
#Difference_Emissions_df = df_Emissions[['Difference']]
#Global_Emissions_df = df_Emissions[['Global']]

TLO_Global_Violations_df = df_Violations[['TLO Global']]
Difference_Violations_Linear_df = df_Violations[['Difference (+)']]
Difference_Violations_Hypervolume_df = df_Violations[['Difference (λ)']]
Difference_Violations_df = df_Violations[['Difference']]
Global_Violations_Linear_df = df_Violations[['Global (+)']]
Global_Violations_Hypervolume_df = df_Violations[['Global (λ)']]
Global_Violations_df = df_Violations[['Global']]


span = 20000 / 200


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

Difference_Cost_array = []
for x in Difference_Cost_df.values:
    Difference_Cost_array.append(float (x))

Global_Cost_array = []
for x in Global_Cost_df.values:
    Global_Cost_array.append(float (x))

TLO_Global_Violations_array = []
for x in TLO_Global_Violations_df.values:
    TLO_Global_Violations_array.append(float(math.log(x)))

Difference_Violations_Linear_array = []
for x in Difference_Violations_Linear_df.values:
    Difference_Violations_Linear_array.append(float(math.log(x)))

Difference_Violations_Hypervolume_array = []
for x in Difference_Violations_Hypervolume_df.values:
    Difference_Violations_Hypervolume_array.append(float(math.log(x)))

Global_Violations_Linear_array = []
for x in Global_Violations_Linear_df.values:
    Global_Violations_Linear_array.append(float(math.log(x)))

Global_Violations_Hypervolume_array = []
for x in Global_Violations_Hypervolume_df.values:
    Global_Violations_Hypervolume_array.append(float(math.log(x)))

Difference_Violations_array = []
for x in Difference_Violations_df.values:
    Difference_Violations_array.append(float(math.log(x)))

Global_Violations_array = []
for x in Global_Violations_df.values:
    Global_Violations_array.append(float(math.log(x)))



TLO_Global_Cost_Average = list(split(TLO_Global_Cost_array, int(span)))
Global_Cost_Linear_df_Average = list(split(Global_Cost_Linear_array, int(span)))
Global_Cost_Hypervolume_df_Average = list(split(Global_Cost_Hypervolume_array, int(span)))
Difference_Cost_Linear_df_Average = list(split(Difference_Cost_Linear_array, int(span)))
Difference_Cost_Hypervolume_df_Average = list(split(Difference_Cost_Hypervolume_array, int(span)))
Global_Cost_df_Average = list(split(Global_Cost_array, int(span)))
Difference_Cost_df_Average = list(split(Difference_Cost_array, int(span)))

TLO_Global_Violations_Average = list(split(TLO_Global_Violations_array, int(span)))
Global_Violations_Linear_df_Average = list(split(Global_Violations_Linear_array, int(span)))
Global_Violations_Hypervolume_df_Average = list(split(Global_Violations_Hypervolume_array, int(span)))
Difference_Violations_Linear_df_Average = list(split(Difference_Violations_Linear_array, int(span)))
Difference_Violations_Hypervolume_df_Average = list(split(Difference_Violations_Hypervolume_array, int(span)))
Global_Violations_df_Average = list(split(Global_Violations_array, int(span)))
Difference_Violations_df_Average = list(split(Difference_Violations_array, int(span)))


TLO_Global_Cost_Average = computeAverage(TLO_Global_Cost_Average)
Difference_Cost_Linear_df_Average = computeAverage(Difference_Cost_Linear_df_Average)
Difference_Cost_Hypervolume_df_Average = computeAverage(Difference_Cost_Hypervolume_df_Average)
Global_Cost_Linear_df_Average = computeAverage(Global_Cost_Linear_df_Average)
Global_Cost_Hypervolume_df_Average = computeAverage(Global_Cost_Hypervolume_df_Average)
Global_Cost_df_Average = computeAverage(Global_Cost_df_Average)
Difference_Cost_df_Average = computeAverage(Difference_Cost_df_Average)

TLO_Global_Violations_Average = computeAverage(TLO_Global_Violations_Average)
Difference_Violations_Linear_df_Average = computeAverage(Difference_Violations_Linear_df_Average)
Difference_Violations_Hypervolume_df_Average = computeAverage(Difference_Violations_Hypervolume_df_Average)
Global_Violations_Linear_df_Average = computeAverage(Global_Violations_Linear_df_Average)
Global_Violations_Hypervolume_df_Average = computeAverage(Global_Violations_Hypervolume_df_Average)
Global_Violations_df_Average = computeAverage(Global_Violations_df_Average)
Difference_Violations_df_Average = computeAverage(Difference_Violations_df_Average)



costDataFrame = pd.DataFrame({
                              'TLO Global': TLO_Global_Cost_Average,
                              'Global (+)': Global_Cost_Linear_df_Average,
                              'Global (λ)': Global_Cost_Hypervolume_df_Average,
                              #'Difference (+)': Difference_Cost_Linear_df_Average,
                              #'Difference (λ)': Difference_Cost_Hypervolume_df_Average,
                              'Global': Global_Cost_df_Average
                              #'Difference': Difference_Cost_df_Average
                            })

violationsDataFrame = pd.DataFrame({
                              'TLO Global': TLO_Global_Violations_Average,
                              'Global (+)': Global_Violations_Linear_df_Average,
                              'Global (λ)': Global_Violations_Hypervolume_df_Average,
                              #'Difference (+)': Difference_Violations_Linear_df_Average,
                              #'Difference (λ)': Difference_Violations_Hypervolume_df_Average,
                              'Global': Global_Violations_df_Average
                              #'Difference': Difference_Violations_df_Average
                              })





violationsGraph(violationsDataFrame, 20000)
costGraph(costDataFrame, 20000)
