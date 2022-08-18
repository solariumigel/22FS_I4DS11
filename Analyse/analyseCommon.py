from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def CountOfCromosoneFigure(data):
    # roupby terminal exon + direction oder Chromoson
    groupby = data.groupby("terminal exon").agg(CountCleavageSite=('cleavage site', 'size'))
    sns.kdeplot(groupby["CountCleavageSite"], shade = True, color="blue")
    plt.show()
    plt.figure()

def AmountOfCGInSequence(data):
    data = AppendCountOfCG(data)
    sns.kdeplot(data["CountOfCG"], shade = True, color="blue")
    plt.show()
    plt.figure()
    
def DencityOfColumnPlot(data, column, function=lambda x : x):
    data = function(data)
    sns.kdeplot(data[column], shade = True, color="blue")
    plt.show()
    plt.figure()

def DencityOfColumnPlotByDirection(data, column, directionColumn, usageText, function=lambda x : x, levels=10):
    data = function(data)
    
    sns.kdeplot(data.loc[data[directionColumn] == '+'][column], shade = True, color="green", label='Positiver Strang', levels=levels)
    ax = sns.kdeplot(data.loc[data[directionColumn] == '-'][column], shade = True, color="blue", label="Negativer Strang", levels=levels)
    ax.legend(loc="upper center")
    ax.set_xlabel('{}, % Verwendung für mRNA'.format(usageText))
    ax.set_ylabel('density in {} levels'.format(levels))
    plt.show()
    plt.figure()
    
def ScatterPlot(data, xcolumn, ycolumn):
    sns.scatterplot(x=xcolumn,y=ycolumn,data=data,
               legend="full")
    plt.show()
    plt.figure()

def AppendCountOfCG(data):
    data["CountOfCG"] = data["sequence"].str.count("CG")
    return data

def Crosstable(data, indexColumn, columnsColumn):
    print(pd.crosstab(index=data[indexColumn], columns=data[columnsColumn]).to_string())

def PlotAnzahlCleavageSitesByChromosome(data):
    groupby = data.groupby('chromosome')
    groupby.size().plot(kind = "bar", ylabel="Anzahl Cleavage Site")
    plt.show()
    plt.figure()

def PlotBar(data, groupbyColumn, label):
    groupby = data.groupby(groupbyColumn)
    groupby.size().plot(kind = 'bar', ylabel=label)
    plt.show()
    plt.figure()

def PlotAnzahlTerminalExonsByChromosome(data):
    data = data.drop_duplicates(['chromosome', 'terminal exon'])
    groupby = data.groupby(['chromosome'])
    groupby.size().plot(kind = "bar", ylabel="Anzahl Terminal Exons")
    plt.show()
    plt.figure()

def PrintLenOfDistinct(data, dupilcateColumn):
    dataLength = len(data)
    uniqueLength = len(data.drop_duplicates(dupilcateColumn))
    print("all: {}".format(dataLength))
    print("Len distinct of {}: {}".format(dupilcateColumn, uniqueLength))
    print("Percent: {}%".format(uniqueLength / dataLength * 100))