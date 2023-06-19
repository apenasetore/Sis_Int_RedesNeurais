import pandas as pd
import numpy as np
import re 
import random

from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from sklearn import tree

import plotly.express as px
import plotly.graph_objects as go
import graphviz

def data_txt_to_matrix():

    f=open("treinoSV.txt", "r")
    L = f.readlines()
    m = []
    lp = []

    for l in L:
        pieces = re.split(",", l)
        lp = []
        for p in pieces:
            p = float(p)
            lp.append(p)
    
        m.append(lp)
    return m

def generate_data(m):

    classes = ["critico", "instavel", "potencialmente estavel", "estavel"]

    train_data = []
    res_train_data =[]
    test_data = []
    res_test_data = []
    n = []
    
    for i in range(0,800):
        n.append(i)
    
    corte = random.sample(n,k = 160)
    for line in m:
        if int(line[0]) in corte:
            test_data.append(line[3:6])
            #res_test_data.append(int(line[7]))
            res_test_data.append(classes[int(line[7])-1])
        else:
            train_data.append(line[3:6])
            #res_train_data.append(int(line[7]))
            res_train_data.append(classes[int(line[7])-1])
        

    return (train_data, res_train_data, 
            test_data,  res_test_data)
