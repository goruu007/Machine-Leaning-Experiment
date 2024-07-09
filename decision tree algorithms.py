print("GauravRaikwar (0901AI223D04)")


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # Import accuracy_score function import matplotlib.pyplot as plt
 
dataset = pd.read_csv("drug200.csv") dataset.head(10)
