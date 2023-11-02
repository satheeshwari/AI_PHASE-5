# AI_PHASE-5
Data set link:https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis


# Project Overview
This project focuses on conducting market basket analysis to gain insights into customer purchasing patterns. Market basket analysis is a technique that identifies associations between products frequently purchased together, helping businesses make informed decisions about product placement, promotions, and inventory management.

## DATA PREPROCESSING

# import the dataset

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("DATASET.csv")
print(data) 

# display the first few rows of the dataset to get an overview
print(data.head(10))

# Get basic statistics about the dataset
summary=data.describe()
print(summary)

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# If there are missing values, you may choose to drop rows with missing data
data.dropna()

# Check for duplicates based on all columns
duplicates = data.duplicated()
print("Number of duplicate rows:", duplicates.sum())

# Remove duplicate rows
data.drop_duplicates()


## ASSOCIATION ANALYSIS TECHNQUIES

# use Apriori algorithm in data mining using association rules
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules
transactions=data.groupby(["CustomerID","Itemname"]).sum().unstack().reset_index().set_index("CustomerID")
print(transactions)

# use Apriori algorithm in data mining using association rules for few values
print(transactions.head(10))

## DATA VISUALISATION

# Bar chart for top 10 sold items in dataset
Item_distr=data.groupby(by="Itemname").size().reset_index(name="Frequency").sort_values(by="Frequency", ascending=False).head(10)
bars=Item_distr["Itemname"]
height=Item_distr["Frequency"]
x_pos=np.arange(len(bars))
plt.figure(figsize=(16,9))
plt.bar(x_pos, height, color=(0.2,0.3,0.5,0.5))
plt.title("Top 10 sold item")
plt.xlabel("Item names")
plt.ylabel('numer of quantity sold')
plt.xticks(x_pos, bars)
plt.show()

## CONCLUSION
In conclusion, the market basket analysis conducted in this project has provided valuable insights into customer purchasing behaviors. By analyzing transaction data and applying association rule mining techniques, we have uncovered meaningful patterns and relationships among products.
 The findings and results presented in this analysis offer a roadmap for making data-informed decisions in various industries, including retail, e-commerce, and beyond. As we continue to evolve and generate more complex datasets, the significance of market basket analysis in uncovering hidden relationships among items will only grow.
We encourage businesses and analysts to use these insights to create more effective strategies and improve the overall shopping experience for customers. In an increasingly competitive landscape, understanding market basket dynamics is a valuable asset for any organization.
