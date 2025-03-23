import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


trainset = pd.read_csv('eval/benchmarks/Pakistan/data/Tuple30K/trainset.csv')
testset = pd.read_csv('eval/benchmarks/Pakistan/data/Tuple30K/testset.csv')

testset['GenerationTime'] = testset['GenerationTime'] + trainset['GenerationTime'].max()


data = pd.concat([trainset, testset])

print(data.head())


print(data.describe())