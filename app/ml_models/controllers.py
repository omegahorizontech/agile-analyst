from flask import jsonify
import requests, operator, math, json, csv, os

from sklearn.tree import DecisionTreeClassifier as DTC
import pandas as pd
import numpy as np

def hello():
    return 'hello machines'
