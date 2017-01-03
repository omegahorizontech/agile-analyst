
from app import app

from flask.ext.pymongo import PyMongo, MongoClient

# Affective Computing Databases
app.config['ANALYSIS_DBNAME'] = 'affect-analysis'
affect_analysis = PyMongo(app, config_prefix='ANALYSIS')
