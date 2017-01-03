from flask import jsonify
import requests, operator, math, json

# databases
from config.databases import affect_analysis

# mongo dependencies
from flask.ext.pymongo import ObjectId

api_ip = '0.0.0.0'
port = '7000'

def hello():
    return 'hello helpers'

def analyze_emotion_set(request_body):

    data = request_body
    endpoint = 'http://' + api_ip + ':' + port + '/helpers/analyze_emotion_set/emotion_ml/'
    r = requests.post(endpoint, json=data)
    return json.loads(r.content)

def save_record(collection_name, data):

    r = analyze_emotion_set(data)
    try:
        affect_analysis.db.create_collection(collection_name)
    except Exception as e:
        pass

    collection = affect_analysis.db[collection_name]
    collection.insert(r)
    return "Success"
