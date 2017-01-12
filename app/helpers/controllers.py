from flask import jsonify
import requests, operator, math, json, csv, os

from nltk.corpus import gutenberg
from nltk.corpus import nps_chat

from datetime import datetime

# databases
from config.databases import affect_analysis

# mongo dependencies
from flask.ext.pymongo import ObjectId


utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
utc_datetime = datetime.utcnow()
api_ip = '0.0.0.0'
port = '7000'

def hello():
    return 'hello helpers'

def analyze_emotion_set(request_body):

    data = request_body
    endpoint = 'http://' + api_ip + ':' + port + '/helpers/analyze_emotion_set/all_emotions/'
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

def write_csv_from_json(collection_name, corpus_name, use_json_sentence, corpus_type):

    corpus_sents = []
    if corpus_type == '0':
        corpus_sents = gutenberg.sents(corpus_name + '.txt')
    if corpus_type == '1':
        corpus_sents = nps_chat.posts(corpus_name + '.xml')


    print "===="
    print "Writing CSV file from: '" + collection_name + "'"
    print "==== Total records: " + str(affect_analysis.db[collection_name].count()) + " ===="
    csv_file = open(os.path.dirname(__file__) + '/../../data/' + collection_name + '(' + utc + ')' + '.csv', 'w')
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    collection = affect_analysis.db[collection_name]
    cursor = collection.find()
    emotion_list = ['sample_doc'] # This is the first coloumn header, more are appended
    for i in range(cursor.count()):
        if (i % 50) == 0:
            print 'Processed record: ' + str(i)
        emotion_row_scores = []
        sample_sent = ''
        if use_json_sentence == '1':
            sample_sent = cursor[i]['doc']
        if use_json_sentence == '0':
            sample_sent = ' '.join(corpus_sents[i])
        emotion_row_scores.append(sample_sent)
        for j in range(len(cursor[i]['emotion_set'])):
            if i < 1:
                emotion_list.append(cursor[i]['emotion_set'][j]['emotion'])
                if (j + 1) == len(cursor[i]['emotion_set']):
                    # CSV Write Head
                    csv_writer.writerow(emotion_list)
            emotion_score = cursor[i]['emotion_set'][j]['normalized_r_score']
            emotion_row_scores.append(emotion_score)
        # CSV Write Data Row
        csv_writer.writerow(emotion_row_scores)
    csv_file.close()
    print "===="
    print "Finished CSV file from: '" + collection_name + "'"
    print "===="

    return "Success"
