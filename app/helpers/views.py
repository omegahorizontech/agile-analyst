from flask import Blueprint
from flask import render_template, redirect, url_for, jsonify, request

from nltk.corpus import gutenberg
from nltk.corpus import nps_chat
from nltk.corpus import brown

import controllers

helpers = Blueprint('helpers', __name__)

@helpers.route('/')
def verify_api_01():
    response = controllers.hello()
    return response

'''
save_full_record
===
This method always does the entire emotion_ml set (400 as of 03-Jan-2016)
'''
@helpers.route('/save-record/<collection_name>/', methods=['POST'])
def save_full_record(collection_name):
    r = request.get_json()

    doc = r.get('doc')
    lang = r.get('lang')
    upper_bound = r.get('ub')
    lower_bound = r.get('lb')
    # TODO: Add Error Handling
    natural = r.get('natural')
    stemmer = r.get('stemmer')
    lemma = r.get('lemma')

    data = {
        "doc": doc,
        "lang": lang,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "natural": natural,
        "stemmer": stemmer,
        "lemma": lemma,
    }

    response = controllers.save_record(collection_name, data)
    return response


'''
save_full_records
===
This method always does the entire emotion_ml set (400 as of 03-Jan-2016)
'''
@helpers.route('/save-records/<collection_name>/', methods=['POST'])
def save_many_full_records(collection_name):
    r = request.get_json()

    docs = r.get('docs')
    lang = r.get('lang')
    upper_bound = r.get('ub')
    lower_bound = r.get('lb')
    # TODO: Add Error Handling
    natural = r.get('natural')
    stemmer = r.get('stemmer')
    lemma = r.get('lemma')

    for doc in docs:
        data = {
            "doc": doc,
            "lang": lang,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "natural": natural,
            "stemmer": stemmer,
            "lemma": lemma,
        }
        controllers.save_record(collection_name, data)

    return "Success"

'''
save_full_records_from_gutenberg
===
This method looks at gutenberg from NLTK and saves the records to use as
training/testing data for ML purposes
'''
@helpers.route('/save-gutenberg-records/<collection_name>/', methods=['POST'])
def save_full_records_from_gutenberg(collection_name):
    r = request.get_json()

    corpus_name = r.get('corpus_name')
    lang = r.get('lang')
    upper_bound = r.get('ub')
    lower_bound = r.get('lb')
    # TODO: Add Error Handling
    natural = r.get('natural')
    stemmer = r.get('stemmer')
    lemma = r.get('lemma')

    guten_sents = gutenberg.sents(corpus_name + '.txt')

    count = 0
    for sent in guten_sents:
        sent = ' '.join(sent)
        if (count % 50) == 0:
            print '===='
            print 'Processed record: ' + str(count)
            print '===='
        count = count + 1
        print sent
        data = {
            "doc": sent,
            "lang": lang,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "natural": natural,
            "stemmer": stemmer,
            "lemma": lemma,
        }
        controllers.save_record(collection_name, data)

    return "Success"

'''
save_full_records_from_nps_chat
===
This method looks at nps_chat from NLTK and saves the records to use as
training/testing data for ML purposes
'''
@helpers.route('/save-nps_chats-records/<collection_name>/', methods=['POST'])
def save_full_records_from_nps_chat(collection_name):
    r = request.get_json()

    corpus_name = r.get('corpus_name')
    lang = r.get('lang')
    upper_bound = r.get('ub')
    lower_bound = r.get('lb')
    # TODO: Add Error Handling
    natural = r.get('natural')
    stemmer = r.get('stemmer')
    lemma = r.get('lemma')

    chatroom = nps_chat.posts(corpus_name + '.xml')

    count = 0
    for post in chatroom:
        post = ' '.join(post)
        if (count % 50) == 0:
            print '===='
            print 'Processed record: ' + str(count)
            print '===='

        if post == 'JOIN':
            pass
        elif post == 'PART':
            pass
        else:
            print post
            data = {
                "doc": post,
                "lang": lang,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "natural": natural,
                "stemmer": stemmer,
                "lemma": lemma,
            }
            controllers.save_record(collection_name, data)

        count = count + 1

    return "Success"

'''
save_full_records_from_nps_chat
===
This method looks at brown corpus from NLTK and saves the records to use as
training/testing data for ML purposes
'''
@helpers.route('/save-brown-records/<collection_name>/', methods=['POST'])
def save_full_records_from_brown(collection_name):
    r = request.get_json()

    corpus_name = r.get('corpus_name')
    lang = r.get('lang')
    upper_bound = r.get('ub')
    lower_bound = r.get('lb')
    # TODO: Add Error Handling
    natural = r.get('natural')
    stemmer = r.get('stemmer')
    lemma = r.get('lemma')

    news_text_sent = brown.sents(categories=corpus_name)

    count = 0
    for sent in news_text_sent:
        sent = ' '.join(sent)
        if (count % 50) == 0:
            print '===='
            print 'Processed record: ' + str(count)
            print '===='
        count = count + 1
        print sent
        data = {
            "doc": sent,
            "lang": lang,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "natural": natural,
            "stemmer": stemmer,
            "lemma": lemma,
        }
        controllers.save_record(collection_name, data)

    return "Success"

'''
write_csv_from_json
===
Converts json records to csv file(s)
from mongo database (based on collection name)
'''
@helpers.route('/convert-gutenberg-records/<collection_name>/', methods=['POST'])
def write_csv_from_json_gutenberg(collection_name):
    r = request.get_json()

    corpus_name = r.get('corpus_name')
    use_json_sentence = r.get('use_json_sentence')
    controllers.write_csv_from_json(collection_name, corpus_name, use_json_sentence, '0')

    return "Success"

@helpers.route('/convert-nps_chats-records/<collection_name>/', methods=['POST'])
def write_csv_from_json_chat(collection_name):
    r = request.get_json()

    corpus_name = r.get('corpus_name')
    use_json_sentence = r.get('use_json_sentence')
    controllers.write_csv_from_json(collection_name, corpus_name, use_json_sentence, '1')

    return "Success"

@helpers.route('/convert-brown-records/<collection_name>/', methods=['POST'])
def write_csv_from_json_brown(collection_name):
    r = request.get_json()

    corpus_name = r.get('corpus_name')
    use_json_sentence = r.get('use_json_sentence')
    controllers.write_csv_from_json(collection_name, corpus_name, use_json_sentence, '2')

    return "Success"
