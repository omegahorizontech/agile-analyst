from flask import Blueprint
from flask import render_template, redirect, url_for, jsonify, request

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
