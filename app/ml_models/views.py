from flask import Blueprint
from flask import render_template, redirect, url_for, jsonify, request

import controllers

ml_models = Blueprint('ml_models', __name__)

@ml_models.route('/')
def verify_api_01():
    response = controllers.hello()
    return response

@ml_models.route('/prepare')
def prep_data():
    response = controllers.prepare_data()
    return response

@ml_models.route('/train')
def train_model():
    response = controllers.train_model()
    return response

@ml_models.route('/validate')
def validate_model():
    response = controllers.validate_model()
    return response

@ml_model.route('/simple_score')
def simple_score():
    r = request.get_json()
    sample = r.get('sample')

    response = controllers.simple_score(sample)

    return response
