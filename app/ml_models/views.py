from flask import Blueprint
from flask import render_template, redirect, url_for, jsonify, request

import controllers

ml_models = Blueprint('ml_models', __name__)

@ml_models.route('/')
def verify_api_01():
    response = controllers.hello()
    return response
