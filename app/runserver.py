from app import app
import sys
from helpers.views import helpers
from ml_models.views import ml_models

app.register_blueprint(helpers, url_prefix='/helpers')
app.register_blueprint(ml_models, url_prefix='/ml_models')

# Sets the port, or defaults to 80
if (len(sys.argv) > 1):
    port = int(sys.argv[1])
else:
    port=80

app.run(debug=True, host='0.0.0.0', port=port)
