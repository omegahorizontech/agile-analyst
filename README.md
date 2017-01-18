# agile-analyst
A natural language processing (NLP) project which makes use of machine learning to understand affect.

# Getting Started
* First, install virtualenv if not done so already -- https://virtualenv.pypa.io/en/latest/installation.html(https://virtualenv.pypa.io/en/latest/installation.html)
* Then, run this command:
<pre>
  <code>
    $ virtualenv venv
  </code>
</pre>
* Next, activate the virtual environment (make sure you get the'.'):
<pre>
  <code>
    $ . venv/bin/activate
  </code>
</pre>
* Last, install the requirements with pip:
<pre>
  <code>
    $ pip install -r requirements.txt
  </code>
</pre>

# Start database - if it is not already running
_From a terminal, start mongo:_
<pre>
  <code>
    mongod
  </code>
</pre>

# Run the application
<pre>
  <code>
    python app/runserver.py 5000
  </code>
</pre>

# Known hic-ups
You might need to install LAPACK (Linear Algebra Package) to solve errors like this one:
*numpy.distutils.system_info.NotFoundError: no lapack/blas resources found*

On Ubuntu 14.04, this solved that error:
<pre>
  <code>
    sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
  </code>
</pre>

Then, this installing scipy with pip should work:

<pre>
  <code>
    pip install scipy
  </code>
</pre>

# Stats for processing/labeling corpora
Re-running any of the data is requires processing time, but the CSV output is saved in the data directory for convenience. The statistics for my computer are listed below.

Running ubuntu 14.04 LTS
Memory: 15.6
Processor: AMD FX(tm)-9370 Eight-Core Processor × 8
Graphics: GeForce GTX 760/PCIe/SSE2
OS type: 64-bit

Running a single processor core: 

* austen-sense.txt - 10.37hr (phase 1 JSON) - 12.60hr (phase 2 CSV)
* milton-paradise.txt	- 3.06hr (phase 1 JSON) -	4.36hr (phase 2 CSV)
* shakespeare-macbeth.txt	- 5.95hr (phase 1 JSON) -	4.40hr (phase 2 CSV)
* 10-19-20s_706posts.xml - 0.69hr	(phase 1 JSON) -	1.10hr (phase 2 CSV)
* 10-19-30s_705posts.xml - 0.74hr	(phase 1 JSON) -	1.19hr (phase 2 CSV)
* 10-19-40s_686posts.xml - 0.73hr (phase 1 JSON) - 1.17hr (phase 2 CSV)
* 10-19-adults_706posts.xml	- 0.79hr (phase 1 JSON) - 1.19hr (phase 2 CSV)


# Requirements

* Flask==0.10.1
* Flask-Cors==2.1.0
* itsdangerous==0.24
* Jinja2==2.8
* MarkupSafe==0.23
* six==1.10.0
* Werkzeug==0.11.3
* wheel==0.24.0
* requests==2.10.0
* Flask-PyMongo==0.3.1
* pymongo==2.9.3
* scipy==0.18.1
* numpy==1.11.1
* scikit-learn==0.18.1
* pandas==0.18.1

# License
MIT
