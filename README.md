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
1)
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

2)
If you upgrade your Ubuntu operating system, you might have packages that have been moved.
One solution is to remove your virtual environment directory and rebuild it. Another is to do
the following:

<pre>
  <code>
    virtualenv --no-site-packages
  </code>
</pre>

This method might not work very well for scikit-learn, and I decided to destroy and rebuild
my virtualenv directory to make it work again.
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

## Future Paths
### Feature Words Only
  #### ML Approach
  This strategy would use only those words relevant to scoring a passage. Using 'wild' passages that have been scored, and only those words which matter for its score, a ML model would be trained. Our inputs would be relevant words only, meaning only true features would be regarded as features. The data would need to be given to the system in a form that has only true features and scores. The system would use these true features to formulate a model capable of accurately estimating and reproducing the scores assigned.

  #### AI Approach
  This strategy would use clever algorithms and data structures (hash tables, ordered dictionaries, etc.) to fully encode relevant words from the original R-emotion corpora. Whereas the conventional approach taken by speedy affect scorer has a complexity near O(nmc), where 'n' is the number of input words, 'm' is the average number of words in a corpus, and 'c' is the number of corpora, this approach would have a compute complexity closer to O(n), the amount of time would only scale with the inputs, since lookups and scoring would essentially be constant complexity with the hash tables and ordered dictionaries. The key challenge would be to balance the hash algorithms and dictionary sizes to accommodate just enough words without having hash crashes.

    ##### Requirements

    Using only feature words, and a more AI rather than ML based approach to building a fast scorer seems the ideal choice. It's simpler than the others and builds on what we already know about the problem. Basically, it involves building a programmatically simpler version of what we have that uses clever algorithms and optimizations. To begin this approach, we'll need csv or json objects of all the r-emotions corpora. We'll then run basic statistics on these corpora to determine appropriate hashing techniques, and the exact ratio of ordered dictionaries, hash addresses, and words. We will seek to reproduce 'speedy affect scorer' but do so with much lower compute overhead, and slightly more memory overhead, essentially eliminating the need to search the corpora altogether. Our benchmark will be something like a hash algorithm, a hash table lookup, and a check of a small ordered dictionary. If the word of interest is found, we get which corpora it's in, and which levels. If it's not found, we don't have an r-emotion word. Once we find the word, we use the corpus and tier its on to assign a partial score. We then carry on with classifying the remaining words in the sample.

    Begin with multiple CSV files, each containing a feature word in the first column and the parent corpus or corpora in the other columns. Based on the total size of these csv files (number of rows across all csv's), we determine an appropriate hash table size. We should also use appropriately sized ordered dictionaries. The ordered dictionaries will be stored at hash addresses. This ensures that we have a small enough number of hash addresses that we can manipulate them easily, and having 10 or 100 words per ordered dictionary will allow the hash table to be an equal factor smaller than it would otherwise be. The ordered dictionaries having a constant size, along with the use of hash tables, will allow fully constant look up times. The memory overhead for small dictionaries should be lower than for larger dictionaries, since there are fewer relationships between items in the dictionary the smaller it is. Smalller hash tables have a similar rationale.

    Our hashing algorithm will be designed so that every group of 100 words will occupy the same address, and stored at that address will be an ordered dictionary where the word and its corresponding r-emotion values are stored. 

### Horizontal Feature Segmentation
  Another strategy would be to segment the full feature space. Feature segments of ~10k words each would be given to distinct models for regression learning. Each of these models would be responsible for only a portion of the final score for a given estimation, and some method would be used to combine their predictions together to get the final score. This feature segmentation modeling could be used in tandem with multioutput regression, leading to groups of several models for each output. The main challenge here would be mediating the complexity of many models working in tandem, and finding a way to consistently reduce error in what are known to be obligate suboptimal predictors, given their usage of only a portion of the feature space. This would be similar in practice to an ensemble method, but instead of subsequent ensemble members focusing on increasingly difficult predictions, the predictions are of approximately equal difficulty for each sub-model.
