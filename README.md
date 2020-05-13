# Semantic Rule Model Browser
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This Python experimental command line tool has been developed for the purposes of my Master's thesis regarding Psychological phenomena influencing perception of business rules in business practice. Series of experiments has been designed based on the outcomes of this tool.

It can be used to mine association rule based decision models from data using the [pyARC](https://github.com/jirifilip/pyARC) package. These rules are afterwards introduced to the concept of semantic coherence.

Similarity between two individual words is obtained using the [gensim](https://radimrehurek.com/gensim/) library with pre-trained word2vec word embedding models by

```
Kutuzov, A., Fares, M., Oepen, S. & Velldal, E. (2017). Word vectors, reuse, and replicability:
Towards a community repository of large-text resources, In Proceedings of the 58th
Conference on Simulation and Modelling. Linköping University Electronic Press.
```

Exact models used may be downloaded from these URLs respectively and are necessary to use the tool:

- EN: [http://vectors.nlpl.eu/repository/20/200.zip](http://vectors.nlpl.eu/repository/20/200.zip)
- CZ: [http://vectors.nlpl.eu/repository/20/37.zip](http://vectors.nlpl.eu/repository/20/37.zip)

Semantic coherence of a rule is afterwards determined based on the proposed principles by

```
Gabriel, A., Paulheim, H. & Janssen, F. (2014). Learning semantically coherent rules.
DMNLP@ PKDD/ECML.
```

using a weighted heuristic.

`pyARC` library depends on the [fim](http://www.borgelt.net/pyfim.html) package for frequent item set mining. This library needs to be present on your system as well to run the tool.

## Setting up the tool
After cloning the repository onto your machine, it is necessary to install all the required dependencies by running

```pip install -r requirements.txt```

Afterwards, you must ensure that the [fim](http://www.borgelt.net/pyfim.html) package is present on your system. You can obtain pre-compiled Python libraries from the linked site as well as source codes necessary to compile the library yourself.

Lastly, it is necessary to download the respecitve word embedding models from sites linked above and place the word embedding `.bin` files into their respective folders in the `word_emebddings` directory.

## Using the tool
Two modules in this repository are runnable. You can start the model browser by running

```python rule_browser.py```

The tool will then prompt you to choose from one of the supplied data sets (using your own data sets is not supported at this time) and the whole workflow will begin.

Another runnable module is `semantics.py`. You can start by running

```python rule_browser.py```

which will enable you to continually determine word similarities between arbitrary words using the aforementioned word2vec model.
