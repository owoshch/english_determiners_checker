# English determiners checker

This repo implements a window classification model using Tensorflow.
It's build as a baseline for CS224n project: https://owoshch.github.io/grammar_checker/

## Task

Given a paragraph, place the determiners (a, an, the) correctly. 

Determiners are strongly connected with the words around them. Thus, we decided to take a window classification model as a baseline. 

## Model

Window model is the one discussed in [CS224n Lecture 4, slide 17](http://web.stanford.edu/class/cs224n/lectures/lecture4.pdf).
The model is built on top of the assignment 2 from CS224d.

* Embed a word and its neightboors using [GloVe](https://nlp.stanford.edu/projects/glove/) vectors. We made experiments for window sizes 3, 5 and 7, which corresponds to 1, 2 or 3 neighboor words for a given center word. 

* Apply a one-hidden-layer neural network to classify a given word. We introduces four classes with respect to particular determiners before a given word: O for a blank space, A, AN and THE.


## Getting started

I took data preprocessing part from the wonderful repo explaining how to build your own Named Entity Recognition system from scratch. Check out and star the repo [Sequence tagging](https://github.com/guillaumegenthial/sequence_tagging).


1. Download the GloVe vectors with

```
make glove
```

Alternatively, you can download them manually [here](https://nlp.stanford.edu/projects/glove/) and update the `glove_filename` entry in `config.py`. You can also choose not to load pretrained word vectors by changing the entry `use_pretrained` to `False` in `model/config.py`.

2. Build the training data, train and evaluate the model with
```
make run
```

## Details


Here is the breakdown of the commands executed in `make run`:

1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```
python build_data.py
```

2. Train the model with

```
python THE_model.py
```

3. Evaluate and interact with the model with
```
python test.py
```

## Training Data


We used [Cornell Movie Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). We store each given utterance (no matter how sentences are there) in a text file with one word and its class per line. We split data is a way to make train, dev and test dataset uniform in terms of lenghts of utterances.

Example: `I have a ball. The ball is red.`

```
I O
have O
ball A
. O
ball THE
is O
red O
. O
```





