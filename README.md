# English determiners checker


Determiners are strongly connected with the words around them. Thus, we decided to take a window classification model as a baseline. 
I took a model from the second assignment of [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/), 
a precursor of [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/syllabus.html)
A brief overview of window models you can find in [CS224n Lecture 4, slide 17](http://web.stanford.edu/class/cs224n/lectures/lecture4.pdf).

We used the following configuration:

* Embed a word and its neightboors using [GloVe](https://nlp.stanford.edu/projects/glove/) vectors. We made experiments for window sizes 3, 5 and 7,
which corresponds to 1, 2 or 3 neighboor words for a given center word. 

* Apply a one-hidden-layer neural network to classify a given word. We introduces four classes with respect to particular determiners before a given word: 
O for a blank space, A, AN and THE.
