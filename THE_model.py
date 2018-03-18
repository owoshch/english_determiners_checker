import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from model.q2_initialization import xavier_weight_init
import model.window_utils as du
import model.window_ner as ner
from model.utils import data_iterator
from model.model import LanguageModel
from model.config import Config
from sklearn.metrics import f1_score, accuracy_score, recall_score

class NERModel(LanguageModel):
  """Implements a NER (Named Entity Recognition) model.

  This class implements a deep network for named entity recognition. It
  inherits from LanguageModel, which has an add_embedding method in addition to
  the standard Model method.

  This is assignment 2 from CS224d with some additional methods
  for a determiners correction tasks. Because of that NERModel is not 
  an appropriate name but I decided to keep it for simplicity of code.
  """

  def load_embeddings(self):
    self.wv = self.config.embeddings
    self.num_to_tag = {v:k for k,v in self.config.vocab_tags.items()}

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the training set

    docs = du.load_dataset(self.config.filename_train)
    self.X_train, self.y_train = du.docs_to_windows(
        docs, self.config.vocab_words, self.config.vocab_tags, wsize=self.config.window_size)
    if debug:
      self.X_train = self.X_train[:1024]
      self.y_train = self.y_train[:1024]

    # Load the dev set (for tuning hyperparameters)
    docs = du.load_dataset(self.config.filename_dev)
    self.X_dev, self.y_dev = du.docs_to_windows(
        docs, self.config.vocab_words, self.config.vocab_tags, wsize=self.config.window_size)
    if debug:
      self.X_dev = self.X_dev[:1024]
      self.y_dev = self.y_dev[:1024]

    # Load the test set (dummy labels only)
    docs = du.load_dataset(self.config.filename_test)
    self.X_test, self.y_test = du.docs_to_windows(
        docs, self.config.vocab_words, self.config.vocab_tags, wsize=self.config.window_size)

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, label_size), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.window_size])
    self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.ntags])
    self.dropout_placeholder = tf.placeholder(tf.float32)

  def create_feed_dict(self, input_batch, dropout, label_batch=None):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }


    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    Hint: When label_batch is None, don't add a labels entry to the feed_dict.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    feed_dict = {self.input_placeholder: input_batch, self.dropout_placeholder: dropout}
    if label_batch is not None:
      feed_dict[self.labels_placeholder] = label_batch
    return feed_dict

  def add_embedding(self):
    """Add embedding layer that maps from vocabulary to vectors.

    Creates an embedding tensor (of shape (len(self.wv), embed_size). Use the
    input_placeholder to retrieve the embeddings for words in the current batch.

    (Words are discrete entities. They need to be transformed into vectors for use
    in deep-learning. Although we won't do so in this problem, in practice it's
    useful to initialize the embedding with pre-trained word-vectors. For this
    problem, using the default initializer is sufficient.)

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: See following link to understand what -1 in a shape means.
      https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.wv), embed_size)

    Returns:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    """
    self.embeddings = tf.Variable(tf.constant(self.wv, dtype=tf.float32))
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      return tf.reshape(tf.nn.embedding_lookup(self.embeddings, self.input_placeholder),
                        (-1, self.config.window_size * self.config.embed_size))

  def add_model(self, window):
    """Adds the 1-hidden-layer NN.

    Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer, and
          another variable_scope (e.g. "Softmax") for the linear transformation
          preceding the softmax. Make sure to use the xavier_weight_init you
          defined in the previous part to initialize weights.
    Hint: Make sure to add in regularization and dropout to this network.
          Regularization should be an addition to the cost function, while
          dropout should be added after both variable scopes.
    Hint: You might consider using a tensorflow Graph Collection (e.g
          "total_loss") to collect the regularization and loss terms (which you
          will add in add_loss_op below).
    Hint: Here are the dimensions of the various variables you will need to
          create

          W:  (window_size*embed_size, hidden_size)
          b1: (hidden_size,)
          U:  (hidden_size, label_size)
          b2: (label_size)

    https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#graph-collections
    Args:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """
    with tf.variable_scope("HiddenLayer"):
      self.W = tf.Variable(xavier_weight_init()(shape=[self.config.window_size * self.config.embed_size, self.config.hidden_size]))
      self.b1 = tf.Variable(xavier_weight_init()(shape=[self.config.hidden_size,]))
    with tf.variable_scope("SoftmaxLayer"):
      self.U = tf.Variable(xavier_weight_init()(shape=[self.config.hidden_size, self.config.ntags]))
      self.b2 = tf.Variable(xavier_weight_init()(shape=[self.config.ntags,]))
    tf.add_to_collection("regularization", tf.reduce_sum(tf.square(self.W)))
    tf.add_to_collection("regularization", tf.reduce_sum(tf.square(self.b1)))
    tf.add_to_collection("regularization", tf.reduce_sum(tf.square(self.U)))
    tf.add_to_collection("regularization", tf.reduce_sum(tf.square(self.b2)))
    hidden_outputs = tf.nn.sigmoid(tf.matmul(tf.to_float(window), self.W) + self.b1)
    output = tf.matmul(tf.nn.dropout(hidden_outputs, self.dropout_placeholder), self.U) + self.b2
    return output

  def add_loss_op(self, pred):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
          implementation. You might find tf.reduce_mean useful.
    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))
    loss += self.config.l2 * sum(tf.get_collection("regularization"))
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    return tf.train.AdamOptimizer(self.config.lr).minimize(loss)

  def __init__(self, config):
    """Constructs the network using the helper functions defined above."""
    self.config = config
    self.load_embeddings()
    self.add_placeholders()
    window = self.add_embedding()
    y = self.add_model(window)

    self.loss = self.add_loss_op(y)
    self.predictions = tf.nn.softmax(y)
    one_hot_prediction = tf.argmax(self.predictions, 1)
    correct_prediction = tf.equal(
        tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
    self.train_op = self.add_training_op(self.loss)


  def run_epoch(self, session, input_data, input_labels,
                shuffle=True, verbose=True):
    orig_X, orig_y = input_data, input_labels
    dp = self.config.dropout
    # We're interested in keeping track of the loss and accuracy during training
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    total_steps = len(orig_X) / self.config.batch_size
    for step, (x, y) in enumerate(
      data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                   label_size=self.config.ntags, shuffle=shuffle)):
      feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
      loss, total_correct, _ = session.run(
          [self.loss, self.correct_predictions, self.train_op],
          feed_dict=feed)
      total_processed_examples += len(x)
      total_correct_examples += total_correct
      total_loss.append(loss)
      ##
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
            step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
        sys.stdout.flush()
    return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

  def predict(self, session, X, y=None):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator(X, y, batch_size=self.config.batch_size,
                             label_size=self.config.ntags, shuffle=False)
    else:
        data = data_iterator(X, batch_size=self.config.batch_size,
                             label_size=self.config.ntags, shuffle=False)
    for step, (x, y) in enumerate(data):
      feed = self.create_feed_dict(input_batch=x, dropout=dp)
      if np.any(y):
        feed[self.labels_placeholder] = y
        loss, preds = session.run(
            [self.loss, self.predictions], feed_dict=feed)
        losses.append(loss)
      else:
        preds = session.run(self.predictions, feed_dict=feed)
      predicted_indices = preds.argmax(axis=1)
      results.extend(predicted_indices)
    return np.mean(losses), results



  def predict_labels(self, session, sequence):
      # deactivate dropout
      dp = 1
      X = du.to_windows(sequence, self.config.vocab_words, 
                        self.config.vocab_tags, self.config.window_size)
      results = []
      feed = self.create_feed_dict(input_batch=X, dropout=dp)
      preds = session.run(self.predictions, feed_dict=feed)
      predicted_indices = preds.argmax(axis=1)

      predicted_labels = [self.num_to_tag[num] for num in predicted_indices]

      return predicted_labels

def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print
    print confusion
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print 'Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall)

def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.ntags, config.ntags), dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion

def save_predictions(predictions, filename):
  """Saves predictions to provided file."""
  with open(filename, "wb") as f:
    for prediction in predictions:
      f.write(str(prediction) + "\n")

def test_NER():
  """Test NER model implementation.

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """
  config = Config()
  with tf.Graph().as_default():
    model = NERModel(config)
    model.load_data()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0

      session.run(init)
      for epoch in xrange(config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()
        ###
        train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                model.y_train)
        val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)

        print 'Training loss: {}'.format(train_loss)
        print 'Training acc: {}'.format(train_acc)
        print 'Validation loss: {}'.format(val_loss)
        dev_presicion = accuracy_score(model.y_dev, predictions)

        dev_f1 = f1_score(model.y_dev, predictions, average='macro')

        print 'Dev acc:', dev_presicion
        print 'Dev f1:', dev_f1
        with open(model.config.path_log, 'w+') as logs:
          logs.write('epoch %s' % (epoch))
          logs.write('Dev acc: %s' % (dev_presicion))
          logs.write('Dev f1: %s \n' % (dev_f1))


        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_epoch = epoch
          if not os.path.exists(config.dir_output):
            os.makedirs(config.dir_output)
          saver.save(session, config.dir_model)
        if epoch - best_val_epoch > config.early_stopping:
          break
        ###
        confusion = calculate_confusion(config, predictions, model.y_dev)
        print_confusion(confusion, model.num_to_tag)
        print 'Total time: {}'.format(time.time() - start)

      saver.restore(session, config.model)
      print 'Test'
      print '=-=-='
      print 'Writing predictions to q2_test.predicted'
      _, predictions = model.predict(session, model.X_test, model.y_test)

      test_presicion = accuracy_score(model.y_test, predictions)
      test_f1 = f1_score(model.y_test, predictions, average='macro')


      print 'Test acc:', test_presicion
      print 'Test f1:', test_f1


      save_predictions(predictions, "q2_test.predicted")

if __name__ == "__main__":
  test_NER()