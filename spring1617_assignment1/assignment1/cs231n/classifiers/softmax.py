import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    h = X[i].dot(W)
    pred = np.exp(h) / np.sum(np.exp(h))
    loss += -np.log(pred[y[i]]) + 0.5 * reg * np.sum(W ** 2)
    pred[y[i]] -= 1
    dW += X[i][..., None].dot(pred[None, ...]) + reg * W
  loss /= X.shape[0]
  dW /= X.shape[0]
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  h = X.dot(W)
  pred = np.exp(h) / np.sum(np.exp(h), axis=1, keepdims=True)
  loss += np.sum(-np.log(pred[np.arange(X.shape[0]), y])) / X.shape[0] + 0.5 * reg * np.sum(W ** 2)
  pred[np.arange(X.shape[0]), y] -= 1
  dW += X.T.dot(pred) / X.shape[0] + reg * W
  #############################################################################

  return loss, dW

