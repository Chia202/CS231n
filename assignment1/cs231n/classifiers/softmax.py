from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass
    for i in range(X.shape[0]):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        correct_class_score = scores[y[i]]
        sum_exp_scores = np.sum(np.exp(scores))
        # dW += np.exp(scores).reshape(-1, 1) * X[i].reshape(1, -1) / sum_exp_scores
        dW += X[i].reshape(-1, 1) * np.exp(scores) / sum_exp_scores
        dW[:, y[i]] -= X[i]
        loss += -correct_class_score + np.log(sum_exp_scores)

    loss = loss / X.shape[0] + reg * np.sum(W * W)

    dW = dW / X.shape[0] + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass
    scores = X @ W
    scores -= np.max(scores, axis=1).reshape(-1, 1)
    correct_class_score = scores[np.arange(X.shape[0]), y]
    sum_exp_scores = np.sum(np.exp(scores), axis=1)
    loss = np.sum(-correct_class_score + np.log(sum_exp_scores)) / X.shape[0] + reg * np.sum(W * W)

    dW = X.T @ (np.exp(scores) / sum_exp_scores[:, np.newaxis])
    dW -= X.T @ np.eye(W.shape[1])[y]
    dW = dW / X.shape[0] + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


if __name__=='__main__':
    # Generate a random softmax weight matrix and use it to compute the loss.
    W = np.random.randn(3073, 10) * 0.0001
    X_dev = np.random.randn(500, 3073)
    y_dev = np.random.randint(10, size=500)
    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

    # As a rough sanity check, our loss should be something close to -log(0.1).
    print('loss: %f' % loss)
    print('sanity check: %f' % (-np.log(0.1)))

    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

    # As we did for the SVM, use numeric gradient checking as a debugging tool.
    # The numeric gradient should be close to the analytic gradient.
    # from gradient_check import grad_check_sparse

    # f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
    # grad_numerical = grad_check_sparse(f, W, grad, 10)

    # # similar to SVM case, do another gradient check with regularization
    # loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
    # f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
    # grad_numerical = grad_check_sparse(f, W, grad, 10)

    loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
    print('Gradient difference: %f' % grad_difference)
