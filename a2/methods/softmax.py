
import numpy as np
import random

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x


#(ii) Implement the softmax loss and gradient in the naiveSoftmaxLossAndGradient method.
def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 

    #np.zeros_like: This is a NumPy function that creates a new array with the same shape as the given array. The new array is filled with zeros.
    gradOutsideVecs = np.zeros_like(outsideVectors)
    # obtain y_hat (i.e., the conditional probability distribution p(O = o | C = c))
    # by taking vector dot products and applying softmax
    y_hat = softmax(np.dot(outsideVectors, centerWordVec)) # (N,) N x 1
    # can also get y_hat in a single line: y_hat = softmax(outsideVectors @ centerWordVec)

    # for a single pair of words c and o, the loss is given by:
    # J(v_c, o, U) = -log P(O = o | C = c) = -log [y_hat[o]]
    loss = -np.log(y_hat[outsideWordIdx])

    # grad calc
    # generate the ground-truth one-hot vector, [..., 0, outsideWordIdx=1, 0, ...]
    y = np.zeros_like(y_hat)
    y[outsideWordIdx] = 1
    # can also get loss as -np.dot(y, np.log(y_hat))    
    
    gradCenterVec = np.dot(y_hat - y, outsideVectors) # inner product results in a scalar
    # or gradCenterVec = np.dot(outsideVectors.T, y_hat - y)
    
    gradOutsideVecs = np.outer(y_hat - y, centerWordVec) # outer product results in a matrix
    # or gradOutsideVecs = np.dot((y_hat - y)[:, np.newaxis], centerWordVec[np.newaxis, :]) 
    
    # sanity check the dimensions
    assert gradCenterVec.shape == centerWordVec.shape
    assert gradOutsideVecs.shape == outsideVectors.shape  

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = 1 / (1 + np.exp(-x))

    ### END YOUR CODE
    print(s)

    return s

def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.
    
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    
    # Calculate the first term
    y_hat = sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec))
    loss = -np.log(y_hat)
    
    gradCenterVec = np.dot(y_hat - 1, outsideVectors[outsideWordIdx])
    gradOutsideVecs[outsideWordIdx] = np.dot(y_hat - 1, centerWordVec)

    # Calculate the second term
    for i in range(K):
        w_k = indices[i+1]
        y_k_hat = sigmoid(-np.dot(outsideVectors[w_k], centerWordVec))
        loss += -np.log(y_k_hat)
        gradOutsideVecs[w_k] += np.dot(1.0 - y_k_hat, centerWordVec)
        gradCenterVec += np.dot(1.0 - y_k_hat, outsideVectors[w_k])

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def test_naiveSoftmaxLossAndGradient():
    """ Test naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for naiveSoftmaxLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "naiveSoftmaxLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")

    

def grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient):
    print ("======Skip-Gram with negSamplingLossAndGradient======")  

    # first test
    output_loss, output_gradCenterVecs, output_gradOutsideVectors = \
                skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
                dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)

    assert np.allclose(output_loss, 16.15119285363322), \
           "Your loss does not match expected loss."
    expected_gradCenterVecs = [[ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ],
                               [-4.54650789, -1.85942252,  0.76397441],
                               [ 0.,          0.,          0.        ],
                               [ 0.,          0.,          0.        ]]
    expected_gradOutsideVectors = [[-0.69148188,  0.31730185,  2.41364029],
                                   [-0.22716495,  0.10423969,  0.79292674],
                                   [-0.45528438,  0.20891737,  1.58918512],
                                   [-0.31602611,  0.14501561,  1.10309954],
                                   [-0.80620296,  0.36994417,  2.81407799]]
                     
    assert np.allclose(output_gradCenterVecs, expected_gradCenterVecs), \
           "Your gradCenterVecs do not match expected gradCenterVecs."
    assert np.allclose(output_gradOutsideVectors, expected_gradOutsideVectors), \
           "Your gradOutsideVectors do not match expected gradOutsideVectors."
    print("The first test passed!")

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    return x


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    
    gradOutsideVecs = np.zeros_like(outsideVectors)
    # obtain y_hat (i.e., the conditional probability distribution p(O = o | C = c))
    # by taking vector dot products and applying softmax
    y_hat = softmax(np.dot(outsideVectors, centerWordVec)) # (N,) N x 1
    # can also get y_hat in a single line: y_hat = softmax(outsideVectors @ centerWordVec)

    # for a single pair of words c and o, the loss is given by:
    # J(v_c, o, U) = -log P(O = o | C = c) = -log [y_hat[o]]
    loss = -np.log(y_hat[outsideWordIdx])

    # grad calc
    # generate the ground-truth one-hot vector, [..., 0, outsideWordIdx=1, 0, ...]
    y = np.zeros_like(y_hat)
    y[outsideWordIdx] = 1
    # can also get loss as -np.dot(y, np.log(y_hat))    
    
    gradCenterVec = np.dot(y_hat - y, outsideVectors) # inner product results in a scalar
    # or gradCenterVec = np.dot(outsideVectors.T, y_hat - y)
    
    gradOutsideVecs = np.outer(y_hat - y, centerWordVec) # outer product results in a matrix
    # or gradOutsideVecs = np.dot((y_hat - y)[:, np.newaxis], centerWordVec[np.newaxis, :]) 
    
    # sanity check the dimensions
    assert gradCenterVec.shape == centerWordVec.shape
    assert gradOutsideVecs.shape == outsideVectors.shape  

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs



def getDummyObjects():
    """ Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    return dataset, dummy_vectors, dummy_tokens
    

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x, gradientText):
   
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        x[ix] += h # increment by h
        random.setstate(rndstate)
        fxh, _ = f(x) # evalute f(x + h)
        x[ix] -= 2 * h # restore to previous value (very important!)
        random.setstate(rndstate)
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh) / 2 / h

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed for %s." % gradientText)
            print("First gradient error found at index %s in the vector of gradients" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!. Read the docstring of the `gradcheck_naive`"
    " method in utils.gradcheck.py to understand what the gradient check does.")

def test_negSamplingLossAndGradient():
    """ Test negSamplingLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for negSamplingLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")


if __name__ == "__main__":
    test_naiveSoftmaxLossAndGradient()
    test_negSamplingLossAndGradient()
