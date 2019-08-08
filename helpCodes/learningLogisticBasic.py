import io

'''
    Learning from a good example from scilab:
    https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#decision-boundary
'''

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

def main():

    # Normalize grades to values between 0 and 1 for more efficient computation
    normalized_range = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

    # Extract Features + Labels
    labels.shape =  (100,) #scikit expects this
    features = normalized_range.fit_transform(features)

    # Create Test/Train
    features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.4)

    # Scikit Logistic Regression
    scikit_log_reg = LogisticRegression()
    scikit_log_reg.fit(features_train,labels_train)

    #Score is Mean Accuracy
    scikit_score = clf.score(features_test,labels_test)
    print ('Scikit score: ', scikit_score)

    #Our Mean Accuracy
    observations, features, labels, weights = run()
    probabilities = predict(features, weights).flatten()
    classifications = classifier(probabilities)
    our_acc = accuracy(classifications,labels.flatten())
    print ('Our score: ',our_acc)


def predict(features, weights):
  '''
  Returns 1D array of probabilities
  that the class label == 1
  '''
  z = np.dot(features, weights)
  return sigmoid(z)

def cost_function(features, labels, weights):
    '''
    Using Mean Absolute Error

    Features:(100,3)
    Labels: (100,1)
    Weights:(3,1)
    Returns 1D matrix of predictions
    Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)

    predictions = predict(features, weights)

    #Take the error when label=1
    class1_cost = -labels*np.log(predictions)

    #Take the error when label=0
    class2_cost = (1-labels)*np.log(1-predictions)

    #Take the sum of both costs
    cost = class1_cost + class2_cost

    #Take the average cost
    cost = cost.sum() / observations

    return cost

def update_weights(features, labels, weights, lr):
    '''
    Vectorized Gradient Descent

    Features:(200, 3)
    Labels: (200, 1)
    Weights:(3, 1)
    '''
    N = len(features)

    #1 - Get Predictions
    predictions = predict(features, weights)

    #2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(features.T,  predictions - labels)

    #3 Take the average cost derivative for each feature
    gradient /= N

    #4 - Multiply the gradient by our learning rate
    gradient *= lr

    #5 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights

def decision_boundary(prob):
  return 1 if prob >= .5 else 0

def classify(predictions):
  '''
  input  - N element array of predictions between 0 and 1
  output - N element array of 0s (False) and 1s (True)
  '''
  decision_boundary = np.vectorize(decision_boundary)
  return decision_boundary(predictions).flatten()

def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        #Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 1000 == 0:
            print ("iter: "+str(i) + " cost: "+str(cost))

    return weights, cost_history

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

def plot_decision_boundary(trues, falses):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    no_of_preds = len(trues) + len(falses)

    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

    plt.legend(loc='upper right');
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()

