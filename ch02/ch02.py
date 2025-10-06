# coding: utf-8


import sys
import numpy as np
import os
import pandas as pd
import matplotlib
# Use a non-interactive backend to avoid Qt/Wayland plugin errors when running headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')

# Ensure output directory exists so plt.savefig won't fail
os.makedirs('figures', exist_ok=True)

# # Chapter 2 - Training Machine Learning Algorithms for Classification

# ### Overview
# 

# - [Artificial neurons – a brief glimpse into the early history of machine learning](#Artificial-neurons-a-brief-glimpse-into-the-early-history-of-machine-learning)
#     - [The formal definition of an artificial neuron](#The-formal-definition-of-an-artificial-neuron)
#     - [The perceptron learning rule](#The-perceptron-learning-rule)
# - [Implementing a perceptron learning algorithm in Python](#Implementing-a-perceptron-learning-algorithm-in-Python)
#     - [An object-oriented perceptron API](#An-object-oriented-perceptron-API)
#     - [Training a perceptron model on the Iris dataset](#Training-a-perceptron-model-on-the-Iris-dataset)
# - [Adaptive linear neurons and the convergence of learning](#Adaptive-linear-neurons-and-the-convergence-of-learning)
#     - [Minimizing cost functions with gradient descent](#Minimizing-cost-functions-with-gradient-descent)
#     - [Implementing an Adaptive Linear Neuron in Python](#Implementing-an-Adaptive-Linear-Neuron-in-Python)
#     - [Improving gradient descent through feature scaling](#Improving-gradient-descent-through-feature-scaling)
#     - [Large scale machine learning and stochastic gradient descent](#Large-scale-machine-learning-and-stochastic-gradient-descent)
# - [Summary](#Summary)






# # Artificial neurons - a brief glimpse into the early history of machine learning





# ## The formal definition of an artificial neuron





# ## The perceptron learning rule










# # Implementing a perceptron learning algorithm in Python

# ## An object-oriented perceptron API





class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        
        if not hasattr(self, 'w_') or self.w_.shape[0] != X.shape[1]:
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        if not hasattr(self, 'b_'):
            self.b_ = np.float64(0.)
        
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    def set_weights(self, w,b):
      """A function that is used to set the values of w,b """
      w = np.asarray(w)
      if w.ndim != 1:
          raise ValueError("w must be a 1-D array")
      self.w_ = w
      self.b_ = float(b)

v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))



# ## Training a perceptron model on the Iris dataset

# ...

# ### Reading-in the Iris data




try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
    
except HTTPError:
    s = 'iris.data'
    print('From local Iris path:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
    
df.tail()



# ### Plotting the Iris data


# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='s', label='Versicolor')

plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
#plt.show()



# ### Training the perceptron model



ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
#plt.show()



# ### A function for plotting decision regions





def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')




plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')


#plt.savefig('images/02_08.png', dpi=300)
#plt.show()



# # Adaptive linear neurons and the convergence of learning

# ...

# ## Minimizing cost functions with gradient descent










# ## Implementing an adaptive linear neuron in Python



class AdalineGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
      Mean squared eror loss function values in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            
            #for w_j in range(self.w_.shape[0]):
            #    self.w_[w_j] += self.eta * (2.0 * (X[:, w_j]*errors)).mean()
            
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
    def set_weights(self, w,b):
      """A function that is used to set the values of w,b """
      w = np.asarray(w)
      self.w_ = w
      self.b_ = float(b)




fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.savefig('images/02_11.png', dpi=300)
#plt.show()








# ## Improving gradient descent through feature scaling







# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()




ada_gd = AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/02_14_1.png', dpi=300)
#plt.show()

plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')

plt.tight_layout()
#plt.savefig('images/02_14_2.png', dpi=300)
#plt.show()



# ## Large scale machine learning and stochastic gradient descent



class AdalineSGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
      Mean squared error loss function value averaged over all
      training examples in each epoch.

        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
    def set_weights(self, w,b):
      """A function that is used to set the values of w,b """
      w = np.asarray(w)
      self.w_ = w
      self.b_ = float(b)



ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figures/02_15_1.png', dpi=300)
#plt.show()

plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')

plt.savefig('figures/02_15_2.png', dpi=300)
#plt.show()




ada_sgd.partial_fit(X_std[0, :], y[0])



# # Summary

# ...

# --- 
# 
# Readers may ignore the following cell


def question1():
    
    print("QUESTION 1")
    
    # Data Input
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    #AND
    y_and = np.array([0,0,0,1])

    #IMPLIES
    y_implies = np.array([1, 1, 0, 1])

    #OR
    y_or = np.array([0,1,1,1])

    #Store results
    functions = [('AND', y_and), ('OR', y_or), ('IMPLIES', y_implies)]
    results = {}


    #Iterate through the different y___
    for function_name, y in functions:
        print(f"\n--- Testing {function_name} Function ---")
        print(f"Expected X: {X}")
        print(f"Expected output: {y}")

        #Init perceptron
        perceptron = Perceptron(eta=0.1, n_iter= 10, random_state=0)
        perceptron.set_weights([0,25,-0.125], 0.0)

        #Train perceptron
        perceptron.fit(X,y)

        #Predict
        predictions = perceptron.predict(X)
        
        final_errors = perceptron.errors_[-1] if perceptron.errors_ else 0
        converged = final_errors == 0
        epochs_to_converge = len(perceptron.errors_) if converged else "Did not converge"
        correct_boundary = np.array_equal(predictions, y)
        
        # Store results
        results[function_name] = {
            'converged': converged,
            'epochs': epochs_to_converge,
            'final_weights': perceptron.w_.copy(),
            'final_bias': perceptron.b_,
            'correct_boundary': correct_boundary,
            'predictions': predictions,
            'expected': y,
            'errors_per_epoch': perceptron.errors_
        }

        # Print detailed results - exactly what assignment asks for
        print(f"\nResults for {function_name}:")
        print(f"(i) Converged over 10 epochs: {converged}")
        print(f"(ii) Epochs to convergence: {epochs_to_converge}")
        print(f"(iii) Final weights w: {perceptron.w_}")
        print(f"      Final bias b: {perceptron.b_}")
        print(f"(iv) Predictions: {predictions}")
        print(f"     Expected:    {y}")
        print(f"     Correct boundary: {correct_boundary}")
        print(f"Errors per epoch: {perceptron.errors_}")

#In order to properly do question 2 we need to read in rectangle data 
def load_rectangle_data(filename='rectangle.data'):
  script_dir = os.path.dirname(os.path.abspath(__file__))
  candidates = [
    os.path.join(script_dir, filename),  # Give path
  ]

  for path in candidates:
    try:
      data = pd.read_csv(path, header=None)

      X = data.iloc[:, :-1].values  # All columns except last (features)
      y = data.iloc[:, -1].values   # Last column (labels)

      print(f"Successfully loaded {X.shape[0]} samples with {X.shape[1]} features from {path}")
      try:
        print(f"Class distribution: {np.bincount(y)}")
      except Exception:
        pass
      return X, y

    except Exception as e:
      print(f"Error reading {path}: {e}")
      return None, None

  # If we get here none of the candidate paths existed/readable
  print(f"Could not find rectangle data at any of: {candidates}")
  return None, None


def question2a():
    """Problem 2: Learning an axis-aligned rectangle as a Perceptron"""
    print("PROBLEM 2: Rectangle Learning with Perceptron")
    
    #Load in inputs
    w = [0.25, -0.125, 0.0625]
    b = 0

    # Load rectangle data
    X, y = load_rectangle_data()

    
    if X is None or y is None:
        print("Failed to load rectangle data.")
    
    # Part a: Different epoch counts [10, 20, 30, 100]

      # Init the perceptron
    q2i_perceptron = Perceptron(eta=0.1,n_iter=10,random_state=0)
    q2i_perceptron.set_weights(w,b)


    
    q2ii_perceptron = Perceptron(eta=0.1,n_iter=20,random_state=0)
    q2ii_perceptron.set_weights(w,b)

    q2iii_perceptron = Perceptron(eta=0.1,n_iter=30,random_state=0)
    q2iii_perceptron.set_weights(w,b)


    q2iv_perceptron = Perceptron(eta=0.1,n_iter=100,random_state=0)
    q2iv_perceptron.set_weights(w,b)



  # Part (a) evaluation: fit each perceptron on the whole dataset and report results
    perceptrons = [
      (q2i_perceptron, 10),
      (q2ii_perceptron, 20),
      (q2iii_perceptron, 30),
      (q2iv_perceptron, 100)
    ]

    results_a = {}

    if X is None or y is None:
      print("No rectangle data available — skipping Part (a) evaluation.")
      return None

    for p, epochs in perceptrons:
      print('\n' + '-'*50)
      print(f"Training perceptron with n_iter={epochs} on full dataset (Part a)")
      # Ensure perceptron has the initial weights set (set_weights called above)
      # Call fit to run the specified number of epochs
      p.fit(X, y)

      preds = p.predict(X)
      accuracy = np.mean(preds == y)
      final_errors = p.errors_[-1] if p.errors_ else 0
      converged = final_errors == 0

      results_a[epochs] = {
        'final_weights': p.w_.copy(),
        'final_bias': float(p.b_),
        'errors_per_epoch': p.errors_.copy(),
        'accuracy': float(accuracy),
        'converged': converged
      }

      print(f"n_iter={epochs}: accuracy={accuracy:.3f}, converged={converged}")
      print(f" final w: {p.w_}, b: {p.b_}")

      # attempt to plot decision regions if X has 2 features
      try:
        if X.shape[1] == 2:
          plot_decision_regions(X, y, classifier=p)
          plt.title(f"Rectangle data — perceptron n_iter={epochs}")
          plt.xlabel('x1')
          plt.ylabel('x2')
          plt.tight_layout()
          out = f"figures/rectangle_partA_niter_{epochs}.png"
          plt.savefig(out, dpi=200)
          plt.clf()
          print(f" Saved decision-region plot to {out}")
      except Exception as e:
        print(f"Could not plot decision region for n_iter={epochs}: {e}")

    # Print a compact summary table for Part (a)
    print('\n' + '='*60)
    print('Part (a) summary — perceptron trained on full rectangle dataset')
    print('epochs | accuracy | converged | final_w | final_b')
    for epochs, r in results_a.items():
      print(f"{epochs:6d} | {r['accuracy']:.3f}    | {str(r['converged']):9s} | {r['final_weights']} | {r['final_bias']}")

    return results_a

def question2b():
    # Load data
    X, y = load_rectangle_data()
    # Split data
    train_idx = range(60)
    test_idx = range(60, 100)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    # Group test samples into 8 groups of 5
    group_indices = [range(i, i+5) for i in range(0, 40, 5)]
    X_groups = [X_test[indices] for indices in group_indices]
    y_groups = [y_test[indices] for indices in group_indices]
    # Initialize perceptron
    p = Perceptron(eta=0.1, n_iter=10, random_state=0)
    p.set_weights([0.25, -0.125, 0.0625], 0.0)
    # Train on first 60 samples
    p.fit(X_train, y_train)
    # Evaluate groups
    alpha = 0.2
    delta = 0.25
    success_count = 0
    for i, (Xg, yg) in enumerate(zip(X_groups, y_groups)):
        preds = p.predict(Xg)
        error_rate = np.mean(preds != yg)
        print(f"Group {i+1}: error={error_rate:.2f}")
        if error_rate <= alpha:
            success_count += 1
    # Check if at least (1 - delta) of groups succeed
    sufficient_groups = int(np.ceil((1 - delta) * len(X_groups)))
    print(f"\nNumber of groups passing: {success_count} (Threshold: {sufficient_groups})")
    if success_count >= sufficient_groups:
        print("YES")
    else:
        print("NO")

def question3():
    """
    Problem 3: Learn boolean functions using Adaline model
    Same initial values as Problem 1: w = [0.25, -0.125], b = 0
    Test different learning rates: 0.01, 0.05, 0.1, 0.5, 0.75
    10 epochs, threshold 0.5 for predict()
    """
    print("="*60)
    print("PROBLEM 3: Boolean Functions with Adaline")
    print("="*60)
    
    # Same boolean function data as Problem 1
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    # Define y values for each function
    y_and = np.array([0, 0, 0, 1])
    y_or = np.array([0, 1, 1, 1])
    y_implies = np.array([1, 1, 0, 1])
    
    # Test parameters from assignment
    functions = [('AND', y_and), ('OR', y_or), ('IMPLIES', y_implies)]
    learning_rates = [0.01, 0.05, 0.1, 0.5, 0.75]  # Five learning rates to test
    n_epochs = 10
    
    print(f"Testing {len(functions)} boolean functions with {len(learning_rates)} learning rates")
    print(f"Learning rates: {learning_rates}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Initial weights: [0.25, -0.125], bias: 0")
    print(f"Prediction threshold: 0.5")
    
    results = {}
    
    for func_name, y in functions:
        print(f"\n{'='*50}")
        print(f"Testing {func_name} Function: f(x1,x2) = x1 {func_name} x2")
        print(f"{'='*50}")
        
        print(f"Input data X:\n{X}")
        print(f"Expected output y: {y}")
        
        results[func_name] = {}
        
        for eta in learning_rates:
            print(f"\n--- Learning Rate η = {eta} ---")
            
            # Initialize Adaline with specified values
            adaline = AdalineGD(eta=eta, n_iter=n_epochs, random_state=0)
            
            # Set initial weights as specified in assignment (same as Problem 1)
            adaline.set_weights([0.25, -0.125], 0.0)
            
            print(f"Initial weights w: {adaline.w_}")
            print(f"Initial bias b: {adaline.b_}")
            
            # Train the Adaline model
            adaline.fit(X, y)
            
            # Make predictions using threshold of 0.5
            predictions = adaline.predict(X)
            
            # Calculate important metrics
            initial_mse = adaline.losses_[0] if adaline.losses_ else float('inf')
            final_mse = adaline.losses_[-1] if adaline.losses_ else float('inf')
            mse_decreased = final_mse < initial_mse
            mse_converged_to_zero = final_mse < 0.01  # Consider converged if MSE < 0.01
            mse_increasing = final_mse > initial_mse
            correct_predictions = np.array_equal(predictions, y)
            
            # Store results
            results[func_name][eta] = {
                'initial_weights': np.array([0.25, -0.125]).copy(),
                'initial_bias': 0.0,
                'final_weights': adaline.w_.copy(),
                'final_bias': adaline.b_,
                'initial_mse': initial_mse,
                'final_mse': final_mse,
                'mse_decreased': mse_decreased,
                'mse_converged_to_zero': mse_converged_to_zero,
                'mse_increasing': mse_increasing,
                'correct_predictions': correct_predictions,
                'predictions': predictions,
                'expected': y,
                'losses_per_epoch': adaline.losses_.copy()
            }
            
            # Print detailed results for this learning rate
            print(f"Training completed:")
            print(f"- Final weights w: {adaline.w_}")
            print(f"- Final bias b: {adaline.b_}")
            print(f"- Initial MSE: {initial_mse:.6f}")
            print(f"- Final MSE: {final_mse:.6f}")
            print(f"- MSE decreased: {mse_decreased}")
            print(f"- MSE converged to ~0: {mse_converged_to_zero}")
            print(f"- MSE increased: {mse_increasing}")
            print(f"- Predictions: {predictions}")
            print(f"- Expected:    {y}")
            print(f"- Correct predictions: {correct_predictions}")
            print(f"- MSE per epoch: {[f'{loss:.4f}' for loss in adaline.losses_[:5]]}")
            if len(adaline.losses_) > 5:
                print(f"  ... (showing first 5 epochs)")
    
    # Comprehensive Analysis Section
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS - Problem 3")
    print("="*80)
    
    # Analysis 1: Which learning rates work well?
    print("\n1. LEARNING RATES THAT WORK WELL:")
    print("-" * 50)
    for func_name, func_results in results.items():
        good_rates = []
        for eta, result in func_results.items():
            if result['correct_predictions'] and result['mse_decreased']:
                good_rates.append(eta)
        print(f"{func_name:>8}: {good_rates if good_rates else 'None'}")
    
    # Analysis 2: MSE convergence to 0
    print("\n2. LEARNING RATES WHERE MSE CONVERGES TO ~0:")
    print("-" * 50)
    for func_name, func_results in results.items():
        converged_rates = []
        for eta, result in func_results.items():
            if result['mse_converged_to_zero']:
                converged_rates.append(eta)
        print(f"{func_name:>8}: {converged_rates if converged_rates else 'None'}")
    
    # Analysis 3: MSE increases
    print("\n3. LEARNING RATES WHERE MSE INCREASES:")
    print("-" * 50)
    for func_name, func_results in results.items():
        increasing_rates = []
        for eta, result in func_results.items():
            if result['mse_increasing']:
                increasing_rates.append(eta)
        print(f"{func_name:>8}: {increasing_rates if increasing_rates else 'None'}")
    
    # Analysis 4: Correct predictions for all input combinations
    print("\n4. LEARNING RATES WITH CORRECT PREDICTIONS:")
    print("-" * 50)
    for func_name, func_results in results.items():
        correct_rates = []
        for eta, result in func_results.items():
            if result['correct_predictions']:
                correct_rates.append(eta)
        print(f"{func_name:>8}: {correct_rates if correct_rates else 'None'}")
    
    return results


def question4():
    """
    Problem 4: Repeat rectangle learning from Problem 2(a) using Adaline
    Same initial values as Problem 2: w = [0.25, -0.125, 0.0625], b = 0
    Learning rate η = 0.01, 10 epochs
    Train on ALL 100 samples (unlike Problem 2b which used only 60)
    """
    print("="*60)
    print("PROBLEM 4: Rectangle Learning with Adaline")
    print("="*60)
    
    # Load rectangle data (same as Problem 2)
    X, y = load_rectangle_data('rectangle.data')
    
    if X is None or y is None:
        print("Failed to load rectangle data. Cannot proceed with Problem 4.")
        return None
    
    # Display dataset info
    print(f"Dataset Information:")
    print(f"- Shape: {X.shape} (samples: {X.shape[0]}, features: {X.shape[1]})")
    print(f"- Class distribution: {np.bincount(y)}")
    
    # Problem 4 parameters (from assignment)
    initial_weights = [0.25, -0.125, 0.0625]  # Same as Problem 2
    initial_bias = 0.0
    learning_rate = 0.01  # Different from Problem 2 (was 0.1)
    n_epochs = 10
    
    print(f"\nTraining Parameters:")
    print(f"- Algorithm: Adaline (ADAptive LInear NEuron)")
    print(f"- Initial weights: {initial_weights}")
    print(f"- Initial bias: {initial_bias}")
    print(f"- Learning rate η: {learning_rate}")
    print(f"- Number of epochs: {n_epochs}")
    print(f"- Training on ALL {X.shape[0]} samples")
    
    # Initialize Adaline with specified parameters
    adaline = AdalineGD(eta=learning_rate, n_iter=n_epochs, random_state=0)
    
    # Set initial weights as specified in assignment
    adaline.set_weights(initial_weights, initial_bias)
    
    print(f"\nInitial state:")
    print(f"- Weights w: {adaline.w_}")
    print(f"- Bias b: {adaline.b_}")
    
    # Train the Adaline model on ALL samples
    print(f"\nTraining Adaline on all {X.shape[0]} samples...")
    adaline.fit(X, y)
    
    # Analyze training results
    predictions = adaline.predict(X)
    training_accuracy = np.mean(predictions == y)
    
    # MSE evolution analysis
    initial_mse = adaline.losses_[0] if adaline.losses_ else float('inf')
    final_mse = adaline.losses_[-1] if adaline.losses_ else float('inf')
    mse_decreased = final_mse < initial_mse
    
    print(f"\nTraining Results:")
    print(f"- Final weights w: {adaline.w_}")
    print(f"- Final bias b: {adaline.b_}")
    print(f"- Training accuracy: {training_accuracy:.3f} ({int(training_accuracy*100)}%)")
    print(f"- Correct predictions: {np.sum(predictions == y)}/{len(y)}")
    print(f"- Incorrect predictions: {np.sum(predictions != y)}/{len(y)}")
    
    # MSE Evolution Analysis - This answers the assignment questions
    print(f"\n" + "="*50)
    print("MSE EVOLUTION ANALYSIS")
    print("="*50)
    
    print(f"Initial MSE (epoch 1): {initial_mse:.6f}")
    print(f"Final MSE (epoch {len(adaline.losses_)}): {final_mse:.6f}")
    print(f"MSE decreased over epochs: {mse_decreased}")
    print(f"MSE reduction: {initial_mse - final_mse:.6f}")
    print(f"Relative MSE reduction: {((initial_mse - final_mse) / initial_mse * 100):.1f}%")
    
    # Epoch-by-epoch MSE evolution
    print(f"\nEpoch-by-epoch MSE evolution:")
    print(f"{'Epoch':<8} {'MSE':<12} {'Change':<12}")
    print("-" * 35)
    
    for i, mse in enumerate(adaline.losses_):
        epoch = i + 1
        if i == 0:
            change_str = "0.000000"
        else:
            change = mse - adaline.losses_[i-1]
            change_str = f"{change:+.6f}"
        
        print(f"{epoch:<8} {mse:<12.6f} {change_str:<12}")
    
    # Prediction Quality Analysis
    print(f"\n" + "="*50)
    print("PREDICTION QUALITY ANALYSIS")
    print("="*50)
    
    # Analyze prediction errors by class
    class_0_indices = y == 0
    class_1_indices = y == 1
    
    class_0_accuracy = np.mean(predictions[class_0_indices] == y[class_0_indices])
    class_1_accuracy = np.mean(predictions[class_1_indices] == y[class_1_indices])
    
    print(f"Class 0 (outside rectangle): {class_0_accuracy:.3f} accuracy")
    print(f"Class 1 (inside rectangle): {class_1_accuracy:.3f} accuracy")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: MSE evolution over epochs
    plt.subplot(1, 3, 1)
    epochs = range(1, len(adaline.losses_) + 1)
    plt.plot(epochs, adaline.losses_, 'bo-', linewidth=2, markersize=6)
    plt.title('MSE Evolution Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    
    # Plot 2: Training accuracy
    plt.subplot(1, 3, 2)
    plt.bar(['Training Accuracy'], [training_accuracy], color='skyblue', alpha=0.7)
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    # Plot 3: Prediction distribution
    plt.subplot(1, 3, 3)
    correct = np.sum(predictions == y)
    incorrect = np.sum(predictions != y)
    plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'], 
            colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
    plt.title('Prediction Distribution')
    
    plt.tight_layout()
    plt.savefig('figures/problem4_adaline_rectangle.png', dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plots saved to figures/problem4_adaline_rectangle.png")
    
    # Final Summary - Answers assignment questions
    print(f"\n" + "="*60)
    print("PROBLEM 4 SUMMARY - ASSIGNMENT ANSWERS")
    print("="*60)
    
    print(f"1. What do you observe?")
    if mse_decreased:
        print("   ✓ MSE consistently decreased over 10 epochs")
    else:
        print("   ✗ MSE did not decrease properly")
    
    if training_accuracy > 0.85:
        print("   ✓ Good training accuracy achieved")
    else:
        print("   ⚠ Moderate training accuracy")
    
    print(f"\n2. Evolution of MSE over 10 epochs:")
    print(f"   • Initial MSE: {initial_mse:.6f}")
    print(f"   • Final MSE: {final_mse:.6f}")
    print(f"   • Total reduction: {initial_mse - final_mse:.6f}")
    print(f"   • Pattern: {'Decreasing' if mse_decreased else 'Not decreasing'}")
    
    print(f"\n3. Quality of resultant w, b for prediction:")
    print(f"   • Final weights: {adaline.w_}")
    print(f"   • Final bias: {adaline.b_:.6f}")
    print(f"   • Training accuracy: {training_accuracy:.1%}")
    print(f"   • Prediction quality: {'Good' if training_accuracy > 0.8 else 'Moderate'}")
    
    return {
        'final_weights': adaline.w_.copy(),
        'final_bias': adaline.b_,
        'training_accuracy': training_accuracy,
        'initial_mse': initial_mse,
        'final_mse': final_mse,
        'mse_decreased': mse_decreased,
        'losses_per_epoch': adaline.losses_.copy()
    }

def question4():
    """
    Problem 4: Repeat rectangle learning from Problem 2(a) using Adaline
    Same initial values as Problem 2: w = [0.25, -0.125, 0.0625], b = 0
    Learning rate η = 0.01, 10 epochs
    Train on ALL 100 samples (unlike Problem 2b which used only 60)
    """
    print("="*60)
    print("PROBLEM 4: Rectangle Learning with Adaline")
    print("="*60)
    
    # Load rectangle data (same as Problem 2)
    X, y = load_rectangle_data('rectangle.data')
    
    if X is None or y is None:
        print("Failed to load rectangle data. Cannot proceed with Problem 4.")
        return None
    
    # Display dataset info
    print(f"Dataset Information:")
    print(f"- Shape: {X.shape} (samples: {X.shape[0]}, features: {X.shape[1]})")
    print(f"- Class distribution: {np.bincount(y)}")
    
    # Problem 4 parameters (from assignment)
    initial_weights = [0.25, -0.125, 0.0625]  # Same as Problem 2
    initial_bias = 0.0
    learning_rate = 0.01  # Different from Problem 2 (was 0.1)
    n_epochs = 10
    
    print(f"\nTraining Parameters:")
    print(f"- Algorithm: Adaline (ADAptive LInear NEuron)")
    print(f"- Initial weights: {initial_weights}")
    print(f"- Initial bias: {initial_bias}")
    print(f"- Learning rate η: {learning_rate}")
    print(f"- Number of epochs: {n_epochs}")
    print(f"- Training on ALL {X.shape[0]} samples")
    
    # Initialize Adaline with specified parameters
    adaline = AdalineGD(eta=learning_rate, n_iter=n_epochs, random_state=0)
    
    # Set initial weights as specified in assignment
    adaline.set_weights(initial_weights, initial_bias)
    
    print(f"\nInitial state:")
    print(f"- Weights w: {adaline.w_}")
    print(f"- Bias b: {adaline.b_}")
    
    # Train the Adaline model on ALL samples
    print(f"\nTraining Adaline on all {X.shape[0]} samples...")
    adaline.fit(X, y)
    
    # Analyze training results
    predictions = adaline.predict(X)
    training_accuracy = np.mean(predictions == y)
    
    # MSE evolution analysis
    initial_mse = adaline.losses_[0] if adaline.losses_ else float('inf')
    final_mse = adaline.losses_[-1] if adaline.losses_ else float('inf')
    mse_decreased = final_mse < initial_mse
    
    print(f"\nTraining Results:")
    print(f"- Final weights w: {adaline.w_}")
    print(f"- Final bias b: {adaline.b_}")
    print(f"- Training accuracy: {training_accuracy:.3f} ({int(training_accuracy*100)}%)")
    print(f"- Correct predictions: {np.sum(predictions == y)}/{len(y)}")
    print(f"- Incorrect predictions: {np.sum(predictions != y)}/{len(y)}")
    
    # MSE Evolution Analysis - This answers the assignment questions
    print(f"\n" + "="*50)
    print("MSE EVOLUTION ANALYSIS")
    print("="*50)
    
    print(f"Initial MSE (epoch 1): {initial_mse:.6f}")
    print(f"Final MSE (epoch {len(adaline.losses_)}): {final_mse:.6f}")
    print(f"MSE decreased over epochs: {mse_decreased}")
    print(f"MSE reduction: {initial_mse - final_mse:.6f}")
    print(f"Relative MSE reduction: {((initial_mse - final_mse) / initial_mse * 100):.1f}%")
    
    # Epoch-by-epoch MSE evolution
    print(f"\nEpoch-by-epoch MSE evolution:")
    print(f"{'Epoch':<8} {'MSE':<12} {'Change':<12}")
    print("-" * 35)
    
    for i, mse in enumerate(adaline.losses_):
        epoch = i + 1
        if i == 0:
            change_str = "0.000000"
        else:
            change = mse - adaline.losses_[i-1]
            change_str = f"{change:+.6f}"
        
        print(f"{epoch:<8} {mse:<12.6f} {change_str:<12}")
    
    # Prediction Quality Analysis
    print(f"\n" + "="*50)
    print("PREDICTION QUALITY ANALYSIS")
    print("="*50)
    
    # Analyze prediction errors by class
    class_0_indices = y == 0
    class_1_indices = y == 1
    
    class_0_accuracy = np.mean(predictions[class_0_indices] == y[class_0_indices])
    class_1_accuracy = np.mean(predictions[class_1_indices] == y[class_1_indices])
    
    print(f"Class 0 (outside rectangle): {class_0_accuracy:.3f} accuracy")
    print(f"Class 1 (inside rectangle): {class_1_accuracy:.3f} accuracy")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: MSE evolution over epochs
    plt.subplot(1, 3, 1)
    epochs = range(1, len(adaline.losses_) + 1)
    plt.plot(epochs, adaline.losses_, 'bo-', linewidth=2, markersize=6)
    plt.title('MSE Evolution Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    
    # Plot 2: Training accuracy
    plt.subplot(1, 3, 2)
    plt.bar(['Training Accuracy'], [training_accuracy], color='skyblue', alpha=0.7)
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    # Plot 3: Prediction distribution
    plt.subplot(1, 3, 3)
    correct = np.sum(predictions == y)
    incorrect = np.sum(predictions != y)
    plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'], 
            colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
    plt.title('Prediction Distribution')
    
    plt.tight_layout()
    plt.savefig('figures/problem4_adaline_rectangle.png', dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plots saved to figures/problem4_adaline_rectangle.png")
    
    # Final Summary - Answers assignment questions
    print(f"\n" + "="*60)
    print("PROBLEM 4 SUMMARY - ASSIGNMENT ANSWERS")
    print("="*60)
    
    print(f"1. What do you observe?")
    if mse_decreased:
        print("   ✓ MSE consistently decreased over 10 epochs")
    else:
        print("   ✗ MSE did not decrease properly")
    
    if training_accuracy > 0.85:
        print("   ✓ Good training accuracy achieved")
    else:
        print("   ⚠ Moderate training accuracy")
    
    print(f"\n2. Evolution of MSE over 10 epochs:")
    print(f"   • Initial MSE: {initial_mse:.6f}")
    print(f"   • Final MSE: {final_mse:.6f}")
    print(f"   • Total reduction: {initial_mse - final_mse:.6f}")
    print(f"   • Pattern: {'Decreasing' if mse_decreased else 'Not decreasing'}")
    
    print(f"\n3. Quality of resultant w, b for prediction:")
    print(f"   • Final weights: {adaline.w_}")
    print(f"   • Final bias: {adaline.b_:.6f}")
    print(f"   • Training accuracy: {training_accuracy:.1%}")
    print(f"   • Prediction quality: {'Good' if training_accuracy > 0.8 else 'Moderate'}")
    
    return {
        'final_weights': adaline.w_.copy(),
        'final_bias': adaline.b_,
        'training_accuracy': training_accuracy,
        'initial_mse': initial_mse,
        'final_mse': final_mse,
        'mse_decreased': mse_decreased,
        'losses_per_epoch': adaline.losses_.copy()
    }


if __name__ == "__main__":
    # question1()
    question4()







