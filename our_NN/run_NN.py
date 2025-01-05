import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.io import arff 
import pandas as pd         #estas dos para el EMNIST

from NN_entrega import Neural_Network

# Load the MNIST dataset
MNIST = datasets.fetch_openml('mnist_784', version=1)

# Features: Divide each pixel by 255 to normalize between 0 and 1
X = MNIST.data / 255.0  

# Targets: Convert to integer type
y = MNIST.target.astype(np.int8)
y = np.array(y)  # Convert targets to a NumPy array for consistent indexing

# First split: separate out the 10% for future predictions
# X_part, X_user, y_part, y_user = train_test_split(X, y, test_size=0.1, random_state=42)
# Second split: split the remaining 90% into 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("ENTROPY")
net = Neural_Network([784, 30, 10])
net.SGD_entropy(X_train_scaled, y_train, 30, 10, 0.3, 5,test_data=X_test_scaled, test_solutions=y_test) #Train the net
#print("QUADRATIC")
#net = Neural_Network([784, 30, 10])
#net.SGD_quad(X_train_scaled, y_train, 30, 10, 3, test_data=X_test_scaled, test_solutions=y_test) #Train the net
#net.menu(X_test_scaled, y_test, scaler)  #Display menu
#net.setup_gui() # Set up the GUI and start the application (emergent window)
