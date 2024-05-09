# Reference link - https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from numpy import loadtxt
from sklearn.model_selection import train_test_split

# Change the path to the data to where its located on your computer
dataset = loadtxt('C:\\Users\\Emil\\PycharmProjects\\CapstoneKerasML\\Training\\2024newdata.csv', delimiter=',') # load the dataset

# Split into input (X) and output (y) variables
X = dataset[:, :-1]  # everything except last column
y = dataset[:, -1]  # labels

y_one_hot = to_categorical(y, num_classes=2)

# Split into training and testing sets
# Test size is 15% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.15, random_state=42)

# Define the Keras model
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))  # First layer 64 nodes
model.add(Dense(32, activation='relu'))  # Second layer 32 nodes
model.add(Dense(2, activation='sigmoid'))  # output layer 2 nodes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras model on the dataset
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
model.save('C:\\Users\\Emil\\PycharmProjects\\CapstoneKerasML\\Testing\\trained_model.h5')  # Save model

# Check performance on test set (splitting)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')