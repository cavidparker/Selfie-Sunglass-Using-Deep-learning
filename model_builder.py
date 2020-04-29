from utilts import load_data
from my_CNN_model import *
import cv2

# loading training set
X_train, y_train = load_data()


# Set the CNN architecture :

my_model = get_my_CNN_model_arch()

# compile the CNN model with an approprite optimizer and loss and metrics
compile_my_CNN_model(my_model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# tarining the model

hist = train_my_CNN_model(my_model, X_train, y_train)

# train_my_CNNmodel returns a history object

# Saving the model:

save_my_CNN_model(my_model, 'my_model')