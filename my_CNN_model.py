def get_my_CNN_model_arch():
	model = Sequential()
	model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))

	model.add(Convolution2D(128, (3, 3), activation= 'relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Convolution2D(30, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))

	model.add(Flatten())

	model.add(Dense(64, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(30))

	return model;

def compile_my_cnn_model(model, optimizer, loss, metrics):
	model.compile(optimizer=optimizer,loss=loss, metrics=metrics)

def train_my_CNN_model(model, X_train, y_train):
	return model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split = 0.2
		

