from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

# Ptah Variables
IMAGE_FILE = './data/driving_log.csv'
IMAGE_PATH = './data/'

def collectImages():
	
	right_image_paths = []
	left_image_paths = []
	center_image_paths = []
	steering_angles = []

	# Import driving data from the log file
	with open(IMAGE_FILE, newline='') as image_file:
		driving_data = list(csv.reader(image_file, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

	# Fill data of image paths and angles for center, left, right cameras in each row
	for row in driving_data[1:]:
		# skip it if ~0 speed - not representative of driving behavior
		if float(row[6]) < 0.1 :
			continue
		# get center image path and angle
		center_image_paths.append(row[0])
		# get left image path and angle
		left_image_paths.append(row[1])
		# get left image path and angle
		right_image_paths.append(row[2])
		# Get steering angle for these images
		steering_angles.append(float(row[3]))
		
	total_image_paths = []
	total_steering_angles = []
	steering_correction = 0.25
	
	total_image_paths.extend(center_image_paths)
	total_image_paths.extend(left_image_paths)
	total_image_paths.extend(right_image_paths)
	
	total_steering_angles.extend(steering_angles)
	total_steering_angles.extend([x + steering_correction for x in steering_angles])
	total_steering_angles.extend([x - steering_correction for x in steering_angles])
	


def preProcessLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model
	

def nvidiaAuto():
    """
    Creates the Autonomour Car Neural Network by Nvidia
    """
    model = preProcessLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
	
def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
	
# Reading images locations.
all_image_paths, all_steering_angles = collect_images()
print('Total Images: {}'.format( len(all_image_paths)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
all_data = list(zip(all_image_paths, all_steering_angles))
train_set, validation_set = train_test_split(all_data, test_size=0.2)

print('Training set samples: {}'.format(len(train_set)))
print('Validation set Samples: {}'.format(len(validation_set)))

train_generator = generator(train_set, batch_size=32)
validation_generator = generator(validation_set, batch_size=32)

# Model creation
model = nVidiaModel()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
