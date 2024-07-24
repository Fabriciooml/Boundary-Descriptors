import cv2
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from os import listdir
from os.path import isfile, join

np.set_printoptions(threshold=np.inf)


def create_neural_net_statistical_moments(hp):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hp.Choice('units', [8, 16, 32]), hp.Choice('activation', ['relu', 'tanh']),
                              hp.Choice('kernel_initializer', ['random_uniform', 'normal']), input_dim=24),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hp.Choice('units', [8, 16, 32]), hp.Choice('activation', ['relu', 'tanh']),
                              hp.Choice('kernel_initializer', ['random_uniform', 'normal'])),
        tf.keras.layers.Dense(0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])

    model.compile(hp.Choice('optimizer', ['adam', 'sgd']), hp.Choice('loss', ['binary_crossentropy', 'hinge']),
                  metrics=['binary_accuracy'])
    return model


X = []
y = []
def get_breast_cancer_image_paths():
    not_cancer_folder = "images/Mamografia/Treinamento/Correta/"
    cancer_folder = "images/Mamografia/Treinamento/Errada/"
    not_cancer_images = [f for f in listdir(not_cancer_folder) if isfile(join(not_cancer_folder, f))]
    cancer_images = [f for f in listdir(cancer_folder) if isfile(join(cancer_folder, f))]
    return not_cancer_images, cancer_images

breast_cancer_fourier_descriptors = []
breast_cancer_shape_numbers = []
breast_cancer_statistical_moments = []
not_cancer, cancer = get_breast_cancer_image_paths()

for path in not_cancer:
    image = cv2.imread("images/Mamografia/Treinamento/Correta/"+path, cv2.IMREAD_GRAYSCALE)
    canny_output = cv2.Canny(image, 100, 100 * 2)
    statistical_moments = cv2.moments(canny_output)
    X.append(list(statistical_moments.values()))
    y.append(0)

for path in cancer:
    image = cv2.imread("images/Mamografia/Treinamento/Errada/"+path, cv2.IMREAD_GRAYSCALE)
    canny_output = cv2.Canny(image, 100, 100 * 2)
    statistical_moments = cv2.moments(canny_output)
    X.append(list(statistical_moments.values()))
    y.append(1)

X = np.array(X)
print(X)
y = np.array(y)

classifier = kt.Hyperband(create_neural_net_statistical_moments, objective='val_binary_accuracy', max_epochs=100, factor=3)
# parameters = {'batch_size': [10, 30],
#               'epochs': [50, 100],
#               'optimizer': ['adam', 'sgd'],
#               'loos': ['binary_crossentropy', 'hinge'],
#               'kernel_initializer': ['random_uniform', 'normal'],
#               'activation': ['relu', 'tanh'],
#               'neurons': [16, 8]}
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
classifier.search(X, y, epochs=100, validation_split=0.5, callbacks=[stop_early])

best_hps = classifier.get_best_hyperparameters(num_trials=1)[0]

print(f"""
{best_hps.get('units')}
{best_hps.get('activation')}
{best_hps.get('kernel_initializer')}
{best_hps.get('optimizer')}
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = classifier.hypermodel.build(best_hps)
history = model.fit(X, y, epochs=100, validation_split=0.5)

val_acc_per_epoch = history.history['val_binary_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
