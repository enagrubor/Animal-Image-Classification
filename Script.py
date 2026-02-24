import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
tf.get_logger().setLevel(logging.ERROR)

from keras.utils import image_dataset_from_directory

#Koristimo samo podatke iz train foldera za obucavanje i validaciju dok podatke iz test foldera koristimo za testiranje
train_path = './train/'
test_path = './test/'

'''
#Kod koji smo koristili da uklonimo corruptovane slike
num_skipped = 0
for source_name in ("train", "test"):
    for folder_name in ("Cat", "Dog", "Horse", "Elephant","Lion"):
        folder_path = os.path.join(source_name, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print(f"Deleted {num_skipped} images.")
'''

#Unosimo podatke i klasifikujemo ih
img_size = (64,64)
batch_size = 64

Xtrain = image_dataset_from_directory(train_path, subset='training', validation_split = 0.2, image_size = img_size, batch_size = batch_size, seed =123)
Xval = image_dataset_from_directory(train_path, subset='validation', validation_split = 0.2, image_size = img_size, batch_size = batch_size, seed =123)
Xtest = image_dataset_from_directory(test_path, image_size = img_size, batch_size = batch_size, seed =123)

classes = Xtrain.class_names
print(classes)

#Prikazivanje primera slika
N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')


#Klase su nebalansirane pa moramo da dodamo class weight, nalazimo broj img svake klase i racunamo tezine
from sklearn.utils import class_weight
import pandas as pd
import fnmatch


Dog_path = './train/Dog'
Cat_path = './train/Cat'
Elephant_path = './train/Elephant'
Horse_path = './train/Horse'
Lion_path = './train/Lion'
count_Dog = len(fnmatch.filter(os.listdir(Dog_path), '*.*'))
count_Cat = len(fnmatch.filter(os.listdir(Cat_path), '*.*'))
count_Elephant = len(fnmatch.filter(os.listdir(Elephant_path), '*.*'))
count_Horse = len(fnmatch.filter(os.listdir(Horse_path), '*.*'))
count_Lion = len(fnmatch.filter(os.listdir(Lion_path), '*.*'))

Y = pd.Series(np.concatenate((np.repeat('Dog',count_Dog), np.repeat('Cat',count_Cat), np.repeat('Elephant',count_Elephant), np.repeat('Horse',count_Horse), np.repeat('Lion',count_Lion))))

#Prikazivanje nebalansiranosti klasa
plt.figure()
Y.hist()
plt.show()

weights = class_weight.compute_class_weight( class_weight = 'balanced', classes = classes, y=Y)
c_weight={0:weights[0], 1:weights[1], 2:weights[2], 3:weights[3], 4:weights[4]}

#Pravimo neuralnu mrezu
from keras import layers
from keras import Sequential
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.regularizers import l2

        
num_classes = len(classes)


data_augmentation = Sequential(
    [
     layers.RandomFlip("horizontal", input_shape = (img_size[0], img_size[1], 3)),
     layers.RandomRotation(0.25),
     layers.RandomZoom(0.1),
     ]
    )

#Prikazivanje preprocesovanih slika
N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')

def make_model(par):
    
    model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64,64,3)),
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation = par['activation'], kernel_regularizer=l2(par['reg'])),
    layers.Dense(num_classes, activation = 'softmax')
    ])

    model.summary()
    model.compile(Adam(learning_rate = par['lr']), loss = SparseCategoricalCrossentropy(), metrics = 'accuracy')
    return model

def display_train_result(history, model):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    plt.subplot(121)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Accuracy')
    plt.subplot(122)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Loss')
    plt.show()
    
    labels_val = np.array([])
    pred_val = np.array([])
    for img, lab in Xval:
        labels_val = np.append(labels_val, lab)
        pred_val = np.append(pred_val, np.argmax(model.predict(img, verbose=0), axis=1))

    from sklearn.metrics import accuracy_score
    score = 100*accuracy_score(labels_val, pred_val)
    print('Tacnost modela na validaciji je: ' + str(score) + '%')
    print('==========================================================')
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(labels_val, pred_val, normalize='true')
    cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    cmDisplay.plot()
    plt.show()
    
    return score

parameter = {}
parameter['activation'] = 'relu'
parameter['lr'] = 0.001
parameter['reg'] = 0.01

'''
#Aktivacione funkcije
var = 'activation'
parameter[var] = ['relu', 'tanh', 'sigmoid']
for i in parameter[var]:
    setting = parameter
    setting[var] = i
    model = make_model(setting)
    history = model.fit(Xtrain, epochs = 100, validation_data = Xval, class_weight=c_weight, verbose = 2)
    score = display_train_result(history, model)
    
parameter['activation'] = 'relu'
'''
  
#Konstanta ucenja
var = 'lr'
#parameter[var] = [0.001, 0.0001, 0.00001]
parameter[var] = [0.01]
for i in parameter[var]:
    setting = parameter
    setting[var] = i
    model = make_model(setting)
    history = model.fit(Xtrain, epochs = 100, validation_data = Xval, class_weight=c_weight, verbose = 2)
    score = display_train_result(history, model)

parameter['lr'] = 0.001

#Regularizacija
var = 'reg'
parameter[var] = [ 0.1 , 0.001]
for i in parameter[var]:
    setting = parameter
    setting[var] = i
    model = make_model(setting)
    history = model.fit(Xtrain, epochs = 100, validation_data = Xval, class_weight=c_weight, verbose = 2)
    score = display_train_result(history, model)

parameter['reg'] = 0.001

#Broj potrebnih epoha
model = make_model(parameter)
history = model.fit(Xtrain, epochs = 200, validation_data = Xval, class_weight=c_weight, verbose = 2)
score = display_train_result(history, model)

#Testiranje modela
labels_test = np.array([])
pred_test = np.array([])
for img, lab in Xtest:
    labels_test = np.append(labels_test, lab)
    pred_test = np.append(pred_test, np.argmax(model.predict(img, verbose=0), axis=1))
    
from sklearn.metrics import accuracy_score
score = 100*accuracy_score(labels_test, pred_test)
print('Tacnost modela na testu je: ' + str(score) + '%')
print('==========================================================')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels_test, pred_test, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

#Prikazivanje pogresnih klasifikacija
N = 10
count = 0
plt.figure()
for img, lab in Xtest.take(1):
    pred = np.argmax(model.predict(img, verbose=0), axis=1)
    for i in range(pred.size):
        if (pred[i]!=lab[i]):
            plt.subplot(2, int(N/2), count+1)
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.title(classes[pred[i]])
            plt.axis('off')
            count+=1
        if (count>=10): 
            break
