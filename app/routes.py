from app import app
from flask import Flask,redirect,request,url_for
from flask import render_template
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

img_width, img_height = 150, 150

train_data_dir = 'images/train'
validation_data_dir = 'images/valid'
nb_train_samples = 90
nb_validation_samples = 20
epochs = 50
batch_size = 16

@app.route('/')
@app.route('/index')
def index():
    return render_template('Defend.html', title='Ucode2019')

@app.route('/gooddef')
def gooddef():
    return render_template('noeresunmanta.html', title='Ucode2019')

@app.route('/shittydef')
def shittydef(foto):
    return render_template('eresunmanta.html', title='Ucode2019', foto=foto)

@app.route("/upload", methods=['POST'])
def upload():
    # Creamos la ruta donde vamos a guardar las imagenes
    target = os.path.join(APP_ROOT, 'static/subidas/')
    print(target)

    # Si no existe la carpeta, la creamos.
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file) #Debug
        # Cogemos el nombre del archivo como nombre que se va a guardar, por ahora.
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)  #Debug
        file.save(destination)

    if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
    else:
            input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy'])

    red_entrenada = os.path.join(APP_ROOT, 'first_try.h5')
    model.load_weights(red_entrenada)
    img = image.load_img(destination, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)

    K.clear_session()
    if classes[0]:
        print("Meh")
        return shittydef(filename)
    else:
        print("OK")
        return redirect("/gooddef")
