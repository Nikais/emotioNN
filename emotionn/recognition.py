import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', help='train/display')
mode = ap.parse_args().mode

# Root path to project
ROOT = Path(__file__).absolute().parent.parent


def create_image_generator(path: Path, batch_size):
    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_directory(
        path,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')


def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1),
                      len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    image = ROOT / 'images'
    image.mkdir(exist_ok=True)
    fig.savefig(image / 'plot.png')
    plt.show()


if __name__ == '__main__':
    # Hyper parameters
    epochs = 20
    batch_size = 32

    model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(),
        Dropout(0.25),

        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(),
        Dropout(0.25),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, 'softmax')
    ])

    if mode == 'train':
        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001, decay=1e-6), metrics=['accuracy', ])
        (ROOT / 'model/checkpoints').mkdir(parents=True, exist_ok=True)

        train_generator = create_image_generator(ROOT / 'data/fer2013/train', batch_size)
        validation_generator = create_image_generator(ROOT / 'data/fer2013/test', batch_size)
        model_checkpoint_callback = ModelCheckpoint(
            filepath=str(ROOT / 'model/checkpoints/weights-epoch:{epoch:02d}-acc:{val_accuracy:.2f}.hdf5'),
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        model_info = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[model_checkpoint_callback, ])

        plot_model_history(model_info)
        model.save_weights(str(ROOT / 'model/model_weights.h5'))

    elif mode == 'display':
        model.load_weights(str(ROOT / 'model/model_weights.h5'))
        cv2.ocl.setUseOpenCL(False)
        emotions = ['Angry', 'Digusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

        face_cascade = cv2.CascadeClassifier(str(ROOT / 'emotionn/haarcascade_frontalface_default.xml'))
        # start the webcam feed
        cap = cv2.VideoCapture(0)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                cv2.putText(frame, f'{emotions[max_index]}', (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255),
                            2, cv2.LINE_AA)
            cv2.imshow('Emotionn', cv2.resize(frame, (720, 480), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
