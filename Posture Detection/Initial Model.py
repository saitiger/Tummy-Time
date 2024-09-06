import cv2
import os
import pyopenpose as op
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

video_path = '../Vid_1.mp4'

video = cv2.VideoCapture(video_path)

frame_count = 0
extracted_frames = []
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Save every 5th frame to save space and reducing processing time
    if frame_count%5 == 0:
        frame_filename = f'{output_dir}/frame_{frame_count}.jpg'
        cv2.imwrite(frame_filename, frame)
        extracted_frames.append(frame_filename)

    frame_count += 1

video.release()

preview = extracted_frames[:20] 

# Using Openpose for Initial Labelling until all files are handgraded by experts 

params = dict()
params["model_folder"] = "../openpose/models/"  

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

frames_dir = '../extracted/frames/'

for frame_file in os.listdir(frames_dir):
    if frame_file.endswith('.jpg'):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        output_image = datum.cvOutputData
        output_path = os.path.join(output_dir, f"openpose_{frame_file}")
        cv2.imwrite(output_path, output_image)

        print(f"Processed {frame_file}, keypoints: {datum.poseKeypoints}")


data_dir = '../extracted/frames/labeled/'

image_size = (224, 224)
batch_size = 32
num_classes = 3  
epochs = 10

# Feature Engineering 
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

model.save('posture_detection_model.h5')
