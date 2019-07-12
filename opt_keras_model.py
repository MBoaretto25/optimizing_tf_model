"""
Script to convert keras models into tensorflow model
## based on vgg4opencv in tutorial by @rodrigo2019
"""

import os
import glob
import keras
import tensorflow as tf
import keras.backend as K

from tkinter import *
from tkinter.filedialog import askopenfilename

from utils import freeze


def keras2tf(file_path):
    print("[INFO] Freezing model...")

    K.set_learning_phase(0)

    model_file_basename, file_extension = os.path.splitext(os.path.basename(file_path))

    # This needs to be altered depending on the model
    custom_objects = {'tf': tf}

    model = keras.models.load_model(file_path, custom_objects=custom_objects)

    model_input = model.input.name.replace(':0', '')
    model_output = model.output.name.replace(':0', '')

    sess = K.get_session()

    width, height, channels = int(model.input.shape[2]), int(model.input.shape[1]), int(model.input.shape[3])

    freeze(sess, model_file_basename, model_input, width, height, channels, model_output)

    # Remove not important files
    removables = []
    [removables.append(x) for x in glob.glob('*.ckpt*')]
    [removables.append(x) for x in glob.glob('*.binary.pb')]
    [removables.append(x) for x in glob.glob('checkpoint')]
    [removables.append(x) for x in glob.glob('*.sh')]
    for f in removables:
        os.remove(f)


def main():

    print("[INFO] loading file path with Tkinter...")
    root = Tk()
    root.update()
    file_path = askopenfilename()
    root.destroy()

    keras2tf(file_path)


if __name__ == '__main__':
    main()





