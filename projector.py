from tensorflow.contrib.tensorboard.plugins import projector
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import os
import sys
import math
import pickle

import facenet

LOG_DIR = 'projector'
NAME_TO_VISUALISE_VARIABLE = "face_embedding"

sprites_fn = 'faces.jpg'; metadata_fn = 'metadata.tsv'
path_for_sprites =  os.path.join(LOG_DIR, sprites_fn)
path_for_metadata =  os.path.join(LOG_DIR, metadata_fn)

this_dir  = os.path.dirname(os.path.realpath(__file__))
data_dir  = os.path.join(this_dir, "data/dataset/test")
model_dir = os.path.join(this_dir, "data/20180402-114759/")

batch_size = 10
image_size = 160

g = tf.Graph()
with g.as_default():
    with tf.Session() as sess:
        dataset = facenet.get_dataset(data_dir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        # Restoe model
        facenet.load_model(model_dir)
        # Get input and output tensors
        images_placeholder = g.get_tensor_by_name("input:0")
        embeddings = g.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = g.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))

        for i in range(nrof_batches_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)

            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)


# Number of data points
TO_EMBED_COUNT = emb_array.shape[0]

batch_xs = emb_array
batch_ys = np.array(labels)

embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)

grid = int(math.sqrt(len(paths))) + 1
image_height = int(8192 / grid)         # tensorboard supports sprite images up to 8192 x 8192
image_width = int(8192 / grid)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Specify where you find the metadata
embedding.metadata_path = metadata_fn #'metadata.tsv'

# Specify where you find the sprite (we will create this later)
embedding.sprite.image_path = sprites_fn #'png'
embedding.sprite.single_image_dim.extend([image_height,image_width])

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)

big_image = Image.new(
    mode='RGB',
    size=(image_width * grid, image_height * grid),
    color=(0,0,0,0))  # fully transparent

for i in range(len(paths)):
    row = i // grid
    col = i % grid
    img = Image.open(paths[i])
    img = img.resize((image_height, image_width), Image.ANTIALIAS)
    row_loc = row * image_height
    col_loc = col * image_width
    big_image.paste(img, (col_loc, row_loc)) # NOTE: the order is reverse due to PIL saving

big_image.save(path_for_sprites, transparency=0)


class_names = [ cls.name.replace('_', ' ') for cls in dataset]
with open(path_for_metadata,'w') as f:
    f.write("Index\tLabel\tName\n")
    for index,label in enumerate(batch_ys):
        f.write("{}\t{}\t{}\n".format(index,label,class_names[batch_ys[index]]))


