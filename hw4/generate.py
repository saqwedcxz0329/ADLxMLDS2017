
import tensorflow as tf
import numpy as np

import utils
from utils import Data
from improved_WGAN import Improved_WGAN

tf.flags.DEFINE_integer("iter", 1000000, "number of training iter")
tf.flags.DEFINE_integer("z_dim", 100, "noise dimension")
tf.flags.DEFINE_integer("batch_size", 64, "batch size per iteration")
tf.flags.DEFINE_integer("display_every", 20, "predict model on dev set after this many steps (default: 20)")
tf.flags.DEFINE_integer("dump_every", 500, "predict model on dev set after this many steps (default: 500)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 500)")

tf.flags.DEFINE_float("lr", 0.0002, "training learning rate")

tf.flags.DEFINE_string("img_dir", "./samples/", "test image directory")
tf.flags.DEFINE_string("train_dir", "./data/faces", "training data directory")
tf.flags.DEFINE_string("tag_path", "./data/tags_clean.csv", "training data tags")
tf.flags.DEFINE_string("test_path", "./data/testing_text.txt", "sample test format")

tf.flags.DEFINE_string("model_path", "./models", "model folder")
tf.flags.DEFINE_string("model_name", "model", "model name")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def generate():
    img_feat, tag_feat = None, None
    test_tag_feat = utils.load_test(FLAGS.test_path)

    data = Data(img_feat, tag_feat, test_tag_feat, FLAGS.z_dim)

    model = Improved_WGAN(data, FLAGS)

    model.build_model()

    for i in range(500, 50500, 500):
        checkpoint_prefix = '{}-{}'.format(FLAGS.model_name, i)

        model.restore(FLAGS.model_path, checkpoint_prefix)

        model.gen_test_img(i)

if __name__ == '__main__':
    generate()