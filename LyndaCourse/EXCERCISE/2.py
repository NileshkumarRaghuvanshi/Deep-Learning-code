import os
import tensorflow as tf

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define computational graph
X = tf.placeholder(dtype=tf.float32, name="X")
Y = tf.placeholder(dtype=tf.float32, name="Y")

addition = tf.add(X, Y, name="addition")


# Create the session
with tf.Session() as session:

    # we need to init X and Y here, either they can be defined initially or they can
    #be fed at run time using feed_dict
    # see tf.constant for referrence to set constant value
    # placeholder seems to be generic 
    result = session.run(addition, feed_dict={X: [[1,2,3], [3,4,5]], Y:[[3,4,5], [5,6,7]]})

    print(result)

print("done")
session.close()
