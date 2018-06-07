""" Constants : It takes no input. """
import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(5.0)
print(node1,node2)

""" placeholder : Accecpt external input. A placeholder is a promise to provide a value later """
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a+b
sess = tf.Session()
print(sess.run(adder, {a:[1,3], b:[2,4]}))

""" Variable : allow to add trainable parameters to the graph """
w = tf.variable([3],tf.float32)
b = tf.variable([-3]tf.float32)
x = tf.placeholder(tf.float32)

linear_model = w*x+b
init =  tf.global_variables_initializer()

sess =tf.Session()
sess.run(init)
print(sess.run(linear_model, x:{1,2,3,4})
