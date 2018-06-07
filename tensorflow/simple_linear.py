import tensorflow as tf

#       model Parameters

w = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)

#       inputs and outputs

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = w*x+b

#       Loss

squared_data = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_data)

init =  tf.global_variables_initializer()

sess =tf.Session()
sess.run(init)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
