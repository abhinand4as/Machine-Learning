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

#       optimizer

optimizer = tf.train.GradientDescentOptimizer(0.01)     #0.01 is learning rate
train = optimizer.minimize(loss)

init =  tf.global_variables_initializer()

sess =tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train,{x:[1,2,3,4], y:[0,-1,-2,-3]})
    
print(sess.run([w,b]))
    
""" The actual weight and bias required are -1 and 1; we get -0.9999 and 0.9999 for
w and b  """
