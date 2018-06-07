import tensorflow as tf
x1 = tf.constant(5)
x2 = tf.constant(9)
result = tf.multiply(x1,x2)

#result = x1*x2
#print(result)  #will not work

#method 1

#sess = tf.Session() #session object operation objectcts are executed
#print(sess.run(result))
#sess.close()

#OR Method 2

with tf.Session() as sess:
    output = sess.run(result)
    print(output)
