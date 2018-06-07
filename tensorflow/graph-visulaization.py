
File_writer = tf.summary.FileWriter("directory", sess.graph)


""" commandline : tensorboard --logdir="path to the graph"
execute this command in cmd
tensorboard run as a local webapp, on port 6006;
6006 is "goog" upside down."""
