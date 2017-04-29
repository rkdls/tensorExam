import tensorflow as tf

da = tf.random_normal([3, 3])

dd = tf.truncated_normal([3, 3])

d2 = tf.random_uniform([3, 3], -1.0, 1.0)
with tf.Session() as sess:
    df, lk, dfw = sess.run([da, dd, d2])
    print(df)
    print()
    print(lk)
    print()
    print(dfw)
