from tensorflow.keras.losses import MSE
import tensorflow as tf

def adversary_create(model, image, label, eps = 2 / 255.0):
	element = tf.cast(image, tf.float32)
	with tf.GradientTape() as t:
		t.watch(image)
		prediction = model(image)
		ls = MSE(label, prediction)
	grad = tape.gradient(ls, image)
	signedGrad = tf.sign(grad)
	adversary_image = (image + (signedGrad * eps)).numpy()
	return adversary