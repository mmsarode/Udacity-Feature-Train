import pickle
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

nb_classes = 43
epochs = 10
batch_size = 128

# TODO: Load traffic signs data.

# /home/mahesh/udacity/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data


# training_file = "/home/mahesh/udacity/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p"
training_file = './train.p'
testing_file = "/home/mahesh/udacity/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)
    
X_train, y_train = [train['features'], train['labels']]
# X_test, y_test = test['features'], test['labels']

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.33, random_state=0)

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)

resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the traffic signs
# model.

fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)

shape = (fc7.get_shape().as_list()[-1], nb_classes)


fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

init_op = tf.initialize_all_variables()


# init_op = tf.global_variables_initializer()

preds = tf.arg_max(logits, 1)

accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


# def evaluate(X_data, y_data):
#     num_examples = len(X_data)
#     total_accuracy = 0
#     sess = tf.get_default_session()
#     for offset in range(0, num_examples, BATCH_SIZE):
#         batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
#         accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
#         total_accuracy += (accuracy * len(batch_x))
#     return total_accuracy / num_examples

def eval_on_data(X, y, sess):
	total_acc = 0
	total_loss = 0

	for offset in range(0, X.shape[0], batch_size):
		end = offset + batch_size
		X_batch, y_batch = X[offset:end], y[offset:end]
		loss, acc = sess.run([loss_operation, accuracy_operation], feed_dict={features: X_batch, labels: y_batch})
		total_loss += (loss * X_batch.shape[0])
		total_acc += (acc * X_batch.shape[0])
	return total_loss/X.shape[0], total_acc/X.shape[0]

with tf.Session() as sess:
	sess.run(init_op)

	for i in range(epochs):
		X_train, y_train = shuffle(X_train, y_train)
		t0 = time.time()

		for offset in range(0, X_train.shape[0], batch_size) :
			end = offset + batch_size
			print(end)
			sess.run(training_operation, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})
		val_loss, val_acc = eval_on_data(X_validation, y_validation, sess)
		print("Epoch", i+1)
		print("Time: %.3f seconds" % (time.time() - t0))
		print("Validation Loss =", val_loss)
		print("Validation Accuracy =", val_acc)
		print("")
		
        	
            

        



        

















# y = tf.placeholder(tf.int32, (None))
# one_hot_y = tf.one_hot(y, 43)
# # TODO: Split data into training and validation sets.
# rate = 0.001
# # TODO: Define placeholders and resize operation.
# # tf.placeholder(tf.float32, (None, 32, 32, 3))
# # TODO: pass placeholder as first argument to `AlexNet`.
# data_input = tf.placeholder(tf.float32, (None, 227, 227, 3) 
# fc7 = AlexNet(data_input, feature_extract=True)
# # NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# # past this point, keeping the weights before and up to `fc7` frozen.
# # This also makes training faster, less work to do!
# fc7 = tf.stop_gradient(fc7)

# fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
# fc8b = tf.Variable(tf.zeros(nb_classes))

# logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    
# # probs = tf.nn.softmax(logits)





# # TODO: Add the final layer for traffic sign classification.


# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
# loss_operation = tf.reduce_mean(cross_entropy)
# optimizer = tf.train.AdamOptimizer(learning_rate = rate)
# training_operation = optimizer.minimize(loss_operation)

# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
# accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# saver = tf.train.Saver()

# def evaluate(X_data, y_data):
#     num_examples = len(X_data)
#     total_accuracy = 0
#     sess = tf.get_default_session()
#     for offset in range(0, num_examples, BATCH_SIZE):
#         batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
#         accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
#         total_accuracy += (accuracy * len(batch_x))
#     return total_accuracy / num_examples

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(X_train_n)
    
#     print("Training...")
#     print()
#     for i in range(EPOCHS):
#         X_train_n, y_train_n = shuffle(X_train_n, y_train_n)
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = X_train_n[offset:end], y_train_n[offset:end]
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
#         validation_accuracy = evaluate(X_validation_n, y_validation_n)
#         training_accuracy = evaluate(X_train_n, y_train_n)
#         print("EPOCH {} ...".format(i+1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print("Training Accuracy = {:.3f}".format(training_accuracy))
#         print()

#     testing_accuracy = evaluate(X_test_n, y_test)
#     print("Training Accuracy = {:.3f}".format(testing_accuracy))
        
#     saver.save(sess, './P2AWS_3Feb_normalize_05augmented_06')
#     print("Model saved")





# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.
