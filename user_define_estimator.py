import tensorflow.compat.v1 as tf
import os
import sys
if sys.version_info < (3, 0, 0):
    from urllib import urlopen
else:
    from urllib.request import urlopen

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.3" <= tf_version, "TensorFlow r1.3 or later is needed"

# Windows users: You only need to change PATH, rest is platform independent
PATH = ".\data"

# Fetch and store Training and Test dataset files
PATH_DATASET = PATH + os.sep + "dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "iris_training.csv"
FILE_TEST = PATH_DATASET + os.sep + "iris_test.csv"
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"

def downloadDataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)
    if not os.path.exists(file):
        data = urlopen(url).read()
        with open(file, "wb") as f:
            f.write(data)
            f.close()
downloadDataset(URL_TRAIN, FILE_TRAIN)
downloadDataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)

# The CSV features in our training & test data
feature_names = ['SepalLength',
                'SepalWidth',
                'PetalLength',
                'PetalWidth']

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API


def my_input_fn(file_path, repeat_count=1, shuffle_count=1):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
       label = parsed_line[-1]  # Last element is the label
       del parsed_line[-1]  # Delete last element
       features = parsed_line  # Everything but last elements are the features
       d = dict(zip(feature_names, features)), label
       return d
   dataset = (tf.data.TextLineDataset(file_path)  # Read text file
                .skip(1)  # Skip header row
                .map(decode_csv, num_parallel_calls=4)  # Decode each line
                .cache() # Warning: Caches entire dataset, can cause out of memory
                .shuffle(shuffle_count)  # Randomize elems (1 == no operation)
                .repeat(repeat_count)    # Repeats dataset this # times
                .batch(32)
                .prefetch(1)  # Make sure you always have 1 batch ready to serve
   )
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels

feature_columns = [
   tf.feature_column.numeric_column(feature_names[0]),
   tf.feature_column.numeric_column(feature_names[1]),
   tf.feature_column.numeric_column(feature_names[2]),
   tf.feature_column.numeric_column(feature_names[3])
]

def my_model_fn(features, labels, mode):
    #Create the layer of input
    input_layer = tf.feature_column.input_layer(features, feature_columns)

    # Definition of hidden layer: h1
    # (Dense returns a Callable so we can provide input_layer as argument to it)
    h1 = tf.layers.Dense(10, activation=tf.nn.relu)(input_layer)
    # Definition of hidden layer: h2
    # (Dense returns a Callable so we can provide h1 as argument to it)
    h2 = tf.layers.Dense(10, activation=tf.nn.relu)(h1)

    # Output 'logits' layer is three numbers = probability distribution
    # (Dense returns a Callable so we can provide h2 as argument to it)
    logits = tf.layers.Dense(3)(h2)

    # class_ids will be the model prediction for the class (Iris flower type)
    # The output node with the highest value is our prediction
    predictions = { 'class_ids': tf.argmax(input=logits, axis=1) }
    # Return our prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # To calculate the loss, we need to convert our labels
    # Our input labels have shape: [batch_size, 1]
    # labels = tf.squeeze(labels, 1)          # Convert to shape [batch_size]
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Calculate the accuracy between the true labels, and our predictions
    accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])
    # Return our loss (which is used to evaluate our model)
    # Set the TensorBoard scalar my_accurace to the a`ccuracy
    # Obs: This function only sets value during mode == ModeKeys.EVAL
    # To set values during training, see tf.summary.scalar
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops={'my_accuracy': accuracy})
    
    optimizer = tf.train.AdagradOptimizer(0.05)
    train_op = optimizer.minimize(
        loss,
        global_step=tf.train.get_global_step())
    # Set the TensorBoard scalar my_accuracy to the accuracy
    tf.summary.scalar('my_accuracy', accuracy[1])

    # Return training operations: loss and train_op
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op)


next_batch = my_input_fn(FILE_TRAIN, True)  # Will return 32 random elements

# Path to where checkpoints etc are stored
classifier = tf.estimator.Estimator(model_fn=my_model_fn, model_dir=PATH)  

print('###############################')
print('#       start training        #')
print('###############################')

# classifier.train(input_fn=lambda: my_input_fn(FILE_TRAIN, repeat_count=500, shuffle_count=10))



# Predict the type of some Iris flowers.
# Let's predict the examples in FILE_TEST, repeat only once.
predict_results = classifier.predict(input_fn=lambda: my_input_fn(FILE_TEST, 1, 1))

# predict_results = [i for i in predict_results]

print("Predictions on test file")
print(predict_results)
for prediction in predict_results:
    # Will print the predicted class, i.e: 0, 1, or 2 if the prediction
    # is Iris Sentosa, Vericolor, Virginica, respectively.
    print(prediction)
    # print("   {}, was: {}".format(prediction, predict_results[prediction]))

    # print(prediction["class_ids"][0])

# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = classifier.evaluate(input_fn=lambda: my_input_fn(FILE_TEST, 1, 4))
print("Evaluation results")
for key in evaluate_result:
    print(key)
    # print("   {}, was: {}".format(key, evaluate_result[key]))

# Let create a dataset for prediction
# We've taken the first 3 examples in FILE_TEST
prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa


def new_input_fn():
    def decode(x):
        x = tf.split(x, 4)  # Need to split into our 4 features
        return dict(zip(feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # In prediction, we have no labels

# Predict all our prediction_input
predict_results = classifier.predict(input_fn=new_input_fn)

# Print results
print("Predictions on memory")
for idx, prediction in enumerate(predict_results):
    print(idx, prediction)
    # type = prediction["class_ids"][0]  # Get the predicted class (index)
    # if type == 0:
    #     print("I think: {}, is Iris Sentosa".format(prediction_input[idx]))
    # elif type == 1:
    #     print("I think: {}, is Iris Versicolor".format(prediction_input[idx]))
    # else:
    #     print("I think: {}, is Iris Virginica".format(prediction_input[idx]))
