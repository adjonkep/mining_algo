import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import time

# only displays the most important warnings
tf.logging.set_verbosity(tf.logging.FATAL)
start_time = time.time()

train_data = pd.read_csv('C:/Users/atouomo/Downloads/train_data.csv')

obs = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area_1','Wilderness_Area_2','Wilderness_Area_3','Wilderness_Area_4',
       '2702','2703','2704','2705','2706','2717','3501','3502','4201','4703','4704','4744','4758','5101','5151','6101',
       '6102','6731','7101','7102','7103','7201',
       '7202','7700','7701','7702','7709','7710','7745','7746','7755','7756','7757','7790','8703','8707',
       '8708','8771','8772','8776']

cls = ['Cover_Type']

X = train_data.loc[:, obs]
Y = train_data.loc[:, 'Cover_Type']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Specify features
feature_columns = [tf.feature_column.numeric_column(key=column)for column in obs]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=8,
    dropout=0.1,
    model_dir="./tmp/forest_model"
)
with tf.device('/gpu:0'):
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=X_train,
        y=Y_train,
        num_epochs=None,
        batch_size=100,
        shuffle=True
    )

    op = classifier.train(input_fn=train_input_fn, steps=320000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=X_test,
        y=Y_test,
        num_epochs=1,
        shuffle=False
    )

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
print("--- %s seconds ---" % (time.time() - start_time))

