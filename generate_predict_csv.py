import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time

# only displays the most important warnings
tf.logging.set_verbosity(tf.logging.FATAL)
start_time = time.time()

train_data = pd.read_csv('C:/Users/atouomo/Downloads/test_data.csv')

obs = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area_1','Wilderness_Area_2','Wilderness_Area_3','Wilderness_Area_4',
       '2702','2703','2704','2705','2706','2717','3501','3502','4201','4703','4704','4744','4758','5101','5151','6101',
       '6102','6731','7101','7102','7103','7201',
       '7202','7700','7701','7702','7709','7710','7745','7746','7755','7756','7757','7790','8703','8707',
       '8708','8771','8772','8776']

X = train_data.loc[:, obs]

# Specify features
feature_columns = [tf.feature_column.numeric_column(key=column)for column in obs]

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=8,
    dropout=0.1,
    model_dir="./tmp/forest_model"
)


test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=X,
        num_epochs=1,
        shuffle=False
    )

predictions = list(classifier.predict(input_fn=test_input_fn, predict_keys=['class_ids'], yield_single_examples=True))
Y_pred = []
for pred in predictions:
    Y_pred.append(pred['class_ids'][0])

X['Cover_Type'] = Y_pred
X.to_csv("test_data_with_prediction.csv", encoding='utf-8', index=False)

print("--- %s seconds ---" % (time.time() - start_time))