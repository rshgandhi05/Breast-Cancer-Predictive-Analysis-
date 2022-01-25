# import the necessary packages
import argparse
import numpy
import pandas
import tensorflow
from tensorflow.keras.models import load_model
# please keep the below code to avoid an runtime errors from keras
physical_devices = tensorflow.config.list_physical_devices('GPU')                  # This is a necessary configuration to avoid
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
                help="path to input patient data")
ap.add_argument("-m", "--model", required=True,
                help="path to trained model")
args = vars(ap.parse_args())

# Resize test image
data = pandas.read_csv(args["data"])
# create dummy variables for the categorical variables
dummies = pandas.get_dummies(data, columns=['MAR_STAT', 'SEQ_NUM', 'PRIMSITE', 'LATERAL', 'GRADE', 'SURGPRIF', 'SURGSITF', 'NO_SURG','AGE_1REC', 'BEHTREND', 'RAC_RECA', 'STAT_REC', 'ERSTATUS','PRSTATUS', 'INSREC_PUB', 'ADJTM_6VALUE', 'ADJNM_6VALUE', 'ADJM_6VALUE','ADJAJCCSTG', 'her2', 'brst_sub', 'MALIGCOUNT', 'BENBORDCOUNT','RADIATION', 'RAD_SURG_SEQ', 'CHEMO'])
# final dataframe
cancer_data_final = dummies
# Set as an array and resize
data_array = numpy.array(dummies)
#print(data_array.shape)
data_array_resh = data_array.reshape(1, 26, 1)

# Set labels list
labels = ['6 years or less', 'more than 6 years']

# Load saved model
model = load_model(args["model"])
# Predict input image label
pred = (model.predict(data_array_resh) > 0.5).astype("int32")
# Pull label from top prediction probability
#uncomment below code if troubleshooting is necessary to see what value the prediction is returning and fed into the Label
#print(pred.item(0))
i = pred.item(0)
label = labels[i]

# Print output
print("The patient will survive for " + label + ".")
