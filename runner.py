import neural_network
from preprocessing import *

# parent_directory = '/media/vaneesh/Data/Datasets/German/training/training_5_classes/'
parent_directory = '/media/vaneesh/Data/Datasets/German/training/training_5_classes (copy)/'
out_dir = '/media/vaneesh/Data/Datasets/German/training/training_5_classes/34/'

# region code for mnist & other datasets
# digits = datasets.load_digits()
# df_images = pd.DataFrame(digits.data)
# df_labels = pd.DataFrame(digits.target)
# endregion

# region code for iris
# iris = datasets.load_iris()
# df_images = pd.DataFrame(iris.data)
# df_labels = pd.DataFrame(iris.target)
# endregion

# mnist kaggle dataset
#df_images, df_labels = read_csv_file()

# Code for custom dataset
df_images, df_labels = make_dataframe_from_images(parent_directory)

neural_network.calculate_neurons_and_run(df_images, df_labels)
print('---------------------------Done---------------------------------')
