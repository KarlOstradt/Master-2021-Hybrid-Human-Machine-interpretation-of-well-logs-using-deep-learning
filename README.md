The _`master`_ folder contains python files for preprocessing, computing metrics and loading models.

The _`config`_ folder contains a folder for each model type: AE, VAE, LSTM and DeepAnT. These folders contains configuration files of the corresponding model type.

The _`results`_ folder contains a folder for each model type: AE, VAE, LSTM and DeepAnT. These folders contains the results for all configurations of the corresponding model type.

The _`src`_ folder contains Java code for reading raw well logs. This code requires Log I/O which we are not at liberty of sharing.

The _`FindBestThreshold.ipynb`_ notebook finds the best threshold for all configurations of a specified model type.

The _`MissingValuePlots.ipynb`_ notebook plot plots and prints statistics about training and test datasets.

The _`PrepareDatasets.ipynb`_ notebook perform some preprocessing steps on the training datasets. Running this notebook will not work as we cannot share the training dataset.

The _`run.py`_ file creates a model for each configuration file in the _config_ folder. Each model is then trained. Results are then obtained from each model. This script cannot be executed without the training dataset.

The _`scaler_remove.pkl`_ file is the min-max scaler based on the training dataset without missing values.

The _`15_9-F-11 T2.csv`_ file is the test dataset.


