Files to train the model and evaluate on CHTC servers.

## Contents

There are two folders, cnn and test, inside this folder. Each of them contain four files, python script (.py), submit file (.sub), shell script (.sh), and environment.yml. 

The submit file is used to submit jobs to CHTC by running the command "condor_submit file_name.sub". This file includes configs such as input file, output file paths, and environment configs to execute the jobs. the shell scripts (.sh) will be executed by the CHTC servers, which will set up the environment (with environment.yml) and execute the python scripts (.py).

For all the python scripts, users need to update the setting/configs in the top of the main based on their needs.

### cnn

The scripts to train the CNN model for both binary classification or regression. More details are included in cnn.py.

The memory and disk size are large due to the input size and that we use one-hot encoding. In the submit file, setting memory and disk to 50GB is enough for training but not for testing (which require 80GB).

### test

The scripts to test the models. Users need to provides the indices for testing and validation dataset, which are the outputs from cnn scripts. 

