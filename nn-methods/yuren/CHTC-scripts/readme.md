Files to train the model and evaluate on CHTC servers. Possible OOM on some CHTC servers but could run on others.

## Contents

There are two folders, cnn and test, inside this folder. Each of them contain three files, python script (.py), submit file (.sub), and shell script (.sh). 

The submit file is used to submit jobs to CHTC by running the command "condor_submit file_name.sub". This file includes configs such as input file, output file paths, and environment configs to execute the jobs. the shell scripts (.sh) will be executed by the CHTC servers, which will set up the environment and execute the python scripts (.py).

### cnn



### test



