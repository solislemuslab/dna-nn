# to do
- need a script to read model.jl file and create the .py files and run

We will try to have an automated way to plot a neural network architecture

# Understanding PlotNeuralNet

I am using this package for latex: [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).
```shell
cd Dropbox/software
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
```

We have to download `texlive`:
```shell
brew cask install mactex ##takes forever, had to run `brew cask doctor` first
```

Now, we want to try the toy example:
```shell
cd Documents/github/PlotNeuralNet/pyexamples
bash ../tikzmake.sh test_simple
```
It works as it produces `test_simple.pdf`.

We will now create a folder to run the example in the github readme:
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/plot-nn/example
emacs my_arch.py ##copy the lines in github readme
bash /Users/Clauberry/Documents/github/PlotNeuralNet/tikzmake.sh my_arch
```
In mac, we get error `ImportError: No module named pycore.tikzeng` related to python.

## Many errors trying to make python work (horrible nightmare!)

```shell
pip3 install -U pycore
```
This works:
```shell
$ pip list
Package            Version  
------------------ ---------
atomicwrites       1.3.0    
attrs              19.1.0   
certifi            2019.6.16
chardet            3.0.4    
coverage           4.5.4    
idna               2.8      
importlib-metadata 0.19     
more-itertools     7.2.0    
packaging          19.1     
pip                19.2.2   
pluggy             0.12.0   
py                 1.8.0    
pycore             17.4.17  
pyparsing          2.4.2    
pytest             5.0.1    
pytest-cov         2.7.1    
requests           2.22.0   
setuptools         40.8.0   
six                1.12.0   
urllib3            1.25.3   
wcwidth            0.1.7    
zipp               0.5.2  
```
but I still get and error when I change to `python3` in `tikzmake.sh`.

What seems to work is to copy and paste the python script inside python3:
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/plot-nn/example
python3
```
Inside python3, copy and paste `my_arch.py`, but I get the same error.
I even put `export PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages"` in `.bashrc`, but it does not matter. I still get the same error.

Following [this thread](https://stackoverflow.com/questions/37341614/modules-are-installed-using-pip-on-osx-but-not-found-when-importing):
```shell
$ which -a pip
/Library/Frameworks/Python.framework/Versions/3.7/bin/pip
/usr/local/bin/pip
/usr/local/bin/pip
$ which -a pip3
/Library/Frameworks/Python.framework/Versions/3.7/bin/pip3
/usr/local/bin/pip3
/usr/local/bin/pip3
$ which -a python
/usr/local/bin/python
/usr/bin/python
/usr/local/bin/python
$ which -a python3
/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
/usr/local/bin/python3
/usr/local/bin/python3
$ brew reinstall python3
$ brew install python3
$ brew link --overwrite python3
$ brew link --overwrite python
```
Then followed many steps in `brew doctor`.

After `python -v`, apparently [this is my error](https://stackoverflow.com/questions/49991416/keyerror-pythonpath-sites-py-broken-in-python3):
```shell
$ python3 -c 'import sys; print(sys.path)'
['', '/Library/Frameworks/Python.framework/Versions/3.7/lib/python37.zip', '/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7', '/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/lib-dynload', '/Users/Clauberry/Library/Python/3.7/lib/python/site-packages', '/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages']

$ python -c 'import sys; print(sys.path)'
'import sitecustomize' failed; use -v for traceback
['', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python27.zip', '/usr/local/lib/python3.7/site-packages', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-old', '/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload', '/Users/Clauberry/Library/Python/2.7/lib/python/site-packages']
```

### Answer!!!
I found the answer in the first place I checked: [here](https://github.com/HarisIqbal88/PlotNeuralNet/issues/34). The `pycore.tikzeng` refers to the local folder inside the github repo!!! Not the installed `pycore`!!
So, I will rename the local folder.
Before:
```shell
cd Documents/github/PlotNeuralNet
(master) $ ls
LICENSE      examples     pycore       tikzmake.sh
README.md    layers       pyexamples   tikzmake.sh~
```

After:
```shell
(master) $ ls
LICENSE      examples     pycore2       tikzmake.sh
README.md    layers       pyexamples   tikzmake.sh~
```
and then change inside `tikzmake.sh` to python3 and to pycore2 inside `my_arch.py`.

So, let's try again:
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/plot-nn/example
python3
```
In python:
```python
(master) $ python3
Python 3.7.4 (v3.7.4:e09359112e, Jul  8 2019, 14:54:52) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.path.append('/Users/Clauberry/Dropbox/software/PlotNeuralNet/')
>>> from pycore2.tikzeng import *
>>> exit()
```
This works!

So, let's try the whole script:
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/plot-nn/example
bash /Users/Clauberry/Dropbox/software/PlotNeuralNet/tikzmake.sh my_arch
```
It will run, but complain about an `init.tex` file not found.
This is really stupid because you need to run the code inside the `PlotNeuralNet` folder, or it does not work.

We do not want do do this.
So, after the fact, we can change the path in `my_arch.tex` in emacs, and just compile in emacs. This works!

Citation:
```
@misc{haris_iqbal_2018_2526396,
  author       = {Haris Iqbal},
  title        = {HarisIqbal88/PlotNeuralNet v1.0.0},
  month        = dec,
  year         = 2018,
  doi          = {10.5281/zenodo.2526396},
  url          = {https://doi.org/10.5281/zenodo.2526396}
}
```

# Test before using my NN

I will modify the `my_arch.py` into `my_arch2.py` and play around with the layers to see how things look.

```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/plot-nn/example
bash /Users/Clauberry/Dropbox/software/PlotNeuralNet/tikzmake.sh my_arch2
```
Then, we change the path to `init.tex` in `my_arch2.tex` to `/Users/Clauberry/Dropbox/software/PlotNeuralNet/layers/`.
Then, we manually remove the ` zlabel=1,` to ` zlabel= ,`.

# Better code in latex!!

See [here](https://tex.stackexchange.com/questions/153957/drawing-neural-network-with-tikz)

I created the file `tiks-example.tex` with these commands to create a pretty and easy NN.

For staph: `tiks-staph1.tex`. Only done for 500 sites, but will be exactly the same for 700 and for pseudomonas.


# Plotting NN by Sebastian Raschka
use https://nbviewer.jupyter.org/github/JuliaTeX/TikzGraphs.jl/blob/master/doc/TikzGraphs.ipynb to plot NN or http://alexlenail.me/NN-SVG/index.html