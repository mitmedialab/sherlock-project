# HOW-TO: Environment Setup
```
cd source/private-github/sherlock-project/
```

## First time
Due to Python dependencies (notably TensorFlow 1.14), we need Python 3.7. More recent versions are likely to have been installed by `brew`, so we need to ensure 3.7 is available and appropriately linked.

```
brew install python@3.7

# check if python 3.9 (or 3.8) is installed and active
python3 --version
Python 3.9.1

# revert to 3.7 (needed by TensorFlow 1.15)

brew unlink python@3.9
brew unlink python@3.8
brew link python@3.7

python3 --version
Python 3.7.12


pip install virtualenv # install first
python3 -m venv venv
```

Now activate the new virtual environment as detailed below in *"Activate virtual environment"*

```
python3.7 -m pip install --upgrade pip

python3.7 -m pip install -U pip setuptools

pip install -r requirements.txt
pip install jupyter line_profiler pandarallel pympler
```

## Usage
### Activate virtual environment
Activate the virtual environment, and you will now see (venv) in front of your prompt:

```
source venv/bin/activate

(venv) lowecg@Chris-Lowe-MBP-Old sherlock-project (master) $
```

Now launch PyCharm or Jupyter Notebook

```
# If you need fully deterministic results between runs, set the following environment value prior to launching jupyter.
# See comment in sherlock.features.paragraph_vectors.infer_paragraph_embeddings_features for more info.
export PYTHONHASHSEED=13

jupyter notebook
```

### Profiling code from Jupyter

Within your jupyter notebook, call: `%load_ext line_profiler`
Profile as follows:

```
# usage : note the usage %lprun -f FUNCTION_TO_PROFILE CODE_EXPRESSION_TO_RUN
          or             %lprun -m MODULE_TO_PROFILE CODE_EXPRESSION_TO_RUN
# function
%lprun -f prof_function prof_function()

# module
X_test = %lprun -m sherlock.features.preprocessing extract_features(test_samples_converted.head(n=100))
```
