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

# revert to 3.7 (needed by TensorFlow 1.14)

brew unlink python@3.9
brew unlink python@3.8
brew link python@3.7

python3 --version
Python 3.8.6


pip install virtualenv # install first
python3 -m venv venv
```

Now activate the new virtual environment as detailed below in *"Activate virtual environment"*

```
/Users/lowecg/source/private-github/sherlock-project/venv/bin/python3.7 -m pip install --upgrade pip

pip install -r requirements.txt
```

## Usage
### Activate virtual environment
Activate the virtual environment, and you will now see (venv) in front of your prompt:

```
source venv/bin/activate

(venv) lowecg@Chris-Lowe-MBP-Old sherlock-project (master) $
```

Now launch PyCharm or Jupyter Notebook

```jupyter notebook```
