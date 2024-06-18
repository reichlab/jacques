# jacques
Just Another Conditional Quantile Estimator

This Python module provides code to implement Jacques forecasting model.

## Setup

### Prerequisites
Before setting up this project:   
    -  Python Version: Your machine will need to have an installed version of python that mets the `requires-pyhon` constraint in [pyproject.toml](https://github.com/reichlab/jacques/blob/BWRedits/pyproject.toml).  
    -  That version of Python should be set as your current Python interpreter.  

### Setup
Follow the directions below to set this project up on your local machine.

1. Clone this repository to your local computer and navigate to the corresponding directory.

2. Create a Python virtual environment.

```
python m venv .venv
```

3. Activate the virual environment.

```
source .venv/bin/activate
```
**Note:** the command above is for Unix-based systems. If you're using Windows, the command is:

```
.venv\Scripts\activate
```

4. Install the project's dependencies, which include the `jacques` dependencies:

```
# if only need the dependencies necessary to run jacques
pip install -r requirements/requirements.txt && pip install -e .
```

and the dependencies required for running the [demo.ipynb](https://github.com/reichlab/jacques/blob/BWRedits/demo.ipynb):

```
# if you'd like to be able to run the demo notebook
pip install -r requirements/doc-requirements.txt && pip install -e .
```
