# jacques
Just Another Conditional Quantile Estimator

This Python module provides code to implement Jacques forecasting model.

### Prerequisites
Before setting up this project:   
    -  Python Version: Your machine will need to have an installed version of python that mets the `requires-pyhon` constraint in [pyproject.toml](https://github.com/reichlab/jacques/blob/BWRedits/pyproject.toml).  
    -  That version of Python should be set as your current Python interpreter. 

## Installing the package



## Setup for local development

The instructions below outline how to set up a development environment based
on uv tooling.

Prerequisites:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

1. Clone this repository
2. Change to the repo's root directory:

    ```bash
    cd jacques
    ```

3. Create a Python virtual environment and install dependencies. The command
below creates a virtual environment in the `.venv` directory, installs Python
if needed, installs project dependencies (including dev dependencies), and
installs the package in
[editable mode](https://setuptools.pypa.io/en/stable/userguide/development_mode.html):

    ```bash
    uv sync
    ```


### Updating dependencies

Use [`uv add`](https://docs.astral.sh/uv/reference/cli/#uv-add) to include a
new dependency in the project. This command will install the new dependency
into the virtual environment, add it to `uv.lock`, and update the
`dependencies` section of [`pyproject.toml`](pyproject.toml).

```bash
uv add <package-name>
```

To add a dependency to a specific group (adding a dev dependency, for example),
use the `--group` flag:

```bash
uv add <package-name> --group dev
```