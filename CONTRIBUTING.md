# Guidelines for contributing to Bamojax

Thanks you for your interest in contributing to **bamojax**! We value contributions such as:

* Bug fixes
* Documentation
* New model implementations
* New features that improve the pipeline anywhere from model conception to posterior inference

## Reporting issues

If you find a bug or have a feature request, please open a GitHub Issue and describe it clearly.

## Pull requests

If you want to make a code or documentation change, please:
  1. Fork the repository
  2. Create a new branch (`git checkout -b fix-bug` or `feature-new-thing`)
  3. Make your changes
  4. Push to your fork and open a Pull Request (PR)

We will then review your PR and provide feedback or merge it with the main branch.

## Testing your changes

in order to run the available tests in **bamojax**, you can simply install it in developer mode using 
```
pip install -e ".[dev]"
```

You can then run the tests using for example (CPU tests shown here):

```
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -q --maxfail=1
```
