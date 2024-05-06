<!-- This file is a placeholder for customizing description of your plugin 
on the napari hub if you wish. The readme file will be used by default if
you wish not to do any customization for the napari hub listing.

If you need some help writing a good description, check out our 
[guide](https://github.com/chanzuckerberg/napari-hub/wiki/Writing-the-Perfect-Description-for-your-Plugin)
-->

## Installation

### Recommended Installation Method

We highly recommend using a `conda` virtual environment to install and operate this plugin.

To use Python 3.9, for example:

    conda create -n basicpy -c conda-forge python=3.9 napari pyqt && \
    conda activate basicpy && \
    pip install napari-basicpy

For further instructions on installing `napari`, visit their [install guide](https://napari.org/stable/tutorials/fundamentals/installation).

---

**IMPORTANT NOTE FOR APPLE SILICON AND WINDOWS USERS:**

If the above instructions fail with Apple silicon (e.g., M1/M2 chip) or Windows, you may need to install the `jax` and `jaxlib` following the instruction [here](https://github.com/peng-lab/BaSiCPy#installation).

---

### Other Installation Methods

You can also install `napari-basicpy` via [pip]:

    pip install napari-basicpy


To install latest development version:

    pip install git+https://github.com/peng-lab/napari-basicpy.git

or

    pip install git+https://github.com/tdmorello/napari-basicpy.git

## Usage

### General Usage

This plugin expects a stack of tiles as input. Mosaic images should be deconstructed into their tiled components before processing. Individual tiles should be two-dimensional.

There are many options to customize the performance of BaSiCPy. Please refer to the BaSiCPy documentation on parameters [here](https://basicpy.readthedocs.io/en/latest/api.html#basicpy.basicpy.BaSiC) for details.

### Batch Processing

Coming soon...

## License

Distributed under the terms of the [BSD-3] license,
"napari-basicpy" is free and open source software

## Issues

If you encounter any problems, please [file an issue](https://github.com/peng-lab/napari-basicpy/issues) along with a detailed description.

[napari]: https://github.com/napari/napari
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
