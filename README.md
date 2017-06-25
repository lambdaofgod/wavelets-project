## Project for Introduction to wavelets course.

Idea is based on [Fast multiresolution image querying](http://grail.cs.washington.edu/wp-content/uploads/2015/08/jacobs-1995.pdf)

Dependencies: [Anaconda](https://www.continuum.io/)

#### Using environment with conda

To create environment run
```
conda env create -f requirements.txt
```

To load environment run
```
source activate wavelets
```

Notebooks can be used from Jupyter (it's bundled in requirements) on loaded environment.

#### Loading data

To load data run
```
make data
```
in the project directory.
