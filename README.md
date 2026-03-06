# minkowski
Minkowski functional code for flat or curved sky maps

* Run on curved sky healpy maps with scripts/mf_curved_sky.py
* Run on square flat sky patches with mf_flat_sky.py
* Sample usage in example notebook


To run C++ script on nersc:
```
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python -m pybind11 --includes) minkowski.cc -o minkowski$(python3-config --extension-suffix) $(python3-config --ldflags)
```
