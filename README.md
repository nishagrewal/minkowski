# minkowski
### Minkowski functional code for flat or curved sky maps

* Run on curved sky healpy maps or flat sky patches with scripts/mf.py
* Sample usage in example notebook


### Option to run code in c++:

To compile C++ script on nersc:
```
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python -m pybind11 --includes) minkowski.cc -o minkowski$(python3-config --extension-suffix) $(python3-config --ldflags)
```
Then run the following in python:
```
from minkowski import V_012, map_derivatives_curved

dx, dy, dxx, dyy, dxy = map_derivatives_curved(m)
sq = np.sqrt(dx**2 + dy**2)
frac = (2*dx*dy*dxy - (dx**2)*dyy - (dy**2)*dxx) / (dx**2 + dy**2)
V0, V1, V2 = V_012(m, v, sq, frac)
```
