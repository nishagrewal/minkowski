# minkowski
Minkowski functional code for flat or curved sky maps

Clone repository 

```
git clone https://github.com/nishagrewal/minkowski
```


To run C++ script on nersc:
```
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python -m pybind11 --includes) minkowski.cc -o minkowski$(python3-config --extension-suffix) $(python3-config --ldflags)
```
