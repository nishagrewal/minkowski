#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

py::tuple V_012_return(py::array_t<double> k_array,
    py::array_t<double> v_array,
    py::array_t<double> sq_array,
    py::array_t<double> frac_array)
{
    auto k = k_array.unchecked<1>();
    auto v = v_array.unchecked<1>();
    auto sq = sq_array.unchecked<1>();
    auto frac = frac_array.unchecked<1>();

    int N = k_array.shape(0);
    int thr_ct = v_array.shape(0);

    // used for calculating bin index per pixel
    double vmin = v(0);
    double vmax = v(thr_ct - 1);
    double inv_vspace = (thr_ct - 1) / (vmax - vmin);

    // initialise sum per bin (for v0)
    std::vector<int> sum(thr_ct, 0);

    // allocate output MF arrays
    py::array_t<double> v0_array(thr_ct);
    py::array_t<double> v1_array(thr_ct);
    py::array_t<double> v2_array(thr_ct);

    auto v0 = v0_array.mutable_unchecked<1>();
    auto v1 = v1_array.mutable_unchecked<1>();
    auto v2 = v2_array.mutable_unchecked<1>();

    // initialise MFs to zero
    for (int b=0; b<thr_ct; b++) {
        v0(b) = 0;
        v1(b) = 0;
        v2(b) = 0;
    }

    for (int i=0; i<N; i++){
        int bin_index = static_cast<int>((k(i) - vmin) * inv_vspace);
        if (bin_index >= 0 && bin_index < thr_ct){
            sum[bin_index] += 1;
            v1(bin_index) += sq(i);
            v2(bin_index) += frac(i);
        }
    }

    // cumulative sum for v0, counting pixels above threshold
    double cum_sum = 0;
    for (int b = thr_ct - 1; b >= 0; b--) {
        cum_sum += sum[b];
        v0(b) = cum_sum;
    }

    // apply normalisation
    for (int b = 0; b < thr_ct; b++) {
        v0(b) /= N;                 
        v1(b) /= (4.0 * N);       
        v2(b) /= (2.0 * M_PI * N); 
    }

    return py::make_tuple(v0_array, v1_array, v2_array);
}


PYBIND11_MODULE(minkowski, m){
    m.def("V_012", &V_012_return, "Calculate and return V0, V1, V2");
}
