// Fused C++ core for the lap-time march (corner-speed limit, forward accel
// pass, backward brake pass) with friction-ellipse coupling. Replaces the
// numba-jitted equivalents in lap_sim.py with a compiled extension that
// avoids per-call Python/numpy dispatch overhead — meaningful because the
// optimizer calls this thousands of times during line optimization.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

static inline double interp1(double v, const double *xp, const double *fp, npy_intp n) {
    if (v <= xp[0]) return fp[0];
    if (v >= xp[n - 1]) return fp[n - 1];
    npy_intp lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        npy_intp mid = (lo + hi) / 2;
        if (xp[mid] <= v) lo = mid; else hi = mid;
    }
    double x0 = xp[lo], x1 = xp[hi];
    double f0 = fp[lo], f1 = fp[hi];
    double t = (v - x0) / (x1 - x0);
    return f0 + t * (f1 - f0);
}

static inline double ellipse_long_accel(double v_i, double kappa_i,
                                         const double *v_grid, const double *a_lat_grid,
                                         const double *a_long_grid, npy_intp n) {
    double a_lat_max = interp1(v_i, v_grid, a_lat_grid, n);
    double a_long_max = interp1(v_i, v_grid, a_long_grid, n);
    double a_lat_used = v_i * v_i * std::fabs(kappa_i);
    double ratio = (a_lat_max > 1e-9) ? (a_lat_used / a_lat_max) : 1.0;
    if (ratio > 1.0) ratio = 1.0;
    double rem = 1.0 - ratio * ratio;
    if (rem < 0.0) rem = 0.0;
    return a_long_max * std::sqrt(rem);
}

// Get a contiguous double* view of a 1D array-like PyObject; returns a new
// reference to the underlying (possibly newly-created) ndarray via `out_arr`.
static double *as_double_ptr(PyObject *obj, PyArrayObject **out_arr) {
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) return nullptr;
    *out_arr = arr;
    return (double *)PyArray_DATA(arr);
}

// corner_speed_limit(v_grid, a_lat_grid, kappa, v_max_grid, iters) -> ndarray
static PyObject *py_corner_speed_limit(PyObject *self, PyObject *args) {
    PyObject *v_grid_o, *a_lat_o, *kappa_o;
    double v_max_grid;
    int iters;
    if (!PyArg_ParseTuple(args, "OOOdi", &v_grid_o, &a_lat_o, &kappa_o, &v_max_grid, &iters))
        return nullptr;

    PyArrayObject *v_grid_arr, *a_lat_arr, *kappa_arr;
    double *v_grid = as_double_ptr(v_grid_o, &v_grid_arr);
    double *a_lat = as_double_ptr(a_lat_o, &a_lat_arr);
    double *kappa = as_double_ptr(kappa_o, &kappa_arr);
    if (!v_grid || !a_lat || !kappa) return nullptr;

    npy_intp n_grid = PyArray_SIZE(v_grid_arr);
    npy_intp n = PyArray_SIZE(kappa_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *out_data = (double *)PyArray_DATA(out);

    for (npy_intp i = 0; i < n; ++i) {
        double k = std::fabs(kappa[i]);
        if (k < 1e-6) k = 1e-6;
        double vi = v_max_grid;
        for (int it = 0; it < iters; ++it) {
            double a = interp1(vi, v_grid, a_lat, n_grid);
            double v_new = std::sqrt(a / k);
            if (v_new > v_max_grid) v_new = v_max_grid;
            vi = 0.5 * vi + 0.5 * v_new;
        }
        out_data[i] = vi;
    }

    Py_DECREF(v_grid_arr);
    Py_DECREF(a_lat_arr);
    Py_DECREF(kappa_arr);
    return (PyObject *)out;
}

enum class Direction { Forward, Backward };

static PyObject *march(PyObject *args, Direction dir) {
    PyObject *v_grid_o, *a_lat_o, *a_long_o, *v_in_o, *ds_o, *kappa_o;
    if (!PyArg_ParseTuple(args, "OOOOOO", &v_grid_o, &a_lat_o, &a_long_o, &v_in_o, &ds_o, &kappa_o))
        return nullptr;

    PyArrayObject *v_grid_arr, *a_lat_arr, *a_long_arr, *v_in_arr, *ds_arr, *kappa_arr;
    double *v_grid = as_double_ptr(v_grid_o, &v_grid_arr);
    double *a_lat = as_double_ptr(a_lat_o, &a_lat_arr);
    double *a_long = as_double_ptr(a_long_o, &a_long_arr);
    double *v_in = as_double_ptr(v_in_o, &v_in_arr);
    double *ds = as_double_ptr(ds_o, &ds_arr);
    double *kappa = as_double_ptr(kappa_o, &kappa_arr);
    if (!v_grid || !a_lat || !a_long || !v_in || !ds || !kappa) return nullptr;

    npy_intp n_grid = PyArray_SIZE(v_grid_arr);
    npy_intp n = PyArray_SIZE(v_in_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *v = (double *)PyArray_DATA(out);
    std::memcpy(v, v_in, n * sizeof(double));

    if (dir == Direction::Forward) {
        for (npy_intp i = 0; i < n - 1; ++i) {
            double a = ellipse_long_accel(v[i], kappa[i], v_grid, a_lat, a_long, n_grid);
            double v_lim = std::sqrt(v[i] * v[i] + 2 * a * ds[i]);
            if (v_lim < v[i + 1]) v[i + 1] = v_lim;
        }
    } else {
        for (npy_intp i = n - 1; i > 0; --i) {
            double a = ellipse_long_accel(v[i], kappa[i], v_grid, a_lat, a_long, n_grid);
            double v_lim = std::sqrt(v[i] * v[i] + 2 * a * ds[i - 1]);
            if (v_lim < v[i - 1]) v[i - 1] = v_lim;
        }
    }

    Py_DECREF(v_grid_arr);
    Py_DECREF(a_lat_arr);
    Py_DECREF(a_long_arr);
    Py_DECREF(v_in_arr);
    Py_DECREF(ds_arr);
    Py_DECREF(kappa_arr);
    return (PyObject *)out;
}

static PyObject *py_forward_pass(PyObject *self, PyObject *args) {
    return march(args, Direction::Forward);
}

static PyObject *py_backward_pass(PyObject *self, PyObject *args) {
    return march(args, Direction::Backward);
}

// Fused: corner_speed_limit -> forward_pass -> backward_pass -> lap_time,
// with optional lap tiling (reps) for closed-track transient removal, done
// entirely in C++ so no intermediate arrays cross the Python boundary.
static PyObject *py_simulate_core(PyObject *self, PyObject *args) {
    PyObject *v_grid_o, *a_lat_o, *a_acc_o, *a_brk_o, *kappa_o, *ds_o;
    double v_max_grid;
    int iters, reps;
    if (!PyArg_ParseTuple(args, "OOOOOOdii", &v_grid_o, &a_lat_o, &a_acc_o, &a_brk_o,
                           &kappa_o, &ds_o, &v_max_grid, &iters, &reps))
        return nullptr;

    PyArrayObject *v_grid_arr, *a_lat_arr, *a_acc_arr, *a_brk_arr, *kappa_arr, *ds_arr;
    double *v_grid = as_double_ptr(v_grid_o, &v_grid_arr);
    double *a_lat = as_double_ptr(a_lat_o, &a_lat_arr);
    double *a_acc = as_double_ptr(a_acc_o, &a_acc_arr);
    double *a_brk = as_double_ptr(a_brk_o, &a_brk_arr);
    double *kappa = as_double_ptr(kappa_o, &kappa_arr);
    double *ds = as_double_ptr(ds_o, &ds_arr);
    if (!v_grid || !a_lat || !a_acc || !a_brk || !kappa || !ds) return nullptr;

    npy_intp n_grid = PyArray_SIZE(v_grid_arr);
    npy_intp n = PyArray_SIZE(kappa_arr);
    if (reps < 1) reps = 1;
    npy_intp total = n * reps;

    // Fixed-size stack buffer for the common case (avoids a heap alloc on
    // every optimizer objective-function call); falls back to heap only for
    // unusually long/dense tracks.
    constexpr npy_intp kStackCap = 4096;
    double stack_v[kStackCap];
    std::vector<double> heap_v;
    double *v;
    if (total <= kStackCap) {
        v = stack_v;
    } else {
        heap_v.resize(total);
        v = heap_v.data();
    }
    // kappa/ds are tiled virtually via modulo indexing — no copy needed.
    auto kap = [&](npy_intp i) { return kappa[i % n]; };
    auto dsv = [&](npy_intp i) { return ds[i % n]; };

    // corner speed limit (per-point, independent of lap tiling)
    for (npy_intp i = 0; i < total; ++i) {
        double k = std::fabs(kap(i));
        if (k < 1e-6) k = 1e-6;
        double vi = v_max_grid;
        for (int it = 0; it < iters; ++it) {
            double a = interp1(vi, v_grid, a_lat, n_grid);
            double v_new = std::sqrt(a / k);
            if (v_new > v_max_grid) v_new = v_max_grid;
            vi = 0.5 * vi + 0.5 * v_new;
        }
        v[i] = vi;
    }

    // forward accel pass
    for (npy_intp i = 0; i < total - 1; ++i) {
        double a = ellipse_long_accel(v[i], kap(i), v_grid, a_lat, a_acc, n_grid);
        double v_lim = std::sqrt(v[i] * v[i] + 2 * a * dsv(i));
        if (v_lim < v[i + 1]) v[i + 1] = v_lim;
    }
    // backward brake pass
    for (npy_intp i = total - 1; i > 0; --i) {
        double a = ellipse_long_accel(v[i], kap(i), v_grid, a_lat, a_brk, n_grid);
        double v_lim = std::sqrt(v[i] * v[i] + 2 * a * dsv(i - 1));
        if (v_lim < v[i - 1]) v[i - 1] = v_lim;
    }

    // take the middle lap (matches Python's transient-removal approach)
    npy_intp mid = reps / 2;
    npy_intp dims[1] = {n};
    PyArrayObject *out_v = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *out_v_data = (double *)PyArray_DATA(out_v);
    std::memcpy(out_v_data, v + mid * n, n * sizeof(double));

    double lap_time = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        double vi = out_v_data[i];
        if (vi < 0.1) vi = 0.1;
        lap_time += ds[i] / vi;
    }

    Py_DECREF(v_grid_arr);
    Py_DECREF(a_lat_arr);
    Py_DECREF(a_acc_arr);
    Py_DECREF(a_brk_arr);
    Py_DECREF(kappa_arr);
    Py_DECREF(ds_arr);

    PyObject *result = Py_BuildValue("Od", (PyObject *)out_v, lap_time);
    Py_DECREF(out_v);
    return result;
}

static PyMethodDef FastCoreMethods[] = {
    {"corner_speed_limit", py_corner_speed_limit, METH_VARARGS, "Corner speed limit (C++)."},
    {"forward_pass", py_forward_pass, METH_VARARGS, "Forward accel pass (C++)."},
    {"backward_pass", py_backward_pass, METH_VARARGS, "Backward brake pass (C++)."},
    {"simulate_core", py_simulate_core, METH_VARARGS, "Fused corner+forward+backward+lap_time (C++)."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef fastcoremodule = {
    PyModuleDef_HEAD_INIT, "_fastcore", nullptr, -1, FastCoreMethods
};

PyMODINIT_FUNC PyInit__fastcore(void) {
    import_array();
    return PyModule_Create(&fastcoremodule);
}
