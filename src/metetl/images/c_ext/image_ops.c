#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static Py_ssize_t reflect_index(Py_ssize_t index, Py_ssize_t size) {
    if (size <= 1) {
        return 0;
    }

    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * size - index - 2;
        }
    }
    return index;
}

static int validate_2d_buffer(Py_buffer *view, const char *name, char expected_format) {
    if (view->ndim != 2) {
        PyErr_Format(PyExc_ValueError, "%s must be a 2D array", name);
        return 0;
    }
    if (view->itemsize != (expected_format == 'B' ? 1 : 4)) {
        PyErr_Format(PyExc_ValueError, "%s has unexpected item size", name);
        return 0;
    }
    if (view->format == NULL || view->format[0] != expected_format || view->format[1] != '\0') {
        PyErr_Format(PyExc_ValueError, "%s must have format '%c'", name, expected_format);
        return 0;
    }
    if (view->suboffsets != NULL) {
        for (int i = 0; i < view->ndim; ++i) {
            if (view->suboffsets[i] >= 0) {
                PyErr_Format(PyExc_ValueError, "%s must be a direct buffer", name);
                return 0;
            }
        }
    }
    return 1;
}

static PyObject *convolve_gray_u8(PyObject *self, PyObject *args) {
    PyObject *image_obj = NULL;
    PyObject *kernel_obj = NULL;
    Py_buffer image_view;
    Py_buffer kernel_view;
    int image_acquired = 0;
    int kernel_acquired = 0;
    float *output = NULL;
    PyObject *result = NULL;

    if (!PyArg_ParseTuple(args, "OO", &image_obj, &kernel_obj)) {
        return NULL;
    }

    if (PyObject_GetBuffer(image_obj, &image_view, PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return NULL;
    }
    image_acquired = 1;

    if (PyObject_GetBuffer(kernel_obj, &kernel_view, PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        goto cleanup;
    }
    kernel_acquired = 1;

    if (!validate_2d_buffer(&image_view, "image", 'B')) {
        goto cleanup;
    }
    if (!validate_2d_buffer(&kernel_view, "kernel", 'f')) {
        goto cleanup;
    }

    Py_ssize_t height = image_view.shape[0];
    Py_ssize_t width = image_view.shape[1];
    Py_ssize_t kh = kernel_view.shape[0];
    Py_ssize_t kw = kernel_view.shape[1];
    Py_ssize_t pad_h = kh / 2;
    Py_ssize_t pad_w = kw / 2;

    output = (float *)PyMem_Malloc((size_t)(height * width * sizeof(float)));
    if (output == NULL) {
        PyErr_NoMemory();
        goto cleanup;
    }

    const unsigned char *image_data = (const unsigned char *)image_view.buf;
    const float *kernel_data = (const float *)kernel_view.buf;

    for (Py_ssize_t y = 0; y < height; ++y) {
        for (Py_ssize_t x = 0; x < width; ++x) {
            float acc = 0.0f;

            for (Py_ssize_t ky = 0; ky < kh; ++ky) {
                Py_ssize_t yy = reflect_index(y + ky - pad_h, height);
                const unsigned char *image_row = image_data + yy * image_view.strides[0];
                const float *kernel_row = (const float *)((const char *)kernel_data + ky * kernel_view.strides[0]);

                for (Py_ssize_t kx = 0; kx < kw; ++kx) {
                    Py_ssize_t xx = reflect_index(x + kx - pad_w, width);
                    const unsigned char *pixel_ptr = image_row + xx * image_view.strides[1];
                    const float *kernel_ptr = (const float *)((const char *)kernel_row + kx * kernel_view.strides[1]);
                    acc += ((float)(*pixel_ptr)) * (*kernel_ptr);
                }
            }

            output[y * width + x] = acc;
        }
    }

    result = PyBytes_FromStringAndSize((const char *)output, height * width * (Py_ssize_t)sizeof(float));

cleanup:
    if (output != NULL) {
        PyMem_Free(output);
    }
    if (kernel_acquired) {
        PyBuffer_Release(&kernel_view);
    }
    if (image_acquired) {
        PyBuffer_Release(&image_view);
    }
    return result;
}

static PyMethodDef ImageOpsMethods[] = {
    {
        "convolve_gray_u8",
        convolve_gray_u8,
        METH_VARARGS,
        "Convolve a grayscale uint8 image with a float32 kernel using reflect padding.",
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef imageopsmodule = {
    PyModuleDef_HEAD_INIT,
    "image_ops",
    "C accelerators for image processing.",
    -1,
    ImageOpsMethods
};

PyMODINIT_FUNC PyInit_image_ops(void) {
    return PyModule_Create(&imageopsmodule);
}
