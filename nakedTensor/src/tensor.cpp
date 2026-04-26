// https://claude.ai/chat/5f13079d-bd10-4176-a649-e8fbf141feae
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace py = pybind11;
using std::cout;
using std::runtime_error;
using std::vector;
using std::memcpy;
using std::string;
using std::ostringstream;

// struct is similar to class.
// struct members are public by default, class members are private (in cpp).
struct Tensor1D {
    float* data = nullptr; // instance attribute, here data variable is the memory address where values 
    // will be stored.
    size_t size = 0; // instance attribute, size_t is an unsigned integer type used for sizes/counts (array length, byte count).
    // to have a class attribute: static size_t size = 0;
    // this is like a constructor (like __init__ in python).
    // in cpp constructor name must match class/struct name.
    Tensor1D(const vector<float>& input){
        size = input.size();
        if(size==0){
            throw runtime_error("Tensor1D: Input can't be empty.");
        }
        // allocate ALIGNED CPU memory 
        // what does aligned cpu memory means?
        // why void** here, instead of float** - as the API is a generic allocator.
        // note to self: aligned and contiguous are different things.
        // contiguous means elements are stored back to back in memory.
        // aligned means starting address is a multiple of something (64).
        int err = posix_memalign((void**)& data, 64, size* sizeof(float)); // the most important line of code here.
        // posix_memalign tries to allocate memory. 
        // it writes the allocated address into data, so data should point to valid RAM.
        if (err!=0 || data==nullptr){
            throw runtime_error("Tensor1D: Memory allocation failed.");
        }
        cout<<"data: "<<data<<"\n";
        memcpy(data, input.data(), size* sizeof(float)); // check why input.data() and not input
        // input.data() is the pointer to the first element stored in the input vector.
    }

    ~Tensor1D(){
        if(data){
            free(data);
            data = nullptr;
        }
    }

    // these two lines are to prevent copying to prevent undefined behavior.
    // if allowed then two objects would point to same memory. both calls free(data) - crash.
    Tensor1D(const Tensor1D&) = delete;
    Tensor1D& operator=(const Tensor1D&) = delete;

    float get(size_t i) const {
        if(i>=size){
            throw runtime_error("Index out of range");
        }
        return data[i];
    }

    string repr() const{
        ostringstream oss;
        oss << "Tensor1D([";
        for (size_t i = 0; i < size; i++) {
            oss << data[i];
            if (i + 1 < size) oss << ", ";
        }
        oss << "])";
        return oss.str();
    }

    size_t numel() const{
        return size;
    }

    size_t data_ptr() const{
        return reinterpret_cast<size_t>(data);
    }
};

PYBIND11_MODULE(nakedTensor, m){
    py::class_<Tensor1D>(m, "Tensor1D")
    .def(py::init<const vector<float>&>())
    .def("numel", &Tensor1D::numel)
    .def("get", &Tensor1D::get)
    .def("__repr__", &Tensor1D::repr)
    .def("data_ptr", &Tensor1D::data_ptr);
}