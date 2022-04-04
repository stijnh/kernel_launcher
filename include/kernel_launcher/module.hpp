#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel_launcher {

struct CuException: std::runtime_error {
    CuException(
        CUresult err,
        const char* expression,
        const char* filename,
        int line);

    CUresult error() const {
        return _err;
    }

  private:
    CUresult _err;
};

#define KERNEL_LAUNCHER_ASSERT(expr) \
    ::kernel_launcher::assert_(expr, #expr, __FILE__, __LINE__)

static inline void
assert_(CUresult err, const char* expression, const char* filename, int line) {
    if (err != CUDA_SUCCESS) {
        throw CuException(err, expression, filename, line);
    }
}

static inline void
assert_(bool cond, const char* expression, const char* filename, int line) {
    if (!cond) {
        throw std::runtime_error(
            std::string("assertion failed: ") + expression);
    }
}

struct CudaModule {
    CudaModule() = default;
    CudaModule(const CudaModule&) = delete;
    CudaModule& operator=(const CudaModule&) = delete;

    CudaModule(CudaModule&& that) {
        *this = std::move(that);
    }

    CudaModule& operator=(CudaModule&& that) noexcept {
        std::swap(that._module, _module);
        std::swap(that._function, _function);
        return *this;
    }

    CudaModule(const char* image, const char* symbol) {
        KERNEL_LAUNCHER_ASSERT(
            cuModuleLoadDataEx(&_module, image, 0, nullptr, nullptr));
        KERNEL_LAUNCHER_ASSERT(
            cuModuleGetFunction(&_function, _module, symbol));
    }

    bool valid() const {
        return _module != nullptr;
    }

    void launch(
        dim3 grid,
        dim3 block,
        unsigned int shared_mem,
        CUstream stream,
        void** args) const {
        KERNEL_LAUNCHER_ASSERT(cuLaunchKernel(
            _function,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_mem,
            stream,
            args,
            nullptr  // extra
            ));
    }

    ~CudaModule() {
        if (valid()) {
            cuModuleUnload(_module);
            _module = nullptr;
            _function = nullptr;
        }
    }

  private:
    CUmodule _module = nullptr;
    CUfunction _function = nullptr;
};

struct CudaEvent {
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& that) {
        *this = std::move(that);
    }

    CudaEvent& operator=(CudaEvent&& that) noexcept {
        std::swap(that._event, _event);
        return *this;
    }

    CudaEvent(unsigned int flags = CU_EVENT_DEFAULT) {
        KERNEL_LAUNCHER_ASSERT(cuEventCreate(&_event, flags));
    }

    CUevent get() const {
        return _event;
    }

    operator CUevent() const {
        return _event;
    }

    void synchronize() const {
        KERNEL_LAUNCHER_ASSERT(cuEventSynchronize(_event));
    }

    void record(CUstream stream) const {
        KERNEL_LAUNCHER_ASSERT(cuEventRecord(_event, stream));
    }

    float seconds_elapsed_since(CUevent before) const {
        float time = 1337;
        KERNEL_LAUNCHER_ASSERT(cuEventElapsedTime(&time, before, _event));
        return time / 1000;  // milliseconds to seconds
    }

    ~CudaEvent() {
        if (_event) {
            KERNEL_LAUNCHER_ASSERT(cuEventSynchronize(_event));
            KERNEL_LAUNCHER_ASSERT(cuEventDestroy(_event));
            _event = nullptr;
        }
    }

  private:
    CUevent _event = nullptr;
};

template<typename T = char>
struct Memory;

template<typename T = char>
struct MemoryView {
    MemoryView() {
        _device_ptr = nullptr;
        _size = 0;
    }

    MemoryView(const MemoryView<T>& mem) {
        _device_ptr = (T*)mem.get();
        _size = mem.size();
    }

    MemoryView(const Memory<T>& mem) {
        _device_ptr = (T*)mem.get();
        _size = mem.size();
    }

    MemoryView(Memory<T>&& mem) = delete;

    MemoryView(T* device_ptr, size_t n) {
        _device_ptr = device_ptr;
        _size = n;
    }

    void copy_to(MemoryView<T> m) const {
        if (this->size() != m.size()) {
            throw std::runtime_error("size mismatch");
        }

        KERNEL_LAUNCHER_ASSERT(cuMemcpy(
            (CUdeviceptr)m._device_ptr,
            (CUdeviceptr)this->_device_ptr,
            this->size_in_bytes()));
    }

    void copy_from(MemoryView<T> m) {
        m.copy_to(*this);
    }

    void copy_to(std::vector<T>& m) const {
        copy_to(MemoryView<T>(m.data(), m.size()));
    }

    void copy_from(const std::vector<T>& m) {
        copy_from(MemoryView<T>((T*)m.data(), m.size()));
    }

    std::vector<T> to_vector() const {
        std::vector<T> data(size());
        copy_to(data);
        return data;
    }

    T* get() {
        return _device_ptr;
    }

    const T* get() const {
        return _device_ptr;
    }

    operator T*() {
        return get();
    }

    operator const T*() const {
        return get();
    }

    size_t size() const {
        return _size;
    }

    size_t size_in_bytes() const {
        return _size * sizeof(T);
    }

    MemoryView<T> slice(size_t start, size_t len) const {
        if (start + len >= _size) {  // TODO check overflow
            throw std::runtime_error("index out of bounds");
        }

        return MemoryView(_device_ptr + start, len);
    }

  protected:
    T* _device_ptr;
    size_t _size;
};

template<typename T>
struct Memory: MemoryView<T> {
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    Memory(Memory&& that) {
        *this = std::move(that);
    }

    Memory& operator=(Memory&& that) noexcept {
        std::swap(this->_device_ptr, that._device_ptr);
        std::swap(this->_size, that._size);
    }

    Memory(const std::vector<T>& values) {
        allocate(values.size());
        this->copy_from(values);
    }

    explicit Memory(size_t n = 0) {
        allocate(n);
    }

    ~Memory() {
        free();
    }

    void allocate(size_t n) {
        free();

        if (n > 0) {
            KERNEL_LAUNCHER_ASSERT(
                cuMemAlloc((CUdeviceptr*)&(this->_device_ptr), n * sizeof(T)));
            this->_size = n;
        }
    }

    void free() {
        if (this->_device_ptr) {
            KERNEL_LAUNCHER_ASSERT(cuMemFree((CUdeviceptr)this->_device_ptr));
            this->_device_ptr = nullptr;
            this->_size = 0;
        }
    }

    MemoryView<T> view() const {
        return MemoryView<T>(this->_device_ptr, this->_size);
    }
};

}  // namespace kernel_launcher