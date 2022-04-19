#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

struct CuException: std::runtime_error {
    CuException(
        CUresult err,
        const char* expression,
        const char* filename,
        int line);

    CUresult error() const {
        return err_;
    }

  private:
    CUresult err_;
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
    (void)filename;
    (void)line;
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
        std::swap(that.module_, module_);
        std::swap(that.function_, function_);
        return *this;
    }

    CudaModule(const char* image, const char* symbol) {
        KERNEL_LAUNCHER_ASSERT(
            cuModuleLoadDataEx(&module_, image, 0, nullptr, nullptr));
        KERNEL_LAUNCHER_ASSERT(
            cuModuleGetFunction(&function_, module_, symbol));
    }

    bool valid() const {
        return module_ != nullptr;
    }

    void launch(
        dim3 grid,
        dim3 block,
        unsigned int shared_mem,
        CUstream stream,
        void** args) const {
        KERNEL_LAUNCHER_ASSERT(cuLaunchKernel(
            function_,
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
            cuModuleUnload(module_);
            module_ = nullptr;
            function_ = nullptr;
        }
    }

  private:
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;
};

struct CudaEvent {
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& that) {
        *this = std::move(that);
    }

    CudaEvent& operator=(CudaEvent&& that) noexcept {
        std::swap(that.event_, event_);
        return *this;
    }

    CudaEvent(unsigned int flags = CU_EVENT_DEFAULT) {
        KERNEL_LAUNCHER_ASSERT(cuEventCreate(&event_, flags));
    }

    CUevent get() const {
        return event_;
    }

    operator CUevent() const {
        return event_;
    }

    void synchronize() const {
        KERNEL_LAUNCHER_ASSERT(cuEventSynchronize(event_));
    }

    void record(CUstream stream) const {
        KERNEL_LAUNCHER_ASSERT(cuEventRecord(event_, stream));
    }

    float seconds_elapsed_since(CUevent before) const {
        float time = 1337;
        KERNEL_LAUNCHER_ASSERT(cuEventElapsedTime(&time, before, event_));
        return time / 1000;  // milliseconds to seconds
    }

    ~CudaEvent() {
        if (event_) {
            KERNEL_LAUNCHER_ASSERT(cuEventSynchronize(event_));
            KERNEL_LAUNCHER_ASSERT(cuEventDestroy(event_));
            event_ = nullptr;
        }
    }

  private:
    CUevent event_ = nullptr;
};

struct CudaDevice {
    CudaDevice() = default;
    explicit CudaDevice(CUdevice d) : device_(d) {}

    static int count() {
        int n;
        KERNEL_LAUNCHER_ASSERT(cuDeviceGetCount(&n));
        return n;
    }

    static CudaDevice current() {
        CUdevice d;
        KERNEL_LAUNCHER_ASSERT(cuCtxGetDevice(&d));
        return CudaDevice(d);
    }

    std::string name() const {
        char name[1024];
        KERNEL_LAUNCHER_ASSERT(cuDeviceGetName(name, sizeof(name), device_));
        return name;
    }

    CUdevice get() const {
        return device_;
    }

    operator CUdevice() const {
        return get();
    }

  private:
    CUdevice device_ = -1;
};

template<typename T = char>
struct Memory;

template<typename T = char>
struct MemoryView;

namespace detail {
    template<
        typename T,
        size_t N = sizeof(T),
        typename = typename std::enable_if<std::is_trivial<T>::value>::type>
    struct MemoryFill {
        static void call(T* data, size_t n, T value) {
            MemoryView<T> view {data, n};

            // Copy initial 32 elements to device
            size_t initial_size = std::min(n, (size_t)32);
            std::vector<T> host_data(initial_size, value);
            view.slice(0, initial_size).copy_from(host_data);

            // Repeatedly copy data, doubling buffer size each time
            for (size_t k = initial_size; k < n; k *= 2) {
                size_t len = std::min(n - k, k);
                view.slice(k, len).copy_from(view.slice(0, len));
            }
        }
    };

    template<typename T>
    struct MemoryFill<
        T,
        sizeof(unsigned char),
        typename std::enable_if<std::is_trivial<T>::value>::type> {
        static void call(T* data, size_t n, T value) {
            unsigned char raw_value;
            std::memcpy(&raw_value, &value, sizeof(unsigned char));
            KERNEL_LAUNCHER_ASSERT(cuMemsetD8((CUdeviceptr)data, raw_value, n));
        }
    };

    template<typename T>
    struct MemoryFill<
        T,
        sizeof(unsigned short),
        typename std::enable_if<std::is_trivial<T>::value>::type> {
        static void call(T* data, size_t n, T value) {
            unsigned short raw_value;
            std::memcpy(&raw_value, &value, sizeof(unsigned short));
            KERNEL_LAUNCHER_ASSERT(
                cuMemsetD16((CUdeviceptr)data, raw_value, n));
        }
    };

    template<typename T>
    struct MemoryFill<
        T,
        sizeof(unsigned int),
        typename std::enable_if<std::is_trivial<T>::value>::type> {
        static void call(T* data, size_t n, T value) {
            unsigned int raw_value;
            std::memcpy(&raw_value, &value, sizeof(unsigned int));
            KERNEL_LAUNCHER_ASSERT(
                cuMemsetD32((CUdeviceptr)data, raw_value, n));
        }
    };
}  // namespace detail

template<typename Impl, typename T>
struct MemoryBase {
    MemoryView<T> view() {
        return MemoryView<T>(((Impl*)this)->data(), ((Impl*)this)->size());
    }

    MemoryView<const T> view() const {
        return MemoryView<const T>(
            ((Impl*)this)->data(),
            ((Impl*)this)->size());
    }

    size_t size_in_bytes() const {
        return view().size() * sizeof(T);
    }

    operator T*() {
        return view().data();
    }

    operator const T*() const {
        return view().data();
    }

    Memory<T> clone() const {
        Memory<T> new_buffer = Memory<T>(view().size());
        new_buffer.copy_from(*this);
        return new_buffer;
    }

    void fill(T value) {
        detail::MemoryFill<T>::call(view().data(), view().size(), value);
    }

    void fill_zeros() {
        fill((T)0);
    }

    void copy_to(T* ptr, size_t n) const {
        auto v = view();

        if (v.size() != n) {
            throw std::runtime_error("size mismatch");
        }

        KERNEL_LAUNCHER_ASSERT(cuMemcpy(
            (CUdeviceptr)ptr,
            (CUdeviceptr)v.data(),
            v.size_in_bytes()));
    }

    void copy_from(const T* ptr, size_t n) {
        auto v = view();

        if (v.size() != n) {
            throw std::runtime_error("size mismatch");
        }

        KERNEL_LAUNCHER_ASSERT(cuMemcpy(
            (CUdeviceptr)v.data(),
            (CUdeviceptr)ptr,
            v.size_in_bytes()));
    }

    template<typename I>
    void copy_to(MemoryBase<I, T>& c) const {
        auto v = c.view();
        copy_to(v.data(), v.size());
    }

    template<typename I>
    void copy_from(const MemoryBase<I, T>& c) {
        auto v = c.view();
        copy_from(v.data(), v.size());
    }

    void copy_to(std::vector<T>& c) const {
        copy_to(c.data(), c.size());
    }

    void copy_from(const std::vector<T>& c) {
        copy_from(c.data(), c.size());
    }

    std::vector<T> to_vector() const {
        std::vector<T> result(view().size());
        copy_to(result);
        return result;
    }

    MemoryView<T> slice(size_t start, size_t len) {
        auto v = view();
        if (start + len >= v.size()) {  // TODO check overflow
            throw std::runtime_error("index out of bounds");
        }

        return MemoryView<T>(v.data() + start, len);
    }
};

template<typename T>
struct Memory: MemoryBase<Memory<T>, T> {
    using value_type = T;

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    Memory() {
        //
    }

    Memory(const std::vector<T>& values) {
        allocate(values.size());
        this->copy_from(values);
    }

    Memory(Memory&& that) {
        *this = std::move(that);
    }

    Memory& operator=(Memory&& that) noexcept {
        std::swap(this->device_ptr_, that.device_ptr_);
        std::swap(this->size_, that.size_);
    }

    explicit Memory(size_t n) {
        allocate(n);
    }

    ~Memory() {
        free();
    }

    void resize(size_t new_size) {
        if (new_size != this->size()) {
            Memory<T> new_buffer = Memory<T>(new_size);

            size_t n = std::min(new_size, this->size());
            this->slice(0, n).copy_to(new_buffer.slice(0, n));

            *this = std::move(new_buffer);
        }
    }

    void allocate(size_t n) {
        free();

        if (n > 0) {
            KERNEL_LAUNCHER_ASSERT(
                cuMemAlloc((CUdeviceptr*)&(this->device_ptr_), n * sizeof(T)));
            this->size_ = n;
        }
    }

    void free() {
        if (this->device_ptr_) {
            KERNEL_LAUNCHER_ASSERT(cuMemFree((CUdeviceptr)this->device_ptr_));
            this->device_ptr_ = nullptr;
            this->size_ = 0;
        }
    }

    T* data() {
        return device_ptr_;
    }

    const T* data() const {
        return device_ptr_;
    }

    size_t size() const {
        return size_;
    }

  private:
    T* device_ptr_ = nullptr;
    size_t size_ = 0;
};

template<typename T>
struct MemoryView: MemoryBase<MemoryView<T>, T> {
    using value_type = T;

    MemoryView() = default;
    MemoryView(const MemoryView&) = default;
    MemoryView(MemoryView&&) = default;
    MemoryView<T>& operator=(const MemoryView<T>&) = default;
    MemoryView<T>& operator=(MemoryView<T>&&) noexcept = default;

    MemoryView(T* ptr, size_t size) : device_ptr_(ptr), size_(size) {
        //
    }

    MemoryView(Memory<T>& mem) : device_ptr_(mem.data()), size_(mem.size()) {
        //
    }

    MemoryView(const Memory<T>& mem) :
        device_ptr_(mem.data()),
        size_(mem.size()) {
        //
    }

    MemoryView(Memory<T>&& mem) = delete;

    T* data() const {
        return device_ptr_;
    }

    size_t size() const {
        return size_;
    }

  private:
    T* device_ptr_ = nullptr;
    size_t size_ = 0;
};

}  // namespace kernel_launcher

#if KERNEL_LAUNCHER_HEADERONLY
    #include KERNEL_LAUNCHER_IMPL("module.cpp")
#endif