#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"  // for ggml_get_type_traits_cpu (vec_dot, from_float)

#include <chrono>
#include <cuda_profiler_api.h>
#include <thread>
#include <atomic>
#include <functional>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/moe-merge.cuh"
#include "ggml-cuda/moe-down9.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/dsv4.cuh"
#include "ggml-cuda/hc-fuse.cuh"
#include "ggml-cuda/dre.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/smallm-tc.cuh"   // [ALPHA_SMALLM_TC]
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/q8stage.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <unordered_set>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

// Every ALPHA_* boolean in this file is on when set and non-zero; absent, empty or "0" is off.
// No flag is presence-checked, so for example ALPHA_NO_POOL_DIRECT=0 leaves pool-direct enabled.
static inline bool alpha_env_on(const char * name) {
    const char * e = getenv(name);
    return e && e[0] && atoi(e) != 0;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;
    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
    for (int id = 0; id < info.device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.device_count, (size_t)(total_vram / (1024 * 1024)));
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      (size_t)(prop.totalGlobalMem / (1024 * 1024)));
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      (size_t)(prop.totalGlobalMem / (1024 * 1024)));
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      (size_t)(prop.totalGlobalMem / (1024 * 1024)));
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.device_count; ++id) {
            ggml_cuda_set_device(id);
            for (int id_other = 0; id_other < info.device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess((CUdeviceptr)((char *)(pool_addr) + pool_size), reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }

    size_t trim() override {
#if defined(GGML_USE_HIP)
        return 0;
#else
        if (pool_addr == 0 || pool_used != 0 || pool_size == 0) return 0;
        // the physical handles were released right after mapping, so unmapping frees them; the virtual range stays
        // reserved and the next alloc maps from pool_addr again
        CU_CHECK(cuMemUnmap(pool_addr, pool_size));
        const size_t freed = pool_size;
        pool_size = 0;
        return freed;
#endif
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

// runs fn while no CUDA graph is being captured, updated or instantiated (and none can start until it returns).
// Used by the RAM hot tier (ALPHA_DRE_TIER), which registers its arena from a background thread: a cudaHostRegister
// that overlaps a capture + cudaGraphExecUpdate on the decode thread makes the update fail.
void ggml_cuda_outside_capture(void (*fn)(void *), void * arg) {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });
    fn(arg);
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
    }

    // Free the expert cache and mirror GPU memory
    for (auto & e : expert_cache) {
        if (e.gpu_data) cudaFree(e.gpu_data);
    }
    for (auto & m : tensor_mirrors) {
        if (m.gpu_mirror) cudaFree(m.gpu_mirror);
    }
#ifdef ALPHA_DEBUG_BUILD
    if (expert_cache_hits + expert_cache_misses > 0) {
        int64_t total = expert_cache_hits + expert_cache_misses;
        fprintf(stderr, "cache stats: %lld hits, %lld misses (%.1f%% hit rate), %lld fast path, %.1f MB used\n",
            (long long)expert_cache_hits, (long long)expert_cache_misses,
            (float)expert_cache_hits / total * 100.0f,
            (long long)expert_cache_fast_path,
            (float)expert_cache_used / (1024*1024));
    }
#endif
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// ALPHA_H2D_LOG=min_bytes: log every host->device upload of at least min_bytes (first 400 only), to find tensors a
// decode re-uploads from pageable memory.
static void alpha_h2d_log(const ggml_tensor * t, size_t size, const char * path) {
    static const long long lim = [] { const char * e = getenv("ALPHA_H2D_LOG"); return e ? atoll(e) : -1LL; }();
    if (lim < 0 || (long long) size < lim || strstr(t->name, ".weight")) return;
    static int n = 0;
    static const int nmax = [] { const char * e = getenv("ALPHA_H2D_LOG_MAX"); return e ? atoi(e) : 400; }();
    if (n++ < nmax) fprintf(stderr, "[h2d] %s %s %s %zu B\n", path, t->name, ggml_op_desc(t), size);
}

// ALPHA_CUDA_BIG_H2D_KERNEL (default 1): copy large (> 1 MB) input uploads of prompt-sized graphs, such as CPU-split
// token embeddings and PLE rows, with SMs instead of a DMA. A DMA queues behind the prompt pipe's expert prefetch on
// the copy engine (FIFO) and stalls each chunk start much longer than the transfer itself. Copies straight from pinned
// memory, or through two 32 MB pinned staging pieces when the source is pageable. Synchronous like the DMA it replaces.
static __global__ void alpha_h2d_big_copy_k(uint4 * __restrict__ dst, const uint4 * __restrict__ src, const size_t n16) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
static bool alpha_h2d_big_upload(void * dst, const void * src, size_t size, cudaStream_t st) {
    static const bool on = [] { const char * e = getenv("ALPHA_CUDA_BIG_H2D_KERNEL"); return !e || atoi(e) != 0; }();
    if (!on || size <= ((size_t) 1 << 20) || !ggml_backend_sched_alpha_async_inputs_on()) return false;
    if (((uintptr_t) dst & 15) || ((uintptr_t) src & 15) || (size & 15)) return false;
    static const int nblk = [] { const char * e = getenv("ALPHA_CUDA_BIG_H2D_BLOCKS"); return e ? std::max(1, atoi(e)) : 256; }();
    cudaPointerAttributes attr;
    const bool pinned = cudaPointerGetAttributes(&attr, src) == cudaSuccess && attr.type == cudaMemoryTypeHost && attr.devicePointer;
    (void) cudaGetLastError();
    if (pinned) {
        const size_t n16 = size / 16;
        alpha_h2d_big_copy_k<<<(int) std::min<size_t>((size_t) nblk, (n16 + 255) / 256), 256, 0, st>>>((uint4 *) dst, (const uint4 *) attr.devicePointer, n16);
        return true;
    }
    // pageable: double-buffered pinned staging, host memcpy of piece k+1 overlaps the kernel reading piece k
    constexpr size_t PIECE = (size_t) 32 << 20;
    static char * stage[2] = { nullptr, nullptr };
    static char * stage_d[2] = { nullptr, nullptr };
    static cudaEvent_t done[2] = { nullptr, nullptr };
    if (!stage[0]) {
        for (int k = 0; k < 2; k++) {
            if (cudaHostAlloc((void **) &stage[k], PIECE, cudaHostAllocMapped) != cudaSuccess) { (void) cudaGetLastError(); stage[k] = nullptr; return false; }
            CUDA_CHECK(cudaHostGetDevicePointer((void **) &stage_d[k], stage[k], 0));
            CUDA_CHECK(cudaEventCreateWithFlags(&done[k], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(done[k], st));
        }
    }
    int k = 0;
    for (size_t off = 0; off < size; off += PIECE, k ^= 1) {
        const size_t n = std::min(PIECE, size - off);
        CUDA_CHECK(cudaEventSynchronize(done[k]));   // the kernel that read this piece last time is done
        memcpy(stage[k], (const char *) src + off, n);
        const size_t n16 = n / 16;
        alpha_h2d_big_copy_k<<<(int) std::min<size_t>((size_t) nblk, (n16 + 255) / 256), 256, 0, st>>>((uint4 *) ((char *) dst + off), (const uint4 *) stage_d[k], n16);
        CUDA_CHECK(cudaEventRecord(done[k], st));
    }
    return true;
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
    alpha_h2d_log(tensor, size, "buf");

    ggml_cuda_set_device(ctx->device);
    if (!alpha_h2d_big_upload((char *) tensor->data + offset, data, size, cudaStreamPerThread)) {
        CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    const int64_t t_alloc0 = ggml_time_us();
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (size >= ((size_t) 256 << 20) && getenv("ALPHA_ALLOC_LOG")) {
        GGML_LOG_INFO("[alloc] device buffer %.0f MiB in %.1f ms\n", size / 1048576.0, (ggml_time_us() - t_alloc0) / 1000.0);
    }
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

static int64_t get_row_rounding(const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split) {
    int64_t row_rounding = 0;
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            continue;
        }

        const int cc = ggml_cuda_info().devices[id].cc;
        row_rounding = std::max(row_rounding, (int64_t)get_mmq_y_host(cc));
    }
    return row_rounding;
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;

    if (id == ggml_backend_cuda_get_device_count() - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_cuda_split_buffer_type_context {
    int main_device;
    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    std::string name;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
                for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
                    if (extra->events[id][is] != nullptr) {
                        CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
                    }
                }
                if (extra->data_device[id] != nullptr) {
                    CUDA_CHECK(cudaFree(extra->data_device[id]));
                }
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};


static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_cuda_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if cudaMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_cuda_set_device(id);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, size, id));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }

        extra->data_device[id] = buf;

        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
            CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

static const char * ggml_backend_cuda_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

#ifdef GGML_USE_NCCL
struct ggml_backend_cuda_comm_context {
    std::vector<ggml_backend_t> backends;
    std::vector<ncclComm_t> comms;

    ~ggml_backend_cuda_comm_context() {
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
    }
};
#endif // GGML_USE_NCCL

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
#ifdef GGML_USE_NCCL
    if (comm_ctx_v == nullptr) {
        return;
    }
    ggml_backend_cuda_comm_context * comm_ctx = (ggml_backend_cuda_comm_context *) comm_ctx_v;
    delete comm_ctx;
#else
    GGML_UNUSED(comm_ctx_v);
#endif // GGML_USE_NCCL
}

static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
#ifdef GGML_USE_NCCL
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }
    ggml_backend_cuda_comm_context * ret = new ggml_backend_cuda_comm_context;
    std::vector<int> dev_ids;
    ret->backends.reserve(n_backends);
    dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->backends.push_back(backends[i]);
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;
        dev_ids.push_back(cuda_ctx->device);
    }

    ret->comms.resize(n_backends);
    NCCL_CHECK(ncclCommInitAll(ret->comms.data(), n_backends, dev_ids.data()));
    return ret;
#else
    // If NCCL is installed it is used by default for optimal performance.
    // However, NVIDIA does not distribute NCCL with CUDA so users may be unwittingly missing this package.
    // RCCL is disabled by default, users are explicitly opting in.
    // Therefore print no warning for RCCL.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    static bool warning_printed = false;
    if (!warning_printed) {
        GGML_LOG_WARN("%s: NVIDIA Collective Communications Library (NCCL) is unavailable, multi GPU performance will be suboptimal\n", __func__);
        warning_printed = true;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(backends, n_backends);
    return nullptr;
#endif // GGML_USE_NCCL
}

static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
#ifdef GGML_USE_NCCL
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    GGML_ASSERT(comm_ctx_v != nullptr);
    ggml_backend_cuda_comm_context * comm_ctx = (ggml_backend_cuda_comm_context *) comm_ctx_v;
    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());

        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
#else
    GGML_UNUSED_VARS(comm_ctx_v, tensors);
    return false;
#endif // GGML_USE_NCCL
}

ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<std::pair<int, std::array<float, GGML_CUDA_MAX_DEVICES>>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_CUDA_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_cuda_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find({main_device, tensor_split_arr});
    if (it != buft_map.end()) {
        return &it->second;
    }
    auto * ctx = new ggml_backend_cuda_split_buffer_type_context{
        main_device,
        tensor_split_arr,
        GGML_CUDA_NAME + std::to_string(main_device) + "_Split",
    };

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), main_device),
        /* .context = */ ctx,
    };

    auto result = buft_map.emplace(std::make_pair(main_device, tensor_split_arr), buft);
    return &result.first->second;
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

// ALPHA_CUDA_HOST_CACHE_MB (default 4096): pinned host buffers are kept after free and handed out again. The
// scheduler's CUDA_Host compute buffer is freed and re-allocated around every long prompt (the DRE borrow resets the
// scheduler), and each cudaMallocHost/cudaFreeHost pair of a few hundred MB costs tens of ms of host time.
static std::mutex g_cuda_host_cache_mutex;
static std::vector<std::pair<void *, size_t>> g_cuda_host_cache;   // free blocks
static std::unordered_map<void *, size_t> g_cuda_host_size;        // every live or cached block's real size
static size_t g_cuda_host_cache_bytes = 0;
static size_t ggml_cuda_host_cache_cap() {
    static const size_t v = [] { const char * e = getenv("ALPHA_CUDA_HOST_CACHE_MB"); return (size_t) (e ? atoll(e) : 4096) << 20; }();
    return v;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    void * ptr = buffer->context;
    {
        std::lock_guard<std::mutex> lk(g_cuda_host_cache_mutex);
        auto it = g_cuda_host_size.find(ptr);
        if (it != g_cuda_host_size.end() && g_cuda_host_cache_bytes + it->second <= ggml_cuda_host_cache_cap()) {
            g_cuda_host_cache.push_back({ ptr, it->second });
            g_cuda_host_cache_bytes += it->second;
            return;
        }
        if (it != g_cuda_host_size.end()) g_cuda_host_size.erase(it);
    }
    CUDA_CHECK(cudaFreeHost(ptr));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    // ALPHA_CUDA_HOST_PREWARM_MB (default 0; alphaserve sets it for big DRE prompt micro-batches): allocate and cache one
    // pinned block up front, so the first long prompt's scheduler host buffer does not pay a cudaMallocHost inside its
    // time-to-first-token. Checked on every call until seen, since the server sets it after the model's host buffers exist.
    static std::atomic<bool> prewarmed { false };
    if (!prewarmed.load(std::memory_order_relaxed)) {
        const char * e = getenv("ALPHA_CUDA_HOST_PREWARM_MB");
        const size_t mb = e ? (size_t) atoll(e) : 0;
        if (mb > 0 && !prewarmed.exchange(true)) {
            void * p = nullptr;
            if (cudaMallocHost(&p, mb << 20) == cudaSuccess) {
                std::lock_guard<std::mutex> lk(g_cuda_host_cache_mutex);
                g_cuda_host_size[p] = mb << 20;
                g_cuda_host_cache.push_back({ p, mb << 20 });
                g_cuda_host_cache_bytes += mb << 20;
            } else {
                (void) cudaGetLastError();
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_cuda_host_cache_mutex);
        int best = -1;
        for (int i = 0; i < (int) g_cuda_host_cache.size(); i++) {
            const size_t c = g_cuda_host_cache[i].second;
            // a big request (a prompt's scheduler buffer) takes any larger block: a re-allocation costs far more
            // (but not 8x larger, so the base micro-batch's buffer does not take the prewarmed block between prompts)
            const bool fits = c >= size && (c <= 2 * size + (64u << 20) || (size >= ((size_t) 64 << 20) && c <= 8 * size));
            if (fits && (best < 0 || c < g_cuda_host_cache[best].second)) best = i;
        }
        if (best >= 0) {
            void * p = g_cuda_host_cache[best].first;
            g_cuda_host_cache_bytes -= g_cuda_host_cache[best].second;
            g_cuda_host_cache.erase(g_cuda_host_cache.begin() + best);
            return p;
        }
    }

    void * ptr = nullptr;
    const int64_t t_alloc0 = ggml_time_us();
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (size >= ((size_t) 64 << 20) && getenv("ALPHA_ALLOC_LOG")) {
        GGML_LOG_INFO("[alloc] pinned host %.0f MiB in %.1f ms (not cached)\n", size / 1048576.0, (ggml_time_us() - t_alloc0) / 1000.0);
    }
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_cuda_host_cache_mutex);
        g_cuda_host_size[ptr] = size;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // ALPHA_CUDA_HOST_HEADROOM_PCT (default 50): allocate a big pinned buffer (the scheduler's host compute buffer of a
    // long prompt) with headroom. The second prompt chunk's inputs (KQ mask over a longer KV) are larger than the
    // reserve's, and growing the buffer would re-allocate ~1 GB of pinned memory between the chunks.
    size_t want = size;
    {
        static const size_t pct = [] { const char * e = getenv("ALPHA_CUDA_HOST_HEADROOM_PCT"); return (size_t) (e ? atoll(e) : 50); }();
        // only in a server that asked for the prompt-buffer cache, and never for a weight-sized buffer
        // (read per call: the server sets it after the model's buffers exist)
        if (getenv("ALPHA_CUDA_HOST_PREWARM_MB") && size >= ((size_t) 64 << 20) && size <= ((size_t) 4 << 30) && pct > 0) {
            want = ((size + size * pct / 100) + ((size_t) 64 << 20) - 1) & ~(((size_t) 64 << 20) - 1);
            want = std::max(want, (size_t) 512 << 20);   // one reusable size class for every mid-size prompt buffer
        }
    }
    void * ptr = ggml_cuda_host_malloc(want);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }
    {
        // report the block's real size (a cached block can be larger): the allocator then grows it only when needed
        std::lock_guard<std::mutex> lk(g_cuda_host_cache_mutex);
        auto it = g_cuda_host_size.find(ptr);
        if (it != g_cuda_host_size.end() && it->second > want) want = it->second;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, want);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

struct cublas_force_compute_type {
    bool fp32 = false;
    bool fp16 = false;
};

static const cublas_force_compute_type & ggml_cuda_cublas_get_force_compute_type() {
    static const cublas_force_compute_type compute_type = [] {
        cublas_force_compute_type result;

        const bool ggml_cuda_force_cublas_compute_32f_env = getenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F") != nullptr;
        const bool ggml_cuda_force_cublas_compute_16f_env = getenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F") != nullptr;

        GGML_ASSERT(ggml_cuda_force_cublas_compute_16f_env == false || ggml_cuda_force_cublas_compute_32f_env == false);

        if (ggml_cuda_force_cublas_compute_32f_env) {
            GGML_LOG_INFO("Detected GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F\n");
            result.fp32 = true;
        } else if (ggml_cuda_force_cublas_compute_16f_env) {
            GGML_LOG_INFO("Detected GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F\n");
            result.fp16 = true;
        }

        return result;
    }();

    return compute_type;
}

// ik_llama.cpp quant types: row-meta-aware device dequant (convert-ik.cu).
GGML_BACKEND_API_C void ggml_cuda_dequantize_ik_rows_f16(const void * vx, void * y, int64_t nrows, int64_t ne, int64_t row_stride, int type, cudaStream_t stream);
GGML_BACKEND_API_C void ggml_cuda_dequantize_ik_rows_f32(const void * vx, void * y, int64_t nrows, int64_t ne, int64_t row_stride, int type, cudaStream_t stream);
GGML_BACKEND_API_C void ggml_cuda_ik_mmvq_iq1kt(const void * wexpert, int64_t nb01, const void * xrow, void * drow, int64_t ne00, int64_t ne01, cudaStream_t stream);
GGML_BACKEND_API_C bool ggml_cuda_is_ik_type(int type);

// Set once, by ggml_backend_cuda_graph_compute, the first time a graph carries an expert matmul
// (GGML_OP_MUL_MAT_ID). The small-M GEMM tier uses it as the model-type signal: the tier helps dense
// models but slows MoE models. Sticky for the process, since a serve process holds one model.
static bool g_alpha_graph_has_mmid = false;

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int cc = ggml_cuda_info().devices[id].cc;

    const bool supports_bf16 = GGML_CUDA_CC_IS_NVIDIA(cc) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);

    const bool use_fp16 =
        src0->type != GGML_TYPE_NVFP4 &&
        (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] &&
        dst->op_params[0] == GGML_PREC_DEFAULT;

    if (supports_bf16 && src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
            GGML_ASSERT(to_bf16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1_ddf_i, src1_as_bf16.get(), ne, stream);
        }
        const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16 * src0_ptr = (const nv_bfloat16 *)src0_dd_i;

        const float alpha_f32 = 1.0f;
        const float beta_f32  = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

        // ALPHA_BF16_F32_OUT (default on): bf16 x bf16 with an F32 output and F32 compute is a supported cuBLAS
        // combination, so write dst directly instead of a bf16 temporary plus a bf16 -> f32 conversion kernel (one
        // full pass over the output per call). Slightly more accurate too, since the f32 accumulator is not rounded to bf16.
        static const bool f32_out = [] { const char * e = getenv("ALPHA_BF16_F32_OUT"); return !e || atoi(e) != 0; }();
        if (f32_out) {
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f32,  src0_ptr, CUDA_R_16BF, ne00,
                                     src1_ptr, CUDA_R_16BF, ne10,
                        &beta_f32,   dst_dd_i, CUDA_R_32F,  ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            return;
        }
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                                 src1_ptr,       CUDA_R_16BF, ne10,
                    &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff*src1_ncols, stream);
    } else if (fast_fp16_hardware_available(cc) && use_fp16) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            if (ggml_cuda_is_ik_type(src0->type)) {
                // row-meta-aware: dequant row_diff rows of ne00, src row stride = nb[1]
                ggml_cuda_dequantize_ik_rows_f16(src0_dd_i, src0_as_f16.get(), row_diff, ne00, src0->nb[1], src0->type, stream);
            } else {
                const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
                GGML_ASSERT(to_fp16_cuda != nullptr);
                to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
            }
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

        const auto & force_compute_type = ggml_cuda_cublas_get_force_compute_type();

        if (!force_compute_type.fp16 && (GGML_CUDA_CC_IS_CDNA(cc)
                                        || GGML_CUDA_CC_IS_RDNA4(cc)
                                        || cc == GGML_CUDA_CC_VOLTA
                                        || force_compute_type.fp32))
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha, src0_ptr,  CUDA_R_16F, ne00,
                                src1_ptr,  CUDA_R_16F, ne10,
                        &beta,   dst_dd_i, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

            const half alpha_f16 = 1.0f;
            const half beta_f16 = 0.0f;

            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f16, src0_ptr,      CUDA_R_16F, ne00,
                                    src1_ptr,      CUDA_R_16F, ne10,
                        &beta_f16,  dst_f16.get(), CUDA_R_16F, ldc,
                        CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
            to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
        }
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            src0_ddq_as_f32.alloc(row_diff*ne00);
            if (ggml_cuda_is_ik_type(src0->type)) {
                ggml_cuda_dequantize_ik_rows_f32(src0_dd_i, src0_ddq_as_f32.get(), row_diff, ne00, src0->nb[1], src0->type, stream);
            } else {
                const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
                GGML_ASSERT(to_fp32_cuda != nullptr);
                to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
            }
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int64_t i02_divisor = ne12 / ne02;
    const int64_t i03_divisor = ne13 / ne03;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));
    GGML_ASSERT(!(split && ne03 < ne13));

    ggml_tensor_extra_gpu * src0_extra = split ? (ggml_tensor_extra_gpu *) src0->extra : nullptr;


    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(tensor_split);

            if (id != 0) {
                dev[id].row_low  = ne01*tensor_split[id];
                if (dev[id].row_low < ne01) {
                    dev[id].row_low -= dev[id].row_low % rounding;
                }
            }

            if (id != ggml_backend_cuda_get_device_count() - 1) {
                dev[id].row_high  = ne01*tensor_split[id + 1];
                if (dev[id].row_high < ne01) {
                    dev[id].row_high -= dev[id].row_high % rounding;
                }
            }
        }
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char *) src0_extra->data_device[id] : (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer.
            // This buffer needs to be cleared entirely because multiple regions will function as padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            const size_t nbytes_data    = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low)*ne00);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && src1_is_contiguous) {
                quantize_src1(
                    dev[id].src1_ddf, nullptr, dev[id].src1_ddq, src0->type, ne10,
                    nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
                    src1_padded_col_size, ne11, ne12, ne13, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low)*ne1 : ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaEventRecord(src0_extra->events[ctx.device][0], ctx.stream()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            // wait for main GPU data if necessary
            if (split && (id != ctx.device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[ctx.device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                const size_t nbytes_src0_matrix = ne01*ne00*src0_ts / src0_bs;
                char  *  src0_dd_i =  dev[id].src0_dd + ((i03/i03_divisor)*ne02 + (i02/i02_divisor)) * nbytes_src0_matrix;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous) {
                    quantize_src1(
                        src1_ddf_i, nullptr, src1_ddq_i, src0->type, ne10, ne10, ne11*ne10, ne12*ne11*ne10,
                        src1_padded_col_size, src1_ncols, 1, 1, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i03 % i03_divisor == 0 && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                        src0_dd_i, src0, i03/i03_divisor, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[id].row_low;
                        CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(
                            dhf_dst_i, ctx.device, ne0*sizeof(float), dst_dd_i, id, row_diff*sizeof(float), row_diff*sizeof(float), src1_ncols, stream));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_backend_cuda_get_device_count() > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

        ggml_cuda_set_device(ctx.device);
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (dev[id].row_low == dev[id].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), src0_extra->events[id][is], 0));
            }
        }
    }
}

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type src0_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<src0_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft));
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float * dst_ddf = (float *) dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == ts_src1);
    int64_t s11 = nb11 / ts_src1;
    int64_t s12 = nb12 / ts_src1;
    int64_t s13 = nb13 / ts_src1;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // Handle src0
    src0_ptr = (const cuda_t *) src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_alloc.alloc(ne_src1);

        const auto convert_func = traits::get_nc_converter(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        src1_ptr = src1_alloc.get();
        s11 = ne10;
        s12 = ne11*s11;
        s13 = ne12*s12;

        is_src1_cont_2 = true;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const auto & force_compute_type = ggml_cuda_cublas_get_force_compute_type();

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    static constexpr bool is_src0_type_f16 = src0_type == GGML_TYPE_F16;

    // bf16 and fp32 are already being computed in fp32 (ensure it using static_assert),
    // so checking necessity of forced fp32 only for fp16 src0_type
    static_assert(is_src0_type_f16 || traits::compute_type == CUBLAS_COMPUTE_32F);

    const bool need_compute_32f = is_src0_type_f16 && !force_compute_type.fp16 && (GGML_CUDA_CC_IS_CDNA(cc)
                                                                                  || GGML_CUDA_CC_IS_RDNA4(cc)
                                                                                  || cc == GGML_CUDA_CC_VOLTA
                                                                                  || force_compute_type.fp32);

    if (dst->op_params[0] == GGML_PREC_DEFAULT && !need_compute_32f) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_t = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    } else {
        dst_t = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
        const int64_t smb = ne12 == 1 ? s13       : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11,       smb,     // strideB
                beta,     dst_t, cu_data_type,   ne0,       ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        size_t src1_stride_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        dim3 block_dims(threads_x, threads_y);

        dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                (src1->type == src0_type) ? nb12 : s12*src1_stride_size,
                (src1->type == src0_type) ? nb13 : s13*src1_stride_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("Unsupported type");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (glu->src[0] != ffn_gate_bias || glu->src[1] != ffn_up_bias) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up || ffn_up_bias->src[1] == ffn_up;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate || ffn_gate_bias->src[1] == ffn_gate;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up || ffn_gate_bias->src[0] != ffn_gate) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    } else {
        if (glu->src[0] != ffn_gate && glu->src[1] != ffn_up) {
            return false;
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (ffn_up->src[2] && (ffn_up->src[2] != ffn_gate->src[2])) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 3> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    const bool split = ggml_backend_buft_is_cuda_split(ffn_up->src[0]->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(ffn_gate->src[0]->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(src1->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

// Fusion is blocked for passes with CPU-half experts: the fused gate/up kernel writes one combined
// GLU output, so per-tensor dst-row correction has nothing to overwrite. Defined after the pcache globals.
static bool pcache_hybrid_fusion_blocked(void);

int alpha_mmid_split_max();   // mmvq.cu

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    if (tensor->op == GGML_OP_MUL_MAT_ID && pcache_hybrid_fusion_blocked()) {
        return false;
    }

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE &&
                             !ggml_cuda_is_ik_type(src0->type); // ik types have no MMVQ kernel -> cuBLAS dequant path
    // [ALPHA_SMALLM_TC] a product the small-M kernel takes is never fused into MMVQ: plain decode and the verify
    // must run the same kernel for exact == plain (see ggml_cuda_smallm_tc_covers)
    if (tensor->op == GGML_OP_MUL_MAT && ggml_cuda_smallm_tc_covers(src0, src1, dst)) {
        return false;
    }

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    // With ALPHA_MMID_SPLIT, 2..N tokens run as 1-token calls inside ggml_cuda_mul_mat_vec_q, so the
    // fused gate/up+GLU form is valid there too (and is what a decode token uses).
    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1 &&
        !(dst->ne[2] >= 2 && dst->ne[2] <= alpha_mmid_split_max() && dst->ne[3] == 1)) {
        return false;
    }


    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(src1->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    return use_mul_mat_vec_q;
}

// 128-bit-safe comparison of two 64x64 products: returns (a*b) < (c*d) without overflow.
// `unsigned __int128` is a GCC/Clang extension that MSVC does not support, so on MSVC x64
// _umul128 provides the full 128-bit product as hi:lo.
#if defined(_MSC_VER)
#  include <intrin.h>
static inline bool alpha_mul_lt(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    unsigned long long h1 = 0, h2 = 0;
    const unsigned long long l1 = _umul128(a, b, &h1);
    const unsigned long long l2 = _umul128(c, d, &h2);
    return (h1 != h2) ? (h1 < h2) : (l1 < l2);
}
#else
static inline bool alpha_mul_lt(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    return (unsigned __int128)a * b < (unsigned __int128)c * d;
}
#endif

// Multi-user expert cache context (captured on first mul_mat_id call)
static ggml_backend_cuda_context * g_cuda_ctx_for_expert = nullptr;

// Marlin INT4 Tensor Core hook for quantized weights (decode only, M <= 16).
// Weak because the definitions live in a different target (tools/serve/serve-marlin.cu): on ELF a weak
// declaration resolves to null when that target is not linked and the call site checks for null.
// MSVC has no weak symbols, so the hook is compiled out there.
#if defined(_MSC_VER)
#  define ALPHA_MARLIN_HOOK 0
#else
#  define ALPHA_MARLIN_HOOK 1
GGML_BACKEND_API_C int marlin_ggml_mul_mat(void *, const void *, const void *, int, int, int, int, void *) __attribute__((weak));
GGML_BACKEND_API_C bool marlin_hook_active() __attribute__((weak));
#endif

// Multi-user batched-decode offload of dense-FFN weights (ALPHA_DENSE_BATCH_MIN, default 0 = off).
// Concurrent decode tokens co-batch into one llama_decode; offload_op then pulls the offloaded FFN
// weight to the GPU, which copies it H2D once per forward and computes all N tokens in one batched GEMM,
// amortizing the weight DMA. Value N = minimum co-batch width at which to pull the FFN to the GPU; below
// it the FFN stays on the CPU backend (pulling at N=1 is much slower). FFN-only, so safe with DeltaNet.
static int g_alpha_dense_batch_min = []() {
    const char * e = getenv("ALPHA_DENSE_BATCH_MIN");
    int v = e ? atoi(e) : 0;
    return v < 0 ? 0 : v;
}();

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // Marlin INT4 hook: Tensor Core path for quantized decode (M <= 16)
#if ALPHA_MARLIN_HOOK
    if (marlin_hook_active && marlin_hook_active() &&
        ggml_is_quantized(src0->type) && !ggml_cuda_is_ik_type(src0->type) && src1->ne[1] <= 16 &&
        src0->ne[0] % 128 == 0 && src0->ne[1] % 128 == 0) {
        int M = src1->ne[1], K = src0->ne[0], N = src0->ne[1];
        if (marlin_ggml_mul_mat &&
            marlin_ggml_mul_mat(dst->data, src0->data, src1->data, M, N, K, (int)src0->type, ctx.stream()) == 0) {
            return;
        }
    }
#endif

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    bool use_mul_mat_vec_f = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_f     = !ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE
        && !ggml_cuda_is_ik_type(src0->type); // ik types: no MMVQ/MMQ kernel -> cuBLAS dequant
    bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && !ggml_cuda_is_ik_type(src0->type);

    bool any_gpus_with_slow_fp16 = false;

    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            const int cc            = ggml_cuda_info().devices[id].cc;
            const int warp_size     = ggml_cuda_info().devices[id].warp_size;
            use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
            use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
            use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
            any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
        }
    } else {
        const int cc            = ggml_cuda_info().devices[ctx.device].cc;
        const int warp_size     = ggml_cuda_info().devices[ctx.device].warp_size;
        use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
        use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
        use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
        any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
    }

    // ALPHA_OUTPUT_MMQ: small-M route for the LM head only.
    // MMVQ is tested before MMQ in the dispatch chain below and accepts up to MMVQ_MAX_BATCH_SIZE=8, so a
    // 2..8-row output matmul takes the per-row GEMV path and re-reads the whole output matrix once per row,
    // several times the marginal cost of a token in a batched forward. That makes wide MTP verify windows
    // and tree drafting expensive.
    // Scoped to the output tensor and to 2 <= M <= limit: a global "M >= 1 uses MMQ" switch
    // (GGML_CUDA_BATCH_INVARIANT) is far slower because it moves single-token decode onto MMQ.
    // ggml_cuda_should_use_mmq already ran above, so use_mul_mat_q reflects real MMQ eligibility; if MMQ is
    // unavailable this is a no-op and MMVQ stays.
    static const int alpha_out_mmq = []() {
        const char * e = getenv("ALPHA_OUTPUT_MMQ");
        return e ? atoi(e) : 0;
    }();
    // one-shot dispatch dump for the vocab-sized matmul (llm_graph_context::cb never sets
    // tensor names -- it is a no-op without an installed callback -- so dst->name is empty here)
    if (alpha_out_mmq > 0 && src0->ne[1] > 100000) {
        static int shown = 0;
        if (shown < 6) { shown++;
            fprintf(stderr, "[alpha-outmmq] src0=%s type=%s ne=[%d,%d] M=%d vecq=%d mmq=%d split=%d",
                    src0->name, ggml_type_name(src0->type), (int) src0->ne[0], (int) src0->ne[1],
                    (int) src1->ne[1], (int) use_mul_mat_vec_q, (int) use_mul_mat_q, (int) split);
            fputc(10, stderr);
        }
    }
    // ALPHA_OUTPUT_MMQ scopes this to the LM head; ALPHA_SMALLM_MMQ applies it to every quantized
    // matmul in the 2..M window. M == 1 is never touched.
    // Default: on (8) for dense models, off for models with experts, where it slows decode. The model type
    // is read off the graph (g_alpha_graph_has_mmid, set by graph_compute on the first MUL_MAT_ID it sees).
    // An explicit ALPHA_SMALLM_MMQ always wins: =0 disables, =N sets the upper width on either model type.
    static const int alpha_smallm_mmq_env = []() {
        const char * e = getenv("ALPHA_SMALLM_MMQ");
        return e ? atoi(e) : -1;
    }();
    const int alpha_smallm_mmq = alpha_smallm_mmq_env >= 0 ? alpha_smallm_mmq_env
                               : (g_alpha_graph_has_mmid ? 0 : 8);
    const int alpha_mmq_lim = alpha_smallm_mmq > 0 ? alpha_smallm_mmq : alpha_out_mmq;
    // MMQ has the right per-column slope but a fixed intercept at small M. The penalty is shape-dependent
    // (MMQ wins on big projections where the weight read dominates and loses on small ones), so a size gate
    // keeps the slope without the intercept. ALPHA_SMALLM_MMQ_MIN is the minimum src0 weight size in MiB
    // for the MMQ route; 0 = no gate (every quantized matmul).
    static const double alpha_mmq_min_mib = []() {
        const char * e = getenv("ALPHA_SMALLM_MMQ_MIN");
        return e ? atof(e) : 16.0;
    }();
    // MMVQ still wins at the narrowest widths (its intercept is lower), so the tier needs a lower width
    // bound as well as an upper one (crossover around 4 tokens with a 16 MiB size gate).
    static const int alpha_mmq_lo = []() {
        const char * e = getenv("ALPHA_SMALLM_MMQ_LO");
        const int v = e ? atoi(e) : 4;
        return v < 2 ? 2 : v;
    }();
    if (alpha_mmq_lim > 0 && use_mul_mat_vec_q && use_mul_mat_q && !split &&
        src1->ne[1] >= alpha_mmq_lo && src1->ne[1] <= alpha_mmq_lim &&
        (alpha_mmq_min_mib <= 0.0 ||
         (double) ggml_nbytes(src0) >= alpha_mmq_min_mib * 1048576.0) &&
        (alpha_smallm_mmq > 0 ||
         strncmp(src0->name, "output.weight",     13) == 0 ||
         strncmp(src0->name, "token_embd.weight", 17) == 0)) {
        use_mul_mat_vec_q = false;   // fall through to the MMQ branch
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    //TODO update for generic tensor parallelism
    const int cc                 = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_batched_cublas_f16  = src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
    bool use_batched_cublas_bf16 = src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
    bool use_batched_cublas_f32  = src0->type == GGML_TYPE_F32;

    // ALPHA_MM_DISPATCH_LOG: print the chosen path for every mul_mat, so two runs at different token
    // counts can be diffed to find ops that change kernel with the batch width. The choice comes from
    // several predicates with their own arch and shape conditions plus the MTP batch-invariance pins, and a
    // pin silently stops applying if its structural preconditions fail for one tensor.
    // Usage: run the same prompt at two token counts with ALPHA_MM_DISPATCH_LOG=1, `sort -u` each log
    // and diff. Use 2 to also print the per-predicate flags.
    static const int alpha_mm_log = []() {
        const char * e = getenv("ALPHA_MM_DISPATCH_LOG");
        return e ? atoi(e) : 0;
    }();
    if (alpha_mm_log > 0) {
        const char * path =
            (!split && use_mul_mat_vec_f) ? "mmvf" :
            (!split && use_mul_mat_f)     ? "mmf"  :
            (!split && use_mul_mat_vec_q) ? "mmvq" :
            (!split && use_mul_mat_q)     ? "mmq"  :
            (!split && (use_batched_cublas_f16 || use_batched_cublas_bf16 || use_batched_cublas_f32)
                && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1)
                                          ? "batched_cublas" :
            use_mul_mat_vec_f             ? "op_mmvf" :
            use_mul_mat_vec_q             ? "op_mmvq" :
            use_mul_mat_q                 ? "op_mmq"  : "op_cublas";
        // src0->name is reliable here (weights are named at load); dst->name is not -- the graph
        // callback that would name it is a no-op unless a callback is installed.
        if (alpha_mm_log >= 2) {
            fprintf(stderr, "[mm-dispatch] M=%d %-28s %-6s ne0=%d ne1=%d path=%s vecf=%d f=%d vecq=%d q=%d split=%d\n",
                    (int) src1->ne[1], src0->name, ggml_type_name(src0->type),
                    (int) src0->ne[0], (int) src0->ne[1], path,
                    (int) use_mul_mat_vec_f, (int) use_mul_mat_f,
                    (int) use_mul_mat_vec_q, (int) use_mul_mat_q, (int) split);
        } else {
            fprintf(stderr, "[mm-dispatch] M=%d %-28s %-6s ne0=%d ne1=%d path=%s\n",
                    (int) src1->ne[1], src0->name, ggml_type_name(src0->type),
                    (int) src0->ne[0], (int) src0->ne[1], path);
        }
    }

    // [ALPHA_SMALLM_TC] register-dequant tensor-core matmul for 2..8 columns; declines what it does not cover
    if (!split && ggml_cuda_mul_mat_smallm_tc(ctx, src0, src1, dst)) {
        return;
    }
    if (!split && use_mul_mat_vec_f) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_f) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_vec_q) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_q) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && (use_batched_cublas_f16 || use_batched_cublas_bf16 || use_batched_cublas_f32)
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_mul_mat_vec_f) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_f, nullptr);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
}

// Published per-forward-pass offset table for a tensor (defined in ggml-backend.cpp).
// Used by the dispatch to route multi-token pooled passes to MMQ (which reads it).
GGML_BACKEND_API_C void * ggml_backend_get_tensor_offsets(const void * tensor_ptr);
static inline void * extern_get_tensor_offsets(const void * t) {
    return ggml_backend_get_tensor_offsets(t);
}

// Async transfer state
static cudaStream_t g_cmp_s = nullptr;
static cudaEvent_t g_xfr_e = nullptr;

// Hybrid CPU/GPU: visible early for mul_mat_id and graph_check.
// The struct and instance are defined later; these flags bridge the gap.
static bool g_pcache_has_misses = false;  // set by pcache_copy, read by mul_mat_id/graph_check

// Per-tensor CPU miss IDs (keyed by gpu_dst pointer), populated by flush, read by mul_mat_id
struct pcache_cpu_miss_entry {
    const void * host_base;
    size_t       expert_size;
    std::vector<int64_t> cpu_miss_ids;
};
static std::unordered_map<void *, pcache_cpu_miss_entry> g_pcache_cpu_misses;

// Kernel offset table: avoids D2D copies on 100% hit tensors.
// Built during pcache_copy, resolved in flush, consumed by ggml_cuda_mul_mat_vec_q.
struct pcache_tensor_offsets {
    // Byte offset in the pool per expert_id (0 = default/unset), sized by the model's expert count
    // (models can have more than 256 experts). Every write goes through set_offset(),
    // which grow the vector instead of overflowing.
    std::vector<int64_t> pool_byte_offsets;
    // Deferred D2D info (for retroactive D2D if tensor has mixed hit/miss)
    struct deferred_copy { void * dst; const void * src; size_t size; };
    std::vector<deferred_copy> deferred;
    const void * cpu_src = nullptr;  // host weight pointer (unique per tensor, stable key)
    size_t type_size = 0;            // ggml_type_size for block offset conversion
    bool   has_miss  = false;
    // ALPHA_LAYER_FUSE: experts resolved from a placement made by this layer's first tensor.
    // Such a tensor has no misses and no deferred D2D, but its offset table must still be published,
    // otherwise the kernel reads a stale table while input_cpy is never written on the pool-direct path.
    int    n_adopted = 0;

    // Grow to cover expert ids [0, n). Only the first touch per tensor allocates.
    void ensure_experts(int64_t n) {
        if ((int64_t) pool_byte_offsets.size() < n) {
            pool_byte_offsets.resize(n, 0);
        }
    }
    void set_offset(int64_t eid, int64_t byte_off) {
        if (eid < 0) return;
        ensure_experts(eid + 1);
        pool_byte_offsets[eid] = byte_off;
    }
    int64_t n_experts() const { return (int64_t) pool_byte_offsets.size(); }
};
static std::unordered_map<void *, pcache_tensor_offsets> g_pcache_tensor_build; // key = gpu_dst

// Per-tensor GPU offset buffers (persistent, keyed by cpu_src, never reused by ggml-alloc)
static std::unordered_map<const void *, int64_t *> g_pcache_d_offsets_per_tensor; // cpu_src -> device buf
static std::unordered_map<const void *, size_t>    g_pcache_d_offsets_cap;        // entries allocated above


// Last-flushed tensor's offset info (consumed by extra callback)
struct pcache_offset_info {
    void *    pool_base;     // GPU pool base pointer
    int64_t * d_offsets;     // per-expert block offsets (GPU buffer)
};
static pcache_offset_info g_pcache_last_offset = { nullptr, nullptr };
static bool       g_pcache_last_all_hit  = false;      // set by flush, consumed by extra callback
static bool       g_pcache_offset_active = false;      // any tensor uses offset table this token
static bool       g_pcache_new_token     = false;      // set after graph_compute
static pcache_offset_info * g_pcache_last_info_ptr = nullptr;  // per-tensor info for last flushed tensor

// Per-tensor offset info (persistent, allocated on heap, keyed by cpu_src)
static std::unordered_map<const void *, pcache_offset_info *> g_pcache_info_per_tensor;

// Number of tokens in the current MUL_MAT_ID batch (set by the scheduler before the
// expert copy loop). n_tokens==1 guarantees the MMVQ decode path (table-capable);
// multi-token batches may route to MMQ/MMF which read input_cpy, so D2D must run.
static int g_pcache_batch_ntokens = 0;

// The multi-token/prefill MMQ kernel (mul_mat_q, need_check=false) reads input_cpy assuming unwritten
// expert slots and row padding are zero, but the multi-token copy path only writes the used experts.
// Zero the full input_cpy buffer once per tensor-eval before its experts are written. Set true at each
// mul_mat_id entry (consumed by the next tensor's first copy); gated to ntokens >= 2 (single-token decode
// is pool-direct and never reads input_cpy). Ordered before the miss H2Ds by the flush's reverse sync.
static bool g_pcache_input_cpy_needs_zero = true;

// Scratch-ring epoch discipline. A wrap barrier recorded on the compute stream is only sound if every
// kernel that reads an already-published ring slot has been enqueued, and it has not:
// ggml_backend_sched_compute_splits copies all of a split's inputs (up to GGML_SCHED_MAX_SPLIT_INPUTS
// expert tensors, each publishing an offset table into the ring) before it enqueues the split's graph.
// A wrap during input-copy k would record an event that covers none of tensors 0..k-1, and the overwrite
// H2D could land before they run. So the whole ring is recycled once per pass, at the first alloc after
// pass_begin, when every previous-pass kernel is enqueued; a mid-pass overflow still wraps (see
// scratch_alloc). ALPHA_SCRATCH_EPOCH=0 disables the pass-boundary recycle.
static const bool g_pcache_scratch_epoch = []() {
    const char * e = getenv("ALPHA_SCRATCH_EPOCH");
    return e ? atoi(e) != 0 : true;
}();

// ALPHA_MT_MAX: max multi-token batch routed pool-direct (ALPHA_PD_MMVQ: batched MMVQ reading the
// pool via offset tables), so the resident union is read once per unique expert instead of through
// input_cpy or the serial per-expert path. Bounded [8,128].
static const int g_pcache_mt_max = []() {
    // Default 16: covers batched decode of up to 16 concurrent users pool-direct.
    const char * e = getenv("ALPHA_MT_MAX");
    int v = e ? atoi(e) : 16;
    if (v < 8)   v = 8;
    if (v > 128) v = 128;
    return v;
}();

// Diagnostic (always on, printed once per window): how often the miss-count gate disables pool-direct
// on single-token decode, and the miss-count distribution.
static int64_t g_dbg_decode_passes = 0;
static int64_t g_dbg_hybrid_passes = 0;   // passes where use_hybrid tripped (pool_direct off)
static int64_t g_dbg_misses_sum    = 0;
static int64_t g_dbg_misses_max    = 0;
// ALPHA_HYBRID_MISS_THRESHOLD: miss count at/above which single-token decode drops pool-direct
// (fast offset-table read) for the input_cpy path. Default 800. Set very high (e.g. 100000) to keep
// pool-direct always on.
static const int g_pcache_hybrid_miss_threshold = []() {
    const char * e = getenv("ALPHA_HYBRID_MISS_THRESHOLD");
    return e ? atoi(e) : 800;
}();

// Direct expert-tensor pinning (cudaHostRegister the mmap'd tensor), default on. A pinned tensor lets
// a miss DMA single-hop from pinned RAM to VRAM: no staging memcpy and no pageable-sync fallback, which
// is several times faster than staging only. Budget-capped (min(MemAvailable - 8GB, 50% of MemTotal))
// so a model bigger than RAM cannot freeze the machine. ALPHA_TENSOR_PIN=0 restores staging only.
static const bool g_pcache_tensor_pin = []() {
    const char * e = getenv("ALPHA_TENSOR_PIN");
    return !(e && atoi(e) == 0);
}();

// ALPHA_PCACHE_DMA_STATS=N: instrument every Nth decode token with timing-enabled events so the
// report can separate instantaneous link rate, duty cycle and GPU stall. 0/unset = off.
// Sampling is per token, never per flush: stalls are not uniform across layers, so sampling
// individual flushes would smear the within-token fence structure.
// Overhead is ~1% at N=1 (4 event records per flush).
static const int g_pcache_dma_stats_n = []() {
    const char * e = getenv("ALPHA_PCACHE_DMA_STATS");
    if (!e) return 0;
    const int v = atoi(e);
    return (v == 0 && e[0] != '0') ? 1 : (v < 0 ? 0 : v);   // "" / "on" -> 1 ; "0" -> off
}();
static const bool g_pcache_dma_stats = g_pcache_dma_stats_n > 0;
static bool    g_pcache_dma_stats_now = false;   // sampling gate, set once per pass in pass_begin
static int64_t g_pcache_stats_prev_us  = 0;      // wall-clock denominator for duty cycle
static int64_t g_pcache_decode_wall_us = 0;
static int64_t g_pcache_decode_passes  = 0;
// Sampled decode passes, counted rather than derived. pass_ctr includes prefill passes, so
// decode_passes/N is not the sampled-token count. Everything measured by an event divides by
// this; everything counted per call (fences, bytes on unsampled tokens) divides by decode_passes.
static int64_t g_pcache_sampled_passes = 0;

// ALPHA_PCACHE_STATS_EVERY=N: emit the windowed [link]/[fence]/[roof] report every N passes.
static const int64_t g_pcache_stats_every = []() {
    const char * e = getenv("ALPHA_PCACHE_STATS_EVERY");
    const int v = e ? atoi(e) : 256;
    return (int64_t)(v < 1 ? 256 : v);
}();

// ALPHA_EVICT_POLICY: 0 = sampled LRU, 1 = sampled freq/age (GDSF-like, default). The gain grows with
// pool starvation: the worse the hit rate, the more a better victim choice is worth.
static const int g_pcache_evict_policy = []() {
    const char * e = getenv("ALPHA_EVICT_POLICY");
    return e ? atoi(e) : 1;
}();


// ALPHA_TRACE_IDS=<path>: append one line per tensor per decode token,
// "<pass> <tensor_ptr> <topk ids...>" (generation order), for offline analysis of the
// spec-verify M-distribution per draft length.
static FILE * g_pcache_ids_trace = []() -> FILE * {
    const char * p = getenv("ALPHA_TRACE_IDS");
    if (!p) return nullptr;
    FILE * f = fopen(p, "a");
    if (f) setvbuf(f, nullptr, _IOFBF, 1 << 20);
    return f;
}();

// Monotonic forward-pass counter (bumped by the pass-begin hook), the "token index"
// for the trace: decode passes are one token each.
static int64_t g_pcache_pass_ctr = 0;

// ALPHA_INLINE_SPLIT: percent of CPU-inline misses computed on the CPU worker; the rest are DMA'd to
// the GPU and computed by run_batch_mmvq concurrently with the worker. Default 100 = all misses on the
// CPU. While the CPU computes its misses the GPU is otherwise idle, so handing it a fraction of the
// misses (single-hop pinned H2D + batched MMVQ) uses that idle time. Uses the flush cpu_take split and
// the non-blocking inline dispatch. The split is exact: GPU half via MMVQ from input_cpy, CPU half via
// worker writeback, every expert accounted once. 50 = half/half.
// Mutable: the serve auto-mode may override it via ggml_cuda_expert_pcache_set_cpu_inline when the
// operator did not set the env. The env is the default and a hard operator override.
//
// Dynamic auto-split (ALPHA_INLINE_SPLIT_AUTO, opt-in; method described in arXiv 2608.16157):
// split a layer's m misses so the GPU (PCIe, B_P) and CPU (host RAM, B_H) pipes finish together.
// Balancing q/B_P = (m-q)/B_H gives the GPU share q* = m*B_P/(B_P+B_H), i.e. CPU share =
// B_H/(B_P+B_H). cpu_take = n_misses*eff_split/100 is already applied per layer, so only eff_split
// is derived. Bandwidths are measured at first use (the CUDA context must be live, so not in static
// init): B_P via a pinned H2D copy, B_H via a threaded host read matching the inline pool.
// ALPHA_INLINE_SPLIT_KAPPA corrects for GPU-compute contention that the transfer model ignores.
struct alpha_bw_t { double pcie_gbps; double host_gbps; };
static alpha_bw_t alpha_measure_bandwidths() {
    const size_t N = 256ull << 20;               // 256 MB per buffer
    alpha_bw_t bw{ 12.0, 50.0 };                 // conservative fallback (typical PCIe4 / DDR)
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    // B_H: aggregate host read bandwidth. The CPU-inline pipe streams expert weights from RAM, so a
    // pure sequential read (not a read+write memcpy) is the right measurement. Threaded with the same
    // count the inline pool uses.
    {
        int nthr = (int)std::thread::hardware_concurrency() / 2;
        if (nthr < 1) nthr = 1; if (nthr > 12) nthr = 12;
        char * a = (char*)malloc(N);
        if (a) {
            memset(a, 1, N);                     // fault in + populate pages
            const size_t chunk = N / nthr;
            std::vector<uint64_t> sums(nthr, 0);
            auto t0 = now();
            std::vector<std::thread> pool;
            for (int t = 0; t < nthr; t++) pool.emplace_back([&, t]{
                size_t off = (size_t)t * chunk, len = (t == nthr-1) ? N - off : chunk;
                const uint64_t * p = (const uint64_t*)(a + off);
                size_t n8 = len / 8; uint64_t s = 0;
                for (size_t i = 0; i < n8; i++) s += p[i];   // read-only stream
                sums[t] = s;
            });
            for (auto & th : pool) th.join();
            double s = std::chrono::duration<double>(now() - t0).count();
            volatile uint64_t sink = 0; for (auto v : sums) sink += v; (void)sink;  // defeat DCE
            if (s > 0) bw.host_gbps = (double)N / 1e9 / s;   // sequential read GB/s
        }
        free(a);
    }
    // B_P: pinned host -> device H2D bandwidth over PCIe.
    {
        void * hbuf = nullptr; void * dbuf = nullptr;
        if (cudaMallocHost(&hbuf, N) == cudaSuccess && cudaMalloc(&dbuf, N) == cudaSuccess) {
            cudaMemcpy(dbuf, hbuf, N, cudaMemcpyHostToDevice);   // warmup
            cudaDeviceSynchronize();
            auto t0 = now();
            for (int r = 0; r < 3; r++) cudaMemcpy(dbuf, hbuf, N, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            double s = std::chrono::duration<double>(now() - t0).count();
            if (s > 0) bw.pcie_gbps = 3.0 * (double)N / 1e9 / s;
        }
        if (hbuf) cudaFreeHost(hbuf);
        if (dbuf) cudaFree(dbuf);
    }
    return bw;
}
// -1 = unresolved: env override resolves it here immediately; otherwise resolve lazily (needs CUDA).
static int g_pcache_inline_split_pct = []() {
    const char * e = getenv("ALPHA_INLINE_SPLIT");
    if (!e) return -1;                           // auto or legacy, decided at first use
    int v = atoi(e);
    return v < 0 ? 0 : (v > 100 ? 100 : v);
}();
static std::once_flag g_split_auto_once;
// Resolve the single-token CPU-inline split %: operator/serve override (>=0) wins; else
// ALPHA_INLINE_SPLIT_AUTO derives it from measured throughput; else the default (100).
// The pure bandwidth-ratio q* (frac = B_H/(B_P+B_H)) mispredicts because it omits:
//   (1) the CPU pipe's cost is set by kernel throughput, not RAM bandwidth (an LUT-decoded format can
//       be compute-bound at a small fraction of what a memory-stream probe reports);
//   (2) running both pipes costs a fence per pass, so an interior split must earn that cost back;
//   (3) the same fence is negligible at hundreds of misses per pass and dominant at a dozen;
//   (4) miss density (handled separately by ALPHA_INLINE_SPLIT_MT_MIN_MISS).
// So the balance point is used only when it can pay for the fence:
//   R_cpu  = measured achieved CPU decode GB/s (autogate accumulators; falls back to the memory
//            probe only until the first CPU work has been timed)
//   R_pcie = measured pinned H2D GB/s
//   W      = bytes this pass would move (n_misses x expert_size)
//   t_best_extreme = W / max(R_cpu, R_pcie)      t_split = W / (R_cpu + R_pcie)
//   split only if (t_best_extreme - t_split) > fence_cost, else snap to the winning extreme.
// Achieved-CPU-throughput accumulators (also feed the engage-gate further down): bytes the CPU
// pipe actually decoded and the microseconds it took.
static std::atomic<int64_t> g_autogate_bytes{0}, g_autogate_us{0};

static double alpha_measured_cpu_gbps() {
    const int64_t us = g_autogate_us.load(std::memory_order_relaxed);
    const int64_t by = g_autogate_bytes.load(std::memory_order_relaxed);
    if (us < 50000 || by <= 0) return 0.0;          // <50ms of CPU work: not yet a stable estimate
    return (double)by / (double)us / 1e3;           // bytes/us -> GB/s
}
static int alpha_resolve_split_for(int n_misses, size_t expert_size) {
    if (g_pcache_inline_split_pct >= 0) return g_pcache_inline_split_pct;   // explicit override wins
    if (!getenv("ALPHA_INLINE_SPLIT_AUTO")) return 100;                      // legacy default
    static alpha_bw_t bw = alpha_measure_bandwidths();
    static const double fence_us = []() {            // measured coordination cost of one split pass
        const char * e = getenv("ALPHA_INLINE_FENCE_US");
        return e ? atof(e) : 40.0;                   // ~40us: hostfunc + spin + stream-order round trip
    }();
    double kappa = 1.0;
    if (const char * k = getenv("ALPHA_INLINE_SPLIT_KAPPA")) { double kv = atof(k); if (kv > 0) kappa = kv; }

    // R_cpu: the live accumulation of achieved CPU decode throughput.
    const double rc_meas  = alpha_measured_cpu_gbps();
    const char * rc_src   = rc_meas <= 0.0 ? "probe-fallback" : "measured";
    const double Rc = rc_meas > 0.0 ? rc_meas : bw.host_gbps;   // measured throughput when available
    const double Rp = bw.pcie_gbps > 0.0 ? bw.pcie_gbps : 12.0;
    const double W  = (double)n_misses * (double)expert_size;   // bytes this pass must move

    const double t_ext   = W / (std::max(Rc, Rp) * 1e9) * 1e6;  // us, best single pipe
    const double t_split = W / ((Rc + Rp)        * 1e9) * 1e6;  // us, both pipes
    const int    ext_pct = (Rc > Rp) ? 100 : 0;

    int pct;
    const bool worth_splitting = (t_ext - t_split) > fence_us;
    if (worth_splitting) {
        double frac = Rc / (Rc + Rp) * kappa;
        pct = (int)(frac * 100.0 + 0.5);
        pct = pct < 0 ? 0 : (pct > 100 ? 100 : pct);
    } else {
        pct = ext_pct;                                          // fence cannot be earned back
    }
    static int last_logged = -2;
    if (pct != last_logged) {                                   // log only on change
        last_logged = pct;
        fprintf(stderr, "[alpha] q*v2: R_cpu=%.1f GB/s (%s) R_pcie=%.1f GB/s | W=%.2f MB/pass "
                        "t_ext=%.0fus t_split=%.0fus fence=%.0fus -> SPLIT=%d%% (%s)\n",
                Rc, rc_src, Rp, W / 1e6,
                t_ext, t_split, fence_us, pct,
                worth_splitting ? "interior: split pays for the fence" : "snapped to extreme");
    }
    return pct;
}

// ALPHA_CPU_INLINE_MT (default on): extend the async CPU-inline correction to the multi-token
// (co-batch) forward, not just single-token decode. Without it every miss at batch streams over PCIe
// while the CPU/RAM pipe sits idle. Two parts:
//   (1) cpu_take may engage under pool_direct_mt (the zero-slot offset-table trick the correction
//       relies on is per-expert, so it works the same for multi-token), and
//   (2) the worker is expert-major (see pcache_inline_job), so one expert's weights are read once for
//       all tokens routed to it; otherwise the CPU pipe re-reads each expert per token.
// Only affects operators who enabled CPU-inline (ALPHA_CPU_INLINE, default off). The multi-token share
// is miss-density gated (see split_mt_eff in the flush): with a pool that fits the working set the gate
// holds it at 0, since applying it unconditionally slows that case down.
static const bool g_pcache_inline_mt = []() {
    const char * e = getenv("ALPHA_CPU_INLINE_MT");
    return e ? (atoi(e) != 0) : true;
}();
// ALPHA_INLINE_MT_DEBUG=1: report engagement on both sides, the flush's assign decision and the
// dispatch's actual grouped compute. avg_grp is the amortization factor: 1.0 means every pair re-reads
// its expert, 8.0 at 8 users means one weight read serves all eight tokens.
static const bool g_inline_mt_dbg = alpha_env_on("ALPHA_INLINE_MT_DEBUG");
// CPU-inline pool admissions left this pass (refilled in pass_begin, see ALPHA_INLINE_ADMIT).
static int g_pcache_warm_left = 0;

// Host-side routing ids for the current tensor cycle (from the scheduler, which already fetched and
// synced them, so no extra D2H is needed).
static std::vector<int32_t> g_pcache_batch_ids;
static int g_pcache_ids_ne0 = 0, g_pcache_ids_ne1 = 0, g_pcache_ids_s0 = 0, g_pcache_ids_s1 = 0;
GGML_BACKEND_API_C void ggml_cuda_expert_pcache_set_ids(const int32_t * ids, int ne0, int ne1, int s0, int s1) {
    const size_t n = (size_t)std::max(0, (ne1 - 1) * s1 + (ne0 - 1) * s0 + 1);
    g_pcache_batch_ids.assign(ids, ids + n);
    g_pcache_ids_ne0 = ne0; g_pcache_ids_ne1 = ne1;
    g_pcache_ids_s0  = s0;  g_pcache_ids_s1  = s1;

    // Expert-group-size histogram (for the grouped-GEMM crossover decision):
    // for multi-token batches, how many (token,slot) pairs share the same expert?
    // M=1 groups gain nothing from grouped GEMM; the win scales with the M>=2 mass.
    if (ne1 > 1) {
        static int64_t hist[9] = {0};      // group size 1..8+, index 0 = total tensors
        static int32_t cnt[512];
        memset(cnt, 0, sizeof(int32_t) * 512);
        for (int t = 0; t < ne1; t++) {
            for (int s = 0; s < ne0; s++) {
                int32_t e = ids[t * s1 + s * s0];
                if (e >= 0 && e < 512) cnt[e]++;
            }
        }
        for (int e = 0; e < 512; e++) {
            if (cnt[e] > 0) hist[std::min(cnt[e], 8)]++;
        }
        if ((++hist[0] & 4095) == 0) {
            fprintf(stderr, "[expert-cache] group-size hist (ntok=%d): M1=%lld M2=%lld M3=%lld M4=%lld M5+=%lld\n",
                    ne1, (long long)hist[1], (long long)hist[2], (long long)hist[3], (long long)hist[4],
                    (long long)(hist[5]+hist[6]+hist[7]+hist[8]));
        }
    }
}

// Persistent worker pool for the CPU-half vec_dot work (spawning threads per tensor costs
// tens of microseconds each, many times per token).
struct pcache_cpu_pool {
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv_job, cv_done;
    std::function<void(int)> job;      // called with part index
    int n_parts = 0;
    int n_workers = 0;
    std::atomic<int> done_workers{0};  // lock-free completion count
    uint64_t generation = 0;
    bool stop = false;

    void init(int n_threads) {
        if (!workers.empty()) return;
        n_workers = n_threads;
        for (int i = 0; i < n_threads; i++) {
            workers.emplace_back([this, i]() {
                uint64_t seen_gen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv_job.wait(lk, [&]{ return stop || generation != seen_gen; });
                        if (stop) return;
                        seen_gen = generation;
                    }
                    // Static strided partition: worker i owns parts i, i+n_workers, ...
                    // A shared lock per part grab serializes the workers and caps the pipe well
                    // below RAM bandwidth. Parts are equal cost (rows_per_part rows each), so a
                    // static split is well balanced and needs no shared grab state. run() cannot
                    // reuse state until every worker has finished its range (done_workers ==
                    // n_workers), so there is no straggler race.
                    const int np = n_parts, nw = n_workers;
                    for (int p = i; p < np; p += nw) {
                        job(p);
                    }
                    if (done_workers.fetch_add(1, std::memory_order_acq_rel) + 1 == nw) {
                        std::lock_guard<std::mutex> lk(mu);
                        cv_done.notify_one();
                    }
                }
            });
        }
    }
    // Run job(0..parts-1) across the pool; blocks until all parts complete.
    void run(int parts, std::function<void(int)> fn) {
        if (workers.empty()) {
            int n = (int)std::thread::hardware_concurrency() / 2;
            n = (n < 2 ? 2 : (n > 12 ? 12 : n));
            // ALPHA_CPU_INLINE_THREADS: override the default pool size (cores/2, capped at 12).
            if (const char * e = getenv("ALPHA_CPU_INLINE_THREADS")) { int v = atoi(e); if (v >= 1) n = v; }
            init(n);
        }
        if (parts <= 0) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(fn);
            n_parts = parts;
            done_workers.store(0, std::memory_order_relaxed);
            generation++;
        }
        cv_job.notify_all();
        {
            std::unique_lock<std::mutex> lk(mu);
            cv_done.wait(lk, [&]{ return done_workers.load(std::memory_order_acquire) == n_workers; });
        }
    }
};
static pcache_cpu_pool g_pcache_cpu_pool;

// Adaptive granularity: a fixed 512 leaves the pool starved at single-user decode (a few parts on a
// many-thread pool), which made the CPU dot several times slower. Targets enough parts to feed every
// thread; at batch (large n_groups) it naturally stays coarse. The env override wins.
static int alpha_inline_rows_per_part_for(int ne0, int n_groups) {
    static const int env = []() {
        const char * e = getenv("ALPHA_INLINE_ROWS_PER_PART");
        int x = e ? atoi(e) : 0;
        return (x >= 16 && x <= 8192) ? x : 0;   // 0 = adaptive
    }();
    if (env) return env;
    if (ne0 <= 0 || n_groups <= 0) return 128;
    static const int ncores = []() {
        int n = (int) std::thread::hardware_concurrency();
        return n > 0 ? n : 8;
    }();
    // Target ~4x cores of total parts: oversubscription balances uneven per-part completion, while
    // much finer parts fall off a cliff. Batch (large n_groups) still lands coarse.
    const int target_parts = 4 * ncores;
    const int want_parts_per_grp = std::max(1, (target_parts + n_groups - 1) / n_groups);
    int rpp = (ne0 + want_parts_per_grp - 1) / want_parts_per_grp;
    if (rpp < 64)  rpp = 64;    // floor: finer parts only add overhead
    if (rpp > 512) rpp = 512;   // batch stays coarse
    return rpp;
}

// Per-layer CPU inline correction (ALPHA_CPU_INLINE=1, default off).
// Exact decode that bypasses the PCIe wall: the GPU runs the backbone and cached experts (pool hits,
// no DMA); all missing experts are computed on the CPU from host RAM on the same per-layer input, and
// their dst rows are written back before downstream ops, which is mathematically identical to full
// decode. The launch thread never blocks: jobs go to a dedicated worker (which fans out to
// g_pcache_cpu_pool); the stream carries a cudaLaunchHostFunc fence that spins on the job's done flag,
// then the pre-enqueued H2D writebacks land. The only stall is max(0, cpu_time - gpu_queue_depth),
// reported per pass as inline_stall.
static int g_pcache_cpu_inline = []() {
    const char * e = getenv("ALPHA_CPU_INLINE");
    return e && atoi(e) != 0 ? 1 : 0;
}();

// Programmatic override for the serve auto-mode's model-vs-VRAM serve-mode selection.
// enable<0 leaves the current value; split_pct<0 leaves the current split. Serve only calls
// this when ALPHA_CPU_INLINE was not set in the environment, so a hard operator env override wins.
GGML_BACKEND_API_C void ggml_cuda_expert_pcache_set_cpu_inline(int enable, int split_pct) {
    if (enable >= 0) g_pcache_cpu_inline = enable ? 1 : 0;
    if (split_pct >= 0) g_pcache_inline_split_pct = split_pct > 100 ? 100 : split_pct;
}


// Max (token, expert-slot) pairs corrected on the CPU for one tensor in one pass.
// Single-token decode needs n_expert_used (<=16). Multi-token needs n_tokens*n_expert_used;
// the assign side is gated to the same bound so a batch can never overflow this (an
// overflow would silently drop a correction and corrupt output, not just slow it down).
#define PCACHE_INL_MAX 512
struct pcache_inline_job {
    std::atomic<int> state{0};          // 0 free, 1 queued, 2 computed (fence may pass)
    cudaEvent_t ev = nullptr;           // activation D2H completion (worker waits on it)
    const void * host_base = nullptr;   // expert weights base (host mmap)
    size_t expert_size = 0;
    size_t nb01_host = 0;               // weight row stride
    ggml_type wtype = GGML_TYPE_F32;
    int ne0 = 0, ne10 = 0;              // rows per expert, activation length
    int n_slots = 0;                    // number of (token, expert-slot) pairs to correct
    int64_t slot_idx[PCACHE_INL_MAX];   // dst expert-slot per pair
    int64_t tok_idx[PCACHE_INL_MAX];    // dst token per pair (all 0 on single-token decode)
    int64_t expert_id[PCACHE_INL_MAX];
    // Expert-major grouping for multi-token amortization. `order` holds pair indices
    // sorted so every pair sharing one expert is contiguous: group g spans
    // order[grp_off[g] .. grp_off[g+1]) and uses expert grp_expert[g]. The worker reads
    // each expert weight row once and dots it against every activation routed to that
    // expert, instead of re-reading the whole expert once per token (GEMV -> grouped GEMM,
    // like llama.cpp's CPU mul_mat_id). Degenerates to per-pair GEMV when every group has
    // one member (single-token decode).
    int     n_groups = 0;
    int     grp_off[PCACHE_INL_MAX + 1];
    int     order[PCACHE_INL_MAX];
    int64_t grp_expert[PCACHE_INL_MAX];
    int act_rows = 1;                   // number of distinct staged activation rows
    // pair -> staged activation row. Gate/up activations depend only on the token, so at 8 users the
    // 64 pairs map onto just 8 rows; this avoids duplicate D2H + quantization.
    int act_idx[PCACHE_INL_MAX];
    float * act_pin = nullptr;          // pinned activation rows
    float * out_pin = nullptr;          // pinned results [n_slots][ne0]
    size_t act_cap = 0, out_cap = 0;
};
static pcache_inline_job g_inline_ring[8];
static std::atomic<uint64_t> g_inline_produced{0};
static uint64_t g_inline_consumed = 0;  // worker-thread only
static std::mutex g_inline_mu;
static std::condition_variable g_inline_cv;
// telemetry (compute thread + pass_begin reader; relaxed is fine for stats)
static std::atomic<int64_t> g_inline_cpu_us{0}, g_inline_stall_us{0}, g_inline_experts{0};
// Runtime engage-gate: CPU-inline only helps if the CPU decode pipe is faster per byte than DMA'ing
// the expert over PCIe. For compute-bound quants (e.g. emulated NVFP4 decode) the CPU pipe is the
// slower lane and CPU-inline slows decode down, and a raw-RAM bandwidth probe cannot tell those quants
// apart from Q4_K. So the live achieved decode GB/s is compared with the measured PCIe H2D GB/s, and
// CPU-inline disengages (pool-direct) if it is not faster. One-way; an operator ALPHA_CPU_INLINE_SPLIT
// or =0 override always wins. Default on; ALPHA_NO_INLINE_AUTOGATE=1 turns it off.
// g_autogate_bytes / g_autogate_us are defined above alpha_resolve_split_for.
static int    g_autogate_state = 0;      // 0=measuring, 1=decided (one-way)
static double g_autogate_pcie_gbps = 0.0;
// Runtime engage-gate: after a short warmup of CPU-inline decode, compare the live achieved decode
// GB/s to the measured PCIe H2D GB/s. If the CPU pipe is the slower lane (decode <= PCIe), disengage
// CPU-inline (pool-direct). One-shot, one-way. Operator envs (ALPHA_CPU_INLINE / _SPLIT / _SPLIT_AUTO)
// keep full control. Cheap once decided: two atomic loads + return.
static void alpha_inline_autogate_check() {
    if (g_autogate_state != 0) return;
    static const bool disabled =
        getenv("ALPHA_NO_INLINE_AUTOGATE") || getenv("ALPHA_CPU_INLINE") ||
        getenv("ALPHA_INLINE_SPLIT") || getenv("ALPHA_INLINE_SPLIT_AUTO");
    if (disabled) { g_autogate_state = 1; return; }
    if (!g_pcache_cpu_inline) return;                       // only meaningful while engaged
    const int64_t us = g_autogate_us.load(std::memory_order_relaxed);
    if (us < 400000) return;                                // ~0.4s of CPU work -> stable estimate
    const int64_t by = g_autogate_bytes.load(std::memory_order_relaxed);
    const double decode_gbps = (double)by / (double)us / 1e3;   // bytes/us -> GB/s
    if (g_autogate_pcie_gbps <= 0.0) {                      // measure PCIe H2D once
        void * hbuf = nullptr; void * dbuf = nullptr; const size_t N = 128ull << 20;
        if (cudaMallocHost(&hbuf, N) == cudaSuccess && cudaMalloc(&dbuf, N) == cudaSuccess) {
            cudaMemcpy(dbuf, hbuf, N, cudaMemcpyHostToDevice); cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < 3; r++) cudaMemcpy(dbuf, hbuf, N, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            if (s > 0) g_autogate_pcie_gbps = 3.0 * (double)N / 1e9 / s;
        }
        if (hbuf) cudaFreeHost(hbuf);
        if (dbuf) cudaFree(dbuf);
        if (g_autogate_pcie_gbps <= 0.0) g_autogate_pcie_gbps = 12.0;   // fallback
    }
    g_autogate_state = 1;                                   // decide once
    if (decode_gbps <= g_autogate_pcie_gbps) {
        g_pcache_cpu_inline = 0;                            // disengage -> pool-direct
        fprintf(stderr, "[alpha] CPU-inline AUTO-GATE: decode %.1f GB/s <= PCIe %.1f GB/s → DISENGAGED "
                "(pool-direct; CPU is the slower lane for this quant). ALPHA_NO_INLINE_AUTOGATE=1 to keep.\n",
                decode_gbps, g_autogate_pcie_gbps);
    } else {
        fprintf(stderr, "[alpha] CPU-inline AUTO-GATE: decode %.1f GB/s > PCIe %.1f GB/s → KEEP CPU-inline.\n",
                decode_gbps, g_autogate_pcie_gbps);
    }
}
// Compute-bound vs bandwidth-bound decision data for the CPU expert pipe. `experts` above counts
// (token,expert) pairs; groups counts distinct experts. Their ratio is the amortization actually
// available, and bytes/cpu_us gives the achieved RAM bandwidth: far below what the DIMMs can do means
// the kernel is compute-bound.
static std::atomic<int64_t> g_inline_groups{0}, g_inline_wbytes{0};
// Amdahl split: time in the two serial phases of the worker (activation quantization and
// per-group pre-fault/classify) vs total worker time, to show which serial phase is the ceiling.
static std::atomic<int64_t> g_inline_quant_us{0}, g_inline_class_us{0};
// Pipelining diagnostic: count fence calls/token + total stall so per-fence stall can be
// split into (a) hostfunc overhead (reducible by combining tensors) vs (b) compute imbalance
// (worker slower than MMVQ; only reducible by reordering GPU work ahead). ALPHA_INLINE_FENCE_DBG.
static std::atomic<int64_t> g_inline_fence_n{0};
// RAM-tier read-class instrumentation: every CPU-inline expert read falls in one of three latency
// classes; count and time each so the residency policy has real data.
static std::atomic<int64_t> g_inl_pin_n{0},  g_inl_pc_n{0},  g_inl_ssd_n{0};
static std::atomic<int64_t> g_inl_pin_us{0}, g_inl_pc_us{0}, g_inl_ssd_us{0};
// Classify + pre-fault one expert's weight pages: returns 0 pinned, 1 page-cache,
// 2 SSD (any non-resident page); times the touch so SSD read cost is isolated from
// the vec_dot compute. Also warms pages single-threaded so the parallel dot below
// never contends on page faults. host_registered -> pinned (always resident).
// Set once at pool init to the pcache's host_registered set (defined later in the file);
// avoids a forward-declared accessor into the pcache struct.
static const std::unordered_set<const void *> * g_inl_host_registered = nullptr;
static int pcache_inline_classify(const void * host_base, const char * w, size_t bytes) {
#ifdef __linux__
    if (g_inl_host_registered && g_inl_host_registered->count(host_base)) {
        g_inl_pin_n.fetch_add(1, std::memory_order_relaxed); return 0;
    }
    const uintptr_t a  = (uintptr_t)w;
    const uintptr_t al = a & ~(uintptr_t)4095;
    const size_t    len = bytes + (a - al);
    const size_t    npg = (len + 4095) / 4096;
    static thread_local std::vector<unsigned char> mc;
    mc.resize(npg);
    bool all_resident = true;
    if (mincore((void *)al, len, mc.data()) == 0) {
        for (size_t p = 0; p < npg; p++) if (!(mc[p] & 1)) { all_resident = false; break; }
    }
    if (all_resident) {
        g_inl_pc_n.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    // SSD: time the fault-in (touch one byte per page)
    const int64_t t0 = ggml_time_us();
    volatile uint8_t sink = 0;
    for (size_t p = 0; p < npg; p++) sink ^= ((const volatile uint8_t *)al)[p * 4096];
    (void)sink;
    g_inl_ssd_us.fetch_add(ggml_time_us() - t0, std::memory_order_relaxed);
    g_inl_ssd_n.fetch_add(1, std::memory_order_relaxed);
    return 2;
#else
    (void)host_base; (void)w; (void)bytes;
    return 1;
#endif
}

// Predictive RAM prefetch for CPU-inline. SSD faults are a small share of expert reads but a large
// share of CPU time, since each fault is synchronous in the worker. Routing is mostly stable from token
// to token, so the previous token's inline miss set predicts this token's: at each token start the whole
// previous-token set gets POSIX_MADV_WILLNEED, and kernel readahead overlaps the CPU compute so pages
// are resident by the time the worker reaches them. Advisory only (never changes results); skips
// already-resident pages cheaply. Runs on the compute thread (same thread as the enqueue recorder and
// pass_begin), so no locking is needed.
static const bool g_inl_prefetch = []() {
    const char * e = getenv("ALPHA_INLINE_PREFETCH");
    return !e || atoi(e) != 0;   // default on within CPU-inline mode
}();
struct inl_pf_ent { const void * host_base; size_t offset; size_t size; };
static std::vector<inl_pf_ent> g_inl_pf_cur;    // misses recorded this token
static std::vector<inl_pf_ent> g_inl_pf_prev;   // previous token's set (swept at token start)
static int64_t g_inl_pf_issued = 0;             // telemetry: WILLNEED'd experts / summary window
static inline void inl_prefetch_record(const void * host_base, size_t offset, size_t size) {
    if (!g_inl_prefetch) return;
    if (g_inl_pf_cur.size() < 8192) g_inl_pf_cur.push_back({ host_base, offset, size });
}
static void inl_prefetch_token_boundary(void) {
    if (!g_inl_prefetch) return;
#ifdef __linux__
    for (const auto & e : g_inl_pf_prev) {
        uintptr_t a  = (uintptr_t)e.host_base + (uintptr_t)e.offset;
        uintptr_t al = a & ~(uintptr_t)4095;
        posix_madvise((void *)al, e.size + (a - al), POSIX_MADV_WILLNEED);
    }
    g_inl_pf_issued += (int64_t)g_inl_pf_prev.size();
#endif
    g_inl_pf_prev.swap(g_inl_pf_cur);
    g_inl_pf_cur.clear();
}

static void pcache_inline_compute(pcache_inline_job * j) {
    {   // ground truth from the worker itself: how many pairs, and what amortization?
        static const bool w_dbg = alpha_env_on("ALPHA_TOP_DEBUG");
        static int64_t wc = 0;
        if (w_dbg && (wc++ & 511) == 0) {
            fprintf(stderr, "[worker] pairs=%d groups=%d avg_grp=%.2f act_rows=%d\n",
                    j->n_slots, j->n_groups,
                    j->n_groups ? (double)j->n_slots / j->n_groups : 0.0, j->act_rows);
        }
    }
    CUDA_CHECK(cudaEventSynchronize(j->ev));   // activation D2H done (worker blocks, not launcher)
    const int64_t t0 = ggml_time_us();
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(j->wtype);
    ggml_vec_dot_t vec_dot = tr->vec_dot;
    const enum ggml_type vdt = tr->vec_dot_type;
    const ggml_from_float_t q_from_float = ggml_get_type_traits_cpu(vdt)->from_float;
    const size_t qrow = (size_t)(j->ne10 / ggml_blck_size(vdt)) * ggml_type_size(vdt) + 64;
    static std::vector<char> qbuf;             // worker thread only
    qbuf.resize((size_t)(j->act_rows) * qrow);
    // Amdahl split: the two serial phases are timed separately from the parallel dot.
    //   QUANT    : f32 -> Q8_K activation conversion. Required: ggml's K-quant vec_dot consumes
    //              Q8_K, never f32. Scales with act_rows, one thread, pool idle.
    //   CLASSIFY : per-group pre-fault/tiering of expert weights. Memory-touching work, also one
    //              thread, and it scales with n_groups.
    const int64_t t_q0 = ggml_time_us();
    for (int r = 0; r < j->act_rows; r++) {
        q_from_float(j->act_pin + (size_t)r * j->ne10, qbuf.data() + (size_t)r * qrow, j->ne10);
    }
    const int64_t t_q1 = ggml_time_us();
    // RAM-tier: classify + pre-fault each expert's weights (isolates SSD read cost
    // and warms pages so the parallel dot below never page-faults under contention).
    // Per group, not per pair: a batch routing 8 tokens to one expert reads it once.
    for (int g = 0; g < j->n_groups; g++) {
        const char * w_base = (const char *)j->host_base + j->grp_expert[g] * j->expert_size;
        pcache_inline_classify(j->host_base, w_base, j->expert_size);
    }
    const int64_t t_c1 = ggml_time_us();
    g_inline_quant_us.fetch_add(t_q1 - t_q0, std::memory_order_relaxed);
    g_inline_class_us.fetch_add(t_c1 - t_q1, std::memory_order_relaxed);
    const int rows_per_part = alpha_inline_rows_per_part_for(j->ne0, j->n_groups);
    const int parts_per_grp = (j->ne0 + rows_per_part - 1) / rows_per_part;
    {   static const bool rdbg = alpha_env_on("ALPHA_RPP_DEBUG"); static int64_t rc = 0;
        if (rdbg && (rc++ & 1023) == 0)
            fprintf(stderr, "[rpp] ne0=%d n_groups=%d ncores=%d -> rpp=%d parts_per_grp=%d total_parts=%d\n",
                    j->ne0, j->n_groups, (int)std::thread::hardware_concurrency(), rows_per_part,
                    parts_per_grp, j->n_groups * parts_per_grp); }
    const int total_parts = j->n_groups * parts_per_grp;
    pcache_inline_job * jj = j;
    const char * qb = qbuf.data();
    g_pcache_cpu_pool.run(total_parts, [jj, qb, qrow, rows_per_part, parts_per_grp, vec_dot](int part) {
        const int g  = part / parts_per_grp;
        const int r0 = (part % parts_per_grp) * rows_per_part;
        const int r1 = r0 + rows_per_part < jj->ne0 ? r0 + rows_per_part : jj->ne0;
        const char * w_base = (const char *)jj->host_base + jj->grp_expert[g] * jj->expert_size;
        const int k0 = jj->grp_off[g], k1 = jj->grp_off[g + 1];
        // The weight row is the bandwidth cost: read it once (it stays hot in L1/L2) and dot
        // it against every activation routed to this expert. With one member per group
        // this is the plain per-slot loop.
        for (int n = r0; n < r1; n++) {
            const char * wrow = w_base + (size_t)n * jj->nb01_host;
            for (int k = k0; k < k1; k++) {
                const int pi = jj->order[k];
                const char * qa = qb + (size_t)jj->act_idx[pi] * qrow;
                float dot = 0.0f;
                vec_dot(jj->ne10, &dot, 1, wrow, 1, qa, 1, 1);
                jj->out_pin[(size_t)pi * jj->ne0 + n] = dot;
            }
        }
    });
    g_inline_cpu_us.fetch_add(ggml_time_us() - t0, std::memory_order_relaxed);
    g_inline_experts.fetch_add(j->n_slots, std::memory_order_relaxed);
    // bytes the worker must read if each distinct expert's rows are read once (the amortized
    // ideal). Achieved GB/s = wbytes / cpu_us shows how close the worker is to RAM speed.
    g_inline_groups.fetch_add(j->n_groups, std::memory_order_relaxed);
    g_inline_wbytes.fetch_add((int64_t)j->n_groups * (int64_t)j->ne0 * (int64_t)j->nb01_host,
                              std::memory_order_relaxed);
    // Auto-gate accumulators (separate from the debug roofline counters, which get exchanged):
    // the achieved CPU decode GB/s over these = the CPU pipe's real per-byte throughput.
    g_autogate_us.fetch_add(ggml_time_us() - t0, std::memory_order_relaxed);
    g_autogate_bytes.fetch_add((int64_t)j->n_groups * (int64_t)j->ne0 * (int64_t)j->nb01_host,
                               std::memory_order_relaxed);
    j->state.store(2, std::memory_order_release);
}

static void pcache_inline_worker_main(void) {
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_inline_mu);
            g_inline_cv.wait(lk, [] {
                return g_inline_produced.load(std::memory_order_acquire) > g_inline_consumed;
            });
        }
        while (g_inline_produced.load(std::memory_order_acquire) > g_inline_consumed) {
            pcache_inline_job * j = &g_inline_ring[g_inline_consumed % 8];
            pcache_inline_compute(j);
            g_inline_consumed++;
        }
    }
}
static void pcache_inline_worker_start(void) {
    static bool started = false;
    if (!started) {
        started = true;
        std::thread(pcache_inline_worker_main).detach();
    }
}

// stream-ordered fence: spins until the worker publishes the job's results (host
// callback: no CUDA calls allowed here, atomic spin only)
static void CUDART_CB pcache_inline_fence(void * ud) {
    pcache_inline_job * j = (pcache_inline_job *)ud;
    const int64_t t0 = ggml_time_us();
    bool waited = j->state.load(std::memory_order_acquire) != 2;   // was the worker still busy on arrival?
    while (j->state.load(std::memory_order_acquire) != 2) {
#if defined(__x86_64__) && defined(__GNUC__)
        __builtin_ia32_pause();
#endif
    }
    g_inline_stall_us.fetch_add(ggml_time_us() - t0, std::memory_order_relaxed);
    // Diagnostic: count fences, and how many arrived after the worker had already finished
    // (state==2 on entry: the GPU was not waiting on compute, so any stall there is pure
    // hostfunc/spin overhead). A high not-waited fraction means combining fences would help.
    g_inline_fence_n.fetch_add(waited ? 1 : ((int64_t)1 << 32), std::memory_order_relaxed);
}
// stream-ordered release: after the H2D writebacks executed, the ring slot is reusable
static void CUDART_CB pcache_inline_release(void * ud) {
    ((pcache_inline_job *)ud)->state.store(0, std::memory_order_release);
}

static bool pcache_hybrid_fusion_blocked(void) {
    // Only when CPU-inline assigned CPU-half experts this pass (has_misses set by flush). Costs
    // unfused gate/up launches for the pass; graphs are off anyway.
    return g_pcache_cpu_inline && g_pcache_has_misses;
}
static int g_pcache_batch_src_quantized = 0;
// Current expert tensor is an ik-quant (IQ1_KT et al.). ik has no MMVQ/MMQ kernel, so it is
// decoded by ggml_cuda_mul_mat_id_ik_pool. The flush forces the non-pool-direct path for ik on every
// pass (including single-token decode), so input_cpy is always populated for that decoder.
static int g_pcache_batch_src_is_ik = 0;
// ggml_type_size (bytes per quant block) of the current expert tensor, threaded from the
// scheduler. Authoritative for offset-table strides: inferring the size from the byte count is
// ambiguous for IQ quants (iq3_xxs=98, iq4_xs=136) and gives a wrong stride.
static int g_pcache_cur_block_size = 0;
// Only valid for the tensor being copied. bulk_insert_hot runs inside one tensor's copy call but
// inserts experts of all tensors, so it uses freq_entry.block_size, captured at observation time
// inside each expert's own tensor copy; using this global there would mis-align slots in any
// mixed-quant MoE.
// Set true once any multi-token (prefill) batch has been seen. Warmup is finalized at the end of
// the first prefill instead of after WARMUP_CALLS: a short prompt + short generation otherwise never
// reaches WARMUP_CALLS, so the whole request runs in the observation regime (pool observation-only,
// pool_direct forced off, every decode token reads input_cpy). That warmup-phase input_cpy decode
// produces NaN for gemma-4 MoE (dual dense+MoE FFN). Long prompts already reach WARMUP_CALLS during
// prefill, so only short-prefill requests are affected.
static bool g_pcache_seen_prefill = false;
GGML_BACKEND_API_C void ggml_cuda_expert_pcache_set_ntokens(int n, int src_quantized, int block_size, int wtype) {
    g_pcache_batch_ntokens = n;
    g_pcache_batch_src_quantized = src_quantized;
    g_pcache_cur_block_size = block_size;
    g_pcache_batch_src_is_ik = ggml_cuda_is_ik_type(wtype);
    if (n >= 2) g_pcache_seen_prefill = true;
    if (n == 1) alpha_inline_autogate_check();   // decode: auto-disengage CPU-inline if it is the slow lane
}

// Resolve the quant block size (bytes) for a tensor's experts. Prefer the authoritative
// per-tensor value threaded from the scheduler; fall back to byte-size inference for
// callers/paths that do not set it (prefetch, legacy). The fallback lists K-quants first, then IQ
// quants (iq4_xs=136, q3_K=110, iq3_xxs=98, q2_K=84, iq2_xs=74, iq2_xxs=66), then F16.
// Authoritative quant block size per expert tensor, keyed by its CPU source pointer.
// pcache_block_size() resolves against g_pcache_cur_block_size, which is only correct for the
// tensor currently being processed. Callers resolving a different tensor's experts (the union
// prefetch walking dropped records, the successor prefetch walking prefetch_info) would otherwise
// fall through to the byte-size guess, which is ambiguous: iq3_xxs blocks are 98 = 2*7^2, so a
// multiple of 98 can also divide 34 or 18, which the list tests first. A wrong stride makes the
// insert land at a wrong offset, so the pool itself would hold garbage.
static std::unordered_map<const void *, uint32_t> g_pcache_bs_by_src;
static const bool g_pcache_bs_telem = alpha_env_on("ALPHA_PCACHE_BS_TELEM");

static inline size_t pcache_block_size(size_t expert_size) {
    if (g_pcache_cur_block_size > 0 && expert_size % (size_t)g_pcache_cur_block_size == 0)
        return (size_t)g_pcache_cur_block_size;
    // IQ1_S=50, IQ1_M=56 included for 1-bit requants.
    for (size_t ts : {(size_t)144,(size_t)176,(size_t)210,(size_t)34,(size_t)18,
                      (size_t)136,(size_t)110,(size_t)98,(size_t)84,(size_t)74,(size_t)66,
                      (size_t)56,(size_t)50,(size_t)2}) {
        if (expert_size % ts == 0) return ts;
    }
    return 0;
}

// Resolve a block size for a specific tensor rather than "whatever is current". Falls back to the
// guess only when that tensor was never seen by the copy path.
static inline size_t pcache_block_size_for(const void * src, size_t expert_size) {
    if (src) {
        auto it = g_pcache_bs_by_src.find(src);
        if (it != g_pcache_bs_by_src.end() && it->second > 0 &&
            expert_size % (size_t) it->second == 0) {
            if (g_pcache_bs_telem) {
                const size_t guess = pcache_block_size(expert_size);
                if (guess != (size_t) it->second) {
                    static int shown = 0;
                    if (shown < 16) { shown++;
                        fprintf(stderr, "[pcache-bs] MISMATCH src=%p expert_size=%zu authoritative=%u guess=%zu",
                                src, expert_size, it->second, guess);
                        fputc(10, stderr);
                    }
                }
            }
            return (size_t) it->second;
        }
    }
    return pcache_block_size(expert_size);
}

// Called by the scheduler at the start of every forward pass (before any expert
// copies). Snapshots the LRU counter as the pass floor: entries touched at/after it
// are protected from eviction (their pool slots may be referenced by this pass's
// offset tables). Also the safe point for cross-pass state resets.
GGML_BACKEND_API_C void ggml_cuda_expert_pcache_pass_begin(void);

// IQ1_KT (ik-quant) expert-pool decode. The pool can hold routed experts in a native ik quant
// (IQ1_KT et al.) that has no MMVQ/MMQ kernel. This MUL_MAT_ID dequantizes each gathered expert's
// weight rows to F32 on the GPU (row-meta-aware device dequant) and runs one cuBLAS GEMV per
// (token, expert-slot), or uses the fused trellis MMVQ for IQ1_KT. Results match the CPU vec_dot path
// to rounding (F32 cuBLAS vs Q8_K integer dot). On CPU-inline/hybrid passes the miss slots computed
// here are overwritten by the CPU worker, as with the native MMVQ zero-slot semantics.
static bool ggml_cuda_mul_mat_id_ik_pool(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor * src0, const ggml_tensor * src1,
                                         const ggml_tensor * ids, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];   // input dim (elements per weight row)
    const int64_t ne01 = src0->ne[1];   // output rows (== dst->ne[0])
    const int64_t ne02 = src0->ne[2];   // total experts
    const int64_t ne0  = dst->ne[0];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];   // activation broadcast dim (1 or n_expert_used)
    const int64_t ne12 = src1->ne[2];   // n_tokens
    GGML_ASSERT(ne0  == ne01);
    GGML_ASSERT(ne10 == ne00);

    const ggml_type wtype = src0->type;
    const size_t    ts    = ggml_type_size(wtype);
    const size_t    nb01  = src0->nb[1];        // quant row stride (bytes)
    cudaStream_t    stream = ctx.stream();
    const int       id    = ggml_cuda_get_device();

    // ik is forced non-pool-direct (see the flush), so the scheduler always materializes
    // this tensor's used experts into src0 (input_cpy) at the natural full-tensor stride
    // (deferred D2D from the pool for hits + H2D for misses). Read input_cpy directly:
    // no offset table, no D2H, no host sync (CUDA-graph safe). The per-expert dequant and
    // GEMV run stream-ordered before the next op, so no synchronization is needed.
    const char * wbase = (const char *) src0->data;
    const size_t src0_alloc = (src0->buffer)
        ? ggml_backend_buffer_get_alloc_size(src0->buffer, (ggml_tensor *)src0)
        : ggml_nbytes(src0);

    // Routing ids: prefer the scheduler's host-side copy (set right before this tensor's
    // copy loop), so no D2H. Fall back to a D2H+sync only if the shape does not match (that
    // fallback is not graph-capture safe, but it does not occur on the decode path).
    const bool ids_host_ok = !g_pcache_batch_ids.empty()
                          && g_pcache_ids_ne0 == (int)n_expert_used
                          && g_pcache_ids_ne1 == (int)ne12;
    std::vector<int32_t> ids_fallback;
    if (!ids_host_ok) {
        ids_fallback.resize((size_t)ne12 * n_expert_used);
        std::vector<char> raw(ggml_nbytes(ids));
        CUDA_CHECK(cudaMemcpyAsync(raw.data(), ids->data, ggml_nbytes(ids),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int64_t t = 0; t < ne12; t++)
            for (int64_t s = 0; s < n_expert_used; s++)
                ids_fallback[t*n_expert_used + s] =
                    *(const int32_t *)(raw.data() + t*ids->nb[1] + s*ids->nb[0]);
    }
    auto get_eid = [&](int64_t t, int64_t s) -> int32_t {
        if (ids_host_ok)
            return g_pcache_batch_ids[(size_t)t * g_pcache_ids_s1 + (size_t)s * g_pcache_ids_s0];
        return ids_fallback[t*n_expert_used + s];
    };

    // Fused trellis MMVQ (decode-in-register dot) for IQ1_KT, default on; ALPHA_IK_FUSED=0
    // falls back to the F32-dequant + cuBLAS GEMV path. Only IQ1_KT has the fused kernel;
    // other ik types always use the dequant path.
    const bool ik_fused = (wtype == GGML_TYPE_IQ1_KT)
                       && !(getenv("ALPHA_IK_FUSED") && atoi(getenv("ALPHA_IK_FUSED")) == 0);

    // The F32 scratch + cuBLAS handle are only needed for the dequant path.
    ggml_cuda_pool_alloc<float> w_f32(ctx.pool(id), ik_fused ? 0 : (size_t)ne01 * ne00);
    cublasHandle_t handle = nullptr;
    const float alpha = 1.0f, beta = 0.0f;
    if (!ik_fused) {
        handle = ctx.cublas_handle(id);
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

    static const bool ik_dbg = alpha_env_on("ALPHA_IK_DEBUG");
    if (ik_dbg) {
        static int64_t dc = 0;
        if ((dc++ & 255) == 0) {
            fprintf(stderr, "[ik-pool] ne00=%lld ne01=%lld ne02=%lld ne12=%lld nexp=%lld nb01=%zu nb02=%zu ts=%zu alloc=%zu ids_host=%d\n",
                    (long long)ne00, (long long)ne01, (long long)ne02, (long long)ne12,
                    (long long)n_expert_used, nb01, (size_t)src0->nb[2], ts, src0_alloc, (int)ids_host_ok);
        }
    }

    for (int64_t t = 0; t < ne12; t++) {
        for (int64_t s = 0; s < n_expert_used; s++) {
            const int32_t e = get_eid(t, s);
            if (e < 0 || e >= ne02) continue;
            const char * wsrc = wbase + (size_t)e * src0->nb[2];
            {
                // Guard against reading past the input_cpy allocation.
                const size_t need    = (size_t)ne01 * nb01;
                const size_t byteoff = (size_t)e * src0->nb[2];
                if (byteoff + need > src0_alloc) {
                    if (ik_dbg) {
                        static int64_t bad = 0;
                        if (bad++ < 20)
                            fprintf(stderr, "[ik-pool] RAW OOB t=%lld s=%lld e=%d byteoff=%zu need=%zu alloc=%zu\n",
                                    (long long)t, (long long)s, (int)e, byteoff, need, src0_alloc);
                    }
                    continue;
                }
            }
            const float * xrow = (const float *)((const char *)src1->data
                + t*src1->nb[2] + (s % ne11)*src1->nb[1]);
            float * drow = (float *)((char *)dst->data + t*dst->nb[2] + s*dst->nb[1]);
            if (ik_fused) {
                // Fused trellis MMVQ: decode-in-register dot, no F32 expansion.
                // dst[j] = sum_i W_expert[j,i] * x[i], one thread per output row.
                ggml_cuda_ik_mmvq_iq1kt(wsrc, (int64_t)nb01, xrow, drow, ne00, ne01, stream);
                if (ik_dbg) CUDA_CHECK(cudaGetLastError());
            } else {
                // Dequantize ne01 rows of ne00 elements -> F32 (stream-ordered before the GEMV).
                ggml_cuda_dequantize_ik_rows_f32(wsrc, w_f32.get(), ne01, ne00,
                                                 (int64_t)nb01, (int)wtype, stream);
                if (ik_dbg) CUDA_CHECK(cudaGetLastError());
                // dst[j] = sum_i w_f32[i + j*ne00] * x[i]  (w_f32 row j at offset j*ne00) -> op_T
                CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, (int)ne00, (int)ne01,
                                         &alpha, w_f32.get(), (int)ne00,
                                         xrow, 1, &beta, drow, 1));
            }
        }
    }
    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    {   // ALPHA_DRE_PF_PIPE: a staged expert buffer still being filled; only the MMQ waits per group, so any other
        // consumer waits for every group first
        const int * pe = nullptr; cudaEvent_t const * pv = nullptr;
        const ggml_tensor * w = dst->src[0];
        const int ng = w ? ggml_cuda_dre_pipe_groups(w->data, &pe, &pv) : 0;
        if (ng > 0 && !(ggml_is_quantized(w->type) && dst->src[2] && dst->src[2]->ne[1] > MMVQ_MAX_BATCH_SIZE &&
                        ggml_cuda_should_use_mmq(w->type, ggml_cuda_info().devices[ctx.device].cc, dst->src[1]->ne[2], w->ne[2]))) {
            for (int g = 0; g < ng; g++) CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), pv[g], 0));
        }
    }
    // Capture compute stream on first call
    if (!g_cmp_s) {
        g_cmp_s = ctx.stream();
    }
    // The DMA barrier wait is done in _acdb() (called from the eval callback, outside CUDA graph
    // capture). Calling cudaStreamWaitEvent here fails with "dependency created on uncaptured work"
    // when CUDA graphs are active.

    // Arm the input_cpy zero-fill for the next tensor-eval's copy phase.
    // The scheduler runs copy(A)->compute(A)->copy(B)->... so this handler entry for A
    // precedes B's copies; the flag it sets is consumed by B's first copy (and A itself
    // was armed by the previous tensor's handler / the static init).
    g_pcache_input_cpy_needs_zero = true;

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    // ALPHA_TOP_DEBUG: which mul_mat_id path does fused-infer take? Everything below (including
    // the CPU-inline correction) is gated on !src0_is_host, so this shows whether the expert tensor
    // is seen as host-resident or as the GPU pool. Function-local static so it cannot be affected
    // by namespace-scope init ordering.
    {
        static const bool top_dbg = alpha_env_on("ALPHA_TOP_DEBUG");
        static int64_t topc = 0;
        if (top_dbg && (topc++ & 511) == 0) {
            const bool s0h = src0->buffer && ggml_backend_buffer_is_host(src0->buffer);
            const int64_t ne12l = src1->ne[2];
            const bool gate_ok = g_pcache_cpu_inline && g_pcache_ids_ne0 == ids->ne[0] &&
                (ne12l == 1 ? g_pcache_ids_ne1 == 1
                            : (g_pcache_inline_mt && g_pcache_ids_ne1 == (int)ne12l &&
                               ne12l * ids->ne[0] <= PCACHE_INL_MAX));
            fprintf(stderr, "[top] src0_host=%d ne12=%lld ids_ne=[%lld,%lld] host_ids=[%d,%d] "
                    "has_misses=%d gate_ok=%d cpu_inline=%d inline_mt=%d\n",
                    (int)s0h, (long long)ne12l,
                    (long long)ids->ne[0], (long long)ids->ne[1],
                    g_pcache_ids_ne0, g_pcache_ids_ne1,
                    (int)g_pcache_has_misses, (int)gate_ok,
                    (int)g_pcache_cpu_inline, (int)g_pcache_inline_mt);
        }
    }

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft) && "mul_mat_id does not support split buffers");

    // Detect host-resident expert tensors (CPU experts via tensor_buft_overrides)
    bool src0_is_host = src0->buffer && ggml_backend_buffer_is_host(src0->buffer);

    // ALPHA_DRE: the scheduler handed over the host tensor itself; MMVQ resolves every expert through
    // the device table (pool slot or pinned host copy), see dre.cu
    if (src0_is_host && ggml_cuda_dre_enabled() && ids->ne[1] <= 8) {
        const void * b = nullptr; const int64_t * t = nullptr;
        if (ggml_cuda_dre_lookup(src0, nullptr, ctx.stream(), &b, &t)) {
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst, nullptr);
            return;
        }
    }

    // GPU tensor mirrors are disabled: they consume too much VRAM (a full expert tensor per mirror)
    // and reduce the LRU cache hit rate. The LRU cache with per-expert DMA works much better.

    // Capture context pointer for extern C API (first call only)
    if (!g_cuda_ctx_for_expert) {
        g_cuda_ctx_for_expert = &ctx;
    }

    // Auto-configure expert LRU cache and PCIe detection on first host expert encounter
    if (src0_is_host && ctx.expert_cache_budget == 0) {
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        // Use up to 80% of free VRAM for expert cache, minimum 512 MB
        ctx.expert_cache_budget = std::max(free_mem * 4 / 5, (size_t)(512 * 1024 * 1024));

        // Measure PCIe bandwidth to decide hybrid mode
        const size_t test_size = 64 * 1024 * 1024; // 64 MB test
        void * test_host = nullptr;
        void * test_gpu  = nullptr;
        cudaError_t err1 = cudaMallocHost(&test_host, test_size);
        cudaError_t err2 = cudaMalloc(&test_gpu, test_size);
        if (err1 == cudaSuccess && err2 == cudaSuccess) {
            memset(test_host, 0, test_size);
            // Warm up
            cudaMemcpy(test_gpu, test_host, test_size, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            // Measure
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaMemcpy(test_gpu, test_host, test_size, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
            ctx.pcie_bw_gbps = (float)(test_size / elapsed_s / 1e9);
        }
        if (test_host) cudaFreeHost(test_host);
        if (test_gpu)  cudaFree(test_gpu);

        // Hybrid mode: if PCIe < 8 GB/s, cache misses use CPU compute instead of DMA
        ctx.hybrid_mode = (ctx.pcie_bw_gbps > 0 && ctx.pcie_bw_gbps < 8.0f);

        fprintf(stderr, "offload budget = %.0f MB (%.0f MB free VRAM)\n",
            (float)ctx.expert_cache_budget / (1024*1024), (float)free_mem / (1024*1024));

        // Initialize per-user expert partitions if multi-user mode was requested
        if (ctx.n_expert_users > 0 && ctx.user_partitions.empty()) {
            ctx.init_user_partitions(ctx.n_expert_users);
        }
        fprintf(stderr, "PCIe bandwidth = %.1f GB/s → %s mode\n",
            ctx.pcie_bw_gbps, ctx.hybrid_mode ? "HYBRID (cache+CPU)" : "DMA (cache+DMA)");

        // The cache warms naturally as experts are accessed during inference.
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    // Skip fast paths when src0 is on host and fall through to the per-expert DMA loop.
    // GPU cache tensors (ne02=K) use the fast paths normally (slot IDs are valid there).
    if (!src0_is_host && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

        // Check if this tensor has CPU-half misses requiring post-MMVQ correction.
        // Guard with a cheap bool to avoid a hash map lookup on every call.
        const bool has_cpu_misses = g_pcache_has_misses
            && g_pcache_cpu_misses.find(src0->data) != g_pcache_cpu_misses.end()
            && !g_pcache_cpu_misses[src0->data].cpu_miss_ids.empty();

        // ALPHA_MMID_DEBUG=1: identify the dispatch branch + shapes taken by the multi-token
        // window verify, throttled to one line per 4096 calls.
        static const bool g_mmid_dbg = getenv("ALPHA_MMID_DEBUG") && atoi(getenv("ALPHA_MMID_DEBUG"));
        if (g_mmid_dbg && ne2 >= 2) {
            static int64_t dbgc = 0;
            if ((dbgc++ & 63) == 0) {
                const bool q = ggml_is_quantized(src0->type);
                const bool mmq = q && ggml_cuda_should_use_mmq(src0->type, cc, ne12, ne02);
                const bool rpm = ne2 > 1 && mmq && extern_get_tensor_offsets(src0) != nullptr;
                fprintf(stderr, "[mmid-dbg] ne0=%lld ne1=%lld ne2=%lld ne12=%lld ne02=%lld type=%d host=%d misses=%d | mmq=%d route_pool=%d mmvq_max=%d\n",
                        (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne12, (long long)ne02,
                        (int)src0->type, (int)src0_is_host, (int)has_cpu_misses,
                        (int)mmq, (int)rpm, MMVQ_MAX_BATCH_SIZE);
            }
        }

        // Lambda to launch batch MMVQ (picks best kernel)
        auto run_batch_mmvq = [&]() {
            // ik-quant experts (IQ1_KT et al.) have no MMVQ/MMQ kernel: decode them from the
            // pool by device dequant -> cuBLAS GEMV. Runs for hits here; on CPU-inline/hybrid
            // passes the CPU worker overwrites the miss slots afterward.
            if (ggml_cuda_is_ik_type(src0->type)) {
                return ggml_cuda_mul_mat_id_ik_pool(ctx, src0, src1, ids, dst);
            }
            static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
            // Pool-direct multi-token routing: when a per-expert pool offset table is published
            // for this tensor and it is a multi-token batch, prefer MMQ, which reads the pool
            // directly via the table (grouped, deduplicated). MMVQ only uses offsets for single
            // tokens (ncols_dst==1), so M >= 2 on MMVQ ignores the table. Only diverts when MMQ is
            // usable; inert otherwise (no multi-token table is published unless ALPHA_PD_MMVQ).
            const bool mmq_ok = ggml_is_quantized(src0->type) &&
                                ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02);
            // Batch-invariant: normally only M>=2 routes to pool-direct MMQ and M=1 falls to MMVQ
            // below, so a token served alone uses a different kernel (MMVQ dp4a) than in a batch
            // (MMQ, which requantizes activations to q8_1) and gets different bits. Route M>=1 to
            // MMQ whenever a table is published so every batch width uses the same kernel.
            // The `offsets != nullptr` guard makes this a no-op if no M=1 table is staged; MMQ is
            // valid for ncols=1 and should_use_mmq is M-independent on turing_mma.
#ifdef GGML_CUDA_BATCH_INVARIANT
            const bool route_pool_mmq = ne2 >= 1 && mmq_ok &&
                                extern_get_tensor_offsets(src0) != nullptr;
#else
            const bool route_pool_mmq = ne2 > 1 && mmq_ok &&
                                extern_get_tensor_offsets(src0) != nullptr;
#endif
            if (ne2 <= MMVQ_MAX_BATCH_SIZE && !route_pool_mmq) {
                if (ggml_is_quantized(src0->type)) {
                    const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                    if (ne2 <= mmvq_mmid_max) {
                        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                        return true;
                    }
                } else {
                    if (GGML_CUDA_CC_IS_AMD(cc)) {
                        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                        return true;
                    }
                }
            }
            if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
                ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
                return true;
            }
            if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
                ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
                return true;
            }
            return false;
        };

        if (!has_cpu_misses) {
            // No CPU correction needed: fast path (CUDA graph compatible)
            if (run_batch_mmvq()) return;
        } else if (g_pcache_cpu_inline && g_pcache_ids_ne0 == ids->ne[0] &&
                   (ne12 == 1 ? g_pcache_ids_ne1 == 1
                              : (g_pcache_inline_mt && g_pcache_ids_ne1 == (int)ne12 &&
                                 ne12 * ids->ne[0] <= PCACHE_INL_MAX))) {
            // Per-layer CPU inline correction (async). All missing experts' dst rows are
            // computed on the CPU worker from host RAM and land via stream-ordered writebacks.
            // The launch thread never blocks: D2H(activations)+event -> MMVQ -> enqueue job ->
            // hostfunc fence -> pre-enqueued H2D writebacks -> hostfunc release. The ordering
            // guarantees the rows are present before the next op consumes dst, so gate/up
            // corrections propagate into the GPU-computed silu-mul that feeds the down tensor.
            auto & tmi = g_pcache_cpu_misses[src0->data];
            cudaStream_t stream = ctx.stream();

            std::unordered_set<int64_t> inl_miss_set(
                tmi.cpu_miss_ids.begin(), tmi.cpu_miss_ids.end());
            // Enumerate the (token, expert-slot) pairs this tensor must correct. ne12==1
            // (single-token decode) collapses to a slot-only loop.
            int n_inl = 0;
            int64_t inl_slot[PCACHE_INL_MAX], inl_eid[PCACHE_INL_MAX], inl_tok[PCACHE_INL_MAX];
            for (int64_t t = 0; t < ne12 && n_inl < PCACHE_INL_MAX; t++) {
                for (int64_t s = 0; s < ids->ne[0] && n_inl < PCACHE_INL_MAX; s++) {
                    int32_t eid = g_pcache_batch_ids[(size_t)t * g_pcache_ids_s1 +
                                                     (size_t)s * g_pcache_ids_s0];
                    if (inl_miss_set.count(eid)) {
                        inl_slot[n_inl] = s; inl_tok[n_inl] = t; inl_eid[n_inl] = eid; n_inl++;
                    }
                }
            }
            if (n_inl == 0) {
                if (run_batch_mmvq()) return;
            } else {
                pcache_inline_worker_start();
                const uint64_t pi = g_inline_produced.load(std::memory_order_relaxed);
                pcache_inline_job * j = &g_inline_ring[pi % 8];
                while (j->state.load(std::memory_order_acquire) != 0) {
                    // ring full (7 tensors in flight): rare; brief yield
                    std::this_thread::yield();
                }
                // Activation staging, deduplicated by source row.
                // The activation address is slot*nb1 (only if src1 has per-slot rows) + token*nb2,
                // so for the gate/up tensors (src1->ne[1]==1) it depends only on the token: at 8 users
                // with top-8 the 64 (token,expert) pairs reference just 8 distinct rows. One staged row
                // per distinct source row, with act_idx[] mapping pair->row, avoids redundant D2H
                // copies and f32->Q8_K quantization. ALPHA_INLINE_ACT_DEDUP=0 forces one row per pair.
                static const bool act_dedup = []() {
                    const char * e = getenv("ALPHA_INLINE_ACT_DEDUP");
                    return !e || atoi(e) != 0;          // default on
                }();
                const bool act_per_slot = src1->ne[1] > 1 || !act_dedup;   // the down tensor
                int act_idx[PCACHE_INL_MAX];
                int64_t a_slot[PCACHE_INL_MAX], a_tok[PCACHE_INL_MAX];
                int n_act = 0;
                if (act_per_slot) {
                    for (int i = 0; i < n_inl; i++) {           // every pair is a distinct row
                        act_idx[i] = i; a_slot[i] = inl_slot[i]; a_tok[i] = inl_tok[i];
                    }
                    n_act = n_inl;
                } else {
                    // gate/up: one row per distinct token
                    int tok_row[PCACHE_INL_MAX];
                    for (int t = 0; t < (int)ne12 && t < PCACHE_INL_MAX; t++) tok_row[t] = -1;
                    for (int i = 0; i < n_inl; i++) {
                        const int t = (int)inl_tok[i];
                        if (tok_row[t] < 0) {
                            tok_row[t] = n_act; a_slot[n_act] = 0; a_tok[n_act] = t; n_act++;
                        }
                        act_idx[i] = tok_row[t];
                    }
                }
                // The worker always indexes activations through act_idx[].
                j->act_rows = n_act;
                memcpy(j->act_idx, act_idx, n_inl * sizeof(int));
                const size_t act_need = (size_t)j->act_rows * ne10 * sizeof(float);
                if (act_need > j->act_cap) {
                    if (j->act_pin) CUDA_CHECK(cudaFreeHost(j->act_pin));
                    j->act_cap = act_need;
                    CUDA_CHECK(cudaMallocHost((void **)&j->act_pin, j->act_cap));
                }
                const size_t out_need = (size_t)n_inl * ne0 * sizeof(float);
                if (out_need > j->out_cap) {
                    if (j->out_pin) CUDA_CHECK(cudaFreeHost(j->out_pin));
                    j->out_cap = out_need;
                    CUDA_CHECK(cudaMallocHost((void **)&j->out_pin, j->out_cap));
                }
                if (!j->ev) CUDA_CHECK(cudaEventCreateWithFlags(&j->ev, cudaEventDisableTiming));
                // one D2H per distinct row (n_act), not per pair
                for (int r = 0; r < n_act; r++) {
                    const char * a_src = (const char *)src1->data
                                       + (act_per_slot ? a_slot[r] : 0) * src1->nb[1]
                                       + a_tok[r] * src1->nb[2];
                    CUDA_CHECK(cudaMemcpyAsync(j->act_pin + (size_t)r * ne10, a_src,
                                               ne10 * sizeof(float), cudaMemcpyDeviceToHost, stream));
                }
                CUDA_CHECK(cudaEventRecord(j->ev, stream));

                // GPU: cached-expert partial for this tensor (misses read zero slots)
                run_batch_mmvq();

                // hand the job to the worker (it waits on j->ev, never this thread)
                j->host_base   = tmi.host_base;
                j->expert_size = tmi.expert_size;
                j->nb01_host   = src0->nb[1];
                j->wtype       = src0->type;
                j->ne0         = (int)ne0;
                j->ne10        = (int)ne10;
                j->n_slots     = n_inl;
                memcpy(j->slot_idx, inl_slot, n_inl * sizeof(int64_t));
                memcpy(j->tok_idx,  inl_tok,  n_inl * sizeof(int64_t));
                memcpy(j->expert_id, inl_eid, n_inl * sizeof(int64_t));
                // Expert-major grouping: sort pair indices by expert so every pair sharing an
                // expert is contiguous, turning the worker's per-pair GEMV into a per-expert GEMM
                // that amortizes the weight read across all tokens routed to that expert.
                {
                    int idx[PCACHE_INL_MAX];
                    for (int i = 0; i < n_inl; i++) idx[i] = i;
                    std::sort(idx, idx + n_inl,
                              [&](int a, int b) { return inl_eid[a] < inl_eid[b]; });
                    int ng = 0;
                    for (int i = 0; i < n_inl; ) {
                        int k = i;
                        while (k < n_inl && inl_eid[idx[k]] == inl_eid[idx[i]]) k++;
                        j->grp_off[ng]    = i;
                        j->grp_expert[ng] = inl_eid[idx[i]];
                        ng++;
                        i = k;
                    }
                    j->grp_off[ng] = n_inl;
                    j->n_groups    = ng;
                    for (int i = 0; i < n_inl; i++) j->order[i] = idx[i];
                    if (g_inline_mt_dbg) {
                        static int64_t dsp_dbg = 0;
                        if ((dsp_dbg++ & 1023) == 0) {
                            fprintf(stderr, "[inline-mt/dispatch] ne12=%lld pairs=%d groups=%d "
                                    "avg_grp=%.2f act_rows=%d\n",
                                    (long long)ne12, n_inl, ng, ng ? (double)n_inl / ng : 0.0,
                                    j->act_rows);
                        }
                    }
                    // Predictive prefetch: record this pass's misses so the next token's
                    // pass_begin can WILLNEED them ahead of the worker. Once per unique expert,
                    // not once per pair.
                    for (int g = 0; g < ng; g++) {
                        inl_prefetch_record(tmi.host_base,
                                            (size_t)j->grp_expert[g] * tmi.expert_size, tmi.expert_size);
                    }
                }
                j->state.store(1, std::memory_order_release);
                // Publish under the mutex. The worker does cv.wait(lk, pred): it takes
                // g_inline_mu, evaluates the predicate, and only then atomically unlocks and
                // sleeps. Publishing + notifying outside the lock can land inside that window, so
                // notify_one() is discarded and the worker sleeps with work pending. The host-func
                // fence below would then spin on state!=2 forever and, since a host callback blocks
                // the stream, the GPU would go idle.
                {
                    std::lock_guard<std::mutex> lk(g_inline_mu);
                    g_inline_produced.store(pi + 1, std::memory_order_release);
                }
                g_inline_cv.notify_one();

                // stream-ordered: wait for results, land them, release the ring slot
                CUDA_CHECK(cudaLaunchHostFunc(stream, pcache_inline_fence, j));
                for (int i = 0; i < n_inl; i++) {
                    // dst row for this pair: expert-slot within the token, plus the token
                    // plane (dst->nb[2] term is 0 on single-token decode).
                    CUDA_CHECK(cudaMemcpyAsync((char *)dst->data + inl_slot[i] * nb1
                                                                 + inl_tok[i] * dst->nb[2],
                                               j->out_pin + (size_t)i * ne0,
                                               ne0 * sizeof(float), cudaMemcpyHostToDevice, stream));
                }
                CUDA_CHECK(cudaLaunchHostFunc(stream, pcache_inline_release, j));
                return;
            }
        } else {
            // Hybrid: batch MMVQ + parallel CPU correction for CPU-half misses.
            // The GPU runs batch MMVQ for all experts (hits correct, CPU-half zeroed -> ~0 output).
            // A CPU thread computes the CPU-half output and overwrites those positions in dst.
            auto & tmi = g_pcache_cpu_misses[src0->data];
            cudaStream_t stream = ctx.stream();

            // Build CPU miss expert set for O(1) lookup
            std::unordered_set<int64_t> cpu_miss_set(
                tmi.cpu_miss_ids.begin(), tmi.cpu_miss_ids.end());

            // Read ids tensor to determine which (token, slot) pairs route to CPU-half experts
            const int64_t n_expert_used = ids->ne[0];
            std::vector<int32_t> ids_host(ne12 * n_expert_used);
            CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data,
                ne12 * n_expert_used * sizeof(int32_t),
                cudaMemcpyDeviceToHost, stream));

            struct cpu_slot { int64_t token_idx, slot_idx, expert_id; };
            std::vector<cpu_slot> cpu_slots;

            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int64_t t = 0; t < ne12; t++) {
                for (int64_t s = 0; s < n_expert_used; s++) {
                    int32_t eid = ids_host[t * n_expert_used + s];
                    if (cpu_miss_set.count(eid)) {
                        cpu_slots.push_back({t, s, eid});
                    }
                }
            }

            if (cpu_slots.empty()) {
                if (run_batch_mmvq()) return;
            } else {
                // Download activations for CPU slots (small: n_slots x ne10 x sizeof(float))
                std::vector<float> cpu_activations(cpu_slots.size() * ne10);
                for (size_t i = 0; i < cpu_slots.size(); i++) {
                    const void * src1_row = (const char *)src1->data +
                        cpu_slots[i].token_idx * src1->nb[1];
                    CUDA_CHECK(cudaMemcpyAsync(
                        cpu_activations.data() + i * ne10,
                        src1_row, ne10 * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<float> cpu_outputs(cpu_slots.size() * ne0);

                // CPU type traits: vec_dot between weight type (e.g., Q4_K) and vec_dot_type (Q8_K)
                auto * cpu_traits = ggml_get_type_traits_cpu(src0->type);
                auto vec_dot_fn   = cpu_traits->vec_dot;
                // Activations are quantized to vec_dot_type, not the weight type.
                auto from_float_fn = ggml_get_type_traits_cpu(cpu_traits->vec_dot_type)->from_float;
                size_t vdt_size   = ggml_type_size(cpu_traits->vec_dot_type);
                int64_t vdt_blck  = ggml_blck_size(cpu_traits->vec_dot_type);
                size_t nb01_host  = src0->nb[1];

                const void * host_base = tmi.host_base;
                size_t expert_sz = tmi.expert_size;
                int local_ne0 = ne0, local_ne10 = ne10;

                // Launch CPU thread for vec_dot (parallel with GPU batch MMVQ)
                std::thread cpu_thread([&]() {
                    size_t q_row_bytes = (size_t)(local_ne10 / vdt_blck) * vdt_size;
                    std::vector<char> q_act_storage(q_row_bytes + 63);
                    char * q_act = (char *)(((uintptr_t)q_act_storage.data() + 63) & ~(uintptr_t)63);

                    for (size_t i = 0; i < cpu_slots.size(); i++) {
                        const float * act = cpu_activations.data() + i * local_ne10;
                        float * out = cpu_outputs.data() + i * local_ne0;

                        from_float_fn(act, q_act, local_ne10);

                        const char * w_base = (const char *)host_base +
                            cpu_slots[i].expert_id * expert_sz;
                        for (int n = 0; n < local_ne0; n++) {
                            float dot = 0.0f;
                            vec_dot_fn(local_ne10, &dot, 1,
                                       w_base + n * nb01_host, 1,
                                       q_act, 1, 1);
                            out[n] = dot;
                        }
                    }
                });

                // GPU: batch MMVQ (hits correct, CPU-half ~0 from zeroed input_cpy)
                run_batch_mmvq();

                cpu_thread.join();

                // Upload CPU results to correct dst positions
                // dst layout: ne[0]=out_dim, ne[1]=n_expert_used, ne[2]=n_tokens
                // Position for (token t, slot s) = dst + t * nb2 + s * nb1
                for (size_t i = 0; i < cpu_slots.size(); i++) {
                    char * dst_pos = (char *)dst->data +
                        cpu_slots[i].token_idx * nb2 +
                        cpu_slots[i].slot_idx * nb1;
                    CUDA_CHECK(cudaMemcpyAsync(
                        dst_pos,
                        cpu_outputs.data() + i * ne0,
                        ne0 * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
                }

                return;
            }
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    // TODO: add asserts to verify this. should work with CUDA, HIP, etc.
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    if (ids_to_sorted_host.size() != size_t(ne_get_rows)) {
        fprintf(stderr, "[MUL_MAT_ID] FAIL: sorted=%zu expected=%lld ne02=%lld ids:",
            ids_to_sorted_host.size(), (long long)ne_get_rows, (long long)ne02);
        for (int64_t iex = 0; iex < std::min(n_expert_used, (int64_t)16); ++iex)
            fprintf(stderr, " %d", *(const int32_t *)(ids_host.data() + iex*ids->nb[0]));
        fprintf(stderr, "\n");
        GGML_ASSERT(false && "duplicate slot IDs in mul_mat_id");
    }

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    // Collect active experts for double-buffered DMA + compute
    struct active_expert {
        int64_t id;
        int64_t n_tokens;
        char *  src1_ptr;
        char *  dst_ptr;
    };
    std::vector<active_expert> active_experts;
    {
        char * s1 = (char *)src1_sorted.ptr;
        char * d  = (char *)dst_sorted.ptr;
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            if (tokens_per_expert[i02] == 0) continue;
            active_experts.push_back({i02, tokens_per_expert[i02], s1, d});
            s1 += tokens_per_expert[i02] * ne10 * (int64_t)ts_src1_sorted;
            d  += tokens_per_expert[i02] * ne0  * (int64_t)ts_dst_sorted;
        }
    }

    if (!src0_is_host || active_experts.empty()) {
        // Non-host path or no active experts: original serial loop
        char * src1_data_cur = (char *) src1_sorted.ptr;
        char *  dst_data_cur = (char *)  dst_sorted.ptr;
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            if (tokens_per_expert[i02] == 0) continue;

            ggml_tensor src0_slice = *src0;
            src0_slice.ne[2] = 1; src0_slice.nb[3] = src0_slice.nb[2];
            src0_slice.op = GGML_OP_VIEW; src0_slice.view_src = dst->src[0];
            src0_slice.data = (char *)src0->data + i02 * nb02;

            ggml_tensor src1_slice; memset(&src1_slice, 0, sizeof(src1_slice));
            src1_slice.buffer = src1->buffer; src1_slice.type = type_src1_sorted;
            src1_slice.ne[0] = ne10; src1_slice.ne[1] = tokens_per_expert[i02]; src1_slice.ne[2] = 1; src1_slice.ne[3] = 1;
            src1_slice.nb[0] = ts_src1_sorted; src1_slice.nb[1] = src1_slice.ne[0]*src1_slice.nb[0];
            src1_slice.nb[2] = src1_slice.ne[1]*src1_slice.nb[1]; src1_slice.nb[3] = src1_slice.nb[2];
            src1_slice.data = src1_data_cur;

            ggml_tensor dst_slice; memset(&dst_slice, 0, sizeof(dst_slice));
            dst_slice.buffer = dst->buffer; dst_slice.type = type_dst_sorted;
            dst_slice.ne[0] = ne0; dst_slice.ne[1] = tokens_per_expert[i02]; dst_slice.ne[2] = 1; dst_slice.ne[3] = 1;
            dst_slice.nb[0] = ts_dst_sorted; dst_slice.nb[1] = dst_slice.ne[0]*dst_slice.nb[0];
            dst_slice.nb[2] = dst_slice.ne[1]*dst_slice.nb[1]; dst_slice.nb[3] = dst_slice.nb[2];
            dst_slice.data = dst_data_cur;

            ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
            CUDA_CHECK(cudaGetLastError());
            src1_data_cur += src1_slice.nb[2];
            dst_data_cur  += dst_slice.nb[2];
        }
    } else {
        // Double-buffered DMA + compute for host experts.
        // A second stream is used for DMA so it overlaps with GPU compute.
        cudaStream_t dma_stream = ctx.stream(ctx.device, 1);  // stream 1 for DMA
        cudaEvent_t dma_done;
        CUDA_CHECK(cudaEventCreateWithFlags(&dma_done, cudaEventDisableTiming));

        const size_t slice_bytes = nb02;

        // Resolve GPU pointers for all active experts (cache lookup or DMA)
        std::vector<void *> expert_gpu_ptrs(active_experts.size(), nullptr);
        std::vector<bool>   expert_needs_dma(active_experts.size(), false);

        for (size_t ei = 0; ei < active_experts.size(); ei++) {
            int64_t eid = active_experts[ei].id;
            void * cached = ctx.expert_cache_lookup(src0->data, eid);
            if (cached) {
                expert_gpu_ptrs[ei] = cached;
            } else {
                expert_needs_dma[ei] = true;
            }
        }

        // Prefetch first uncached expert on DMA stream
        size_t prefetch_idx = SIZE_MAX;
        ggml_cuda_pool_alloc<char> dma_buf_a, dma_buf_b;
        void * prefetch_ptr = nullptr;

        auto start_prefetch = [&](size_t ei, ggml_cuda_pool_alloc<char> & buf) {
            if (ei >= active_experts.size() || !expert_needs_dma[ei]) return;
            int64_t eid = active_experts[ei].id;
            const void * host_src = (const char *)src0->data + eid * nb02;

            if (ctx.expert_cache_budget > 0) {
                void * gpu_ptr = ctx.expert_cache_insert(src0->data, eid, host_src, slice_bytes, dma_stream);
                expert_gpu_ptrs[ei] = gpu_ptr;
            } else {
                buf.alloc(ctx.pool(), slice_bytes);
                CUDA_CHECK(cudaMemcpyAsync(buf.ptr, host_src, slice_bytes, cudaMemcpyHostToDevice, dma_stream));
                expert_gpu_ptrs[ei] = buf.ptr;
            }
            CUDA_CHECK(cudaEventRecord(dma_done, dma_stream));
            prefetch_idx = ei;
        };

        // Kick off first DMA
        for (size_t ei = 0; ei < active_experts.size(); ei++) {
            if (expert_needs_dma[ei]) { start_prefetch(ei, dma_buf_a); break; }
        }

        // Hybrid parallel: GPU cached experts and CPU uncached experts run simultaneously.
        //
        // Phase 1: download activations for uncached experts (small, one copy for all)
        // Phase 2: launch GPU matmuls for cached experts (non-blocking) and, at the same time,
        //          a CPU thread for uncached expert dequant+matmul
        // Phase 3: wait for both to finish, upload CPU results to GPU

        // Collect uncached expert info
        struct cpu_expert_task {
            int64_t eid;
            const void * host_weights;
            char * dst_ptr;
            char * src1_ptr;
            int n_tokens;
        };
        std::vector<cpu_expert_task> cpu_tasks;
        for (size_t ei = 0; ei < active_experts.size(); ei++) {
            if (expert_needs_dma[ei]) {
                cpu_tasks.push_back({
                    active_experts[ei].id,
                    (const char *)src0->data + active_experts[ei].id * nb02,
                    active_experts[ei].dst_ptr,
                    active_experts[ei].src1_ptr,
                    (int)active_experts[ei].n_tokens
                });
            }
        }

        // Timing diagnostics
        static int64_t diag_count = 0;
        auto t_start = std::chrono::high_resolution_clock::now();

        // Phase 1: Download activations for uncached experts
        static thread_local std::vector<float> cpu_src1_all;
        if (!cpu_tasks.empty()) {
            size_t total_src1 = 0;
            for (auto & t : cpu_tasks) total_src1 += (size_t)t.n_tokens * ne10;
            cpu_src1_all.resize(total_src1);

            // Async copy each task's activations on DMA stream (non-blocking on compute stream)
            size_t off = 0;
            for (auto & t : cpu_tasks) {
                size_t bytes = (size_t)t.n_tokens * ne10 * sizeof(float);
                CUDA_CHECK(cudaMemcpyAsync(cpu_src1_all.data() + off, t.src1_ptr,
                                            bytes, cudaMemcpyDeviceToHost, dma_stream));
                off += (size_t)t.n_tokens * ne10;
            }
            // Wait for all downloads before launching CPU thread
            CUDA_CHECK(cudaStreamSynchronize(dma_stream));
        }

        // Phase 2: launch GPU cached experts + CPU uncached experts in parallel
        static thread_local std::vector<float> cpu_dst_all;
        static thread_local std::vector<float> cpu_weights_buf;
        std::atomic<bool> cpu_done{false};
        std::thread cpu_thread;

        if (!cpu_tasks.empty()) {
            // Pre-allocate CPU output buffer
            size_t total_dst = 0;
            for (auto & t : cpu_tasks) total_dst += (size_t)t.n_tokens * ne0;
            cpu_dst_all.resize(total_dst);

            // Launch CPU thread for uncached expert matmuls using ggml's optimized vec_dot
            auto * cpu_traits = ggml_get_type_traits_cpu(src0->type);
            auto vec_dot_fn = cpu_traits->vec_dot;
            // Quantize activations to vec_dot_type (e.g., Q8_K), not the weight type (Q4_K)
            auto from_float_fn = ggml_get_type_traits_cpu(cpu_traits->vec_dot_type)->from_float;
            size_t vdt_size = ggml_type_size(cpu_traits->vec_dot_type);
            int64_t vdt_blck = ggml_blck_size(cpu_traits->vec_dot_type);
            int cpu_ne0 = ne0, cpu_ne10 = ne10;
            size_t cpu_nb01 = nb01;

            auto * p_src1 = &cpu_src1_all;
            auto * p_dst = &cpu_dst_all;

            cpu_thread = std::thread([&cpu_tasks, p_src1, p_dst,
                                       &cpu_done, vec_dot_fn, from_float_fn,
                                       vdt_size, vdt_blck,
                                       cpu_ne0, cpu_ne10, cpu_nb01]() {
                // Quantize activations: FP32 -> vec_dot_type (e.g., Q8_1) for fast dot product
                static thread_local std::vector<char> q_act_buf;
                size_t q_row_bytes = (size_t)(cpu_ne10 / vdt_blck) * vdt_size;

                size_t src1_off = 0, dst_off = 0;
                for (auto & t : cpu_tasks) {
                    int M = t.n_tokens, N = cpu_ne0, K = cpu_ne10;
                    const float * src1 = p_src1->data() + src1_off;
                    float * dst = p_dst->data() + dst_off;

                    // Quantize M activation rows
                    q_act_buf.resize((size_t)M * q_row_bytes);
                    for (int m = 0; m < M; m++) {
                        from_float_fn(src1 + (size_t)m * K,
                                       q_act_buf.data() + (size_t)m * q_row_bytes, K);
                    }

                    // Optimized matmul: vec_dot(quantized_weight_row, quantized_activation)
                    const char * w_base = (const char *)t.host_weights;
                    for (int m = 0; m < M; m++) {
                        const char * q_act = q_act_buf.data() + (size_t)m * q_row_bytes;
                        for (int n = 0; n < N; n++) {
                            float dot = 0.0f;
                            vec_dot_fn(K, &dot, 1,
                                        w_base + (size_t)n * cpu_nb01, 1,
                                        q_act, 1, 1);
                            dst[m * N + n] = dot;
                        }
                    }

                    src1_off += (size_t)M * K;
                    dst_off += (size_t)M * N;
                }
                cpu_done.store(true);
            });
        }

        // GPU: process cached experts (runs while CPU thread works on uncached)
        for (size_t ei = 0; ei < active_experts.size(); ei++) {
            if (expert_needs_dma[ei]) continue;  // skip uncached, the CPU handles these

            auto & ae = active_experts[ei];
            ggml_tensor src0_slice = *src0;
            src0_slice.ne[2] = 1; src0_slice.nb[3] = src0_slice.nb[2];
            src0_slice.op = GGML_OP_VIEW; src0_slice.view_src = dst->src[0];
            src0_slice.data = (char *)expert_gpu_ptrs[ei];
            src0_slice.buffer = dst->buffer;

            ggml_tensor src1_slice; memset(&src1_slice, 0, sizeof(src1_slice));
            src1_slice.buffer = src1->buffer; src1_slice.type = type_src1_sorted;
            src1_slice.ne[0] = ne10; src1_slice.ne[1] = ae.n_tokens; src1_slice.ne[2] = 1; src1_slice.ne[3] = 1;
            src1_slice.nb[0] = ts_src1_sorted; src1_slice.nb[1] = src1_slice.ne[0]*src1_slice.nb[0];
            src1_slice.nb[2] = src1_slice.ne[1]*src1_slice.nb[1]; src1_slice.nb[3] = src1_slice.nb[2];
            src1_slice.data = ae.src1_ptr;

            ggml_tensor dst_slice; memset(&dst_slice, 0, sizeof(dst_slice));
            dst_slice.buffer = dst->buffer; dst_slice.type = type_dst_sorted;
            dst_slice.ne[0] = ne0; dst_slice.ne[1] = ae.n_tokens; dst_slice.ne[2] = 1; dst_slice.ne[3] = 1;
            dst_slice.nb[0] = ts_dst_sorted; dst_slice.nb[1] = dst_slice.ne[0]*dst_slice.nb[0];
            dst_slice.nb[2] = dst_slice.ne[1]*dst_slice.nb[1]; dst_slice.nb[3] = dst_slice.nb[2];
            dst_slice.data = ae.dst_ptr;

            ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
            CUDA_CHECK(cudaGetLastError());
        }

        // Phase 3: Wait for CPU thread, upload results to GPU, warm cache
        if (!cpu_tasks.empty()) {
            cpu_thread.join();

            // Upload CPU results to GPU dst slots
            size_t dst_off = 0;
            for (auto & t : cpu_tasks) {
                size_t bytes = (size_t)t.n_tokens * ne0 * sizeof(float);
                CUDA_CHECK(cudaMemcpyAsync(t.dst_ptr, cpu_dst_all.data() + dst_off,
                                            bytes, cudaMemcpyHostToDevice, stream));
                dst_off += (size_t)t.n_tokens * ne0;

                // Background DMA: warm GPU cache for future hits
                if (ctx.expert_cache_budget > 0) {
                    ctx.expert_cache_insert(src0->data, t.eid, t.host_weights, slice_bytes, dma_stream);
                }
            }
        }

        CUDA_CHECK(cudaEventDestroy(dma_done));

        // Timing diagnostics (first 20 calls)
        if (diag_count < 20) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            int n_cached = 0, n_cpu = (int)cpu_tasks.size();
            for (size_t ei = 0; ei < active_experts.size(); ei++)
                if (!expert_needs_dma[ei]) n_cached++;
            fprintf(stderr, "[hybrid] %d cached(GPU) + %d uncached(CPU) = %.1fms\n",
                    n_cached, n_cpu, ms);
            diag_count++;
        }
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
            ggml_cuda_op_dsv4_hc_split_sinkhorn(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
            ggml_cuda_op_dsv4_hc_weighted_sum(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_EXPAND:
            ggml_cuda_op_dsv4_hc_expand(ctx, dst);
            break;
        case GGML_OP_HC_COMBINE:
            ggml_cuda_op_hc_combine(ctx, dst);
            break;
        case GGML_OP_HC_MIX_TAIL:
            ggml_cuda_op_hc_mix_tail(ctx, dst);
            break;
        case GGML_OP_HC_MIX_FUSED:
            ggml_cuda_op_hc_mix_fused(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

// ALPHA_CUDA_SMALL_H2D_KERNEL (default 1): a 64 KB - 1 MB async upload from pinned host memory is done by a copy
// kernel reading the host bytes directly. On the copy engine it would queue behind other streams' H2D traffic (a
// long prompt's speculative expert copies, hundreds of MB), stalling the compute stream at every attention layer.
static __global__ void alpha_h2d_copy_k(uint4 * __restrict__ dst, const uint4 * __restrict__ src, const size_t n16) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
static bool alpha_h2d_kernel_copy(void * dst, const void * src, size_t size, cudaStream_t st) {
    static const bool on = [] { const char * e = getenv("ALPHA_CUDA_SMALL_H2D_KERNEL"); return !e || atoi(e) != 0; }();
    // (prompt-sized graphs only: as kernels, decode graphs' uploads in this size range make decode slightly slower)
    if (!on || !ggml_backend_sched_alpha_async_inputs_on() || size < ((size_t) 64 << 10) || size > ((size_t) 1 << 20)) return false;
    if (((uintptr_t) dst & 15) || ((uintptr_t) src & 15) || (size & 15)) return false;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, src) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    if (attr.type != cudaMemoryTypeHost || !attr.devicePointer) return false;
    const size_t n16 = size / 16;
    const int nb = (int) std::min<size_t>(64, (n16 + 255) / 256);
    alpha_h2d_copy_k<<<nb, 256, 0, st>>>((uint4 *) dst, (const uint4 *) attr.devicePointer, n16);
    return true;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");
    alpha_h2d_log(tensor, size, "async");

    if (alpha_h2d_kernel_copy((char *) tensor->data + offset, data, size, cuda_ctx->stream())) return;
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

// DMA/compute overlap via stream 1 for cache-miss H2D.
// mmap'd (unpinned) host memory makes cudaMemcpyAsync on non-default streams go through a shared
// staging buffer that subsequent copies can overwrite before the DMA completes, so the expert weight
// pages are registered with cudaHostRegister for true async DMA on stream 1. Attention layers run on
// stream 0 while the H2D DMA is in flight; cudaStreamWaitEvent before MUL_MAT_ID compute ensures the
// DMA completes before expert reads.

#ifdef USE_CUDA_GRAPH
// Mirror of the run_batch_mmvq() decision tree in ggml_cuda_mul_mat_id: true when the node has a
// device-only kernel (MMVQ / MMQ / MMF / ik pool). Otherwise it takes the generic fallback that sorts
// the ids on the host (cudaStreamSynchronize), which is not capturable ("operation not permitted when
// stream is capturing"). Used only to veto CUDA-graph capture for such nodes.
static bool alpha_mmid_has_device_path(const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1 || src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
        return false;
    }
    const int     cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t ne2  = node->ne[2];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne02 = src0->ne[2];
    if (ggml_cuda_is_ik_type(src0->type)) {
        return true;   // ik pool path (vetoed separately below)
    }
    if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (ne2 <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return true;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return true;
        }
    }
    if (ggml_is_quantized(src0->type) && ggml_cuda_should_use_mmq(src0->type, cc, ne12, ne02)) {
        return true;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return true;
    }
    return false;
}

// Mirror of the (non-split) flag chain at the top of ggml_cuda_mul_mat: true when this MUL_MAT would
// fall through to cuBLAS (batched or op_mul_mat_cublas), which is what actually fails under CUDA-graph
// capture (CUBLAS_STATUS_NOT_SUPPORTED). The custom kernels (MMVF/MMF/MMVQ/MMQ) are capture-safe.
// Keep in sync with that chain; the small-M / output-MMQ tiers only move quantized
// matmuls between MMVQ and MMQ, never to cuBLAS. Used by the batched-MUL_MAT graph veto below.
static bool alpha_mul_mat_uses_cublas(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!src0 || !src1 || !src0->buffer || ggml_backend_buft_is_cuda_split(src0->buffer->buft)) {
        return true;
    }
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    const bool f32io = src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    const bool quant = ggml_is_quantized(src0->type);
    const bool ik    = ggml_cuda_is_ik_type(src0->type);
    const int  dev       = ggml_cuda_get_device();
    const int  cc        = ggml_cuda_info().devices[dev].cc;
    const int  warp_size = ggml_cuda_info().devices[dev].warp_size;
    const int64_t ne11   = src1->ne[1];
    bool mmvf = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) && f32io;
    bool mmf  = !quant && f32io;
    bool mmvq = quant && !bad_padding_clear && f32io && ne11 <= MMVQ_MAX_BATCH_SIZE && !ik;
    bool mmq  = quant && !bad_padding_clear && f32io && !ik;
    mmq  = mmq  && ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts=*/0);
    mmf  = mmf  && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id=*/false);
    mmvf = mmvf && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11);
    return !(mmvf || mmf || mmvq || mmq);
}

// ALPHA_GRAPH_STATS: per-graph-key outcome tally (n_nodes, output columns, direct/captured/replayed)
struct alpha_graph_key_stats { int n_nodes; long long ne1; long long direct, captured, replayed; };
static std::map<const void *, alpha_graph_key_stats> g_alpha_graph_keys;

// ALPHA_GRAPH_STATS: which veto site in ggml_cuda_graph_check_compability refused capture, by source line
static std::map<int, long long> g_alpha_graph_veto;
static inline void alpha_graph_veto_note(int line) { g_alpha_graph_veto[line]++; }

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
            use_cuda_graph = false; alpha_graph_veto_note(__LINE__); // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
#endif
        }

        // Multi-token batches (prefill and the batch-GPU window verify) route wide GGML_OP_MUL_MAT
        // through cublasSgemm, which returns CUBLAS_STATUS_NOT_SUPPORTED when issued during stream
        // capture. The batch-GPU verify repeats an identical M-token shape every window, so it warms
        // up and gets captured, unlike prefill. Graphs mostly help launch-bound single-token decode;
        // a batched matmul amortizes launch cost over its columns.
        // ALPHA_BATCH_GRAPH: 0 = veto every batched MUL_MAT (src1 ne[1] > 1); 1 = veto only matmuls
        // that would run on cuBLAS; 2 (default) = no batched veto for batches of <= 64 columns, relying
        // on the cuBLAS workspace set at handle creation (ALPHA_CUBLAS_WS_MB), which makes cuBLAS
        // capturable. This lets MTP verify passes and small multi-user batches replay graphs. Wider
        // batches (prefill chunks) keep the veto: nothing to gain there, and the chunked-GDN /
        // compaction paths are not exercised under capture.
        static const int relax_batch_graph = []() {
            const char * e = getenv("ALPHA_BATCH_GRAPH");
            return e ? atoi(e) : 2;
        }();
        constexpr int64_t relax2_max_cols = 64;
        if (node->op == GGML_OP_MUL_MAT && node->src[1] && node->src[1]->ne[1] > 1) {
            const bool src0_float = node->src[0] &&
                (node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_BF16 ||
                 node->src[0]->type == GGML_TYPE_F32);
            // Blanket disable unless relaxed; when relaxed, only a matmul that would run on cuBLAS is
            // unsafe (f16/f32 src0 at small M goes to MMVF/MMF, not cuBLAS).
            (void) src0_float;
            const bool relax2_ok = relax_batch_graph >= 2 && node->src[1]->ne[1] <= relax2_max_cols;
            if (relax_batch_graph == 0 || (relax_batch_graph == 1 && alpha_mul_mat_uses_cublas(node)) ||
                (relax_batch_graph >= 2 && !relax2_ok)) {
                use_cuda_graph = false; alpha_graph_veto_note(__LINE__);
                static const bool stats_on = alpha_env_on("ALPHA_GRAPH_STATS");
                static int shown = 0;
                if (stats_on && shown < 12) {
                    shown++;
                    fprintf(stderr, "[graph-veto] batched MUL_MAT %s: src0=%s type=%s ne=[%lld,%lld,%lld] M=%lld cublas=%d\n",
                            node->name, node->src[0]->name, ggml_type_name(node->src[0]->type),
                            (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[1], (long long) node->src[0]->ne[2],
                            (long long) node->src[1]->ne[1], (int) alpha_mul_mat_uses_cublas(node));
                }
            }
        }

        // Disable CUDA graphs for MUL_MAT_ID when src[0] is host-resident without the persistent
        // expert cache (the per-expert dispatch path, which needs cudaStreamSynchronize per expert).
        // When the persistent cache is active, src[0] in the split graph is input_cpy (GPU buffer),
        // not the original host tensor, and the scheduler's copy phase + flush() populate it fully
        // before graph launch, so graphs are safe.
        // ALPHA_DRE: a device-resolved expert tensor is also safe: its single-token MMVQ reads a device
        // table and the ids readback is a plain async copy; the table pointer is fixed per tensor.
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] && node->src[0]->buffer &&
            ggml_backend_buffer_is_host(node->src[0]->buffer)) {
            const void * db = nullptr; const int64_t * dt = nullptr;
            const bool dre = ggml_cuda_dre_enabled() && node->src[2] && node->src[2]->ne[1] <= 8 &&
                             ggml_cuda_dre_lookup(node->src[0], nullptr, nullptr, &db, &dt);
            if (!dre) {
                use_cuda_graph = false; alpha_graph_veto_note(__LINE__);
            }
        }

        // A prefill-sized MUL_MAT_ID (> 8 tokens) reads expert weights the scheduler stages per call (DRE
        // prefill streaming redirects the staging tensor to alternating buffers), so a captured graph would
        // replay against a stale pointer. MTP mode captures more shapes (512-token prefill chunks, the MTP
        // head's KV-fill graph), which makes this reachable. Such graphs gain nothing from capture anyway.
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[2] && node->src[2]->ne[1] > 8) {
            use_cuda_graph = false; alpha_graph_veto_note(__LINE__);
        }

        // The generic MUL_MAT_ID fallback sorts the ids on the host (stream sync), so never capture it.
        if (node->op == GGML_OP_MUL_MAT_ID && !alpha_mmid_has_device_path(node)) {
            use_cuda_graph = false; alpha_graph_veto_note(__LINE__);
            static const bool stats_mmid = alpha_env_on("ALPHA_GRAPH_STATS");
            static int shown_mmid = 0;
            if (stats_mmid && shown_mmid < 6) {
                shown_mmid++;
                fprintf(stderr, "[graph-veto] MUL_MAT_ID without device path: %s src0=%s type=%s src1=%s ne=[%lld,%lld,%lld] src1_ne=[%lld,%lld,%lld]\n",
                        node->name, node->src[0]->name, ggml_type_name(node->src[0]->type),
                        node->src[1] ? ggml_type_name(node->src[1]->type) : "-",
                        (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2],
                        node->src[1] ? (long long) node->src[1]->ne[0] : -1LL,
                        node->src[1] ? (long long) node->src[1]->ne[1] : -1LL,
                        node->src[1] ? (long long) node->src[1]->ne[2] : -1LL);
            }
        }

        // ik-quant (IQ1_KT et al.) experts decode via a per-expert dequant+GEMV loop
        // (ggml_cuda_mul_mat_id_ik_pool); the captured graph becomes huge and cudaGraphInstantiate
        // runs out of memory. The path does no host sync, so it runs fine uncaptured.
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] &&
            ggml_cuda_is_ik_type(node->src[0]->type)) {
            use_cuda_graph = false; alpha_graph_veto_note(__LINE__);
        }

        // Hybrid CPU/GPU: disable CUDA graphs when cache misses exist this token, since mul_mat_id
        // needs post-MMVQ CPU correction (stream sync, not capturable).
        if (node->op == GGML_OP_MUL_MAT_ID && g_pcache_has_misses) {
            use_cuda_graph = false; alpha_graph_veto_note(__LINE__);
        }
        // Offset table + CUDA graphs: decode passes (n_tokens==1, non-hybrid) are always
        // pool-direct: every used expert (hit or miss) resolves into the pool or scratch ring
        // and the table is re-uploaded host-side each pass. Kernel params are therefore uniform
        // across decode passes (vx=pool, d_offsets=d_buf) and graph replay is safe. Hybrid and
        // the panic fallback set g_pcache_has_misses, handled by the check above. Multi-token
        // passes never use tables (host gates on ncols_dst==1), so their graphs bake input_cpy
        // mode, which those passes keep populated via D2D/H2D.

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// ALPHA_GRAPH_SHAPE_KEY (default on): key the per-context graph cache by (first node, node count, output
// token count) instead of the first node alone. MTP alternates a 1-token draft pass with a W-token verify
// pass; both rebuild into the same ggml context and share the first-node key, so every pass would read as
// "properties changed" and reset warmup. Separate keys let each shape warm up and replay.
static const void * ggml_cuda_graph_get_key_impl(ggml_cgraph * cgraph) {
    // ALPHA_GRAPH_SHAPE_KEY: 0 = first-node key; 1 (default) = (first node, n_nodes, n_tokens); 2 = 1 plus
    // a hash of every node's op and shape. Level 2 separates graphs whose shape toggles at a fixed verify
    // width (a near-final GET_ROWS selecting 0/1 rows, and an early CONCAT whose allocation shifts with it)
    // into their own cached graphs. It does not speed up the GPU-bound multi-token verify pass, so it is
    // opt-in, and it needs the graph-cache cap (ALPHA_GRAPH_CACHE_MAX) to bound memory.
    static const int shape_key = []() {
        const char * e = getenv("ALPHA_GRAPH_SHAPE_KEY");
        return e ? atoi(e) : 1;
    }();
    if (shape_key <= 0 || cgraph->n_nodes == 0) {
        return cgraph->nodes[0];
    }
    // Level 3: content key, no struct addresses. When llama rebuilds the graph every step (hybrid models)
    // the node structs relocate, so any key containing cgraph->nodes[0] never repeats and nothing is ever
    // captured. Hash what is stable across rebuilds: op, shape, the node's data pointer and src[0]'s data
    // pointer (weights and allocator arena offsets), which also keeps the per-layer splits apart. Use with
    // ALPHA_GRAPH_SEMCMP=1 (field-wise compare).
    if (shape_key >= 3) {
        // ALPHA_GRAPH_OPDUMP=N: list every node (op, name, srcs) of the N-th split graph with > 30 nodes
        static const long long opdump_at = [] { const char * e = getenv("ALPHA_GRAPH_OPDUMP"); return e ? atoll(e) : 0LL; }();
        static long long opdump_n = 0;
        if (opdump_at > 0 && cgraph->n_nodes > 30 && ++opdump_n == opdump_at) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const ggml_tensor * t = cgraph->nodes[i];
                fprintf(stderr, "[opdump] %4d %-14s %-28s <- %s , %s\n", i, ggml_op_desc(t), t->name,
                        t->src[0] ? t->src[0]->name : "-", t->src[1] ? t->src[1]->name : "-");
            }
        }
        static const bool key_dbg = alpha_env_on("ALPHA_GRAPH_KEY_DBG");
        static long long key_dbg_n = 0;
        if (key_dbg && cgraph->n_nodes > 30 && (++key_dbg_n == 2000 || key_dbg_n == 2000 + 140 || key_dbg_n == 2000 + 280)) {
            std::string names;
            for (int i = 0; i < cgraph->n_nodes && i < 12; i++) { names += cgraph->nodes[i]->name; names += " | "; }
            fprintf(stderr, "[graph-key-dbg] call %lld n_nodes=%d first names: %s\n", key_dbg_n, cgraph->n_nodes, names.c_str());
        }
        // Names and types are required: the scheduler reuses one input_cpy address for every layer's expert
        // weights, so without them two layers' MoE splits would hash identically, and a captured graph bakes
        // the per-tensor offset-table pointer into its kernel args, so a shared key could replay layer A's
        // table for layer B. Node names carry the layer index (ffn_moe_gate-12) and are stable across rebuilds.
        uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t) cgraph->n_nodes;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * t = cgraph->nodes[i];
            for (const char * c = t->name; *c && *c != '#'; ++c) h = (h ^ (uint64_t) (unsigned char) *c) * 0x100000001b3ull;   // stop at the scheduler's '#copy' decoration
            h = (h ^ (uint64_t) t->type) * 0x100000001b3ull;
            // src types only: src names of scheduler input copies carry an alternating copy index
            // ("CUDA0#blk.0.ffn_down_exps.weight#0" / "#1"), which would make every call a new key.
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (!t->src[j]) continue;
                h = (h ^ (uint64_t) t->src[j]->type) * 0x100000001b3ull;
            }
            h = (h ^ (uint64_t) t->op) * 0x100000001b3ull;
            for (int d = 0; d < GGML_MAX_DIMS; d++) h = (h ^ (uint64_t) t->ne[d]) * 0x100000001b3ull;
            h = (h ^ (uint64_t) (uintptr_t) t->data) * 0x100000001b3ull;
            if (t->src[0]) h = (h ^ (uint64_t) (uintptr_t) t->src[0]->data) * 0x100000001b3ull;
        }
        return (const void *) (uintptr_t) (h | 1);
    }
    const ggml_tensor * last = cgraph->nodes[cgraph->n_nodes - 1];
    uintptr_t k = (uintptr_t) cgraph->nodes[0];
    k ^= (uintptr_t) cgraph->n_nodes * (uintptr_t) 0x9E3779B97F4A7C15ull;
    k ^= ((uintptr_t) last->ne[1] + 1) * (uintptr_t) 0xC2B2AE3D27D4EB4Full;
    // token count: the first node of a llama graph is the embedding gather [n_embd, n_tokens]; the
    // last node is often an argmax/view with ne[1]==1, which is why it alone did not separate widths
    k ^= ((uintptr_t) cgraph->nodes[0]->ne[1] + 7) * (uintptr_t) 0x94D049BB133111EBull;
    if (shape_key >= 2) {
        uint64_t h = 0xcbf29ce484222325ull;   // FNV-1a over (op, ne[0..3]) of every node
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * t = cgraph->nodes[i];
            h = (h ^ (uint64_t) t->op) * 0x100000001b3ull;
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                h = (h ^ (uint64_t) t->ne[d]) * 0x100000001b3ull;
            }
        }
        k ^= (uintptr_t) h;
    }
    return (const void *) k;
}

// ALPHA_GRAPH_KEY_CACHE (default on): the level-3 content key hashes every node, names included, which costs
// noticeable host time per call on large graphs while the GPU sits idle. A reused graph (same non-zero uid;
// llama's graph reuse hands back the same cgraph) has the same content, so its key is cached per uid.
static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    static const bool on = []() { const char * e = getenv("ALPHA_GRAPH_KEY_CACHE"); return !e || atoi(e) != 0; }();
    if (!on || cgraph->uid == 0) return ggml_cuda_graph_get_key_impl(cgraph);
    struct kent { size_t uid; int n_nodes; const void * key; };
    static kent cache[32] = {};
    static int next = 0;
    for (int i = 0; i < 32; i++) {
        if (cache[i].uid == (size_t) cgraph->uid && cache[i].n_nodes == cgraph->n_nodes) return cache[i].key;
    }
    const void * key = ggml_cuda_graph_get_key_impl(cgraph);
    cache[next] = { (size_t) cgraph->uid, cgraph->n_nodes, key };
    next = (next + 1) % 32;
    return key;
}

// Compare only the semantic node properties that affect kernel execution, not the whole ggml_tensor
// struct. llama's graph reuse fails on multi-seq hybrid decode (the recurrent input head/rs_z view-offset
// check moves via find_slot), so llama rebuilds the graph every step, relocating tensor and src[] struct
// addresses even when topology, data pointers and kernel params are identical. A whole-struct memcmp sees
// those relocations as a change and recaptures every step. Upstream compares field-wise as well.
// Enabled by ALPHA_GRAPH_SEMCMP=1.
static bool node_props_equal_semantic(const ggml_cuda_graph::node_properties & a,
                                      const ggml_cuda_graph::node_properties & b) {
    const ggml_tensor & na = a.node, & nb = b.node;
    if (na.op != nb.op || na.type != nb.type || na.data != nb.data) return false;
    for (int d = 0; d < GGML_MAX_DIMS; ++d)
        if (na.ne[d] != nb.ne[d] || na.nb[d] != nb.nb[d]) return false;
    if (memcmp(na.op_params, nb.op_params, sizeof(na.op_params)) != 0) return false;
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (a.node_src_data_ptrs[j] != b.node_src_data_ptrs[j]) return false;
        if (memcmp(a.node_src_ne[j], b.node_src_ne[j], sizeof(a.node_src_ne[j])) != 0) return false;
        if (memcmp(a.node_src_nb[j], b.node_src_nb[j], sizeof(a.node_src_nb[j])) != 0) return false;
    }
    return true;
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        static const bool semcmp = alpha_env_on("ALPHA_GRAPH_SEMCMP");
        const bool changed = semcmp ? !node_props_equal_semantic(graph->node_props[i], prop)
                                    : (memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0);
        // ALPHA_GRAPH_STATS=1: report which node broke a warmed-up graph's replay (first 24 events), so the
        // property drift can be traced to its op
        static const bool drift_all = alpha_env_on("ALPHA_GRAPH_DRIFT_ALL");   // also report drift that blocks WARMUP
        if (changed && !res && (graph->warmup_complete || (drift_all && graph->node_props.size() > 0 && graph->uid != 0))) {
            static const bool stats_on = alpha_env_on("ALPHA_GRAPH_STATS");
            static int shown = 0;
            if (stats_on && shown < 24) {
                shown++;
                const ggml_tensor & a = graph->node_props[i].node;
                const ggml_tensor & b = prop.node;
                const char * what = a.data != b.data ? "data" : (memcmp(a.ne, b.ne, sizeof(a.ne)) != 0 ? "ne" :
                                    (memcmp(a.nb, b.nb, sizeof(a.nb)) != 0 ? "nb" : (a.view_offs != b.view_offs ? "view_offs" :
                                    (memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0 ? "op_params" :
                                    (memcmp(a.src, b.src, sizeof(a.src)) != 0 ? "src-ptr" :
                                    (memcmp(graph->node_props[i].node_src_data_ptrs, prop.node_src_data_ptrs, sizeof(prop.node_src_data_ptrs)) != 0 ? "src-data" : "other"))))));
                int sj = -1;
                for (int j = 0; j < GGML_MAX_SRC; ++j)
                    if (graph->node_props[i].node_src_data_ptrs[j] != prop.node_src_data_ptrs[j]) { sj = j; break; }
                {   // which semantic field differs (what node_props_equal_semantic trips on)
                    char fld[160] = "";
                    if (a.type != b.type) strcat(fld, "type ");
                    if (memcmp(a.ne, b.ne, sizeof(a.ne)) != 0) strcat(fld, "ne ");
                    if (memcmp(a.nb, b.nb, sizeof(a.nb)) != 0) strcat(fld, "nb ");
                    if (memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0) strcat(fld, "op_params ");
                    for (int j = 0; j < GGML_MAX_SRC; ++j) {
                        char t[24];
                        if (memcmp(graph->node_props[i].node_src_ne[j], prop.node_src_ne[j], sizeof(prop.node_src_ne[j])) != 0) { snprintf(t, sizeof(t), "src%d.ne ", j); strcat(fld, t); }
                        if (memcmp(graph->node_props[i].node_src_nb[j], prop.node_src_nb[j], sizeof(prop.node_src_nb[j])) != 0) { snprintf(t, sizeof(t), "src%d.nb ", j); strcat(fld, t); }
                    }
                    fprintf(stderr, "[graph-drift-sem] %s: %s | src0.nb old=[%zu,%zu,%zu,%zu] new=[%zu,%zu,%zu,%zu] src0.ne=[%lld,%lld,%lld]\n", b.name, fld[0] ? fld : "(none: equal semantically)",
                            graph->node_props[i].node_src_nb[0][0], graph->node_props[i].node_src_nb[0][1], graph->node_props[i].node_src_nb[0][2], graph->node_props[i].node_src_nb[0][3],
                            prop.node_src_nb[0][0], prop.node_src_nb[0][1], prop.node_src_nb[0][2], prop.node_src_nb[0][3],
                            (long long) prop.node_src_ne[0][0], (long long) prop.node_src_ne[0][1], (long long) prop.node_src_ne[0][2]);
                }
                fprintf(stderr, "[graph-drift] node %d %s %s: %s changed (ne=[%lld,%lld,%lld,%lld] data %p -> %p) src-data[%d] %p -> %p (%s)\n",
                        i, ggml_op_name(b.op), b.name, what,
                        (long long) b.ne[0], (long long) b.ne[1], (long long) b.ne[2], (long long) b.ne[3], a.data, b.data,
                        sj, sj >= 0 ? graph->node_props[i].node_src_data_ptrs[sj] : nullptr,
                        sj >= 0 ? prop.node_src_data_ptrs[sj] : nullptr,
                        (sj >= 0 && b.src[sj]) ? b.src[sj]->name : "-");
            }
        }
        if (res || changed) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        if (stat != cudaSuccess) {
            // name the node the update rejected, then rebuild the executable from the captured graph (which is valid
            // whenever instantiation succeeds) instead of aborting the server
            (void) cudaGetLastError();
#if CUDART_VERSION >= 12000
            cudaGraphNodeType nt = cudaGraphNodeTypeEmpty;
            if (result_info.errorNode) (void) cudaGraphNodeGetType(result_info.errorNode, &nt);
            GGML_LOG_ERROR("%s: cudaGraphExecUpdate: %s (result %d, error node type %d)\n", __func__, cudaGetErrorString(stat),
                           (int) result_info.result, (int) nt);
            if (result_info.errorNode && nt == cudaGraphNodeTypeMemcpy) {
                cudaMemcpy3DParms mp = {};
                if (cudaGraphMemcpyNodeGetParams(result_info.errorNode, &mp) == cudaSuccess) {
                    cudaPointerAttributes as = {}, ad = {};
                    (void) cudaPointerGetAttributes(&as, mp.srcPtr.ptr);
                    (void) cudaPointerGetAttributes(&ad, mp.dstPtr.ptr);
                    GGML_LOG_ERROR("%s:   memcpy node: src %p (memory type %d) -> dst %p (memory type %d), kind %d, %zu bytes\n", __func__,
                                   mp.srcPtr.ptr, (int) as.type, mp.dstPtr.ptr, (int) ad.type, (int) mp.kind, mp.extent.width);
                }
                (void) cudaGetLastError();
            }
#else
            GGML_LOG_ERROR("%s: cudaGraphExecUpdate: %s\n", __func__, cudaGetErrorString(stat));
#endif
            CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
            graph->instance = nullptr;
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        if (ggml_get_unary_op(nodes[node_idx]) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        args.sigmoid = true;
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
int ggml_cuda_topk_fuse_maxrows();   // topk-moe.cu

static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // exception for topk-moe, as each row is read entirely before writing
    if (ggml_nrows(cgraph->nodes[node_idx]) == 1 && is_topk_moe) {
        return true;
    }
    // ALPHA_TOPK_FUSE_MAXROWS: small multi-row topk-moe reads its logits from a scratch copy, so an
    // overlap is harmless. Without this, 1-token decode is fused and a 2-token MTP verify is not,
    // and the two paths can disagree on logits, so MTP greedy text would differ from plain greedy text.
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= ggml_cuda_topk_fuse_maxrows()) {
        return true;
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}


// CONCAT(dim 0) -> VIEW(last k columns) -> VIEW(state slot) -> CPY(view -> slot): the recurrent conv-state
// roll. Fused into one launch by ggml_cuda_op_concat_shift; the concat result stays a real output (ssm_conv reads
// it next). ALPHA_FUSE_CONCAT_SHIFT=0 disables. Decode-sized only (T <= 16): the thread-per-channel kernel is
// uncoalesced for long T and the generic path is fine there.
// ALPHA_MOE_MERGE: MUL(experts [n_embd,k,T], weights [1,k,T]) -> k VIEWs (one per expert slot) -> (k-1) ADDs
// summing them in slot order -> up to 4 trailing ADDs of [n_embd,T] addends (shared expert, residual). Returns
// the number of graph nodes the fused kernel replaces (0 = no match). ALPHA_MOE_MERGE=0 disables.
static int ggml_cuda_match_moe_merge(const struct ggml_cgraph * cgraph, int i,
                                     const ggml_tensor ** addends, int * n_add) {
    static const bool on = []() {
        const char * e = getenv("ALPHA_MOE_MERGE");
        return e ? atoi(e) != 0 : true;
    }();
    if (!on) {
        return 0;
    }
    const ggml_tensor * mul = cgraph->nodes[i];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || (mul->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * e = mul->src[0];
    const ggml_tensor * w = mul->src[1];
    if (!e || !w || e->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32) {
        return 0;
    }
    const int64_t n_embd = e->ne[0];
    const int64_t k      = e->ne[1];
    const int64_t T      = e->ne[2];
    if (k < 2 || k > 16 || e->ne[3] != 1 || w->ne[0] != 1 || w->ne[1] != k || w->ne[2] != T || w->ne[3] != 1) {
        return 0;
    }
    if (!ggml_are_same_shape(mul, e) || e->nb[0] % sizeof(float) || w->nb[1] % sizeof(float)) {
        return 0;
    }
    if (i + k + (k - 1) >= cgraph->n_nodes || ggml_node_get_use_count(cgraph, i) != (int32_t) k) {
        return 0;
    }
    // k views, slot j at offset j*nb[1], shaped [n_embd, T] with row stride nb[2]
    for (int64_t j = 0; j < k; ++j) {
        const ggml_tensor * v = cgraph->nodes[i + 1 + j];
        if (v->op != GGML_OP_VIEW || v->view_src != mul || v->ne[0] != n_embd || v->ne[1] != T || v->ne[2] != 1 || v->ne[3] != 1 ||
            v->view_offs != (size_t) j * mul->nb[1] || v->nb[0] != mul->nb[0] || v->nb[1] != mul->nb[2] ||
            ggml_node_get_use_count(cgraph, i + 1 + j) != 1 || (v->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }
    // (k-1) adds: add_1 = v0 + v1, add_j = add_{j-1} + v_j
    int idx = (int) (i + 1 + k);
    const ggml_tensor * prev = nullptr;
    for (int64_t j = 1; j < k; ++j, ++idx) {
        const ggml_tensor * a = cgraph->nodes[idx];
        const ggml_tensor * lhs = j == 1 ? cgraph->nodes[i + 1] : prev;
        if (a->op != GGML_OP_ADD || a->type != GGML_TYPE_F32 || a->src[0] != lhs || a->src[1] != cgraph->nodes[i + 1 + j] ||
            a->ne[0] != n_embd || a->ne[1] != T || a->ne[2] != 1 || a->ne[3] != 1) {
            return 0;
        }
        prev = a;
    }
    // the sum so far is an intermediate unless nothing else can be appended
    int last = idx - 1;
    int na = 0;
    while (idx < cgraph->n_nodes && na < 4) {
        const ggml_tensor * a = cgraph->nodes[idx];
        if (a->op != GGML_OP_ADD || a->type != GGML_TYPE_F32 || a->src[0] != prev || !a->src[1] ||
            a->src[1]->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, a->src[1]) || !ggml_are_same_shape(a, prev) ||
            ggml_node_get_use_count(cgraph, idx - 1) != 1 || (prev->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            break;
        }
        addends[na++] = a->src[1];
        prev = a;
        last = idx;
        ++idx;
    }
    // every node before the last must feed only the chain
    for (int j = (int) (i + 1 + k); j < last; ++j) {
        if (ggml_node_get_use_count(cgraph, j) != 1 || (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }
    const ggml_tensor * dst = cgraph->nodes[last];
    if (!ggml_is_contiguous(dst) || dst->ne[0] != n_embd || dst->ne[1] != T) {
        return 0;
    }
    // the kernel reads e, w and the addends while writing dst at different offsets: no overlap allowed
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    if (overlap(dst, e) || overlap(dst, w)) {
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "[moe-merge] declined: dst overlaps experts/weights (in-place allocation)\n"); }
        return 0;
    }
    for (int j = 0; j < na; ++j) {
        if (overlap(dst, addends[j]) && addends[j]->data != dst->data) {
            return 0;
        }
        if (addends[j]->data == dst->data && (addends[j]->nb[0] != dst->nb[0] || addends[j]->nb[1] != dst->nb[1])) {
            return 0;
        }
    }
    *n_add = na;
    static bool shown = false;
    if (!shown) { shown = true; fprintf(stderr, "[moe-merge] fused: k=%d n_add=%d T=%d nodes=%d\n", (int) k, na, (int) T, last - i + 1); }
    return last - i + 1;
}

// ALPHA_MOE_DOWN9: MUL_MAT_ID(routed down) followed by the moe-merge chain: one kernel does the
// down projection, the routing-weighted sum and the trailing addends, and the per-slot
// [n_embd, k, T] intermediate never exists. Returns the node count replaced (0 = no match).
struct alpha_moe_down9_args {
    const ggml_tensor * down_exps;
    const ggml_tensor * act_routed;
    const ggml_tensor * ids;
    const ggml_tensor * weights;
    // the shared expert as a (k+1)-th path: set only at ALPHA_MOE_DOWN9 >= 2
    const ggml_tensor * down_shexp;
    const ggml_tensor * act_shexp;
    const ggml_tensor * score;
    const ggml_tensor * addends[4];
    int                 n_add;
    ggml_tensor *       dst;
};

// ALPHA_MOE_DOWN9: 0 off, 1 = fuse routed down + merge, >= 2 = also fold the shared expert in as the
// (k+1)-th warp, which additionally removes its down projection, its sigmoid-multiply and its
// activation-quantize prologue.
//
// FIXME: level 2 produces wrong output and must not be made a default; level 1 is correct. A stable
// output hash across runs does not catch it, since that only proves determinism. Suspected cause is the
// lifetime hazard checked below (deferring a read of act_shexp/score past their last graph-order consumer).
static int alpha_moe_down9_level() {
    static const int v = []() {
        const char * e = getenv("ALPHA_MOE_DOWN9");
        return e ? atoi(e) : 0;
    }();
    return v;
}

static int ggml_cuda_match_moe_merge(const struct ggml_cgraph * cgraph, int i,
                                     const ggml_tensor ** addends, int * n_add);

static int ggml_cuda_match_moe_down9(const struct ggml_cgraph * cgraph, int i, alpha_moe_down9_args * a) {
    if (alpha_moe_down9_level() < 1 || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mmid = cgraph->nodes[i];
    if (mmid->op != GGML_OP_MUL_MAT_ID || (mmid->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    if (ggml_node_get_use_count(cgraph, i) != 1) {
        return 0;
    }
    const ggml_tensor * mul = cgraph->nodes[i + 1];
    if (mul->op != GGML_OP_MUL || mul->src[0] != mmid) {
        return 0;
    }
    const ggml_tensor * mm_add[4] = { nullptr, nullptr, nullptr, nullptr };
    int mm_nadd = 0;
    const int mm_n = ggml_cuda_match_moe_merge(cgraph, i + 1, mm_add, &mm_nadd);
    if (mm_n <= 0) {
        return 0;
    }
    a->down_exps  = mmid->src[0];
    a->act_routed = mmid->src[1];
    a->ids        = mmid->src[2];
    a->weights    = mul->src[1];
    a->down_shexp = nullptr;
    a->act_shexp  = nullptr;
    a->score      = nullptr;
    a->n_add      = mm_nadd;
    for (int j = 0; j < 4; ++j) {
        a->addends[j] = j < mm_nadd ? mm_add[j] : nullptr;
    }
    a->dst = cgraph->nodes[i + 1 + mm_n - 1];

    if (!ggml_cuda_moe_down9_supported(a->down_exps, a->act_routed, a->ids, a->weights,
                                       a->down_shexp, a->act_shexp, a->score,
                                       a->addends, a->n_add, a->dst)) {
        return 0;
    }
    // the kernel reads the activations, weights and addends while writing dst: no aliasing with dst,
    // except an addend that is dst (an in-place residual), which is read before the store
    auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * x0 = (const char *) x->data; const char * x1 = x0 + ggml_nbytes(x);
        const char * y0 = (const char *) y->data; const char * y1 = y0 + ggml_nbytes(y);
        return x0 < y1 && y0 < x1;
    };
    if (overlap(a->dst, a->act_routed) || overlap(a->dst, a->weights)) {
        return 0;
    }
    for (int j = 0; j < a->n_add; ++j) {
        if (overlap(a->dst, a->addends[j]) && a->addends[j]->data != a->dst->data) {
            return 0;
        }
        if (a->addends[j]->data == a->dst->data &&
            (a->addends[j]->nb[0] != a->dst->nb[0] || a->addends[j]->nb[1] != a->dst->nb[1])) {
            return 0;
        }
    }
    return 1 + mm_n;
}

// ALPHA_MOE_DOWN9 >= 2: the shared expert's tail sits earlier in the graph than the routed down, so it
// cannot be absorbed by a look-ahead from the routed side. Instead this matches at the shared expert's
// own down projection: MUL_MAT(down_shexp) -> (views) -> UNARY(SIGMOID) -> MUL, then finds the routed
// MUL_MAT_ID that consumes the result as a merge addend. When it matches, the three real nodes are
// skipped and the later fused kernel recomputes them as its (k+1)-th path.
//
// Safety: the decision is taken once, here, and only if the full routed group also matches, using the
// same deterministic matcher that will run again at the MUL_MAT_ID. If it could match here and fail
// there, the skipped nodes would silently never be computed.
static int d9s_no(int code) {
    static bool seen[16] = {};
    if (alpha_moe_down9_level() >= 2 && code < 16 && !seen[code]) {
        seen[code] = true;
        GGML_LOG_INFO("[moe-down9-shexp] declined: code %d\n", code);
    }
    return 0;
}

static int ggml_cuda_match_moe_down9_shexp(const struct ggml_cgraph * cgraph, int i,
                                           alpha_moe_down9_args * a, int * mmid_idx,
                                           int * skip_sig, int * skip_mul) {
    if (alpha_moe_down9_level() < 2 || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("[moe-down9] LEVEL 2 IS UNVERIFIED AND HAS PRODUCED WRONG OUTPUT "
                          "(2026-09-18, 3090). Diagnostic use only - not a shippable result.\n");
        }
    }
    const ggml_tensor * shd = cgraph->nodes[i];
    if (shd->op != GGML_OP_MUL_MAT || (shd->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;   // not logged: every non-matmul node lands here
    }
    if (alpha_moe_down9_level() >= 3 && strstr(shd->name, "shexp")) {
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            for (int d = i - 2; d < cgraph->n_nodes && d < i + 10; ++d) {
                if (d < 0) continue;
                const ggml_tensor * x = cgraph->nodes[d];
                GGML_LOG_INFO("[d9-dump] %c%3d %-14s %-28s uc=%d ne=[%lld,%lld] src0=%s src1=%s\n",
                              d == i ? '*' : ' ', d, ggml_op_name(x->op), x->name,
                              ggml_node_get_use_count(cgraph, d),
                              (long long) x->ne[0], (long long) x->ne[1],
                              x->src[0] ? x->src[0]->name : "-", x->src[1] ? x->src[1]->name : "-");
            }
        }
    }
    if (ggml_node_get_use_count(cgraph, i) != 1) {
        return d9s_no(1);
    }
    // Find the MUL that gates this shared-expert output by a sigmoid. Adjacency is not assumed: the
    // gate's own projection sits between them, and whether it is a MUL_MAT or a view of the fused
    // router output depends on ALPHA_FUSE_PROJ. Bounded search, and nothing in between may consume
    // shd (if anything else reads it, dropping the node would change that consumer's input).
    int j_gat = -1;
    for (int m = i + 1; m < cgraph->n_nodes && m < i + 12; ++m) {
        const ggml_tensor * x = cgraph->nodes[m];
        if (x->op == GGML_OP_MUL && (x->src[0] == shd || x->src[1] == shd)) {
            j_gat = m;
            break;
        }
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            if (x->src[q] == shd) {
                return d9s_no(2);   // consumed by something other than the gating multiply
            }
        }
    }
    if (j_gat < 0) {
        return d9s_no(3);
    }
    const ggml_tensor * gat = cgraph->nodes[j_gat];
    const ggml_tensor * sig = gat->src[0] == shd ? gat->src[1] : gat->src[0];
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) {
        return d9s_no(4);
    }
    // the sigmoid must be its own node in this graph, so that skipping it is well defined
    int j_sig = -1;
    for (int m = i + 1; m < j_gat; ++m) {
        if (cgraph->nodes[m] == sig) {
            j_sig = m;
            break;
        }
    }
    if (j_sig < 0) {
        return d9s_no(5);
    }
    if (ggml_node_get_use_count(cgraph, j_sig) != 1 || ggml_node_get_use_count(cgraph, j_gat) != 1) {
        return d9s_no(6);
    }
    if ((sig->flags & GGML_TENSOR_FLAG_OUTPUT) || (gat->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return d9s_no(10);
    }

    // find the routed down projection that consumes the gated output, and match its whole group
    for (int m = j_gat + 1; m < cgraph->n_nodes && m < j_gat + 96; ++m) {
        if (cgraph->nodes[m]->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        alpha_moe_down9_args t{};
        const int n = ggml_cuda_match_moe_down9(cgraph, m, &t);
        if (n <= 0) {
            // Not a bail-out: the first MUL_MAT_ID after the shared tail is the routed gate/up pair,
            // which never matches this pattern. Keep scanning for the down projection.
            continue;
        }
        int keep = 0;
        bool found_gate = false;
        const ggml_tensor * kept[4] = { nullptr, nullptr, nullptr, nullptr };
        for (int q = 0; q < t.n_add; ++q) {
            if (t.addends[q] == gat) {
                found_gate = true;
            } else {
                kept[keep++] = t.addends[q];
            }
        }
        if (!found_gate) {
            return d9s_no(7);
        }
        *a = t;
        a->down_shexp = shd->src[0];
        a->act_shexp  = shd->src[1];
        a->score      = sig->src[0];
        a->n_add      = keep;
        for (int q = 0; q < 4; ++q) {
            a->addends[q] = q < keep ? kept[q] : nullptr;
        }
        if (!ggml_cuda_moe_down9_supported(a->down_exps, a->act_routed, a->ids, a->weights,
                                           a->down_shexp, a->act_shexp, a->score,
                                           a->addends, a->n_add, a->dst)) {
            {
                static bool once = false;
                if (!once) {
                    once = true;
                    GGML_LOG_INFO("[moe-down9-shexp] unsupported: routed n_ff=%lld n_embd=%lld k=%lld T=%lld "
                                  "type=%d | shared n_ff=%lld n_embd=%lld type=%d act=[%lld,%lld] score=[%lld,%lld] n_add=%d\n",
                                  (long long) a->down_exps->ne[0], (long long) a->down_exps->ne[1],
                                  (long long) a->act_routed->ne[1], (long long) a->act_routed->ne[2],
                                  (int) a->down_exps->type,
                                  (long long) a->down_shexp->ne[0], (long long) a->down_shexp->ne[1],
                                  (int) a->down_shexp->type,
                                  (long long) a->act_shexp->ne[0], (long long) a->act_shexp->ne[1],
                                  (long long) a->score->ne[0], (long long) a->score->ne[1], a->n_add);
                }
            }
            return d9s_no(8);
        }
        // Lifetime: act_shexp and score are read by the fused kernel at node m, which is after their
        // last consumer in graph order, so the allocator may have reused their storage in between.
        {
            auto ov = [](const ggml_tensor * x, const ggml_tensor * y) {
                if (!x || !y || !x->data || !y->data) return false;
                const char * x0 = (const char *) x->data; const char * x1 = x0 + ggml_nbytes(x);
                const char * y0 = (const char *) y->data; const char * y1 = y0 + ggml_nbytes(y);
                return x0 < y1 && y0 < x1;
            };
            for (int p = i + 1; p < m; ++p) {
                const ggml_tensor * mid = cgraph->nodes[p];
                if (mid == sig || mid == gat) {
                    continue;           // skipped, never written
                }
                if (mid == a->score || mid == a->act_shexp) {
                    continue;           // the producer writing its own output is expected, not a hazard
                }
                if (ov(mid, a->act_shexp) || ov(mid, a->score)) {
                    static bool once = false;
                    if (!once) {
                        once = true;
                        GGML_LOG_INFO("[moe-down9-shexp] LIFETIME: node %d (%s, %s) overwrites %s "
                                      "before the fused read at node %d -- declining\n",
                                      p, ggml_op_name(mid->op), mid->name,
                                      ov(mid, a->act_shexp) ? "act_shexp" : "score", m);
                    }
                    return d9s_no(11);
                }
            }
        }
        *mmid_idx  = m;
        *skip_sig  = j_sig;   // the gate's own projection between them still runs
        *skip_mul  = j_gat;
        return n;
    }
    return d9s_no(9);   // shared tail matched but no fusable routed down within the scan window
}

static bool ggml_cuda_can_fuse_concat_shift(const struct ggml_cgraph * cgraph, int i) {
    static const bool on = []() {
        const char * e = getenv("ALPHA_FUSE_CONCAT_SHIFT");
        return e ? atoi(e) != 0 : true;
    }();
    if (!on || i + 3 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * cc = cgraph->nodes[i];
    const ggml_tensor * v1 = cgraph->nodes[i + 1];
    const ggml_tensor * v2 = cgraph->nodes[i + 2];
    const ggml_tensor * cp = cgraph->nodes[i + 3];
    if (cc->op != GGML_OP_CONCAT || v1->op != GGML_OP_VIEW || v2->op != GGML_OP_VIEW || cp->op != GGML_OP_CPY) {
        return false;
    }
    if (((const int32_t *) cc->op_params)[0] != 0) {
        return false;
    }
    const ggml_tensor * s = cc->src[0];
    const ggml_tensor * x = cc->src[1];
    if (!s || !x || cc->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || cp->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t k = s->ne[0], T = x->ne[0], C = s->ne[1], B = s->ne[2];
    if (k < 1 || k > 8 || T < 1 || T > 16 || s->ne[3] != 1 || x->ne[3] != 1 || cc->ne[3] != 1) {
        return false;
    }
    if (x->ne[1] != C || x->ne[2] != B || cc->ne[0] != k + T || cc->ne[1] != C || cc->ne[2] != B) {
        return false;
    }
    if (s->nb[0] != sizeof(float) || x->nb[0] % sizeof(float) != 0 || !ggml_is_contiguous(cc)) {
        return false;
    }
    // the copy takes the last k columns of the concat result ...
    if (cp->src[0] != v1 || v1->view_src != cc || v1->ne[0] != k || v1->ne[1] != C || v1->ne[2] != B || v1->ne[3] != 1 ||
        v1->view_offs != (size_t) T * sizeof(float) || v1->nb[0] != sizeof(float) || v1->nb[1] != cc->nb[1] || v1->nb[2] != cc->nb[2]) {
        return false;
    }
    // ... into a contiguous k*C*B slot (the CPY node is that view)
    if (cp->src[1] != v2 || v2->type != GGML_TYPE_F32 || !ggml_is_contiguous(v2) || ggml_nelements(v2) != k * C * B || cp->data != v2->data) {
        return false;
    }
    // x and the concat result must not alias the slot (s may: the kernel reads its state before writing)
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    if (!x->data || !cc->data || !v2->data || overlap(x, v2) || overlap(cc, v2)) {
        return false;
    }
    return true;
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            return true;
        }
    }

    // ALPHA_UNARY_MUL_BCAST: per-row scalar gate. The shared expert's output [n_embd, T] is multiplied by one
    // sigmoid value per token, so the two MUL operands have different shapes and the generic gate below (which
    // requires every fused node to share a shape, ggml-impl.h) rejects the pair before any pattern branch is
    // reached. The fused kernel broadcasts the unary operand along the row and computes op(x[row]) * g[i],
    // the same arithmetic as the unfused pair, so results are bit-identical. ALPHA_UNARY_MUL_BCAST=0 disables.
    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1 && node_idx + 1 < cgraph->n_nodes) {
        static const bool bcast_on = []() {
            const char * e = getenv("ALPHA_UNARY_MUL_BCAST");
            return !e || atoi(e) != 0;
        }();
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];
        if (bcast_on && unary->op == GGML_OP_UNARY && mul->op == GGML_OP_MUL &&
            (unary->flags & GGML_TENSOR_FLAG_COMPUTE) && (mul->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            ggml_get_unary_op(unary) == unary_ops.begin()[0] &&
            (ggml_get_unary_op(unary) == GGML_UNARY_OP_SILU || ggml_get_unary_op(unary) == GGML_UNARY_OP_SIGMOID ||
             ggml_get_unary_op(unary) == GGML_UNARY_OP_SOFTPLUS) &&
            (mul->src[0] == unary || mul->src[1] == unary) &&
            ggml_node_has_n_uses(cgraph, node_idx, 1) &&
            !ggml_are_same_shape(unary, mul) && unary->ne[0] == 1 && unary->src[0]->ne[0] == 1) {
            const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
            if (unary->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && other->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(other, mul) && ggml_is_contiguous(mul) &&
                ggml_is_contiguous_1(other) && ggml_is_contiguous_1(unary->src[0]) &&
                other->ne[1] == unary->ne[1] && other->ne[2] == unary->ne[2] && other->ne[3] == unary->ne[3]) {
                return true;
            }
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// Event-based per-op timer (ALPHA_OP_PROF=1, default off). Event-based on purpose: a
// cudaDeviceSynchronize per node would serialize the pipelining being measured. One stream sync per
// graph, then elapsed times are read back off already-completed events.
// Requires GGML_CUDA_DISABLE_GRAPHS=1: events cannot be recorded inside a graph capture, so the
// profiler disables itself while use_cuda_graph is set.
struct alpha_op_prof {
    std::vector<cudaEvent_t> ev_beg, ev_end;
    std::vector<int>         ev_op;
    size_t    n      = 0;
    long long passes = 0;
    double    total_us = 0.0;
    std::map<int, std::pair<double, long long>> acc;   // op -> {total_us, calls}
    // ALPHA_OP_PROF_BYNAME=1: also key MUL_MAT / MUL_MAT_ID time by weight name (layer index stripped),
    // weight type and batch rows, so a dense prefill total can be attributed to projections.
    std::vector<std::string> ev_key;
    std::map<std::string, std::pair<double, long long>> acc_name;
    // host time spent inside ggml_cuda_compute_forward per node (launch cost + any blocking sync in the op)
    std::vector<int64_t> ev_host_us;
    int64_t host_t0 = 0;
    double  host_total_us = 0.0;
    double  span_us = 0.0;
    bool    iter_open = false;
    std::map<std::string, std::pair<double, long long>> acc_host;   // op or op+weight key -> {host_us, calls}
};
static bool alpha_op_prof_byname() {
    static const bool v = []() { const char * e = getenv("ALPHA_OP_PROF_BYNAME"); return e && atoi(e) != 0; }();
    return v;
}
static std::string alpha_op_prof_key(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) return std::string();
    const ggml_tensor * w = node->src[0];
    std::string nm = w && w->name[0] ? w->name : "?";
    if (nm.compare(0, 4, "blk.") == 0) {
        size_t d = nm.find('.', 4);
        if (d != std::string::npos) nm = "blk.*" + nm.substr(d);
    }
    char buf[192];
    const ggml_tensor * x = node->src[1];
    snprintf(buf, sizeof(buf), "%s %s %s w=[%lld,%lld] x=[%lld,%lld,%lld,%lld]%s", ggml_op_name(node->op), nm.c_str(),
             w ? ggml_type_name(w->type) : "?", (long long) (w ? w->ne[0] : 0), (long long) (w ? w->ne[1] : 0),
             (long long) (x ? x->ne[0] : 0), (long long) (x ? x->ne[1] : 0), (long long) (x ? x->ne[2] : 0),
             (long long) (x ? x->ne[3] : 0), x && !ggml_is_contiguous(x) ? " noncontig" : "");
    return buf;
}
static alpha_op_prof g_op_prof;
// ALPHA_OP_PROF_ITER=1: time every loop iteration of the evaluate loop (begin at its start, end at the next
// iteration's start) instead of each ggml_cuda_compute_forward call, so fused groups -- which return
// before the per-op timer -- are attributed too. A fused iteration is labelled "<first op>+F".
static const int ALPHA_OP_FUSED_BIT = 0x10000;
static bool alpha_op_prof_iter() {
    static const bool v = []() { const char * e = getenv("ALPHA_OP_PROF_ITER"); return e && atoi(e) != 0; }();
    return v;
}
static std::string alpha_op_label(int op) {
    std::string n = ggml_op_name((ggml_op) (op & (ALPHA_OP_FUSED_BIT - 1)));
    if (op & ALPHA_OP_FUSED_BIT) n += "+F";
    return n;
}
static bool alpha_op_prof_on() {
    static const bool v = []() { const char * e = getenv("ALPHA_OP_PROF"); return e && atoi(e) != 0; }();
    return v;
}
static void alpha_op_prof_begin(cudaStream_t st, const ggml_tensor * node) {
    alpha_op_prof & p = g_op_prof;
    if (p.n >= p.ev_beg.size()) {
        cudaEvent_t a = nullptr, b = nullptr;
        if (cudaEventCreate(&a) != cudaSuccess) return;
        if (cudaEventCreate(&b) != cudaSuccess) { cudaEventDestroy(a); return; }
        p.ev_beg.push_back(a); p.ev_end.push_back(b); p.ev_op.push_back(0); p.ev_key.emplace_back();
        p.ev_host_us.push_back(0);
    }
    p.ev_op[p.n] = (int) node->op;
    if (alpha_op_prof_byname()) p.ev_key[p.n] = alpha_op_prof_key(node);
    // ALPHA_OP_PROF_HOSTSYNC=1: drain the stream before timing the op's host side, so "host" is pure launch
    // cost and not back-pressure from a full kernel queue (diagnostic only; serializes the pass).
    static const bool hostsync = alpha_env_on("ALPHA_OP_PROF_HOSTSYNC");
    if (hostsync) cudaStreamSynchronize(st);
    cudaEventRecord(p.ev_beg[p.n], st);
    p.host_t0 = ggml_time_us();
}
static void alpha_op_prof_mark_fused() {
    alpha_op_prof & p = g_op_prof;
    if (p.n < p.ev_op.size()) p.ev_op[p.n] |= ALPHA_OP_FUSED_BIT;
}
static void alpha_op_prof_end(cudaStream_t st) {
    alpha_op_prof & p = g_op_prof;
    if (p.n < p.ev_end.size()) {
        p.ev_host_us[p.n] = ggml_time_us() - p.host_t0;
        cudaEventRecord(p.ev_end[p.n], st);
        p.n++;
    }
}
static void alpha_op_prof_flush(cudaStream_t st) {
    alpha_op_prof & p = g_op_prof;
    if (p.iter_open && p.n < p.ev_end.size()) {
        p.iter_open = false;
        // close the iteration record left open by the evaluate loop (its last node)
        p.ev_host_us[p.n] = ggml_time_us() - p.host_t0;
        cudaEventRecord(p.ev_end[p.n], st);
        p.n++;
    }
    if (p.n == 0) return;
    cudaStreamSynchronize(st);
    {
        // whole-pass GPU span, first profiled op start -> last profiled op end: includes the fused ops the
        // per-op events skip and any GPU idle between launches
        float span_ms = 0.0f;
        if (cudaEventElapsedTime(&span_ms, p.ev_beg[0], p.ev_end[p.n - 1]) == cudaSuccess) p.span_us += span_ms * 1000.0;
    }
    for (size_t i = 0; i < p.n; i++) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, p.ev_beg[i], p.ev_end[i]) != cudaSuccess) continue;
        std::pair<double, long long> & e = p.acc[p.ev_op[i]];
        e.first += ms * 1000.0; e.second++;
        p.total_us += ms * 1000.0;
        if (alpha_op_prof_byname() && !p.ev_key[i].empty()) {
            std::pair<double, long long> & k = p.acc_name[p.ev_key[i]];
            k.first += ms * 1000.0; k.second++;
        }
        {
            const std::string hk = p.ev_key[i].empty() ? alpha_op_label(p.ev_op[i]) : p.ev_key[i];
            std::pair<double, long long> & h = p.acc_host[hk];
            h.first += (double) p.ev_host_us[i]; h.second++;
            p.host_total_us += (double) p.ev_host_us[i];
        }
    }
    p.n = 0;
    static const int prof_every = []() {
        const char * e = getenv("ALPHA_OP_PROF_EVERY");
        const int v = e ? atoi(e) : 256;
        return v < 1 ? 1 : v;
    }();
    if ((++p.passes % prof_every) == 0) {
        std::vector<std::pair<double,int>> v;
        for (std::map<int, std::pair<double,long long>>::iterator it = p.acc.begin(); it != p.acc.end(); ++it) {
            v.push_back(std::make_pair(it->second.first, it->first));
        }
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[op-prof] passes=%lld  GPU busy=%.3f ms/pass  (top ops: ms/pass, %% of busy, calls/pass)\n",
                (long long) p.passes, p.total_us / (double) p.passes / 1000.0);
        for (size_t i = 0; i < v.size() && i < 12; i++) {
            std::pair<double,long long> & e = p.acc[v[i].second];
            fprintf(stderr, "[op-prof]   %-20s %8.3f  %5.1f%%  %7.1f\n",
                    alpha_op_label(v[i].second).c_str(),
                    e.first / (double) p.passes / 1000.0,
                    100.0 * e.first / p.total_us,
                    (double) e.second / (double) p.passes);
        }
        // ALPHA_OP_PROF_RESET=1 clears the accumulator after each report, so with
        // ALPHA_OP_PROF_EVERY=1 every line is exactly one graph_compute. Without it the running
        // average folds prefill passes into the decode numbers.
        if (alpha_op_prof_byname()) {
            std::vector<std::pair<double, std::string>> vn;
            for (const auto & kv : p.acc_name) vn.push_back(std::make_pair(kv.second.first, kv.first));
            std::sort(vn.rbegin(), vn.rend());
            for (size_t i = 0; i < vn.size() && i < 40; i++) {
                const std::pair<double, long long> & e = p.acc_name[vn[i].second];
                fprintf(stderr, "[op-prof-name] %8.3f ms/pass %5.1f%% %6.1f calls/pass  %s\n",
                        e.first / (double) p.passes / 1000.0, 100.0 * e.first / p.total_us,
                        (double) e.second / (double) p.passes, vn[i].second.c_str());
            }
        }
        {
            std::vector<std::pair<double, std::string>> vh;
            for (const auto & kv : p.acc_host) vh.push_back(std::make_pair(kv.second.first, kv.first));
            std::sort(vh.rbegin(), vh.rend());
            fprintf(stderr, "[op-prof-host] host ms/pass inside compute_forward=%.3f  gpu span first->last op=%.3f (top: ms/pass, calls/pass)\n",
                    p.host_total_us / (double) p.passes / 1000.0, p.span_us / (double) p.passes / 1000.0);
            for (size_t i = 0; i < vh.size() && i < 10; i++) {
                const std::pair<double, long long> & e = p.acc_host[vh[i].second];
                fprintf(stderr, "[op-prof-host] %8.3f ms/pass %6.1f calls/pass  %s\n",
                        e.first / (double) p.passes / 1000.0, (double) e.second / (double) p.passes, vh[i].second.c_str());
            }
        }
        static const bool prof_reset = alpha_env_on("ALPHA_OP_PROF_RESET");
        if (prof_reset) { p.acc.clear(); p.acc_name.clear(); p.acc_host.clear(); p.host_total_us = 0.0; p.span_us = 0.0; p.total_us = 0.0; p.passes = 0; }
    }
}

// ALPHA_Q8_STAGE: decide whether the tensor produced at node `idx` is worth quantizing in its producer's
// epilogue, and if so describe the staged copy. Returns the buffer size in bytes, or 0 to decline.
//
// Declines unless at least one later node is a q8_1-consuming GEMV over exactly this tensor. Staging an
// activation nobody consumes that way would add work and remove no launch.
static size_t alpha_q8_stage_plan(const struct ggml_cgraph * cgraph, int idx,
                                  const ggml_tensor * produced, alpha_q8_staged * e) {
    // One-time decline reason, so that "staged nothing" can be told apart from "the hook never ran".
    auto no = [produced](int code) -> size_t {
        static bool seen[8] = {};
        if (code < 8 && !seen[code]) {
            seen[code] = true;
            GGML_LOG_INFO("[q8-stage] plan declined: code %d (ne=[%lld,%lld,%lld,%lld] pad=%d)\n", code,
                          produced ? (long long) produced->ne[0] : -1,
                          produced ? (long long) produced->ne[1] : -1,
                          produced ? (long long) produced->ne[2] : -1,
                          produced ? (long long) produced->ne[3] : -1,
                          (int) MATRIX_ROW_PADDING);
        }
        return 0;
    };
    if (!produced || produced->type != GGML_TYPE_F32 || !ggml_is_contiguous(produced)) {
        return no(1);
    }
    const int64_t ne10 = produced->ne[0];
    if (ne10 % QK8_1 != 0) {
        return no(2);
    }
    // The reference quantizer zero-fills the padding tail; this epilogue only writes real columns, so
    // only stage when there is no padding (n_embd is usually a multiple of MATRIX_ROW_PADDING).
    if (GGML_PAD(ne10, MATRIX_ROW_PADDING) != ne10) {
        return no(3);
    }

    bool consumed_by_gemv = false;
    for (int m = idx + 1; m < cgraph->n_nodes && m < idx + 128 && !consumed_by_gemv; ++m) {
        const ggml_tensor * n = cgraph->nodes[m];
        if (n->op != GGML_OP_MUL_MAT && n->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        // src1 must be this tensor, not merely a view sharing its bytes: the GEMV matches on the exact
        // pointer plus strides, and a view would have different ones.
        if (n->src[1] != produced || !ggml_is_quantized(n->src[0]->type)) {
            continue;
        }
        consumed_by_gemv = true;
    }
    if (!consumed_by_gemv) {
        return no(4);
    }

    {   // log the first positive decision too, not just the declines
        static bool first = true;
        if (first) {
            first = false;
            GGML_LOG_INFO("[q8-stage] staging ne=[%lld,%lld,%lld,%lld]\n",
                          (long long) produced->ne[0], (long long) produced->ne[1],
                          (long long) produced->ne[2], (long long) produced->ne[3]);
        }
    }
    const size_t ts = ggml_type_size(produced->type);
    e->src1_data = produced->data;
    e->ne10 = ne10;                      // == ne10_padded here, checked above
    e->ne11 = produced->ne[1];
    e->ne12 = produced->ne[2];
    e->ne13 = produced->ne[3];
    e->s11  = produced->nb[1] / ts;
    e->s12  = produced->nb[2] / ts;
    e->s13  = produced->nb[3] / ts;
    return (size_t) e->ne13 * e->ne12 * e->ne11 * ne10 * sizeof(block_q8_1) / QK8_1;
}


// [AI_BOUND_DELETE] v2 kernel-count bound. AI_BOUND_DELETE=norm,glu,unary,copy,rope skips every node of
// those op classes (output left stale), so a decode round prices what deleting that class of kernels is
// worth -- the method that produced the +6.07% quantize bound, extended to the other glue classes.
// Checked BEFORE the fusion shortcuts, and a deleted norm also takes its FUSED FOLLOWERS (the MUL/ADD
// ggml would fold into it): otherwise the follower launches alone, the kernel count does not drop, and
// the class reads as falsely worthless. Per-class counts print at exit: a partially-live knob reads
// exactly like a fully-live one otherwise.
struct ai_bd_state {
    bool norm = false, normrow = false, normhead = false, glu = false, unary = false, copy = false, rope = false, any = false;
    long long n_norm = 0, n_glu = 0, n_unary = 0, n_copy = 0, n_rope = 0, n_follow = 0;
};
static ai_bd_state & ai_bd() {
    static ai_bd_state st = []() {
        ai_bd_state t;
        const char * e = getenv("AI_BOUND_DELETE");
        if (e) {
            const std::string v = std::string(",") + e + ",";
            t.norm  = v.find(",norm,")  != std::string::npos;
            t.normrow  = v.find(",normrow,")  != std::string::npos;
            t.normhead = v.find(",normhead,") != std::string::npos;
            t.glu   = v.find(",glu,")   != std::string::npos;
            t.unary = v.find(",unary,") != std::string::npos;
            t.copy  = v.find(",copy,")  != std::string::npos;
            t.rope  = v.find(",rope,")  != std::string::npos;
            t.any   = t.norm || t.normrow || t.normhead || t.glu || t.unary || t.copy || t.rope;
            if (t.any) {
                atexit([]() {
                    ai_bd_state & q = ai_bd();
                    fprintf(stderr, "[ai-bound-delete] deleted ggml nodes: norm %lld glu %lld unary %lld copy %lld rope %lld (+%lld fused followers)\n",
                            q.n_norm, q.n_glu, q.n_unary, q.n_copy, q.n_rope, q.n_follow);
                });
            }
        }
        return t;
    }();
    return st;
}
// Number of consecutive nodes to skip starting at i; 0 keeps the node.
static int ai_bound_delete_span(ggml_cgraph * cgraph, int i) {
    ai_bd_state & s = ai_bd();
    if (!s.any) {
        return 0;
    }
    ggml_tensor * node = cgraph->nodes[i];
    switch (node->op) {
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM: {
            if (!s.norm && !(s.normrow && node->ne[0] >= 1024) && !(s.normhead && node->ne[0] < 1024)) return 0;
            ++s.n_norm;
            int k = 1;
            const ggml_tensor * prev = node;
            while (i + k < cgraph->n_nodes && k <= 2) {
                const ggml_tensor * nx = cgraph->nodes[i + k];
                if ((nx->op == GGML_OP_MUL || nx->op == GGML_OP_ADD) && (nx->src[0] == prev || nx->src[1] == prev)) {
                    ++s.n_follow;
                    prev = nx;
                    ++k;
                } else {
                    break;
                }
            }
            return k;
        }
        case GGML_OP_GLU:
            if (!s.glu) return 0;
            ++s.n_glu;
            return 1;
        case GGML_OP_UNARY:
            if (!s.unary) return 0;
            ++s.n_unary;
            return 1;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_CONCAT:
        case GGML_OP_SET_ROWS:
            if (!s.copy) return 0;
            ++s.n_copy;
            return 1;
        case GGML_OP_ROPE:
            if (!s.rope) return 0;
            ++s.n_rope;
            return 1;
        default:
            return 0;
    }
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        ggml_cuda_mmvq_q8_cache_begin_graph();   // ALPHA_Q8_CACHE: nothing survives a graph boundary (or a capture retry)
        ggml_cuda_mmq_id_cache_begin_graph();    // ALPHA_MMID_CACHE: same rule
        int q8c_noted = -1;                        // every node index gets note_write()d exactly once, in order
        // ALPHA_MOE_DOWN9 >= 2: one deferred shared-expert tail at a time: set when its nodes are
        // skipped, consumed when its routed down projection is reached. Reset per evaluation pass.
        alpha_moe_down9_args d9_pending{};
        int d9_pending_mmid = -1;
        int d9_pending_n    = 0;
        int d9_skip_sig     = -1;   // the sigmoid and the gating multiply are recomputed inside the
        int d9_skip_mul     = -1;   // fused kernel, so their own nodes are dropped
        // ALPHA_Q8_STAGE: backing store for producer-emitted q8_1 copies, held for the whole pass so a
        // consumer many nodes later still reads live memory. (Same graph-capture caveat as mmvq's own
        // src1_q8_1 pool allocation, which upstream already bakes into captured graphs.)
        // One bump-allocated arena: the CUDA pool requires LIFO frees, and a container of per-staging
        // pool allocations would free in the wrong order; a single allocation avoids that and costs one
        // pool operation.
        ggml_cuda_pool_alloc<char> q8_stage_arena(cuda_ctx->pool());
        char * q8_stage_base = nullptr;
        size_t q8_stage_off  = 0;
        const size_t q8_stage_cap = 64u * 1024 * 1024;  // sized for the largest prefill T, not decode
        ggml_cuda_q8_stage_reset();
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            const bool prof_iter = alpha_op_prof_on() && alpha_op_prof_iter() && !use_cuda_graph;
            int prof_iter_open = -1;   // node index whose iteration record is open
            // ALPHA_NODE_HASH=N (diagnostic, needs GGML_CUDA_DISABLE_GRAPHS=1): on the N-th eager graph with > 30
            // nodes, print an FNV hash of every computed node's output, to find where two builds first diverge.
            static const long long nh_at = [] { const char * e = getenv("ALPHA_NODE_HASH"); return e ? atoll(e) : 0LL; }();
            static long long nh_n = 0;
            static const long long nh_cnt = [] { const char * e = getenv("ALPHA_NODE_HASH_N"); return e ? atoll(e) : 1LL; }();
            const bool nh_on = nh_at > 0 && !use_cuda_graph && cgraph->n_nodes > 30 && ++nh_n >= nh_at && nh_n < nh_at + nh_cnt;
            auto nh_dump = [&](int j) {
                const ggml_tensor * t = cgraph->nodes[j];
                if (!nh_on || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return;
                CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                std::vector<uint32_t> hb(ggml_nelements(t));
                CUDA_CHECK(cudaMemcpy(hb.data(), t->data, hb.size()*4, cudaMemcpyDeviceToHost));
                uint64_t h = 0xcbf29ce484222325ull;
                for (uint32_t v : hb) h = (h ^ v) * 0x100000001b3ull;
                fprintf(stderr, "[nodehash] %-14s %-32s %016llx\n", ggml_op_desc(t), t->name, (unsigned long long) h);
            };
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (nh_on && i > 0) nh_dump(i - 1);   // a fused group leaves i on its last node, the real output
                if (prof_iter) {
                    if (prof_iter_open >= 0) {
                        if (i - prof_iter_open > 1) alpha_op_prof_mark_fused();
                        alpha_op_prof_end(cuda_ctx->stream());
                    }
                    alpha_op_prof_begin(cuda_ctx->stream(), node);
                    g_op_prof.iter_open = true;
                    prof_iter_open = i;
                }
                // ALPHA_Q8_CACHE: a fused group advances i past its intermediates; note them all before any compute
                for (int q = q8c_noted + 1; q <= i; q++) {
                    ggml_cuda_mmvq_q8_cache_note_write(cgraph->nodes[q]->data, ggml_nbytes(cgraph->nodes[q]));
                    ggml_cuda_mmq_id_cache_note_write(cgraph->nodes[q]->data, ggml_nbytes(cgraph->nodes[q]));
                    // ALPHA_Q8_STAGE: a staged q8_1 copy describes specific bytes; if ggml's allocator has
                    // handed those bytes to another node, the copy no longer corresponds to the activation.
                    ggml_cuda_q8_stage_invalidate(cgraph->nodes[q]->data, ggml_nbytes(cgraph->nodes[q]), q);
                }
                q8c_noted = i;
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

#ifdef GGML_CUDA_DEBUG
                const int nodes_fused = i - prev_i - 1;
                if (nodes_fused > 0) {
                    GGML_LOG_INFO("nodes_fused: %d\n", nodes_fused);
                }
#endif
                prev_i = i;

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // [AI_BOUND_DELETE] before any fusion shortcut, so fused groups are deleted whole
                if (const int ai_bd_k = ai_bound_delete_span(cgraph, i); ai_bd_k > 0) {
                    i += ai_bd_k - 1;
                    continue;
                }

                // start of fusion operations
                static bool disable_fusion = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
                if (!disable_fusion) {
                    ggml_cuda_topk_moe_args args;

                    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                        cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                        const bool can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);

                        std::vector<ggml_op> ops;

                        if (can_fuse) {
                            const ggml_tensor * logits  = node->src[0];
                            ggml_tensor *       weights = nullptr;
                            ggml_tensor *       ids     = nullptr;
                            const ggml_tensor * bias    = nullptr;
                            const ggml_tensor * clamp   = nullptr;
                            const ggml_tensor * scale   = nullptr;

                            if (!args.delayed_softmax) {
                                ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
                                int     out_nodes[2];  // nodes which can't be elided

                                if (args.prob_bias) {
                                    bias = cgraph->nodes[i + 2]->src[1];
                                    ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT,
                                                            GGML_OP_VIEW, GGML_OP_GET_ROWS });
                                    out_nodes[0] = i + 4;
                                    ids          = cgraph->nodes[i + 4];
                                } else {
                                    ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                            GGML_OP_GET_ROWS });
                                    out_nodes[0] = i + 3;
                                    ids          = cgraph->nodes[i + 3];
                                }

                                if (args.norm) {
                                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP,
                                                            GGML_OP_DIV, GGML_OP_RESHAPE });
                                    clamp = cgraph->nodes[i + ops.size() - 3];
                                }
                                if (args.scale) {
                                    ops.insert(ops.end(), { GGML_OP_SCALE });
                                    scale = cgraph->nodes[i + ops.size() - 1];
                                }

                                weights      = cgraph->nodes[i + ops.size() - 1];
                                out_nodes[1] = i + ops.size() - 1;

                                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                    ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/ true)) {
                                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                                    i += ops.size() - 1;
                                    continue;
                                }
                            } else if (!args.norm && !args.prob_bias) {
                                //special case gpt-oss, no norm, no bias.
                                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
                                                        GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                                weights                     = cgraph->nodes[i + 5];
                                ids                         = cgraph->nodes[i + 1];
                                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                                int out_nodes[2] = { i + 1, i + 5 };
                                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                    ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/ true)) {
                                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                                    i += ops.size() - 1;
                                    continue;
                                }
                            }
                        }
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
                        ggml_tensor * rope = cgraph->nodes[i];
                        ggml_tensor * set_rows = cgraph->nodes[i + 2];

                        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
                        i += 2;
                        continue;
                    }

                    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
                        int n_fuse = 0;
                        ggml_op ops[8];
                        std::fill(ops, ops + 8, node->op);

                        for (; n_fuse <= 6; ++n_fuse){
                            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                                break;
                            }
                            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                                break;
                            }
                            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                                break;
                            }
                        }

                        n_fuse++;

                        if (n_fuse > 1) {
                            ggml_tensor fused_node;
                            memcpy(&fused_node, node, sizeof(ggml_tensor));
                            for (int j = 0; j < n_fuse - 1; ++j) {
                                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
                            }
                            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
                            if (node->op == GGML_OP_ADD) {
                                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
                            } else {
                                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
                            }
                            i += n_fuse - 1;

                            continue;
                        }
                    }

                    bool fused_mul_mat_vec = false;
                    int fused_node_count = 0;

                    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
                        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

                        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
                            ggml_tensor * glu         = cgraph->nodes[i + 4];
                            ggml_tensor * gate_bias_n = glu->src[0];
                            ggml_tensor * up_bias_n   = glu->src[1];

                            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
                            ggml_tensor * gate_n      = nullptr;
                            ggml_tensor * up_n        = nullptr;

                            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                                gate_n = cgraph->nodes[i];
                                up_n   = cgraph->nodes[i + 2];
                            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                                gate_n = cgraph->nodes[i + 2];
                                up_n   = cgraph->nodes[i];
                            } else {
                                continue;
                            }

                            auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) {
                                if (op_bias == GGML_OP_ADD) {
                                    if (bias_node->src[0] == mul_node) {
                                        return bias_node->src[1];
                                    }
                                    if (bias_node->src[1] == mul_node) {
                                        return bias_node->src[0];
                                    }
                                    return (ggml_tensor *) nullptr;
                                }
                                GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
                                GGML_ASSERT(bias_node->src[0] == mul_node);
                                return bias_node->src[1];
                            };

                            ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
                            ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

                            if (!up_bias_tensor || !gate_bias_tensor) {
                                continue;
                            }

                            // we don't support repeating adds
                            if (bias_op == GGML_OP_ADD &&
                                (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                 !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                                continue;
                            }

                            const ggml_tensor * src0 = up_n->src[0];
                            const ggml_tensor * src1 = up_n->src[1];
                            const ggml_tensor * ids  = up_n->src[2];

                            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate      = gate_n->src[0];
                                fusion_data.x_bias    = up_bias_tensor;
                                fusion_data.gate_bias = gate_bias_tensor;
                                fusion_data.glu_op    = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 5;
                                break;
                            }

                            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate      = gate_n->src[0];
                                fusion_data.x_bias    = up_bias_tensor;
                                fusion_data.gate_bias = gate_bias_tensor;
                                fusion_data.glu_op    = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 5;
                                break;
                            }
                        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
                            ggml_tensor * glu  = cgraph->nodes[i + 2];
                            ggml_tensor * gate = glu->src[0];
                            ggml_tensor * up   = glu->src[1];

                            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1])
                                || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

                            if (!ok) continue;

                            const ggml_tensor * src0 = up->src[0];
                            const ggml_tensor * src1 = up->src[1];
                            const ggml_tensor * ids  = up->src[2];

                            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate   = gate->src[0];
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 3;
                                break;
                            }

                            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate   = gate->src[0];
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 3;
                                break;
                            }
                        }
                    }

                    if (fused_mul_mat_vec) {
                        i += fused_node_count - 1;
                        continue;
                    }

                    fused_mul_mat_vec = false;
                    fused_node_count = 0;

                    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
                        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

                        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
                            continue;
                        }

                        ggml_tensor * mm_node   = cgraph->nodes[i];
                        ggml_tensor * bias_node = cgraph->nodes[i + 1];

                        ggml_tensor * bias_tensor = nullptr;
                        if (bias_op == GGML_OP_ADD) {
                            if (bias_node->src[0] == mm_node) {
                                bias_tensor = bias_node->src[1];
                            } else if (bias_node->src[1] == mm_node) {
                                bias_tensor = bias_node->src[0];
                            } else {
                                continue;
                            }
                        } else {
                            if (bias_node->src[0] != mm_node) {
                                continue;
                            }
                            bias_tensor = bias_node->src[1];
                        }

                        const ggml_tensor * src0 = mm_node->src[0];
                        const ggml_tensor * src1 = mm_node->src[1];
                        const ggml_tensor * ids  = mm_node->src[2];

                        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
                            continue;
                        }

                        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
                            continue;
                        }

                        ggml_cuda_mm_fusion_args_host fusion_data{};
                        fusion_data.x_bias = bias_tensor;

                        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
                            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
                            fused_mul_mat_vec = true;
                            fused_node_count = 2;
                            break;
                        }

                        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
                            fused_mul_mat_vec = true;
                            fused_node_count = 2;
                            break;
                        }
                    }

                    if (fused_mul_mat_vec) {
                        i += fused_node_count - 1;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD}, {})) {
                        // ALPHA_Q8_STAGE: same epilogue as the RMS_NORM+MUL branch; the residual norms
                        // take this path, and they are the ones feeding the expert projections.
                        block_q8_1 * q8a_dst = nullptr;
                        ggml_tensor * add_node = cgraph->nodes[i+2];
                        if (ggml_cuda_q8_stage_enabled()) {
                            alpha_q8_staged e{};
                            const size_t bytes   = alpha_q8_stage_plan(cgraph, i + 2, add_node, &e);
                            const size_t aligned = (bytes + 255) & ~(size_t) 255;
                            if (bytes > 0 && q8_stage_off + aligned <= q8_stage_cap) {
                                if (q8_stage_base == nullptr) {
                                    q8_stage_base = q8_stage_arena.alloc(q8_stage_cap);
                                }
                                q8a_dst        = (block_q8_1 *) (q8_stage_base + q8_stage_off);
                                q8_stage_off  += aligned;
                                e.q8           = (const char *) q8a_dst;
                                e.produced_at  = i + 2;
                                ggml_cuda_q8_stage_put(e);
                            }
                        }
                        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i+1], add_node, q8a_dst);
                        i += 2;
                        continue;
                    }

                    // ALPHA_Q8_STAGE: the two swiglu shapes account for many of the remaining quantize
                    // launches. Whether they reach this standalone GLU node or are absorbed by the
                    // {MUL_MAT_ID, MUL_MAT_ID, GLU} fusion is decided elsewhere, so this logs once either way.
                    if (node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU &&
                        node->type == GGML_TYPE_F32 && ggml_cuda_q8_stage_enabled()) {
                        static bool seen = false;
                        if (!seen) {
                            seen = true;
                            GGML_LOG_INFO("[q8-stage] standalone SWIGLU node reached: ne=[%lld,%lld,%lld,%lld]\n",
                                          (long long) node->ne[0], (long long) node->ne[1],
                                          (long long) node->ne[2], (long long) node->ne[3]);
                        }
                        ggml_cuda_q8_stage_note_producer(node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
                        alpha_q8_staged e{};
                        const size_t bytes = alpha_q8_stage_plan(cgraph, i, node, &e);
                        const size_t aligned = (bytes + 255) & ~(size_t) 255;
                        const bool fits = q8_stage_off + aligned <= q8_stage_cap;
                        if (bytes > 0 && ggml_nelements(node) % QK8_1 == 0 && !fits) {
                            ggml_cuda_q8_stage_note_arena_full();
                        }
                        if (bytes > 0 && ggml_nelements(node) % QK8_1 == 0 && fits) {
                            if (q8_stage_base == nullptr) {
                                q8_stage_base = q8_stage_arena.alloc(q8_stage_cap);
                            }
                            block_q8_1 * q8g = (block_q8_1 *) (q8_stage_base + q8_stage_off);
                            q8_stage_off  += aligned;
                            e.q8           = (const char *) q8g;
                            e.produced_at  = i;
                            ggml_cuda_q8_stage_put(e);
                            ggml_cuda_op_swiglu(*cuda_ctx, node, q8g);
                            continue;
                        }
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) {
                        // ALPHA_Q8_STAGE: if this normed output feeds a q8_1-consuming GEMV, emit the q8_1
                        // copy here as an epilogue instead of paying a separate quantize launch per consumer.
                        // quantize_q8_1 is almost pure launch cost, and the post-attention norm alone feeds
                        // three consumers.
                        block_q8_1 * q8_dst = nullptr;
                        ggml_tensor * mul_node = cgraph->nodes[i+1];
                        if (ggml_cuda_q8_stage_enabled()) {
                            alpha_q8_staged e{};
                            const size_t bytes = alpha_q8_stage_plan(cgraph, i + 1, mul_node, &e);
                            const size_t aligned = (bytes + 255) & ~(size_t) 255;
                            if (bytes > 0 && q8_stage_off + aligned <= q8_stage_cap) {
                                if (q8_stage_base == nullptr) {
                                    q8_stage_base = q8_stage_arena.alloc(q8_stage_cap);
                                }
                                q8_dst         = (block_q8_1 *) (q8_stage_base + q8_stage_off);
                                q8_stage_off  += aligned;
                                e.q8           = (const char *) q8_dst;
                                e.produced_at  = i + 1;   // the MUL node is what writes the activation
                                ggml_cuda_q8_stage_put(e);
                            }
                        }
                        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, mul_node, q8_dst);
                        i++;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
                        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i+1]);
                        i++;
                        continue;
                    }

                    // ALPHA_MOE_DOWN9 >= 2: the shared expert's tail is deferred here and recomputed
                    // inside the fused kernel when its routed down projection is reached
                    if (i == d9_skip_sig || i == d9_skip_mul) {
                        continue;   // folded into the pending fused kernel
                    }

                    if (node->op == GGML_OP_MUL_MAT && d9_pending_mmid < 0) {
                        alpha_moe_down9_args d9s{};
                        int mmid_idx = -1, sk_sig = -1, sk_mul = -1;
                        const int d9s_n = ggml_cuda_match_moe_down9_shexp(cgraph, i, &d9s, &mmid_idx,
                                                                          &sk_sig, &sk_mul);
                        if (d9s_n > 0) {
                            d9_pending      = d9s;
                            d9_pending_n    = d9s_n;
                            d9_pending_mmid = mmid_idx;
                            d9_skip_sig     = sk_sig;
                            d9_skip_mul     = sk_mul;
                            continue;       // the shared down itself is recomputed by the fused kernel
                        }
                    }

                    if (node->op == GGML_OP_MUL_MAT_ID) {
                        alpha_moe_down9_args d9{};
                        int d9_n = 0;
                        if (d9_pending_mmid == i) {
                            d9              = d9_pending;
                            d9_n            = d9_pending_n;
                            d9_pending_mmid = -1;
                            d9_skip_sig     = -1;
                            d9_skip_mul     = -1;
                        } else {
                            d9_n = ggml_cuda_match_moe_down9(cgraph, i, &d9);
                        }
                        if (d9_n > 0) {
                            ggml_cuda_op_moe_down9(*cuda_ctx, d9.down_exps, d9.act_routed, d9.ids, d9.weights,
                                                   d9.down_shexp, d9.act_shexp, d9.score,
                                                   d9.addends, d9.n_add, d9.dst);
                            i += d9_n - 1;
                            continue;
                        }
                    }

                    if (node->op == GGML_OP_MUL) {
                        const ggml_tensor * mm_add[4] = { nullptr, nullptr, nullptr, nullptr };
                        int mm_nadd = 0;
                        const int mm_n = ggml_cuda_match_moe_merge(cgraph, i, mm_add, &mm_nadd);
                        if (mm_n > 0) {
                            ggml_cuda_op_moe_merge(*cuda_ctx, node->src[0], node->src[1], mm_add, mm_nadd, cgraph->nodes[i + mm_n - 1]);
                            i += mm_n - 1;
                            continue;
                        }
                    }

                    if (node->op == GGML_OP_CONCAT && ggml_cuda_can_fuse_concat_shift(cgraph, i)) {
                        ggml_cuda_op_concat_shift(*cuda_ctx, node, cgraph->nodes[i+3]);
                        i += 3;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
                        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
                        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
                        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i+1]);
                        i++;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
                        i += 2;
                        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i], node);
                        continue;
                    }
                }
#ifndef NDEBUG
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               ggml_backend_buft_is_cuda_split(node->src[j]->buffer->buft) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)) ||
                               ggml_backend_buffer_is_host(node->src[j]->buffer)); // host expert tensors accessed via DMA
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                const bool prof_this = alpha_op_prof_on() && !use_cuda_graph && !alpha_op_prof_iter();
                if (prof_this) alpha_op_prof_begin(cuda_ctx->stream(), node);
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (prof_this) alpha_op_prof_end(cuda_ctx->stream());
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }

            if (nh_on) nh_dump(cgraph->n_nodes - 1);

            // ALPHA_MOE_DOWN9 >= 2: a deferred shared-expert tail must be consumed by its routed down
            // projection in this same pass. If another matcher skipped over that node, the three
            // dropped nodes would never be computed and the output would be silently wrong, so this
            // is fatal rather than a warning. It cannot fire on a graph the matcher accepted (the
            // routed node was located in this cgraph).
            GGML_ASSERT(d9_pending_mmid < 0 && "moe-down9: deferred shared-expert tail was never consumed");
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured
            // the capture lock is released after the executable update below (ggml_cuda_outside_capture)
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            static const bool launch_dbg = alpha_env_on("ALPHA_GRAPH_LAUNCH_DBG");
            if (launch_dbg && cgraph->n_nodes > 0) {
                fprintf(stderr, "[graph-inst] key=%p n_nodes=%d first=%s (%s) last=%s (%s) ne1=%lld\n", graph_key, cgraph->n_nodes,
                        cgraph->nodes[0]->name, ggml_op_desc(cgraph->nodes[0]), cgraph->nodes[cgraph->n_nodes - 1]->name,
                        ggml_op_desc(cgraph->nodes[cgraph->n_nodes - 1]), (long long) cgraph->nodes[0]->ne[1]);
            }
            cudaError_t ierr = cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0);
            if (ierr == cudaErrorMemoryAllocation) {
                // Every cached executable graph holds device memory. With many cached shapes (MTP verify widths,
                // adaptive draft depth, DRE prefill/decode switches) the instances alone can use up the headroom
                // the expert pool left. Drop every other cached graph, return the idle temp-pool memory,
                // and try once more.
                (void) cudaGetLastError();
                // earlier launches of the graphs about to be destroyed may still be running
                CUDA_CHECK(cudaDeviceSynchronize());
                // destroy only the other graphs' executable instances (the device memory); the graph objects stay,
                // because other caches (the semantic reuse map) hold raw pointers to them. A dropped instance is
                // re-created on its next use.
                size_t dropped = 0;
                for (auto & kv : cuda_ctx->cuda_graphs) {
                    ggml_cuda_graph * g = kv.second.get();
                    if (g != graph && g->instance != nullptr) {
                        CUDA_CHECK(cudaGraphExecDestroy(g->instance));
                        g->instance = nullptr;
                        dropped++;
                    }
                }
                size_t freed = 0;
                for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
                    for (int s = 0; s < GGML_CUDA_MAX_STREAMS; s++) {
                        if (cuda_ctx->pools[d][s]) freed += cuda_ctx->pools[d][s]->trim();
                    }
                }
                GGML_LOG_WARN("%s: cudaGraphInstantiate out of memory: dropped %zu cached graphs, trimmed %.0f MB of "
                              "temp pools, retrying\n", __func__, dropped, freed / 1048576.0);
                ierr = cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0);
            }
            CUDA_CHECK(ierr);
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        }
        // Launch graph
        static const bool glt = getenv("ALPHA_LAUNCH_TIMERS") != nullptr;
        const int64_t gl0 = glt ? ggml_time_us() : 0;
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
        if (glt && cgraph->n_nodes > 1000 && !cuda_graph_update_required) {
            static double s = 0; static long long n = 0;
            s += ggml_time_us() - gl0;
            if ((++n & 127) == 0) fprintf(stderr, "[launch-t] cudaGraphLaunch (replay, %d nodes) %.3f ms/call (n=%lld)\n",
                                          cgraph->n_nodes, s / n / 1e3, n);
        }
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    // ALPHA_NCU_RANGE=start,count: profile only big graphs [start, start+count) (run under ncu/nsys with
    // --profile-from-start off); every big graph's index and node count is logged so a window can be picked
    {
        static const char * nr = getenv("ALPHA_NCU_RANGE");
        static int nr_start = -1, nr_count = 0, nr_i = 0;
        static bool nr_init = false;
        if (nr && !nr_init) { nr_init = true; sscanf(nr, "%d,%d", &nr_start, &nr_count); }
        if (nr && cgraph->n_nodes > 1000) {
            if (nr_i == nr_start) { cudaDeviceSynchronize(); cudaProfilerStart(); }
            if (nr_i == nr_start + nr_count) { cudaDeviceSynchronize(); cudaProfilerStop(); }
            fprintf(stderr, "[ncu-range] big graph %d n_nodes=%d\n", nr_i, cgraph->n_nodes);
            nr_i++;
        }
    }

    // ALPHA_LAUNCH_TIMERS: host time of the phases of a big graph's compute call (DRE bookkeeping, CUDA-graph
    // property check, the rest = capture/launch/eager dispatch)
    static const bool lt_on = getenv("ALPHA_LAUNCH_TIMERS") != nullptr;
    const bool lt = lt_on && cgraph->n_nodes > 1000;
    const int64_t lt0 = lt ? ggml_time_us() : 0;
    // ALPHA_GRAPH_NAMES=a,b,c: once per listed node count, print that graph's first node names (which graph is it?)
    {
        static const char * gn = getenv("ALPHA_GRAPH_NAMES");
        static std::unordered_set<int> shown;
        if (gn) {
            char buf[32]; snprintf(buf, sizeof(buf), ",%d,", cgraph->n_nodes);
            std::string want = std::string(",") + gn + ",";
            if (want.find(buf) != std::string::npos && shown.insert(cgraph->n_nodes).second) {
                std::string nm;
                for (int i = 0; i < cgraph->n_nodes && i < 10; i++) { nm += ggml_op_desc(cgraph->nodes[i]); nm += ":"; nm += cgraph->nodes[i]->name; nm += " | "; }
                fprintf(stderr, "[graph-names] n_nodes=%d uid=%zu: %s\n", cgraph->n_nodes, (size_t) cgraph->uid, nm.c_str());
            }
        }
    }
    // ALPHA_DRE: residency changes happen here, before any capture begins (uncaptured stream work)
    ggml_cuda_dre_graph_begin(*cuda_ctx, cgraph);
    const int64_t lt1 = lt ? ggml_time_us() : 0;
    int64_t lt2 = lt1;
    struct lt_guard {
        bool on; int64_t t0, * t1, * t2;
        ~lt_guard() {
            if (!on) return;
            static double a = 0, b = 0, c = 0; static long long n = 0;
            const int64_t t3 = ggml_time_us();
            a += *t1 - t0; b += *t2 - *t1; c += t3 - *t2; n++;
            if ((n & 127) == 0) fprintf(stderr, "[launch-t] dre_begin %.3f prop_check %.3f launch %.3f ms/call (n=%lld)\n",
                                        a / n / 1e3, b / n / 1e3, c / n / 1e3, n);
        }
    };
    int64_t lt1v = lt1;
    lt_guard ltg { lt, lt0, &lt1v, &lt2 };

    // Model-type probe for the small-M GEMM tier (see g_alpha_graph_has_mmid). Under offload the
    // backend sees scheduler splits, and the first splits of a pass hold no expert matmul, so this
    // is a per-call scan that stops the moment one is found rather than a first-call check. On a
    // dense model it never finds one and costs n_nodes pointer reads per call.
    if (!g_alpha_graph_has_mmid) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                g_alpha_graph_has_mmid = true;
                if (getenv("ALPHA_SMALLM_MMQ") == nullptr) {
                    GGML_LOG_INFO("[alpha-smallm] expert matmul seen in graph: small-M GEMM tier default OFF "
                                  "for this model (ALPHA_SMALLM_MMQ=N overrides)\n");
                }
                break;
            }
        }
    }

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        // ALPHA_GRAPH_COMPAT_CACHE: the compatibility scan walks every node, which costs noticeable host time
        // per call on large graphs while the GPU is idle. A reused graph (same non-zero uid) has the same nodes,
        // ops and buffers, so its verdict is cached per uid.
        bool graph_compatible;
        {
            static const bool cache_on = []() { const char * e = getenv("ALPHA_GRAPH_COMPAT_CACHE"); return !e || atoi(e) != 0; }();
            struct ent { size_t uid; bool ok; };
            static ent cache[16] = {};
            static int next = 0;
            int hit = -1;
            if (cache_on && cgraph->uid != 0) {
                for (int i = 0; i < 16; i++) if (cache[i].uid == (size_t) cgraph->uid) { hit = i; break; }
            }
            if (hit >= 0) {
                graph_compatible = cache[hit].ok;
            } else {
                graph_compatible = ggml_cuda_graph_check_compability(cgraph);
                if (cache_on && cgraph->uid != 0) { cache[next] = { (size_t) cgraph->uid, graph_compatible }; next = (next + 1) % 16; }
            }
        }
        {
            static const bool sm_dbg = alpha_env_on("ALPHA_GRAPH_SM_DBG");
            static long long sm_n = 0;
            if (sm_dbg && (++sm_n % 4999) == 0) {
                fprintf(stderr, "[graph-sm] call=%lld key=%p compat=%d warm=%d hits=%d n_nodes=%d props=%zu uid=%zu/%zu\n",
                        sm_n, graph_key, (int) graph_compatible, (int) graph->warmup_complete, graph->warmup_hits,
                        cgraph->n_nodes, graph->node_props.size(), (size_t) cgraph->uid, (size_t) graph->uid);
            }
        }
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);
            if (lt) {
                lt2 = ggml_time_us();
                static long long slow = 0, tot = 0;
                tot++;
                if (lt2 - lt1 > 200) {
                    slow++;
                    if (slow <= 6 || slow % 64 == 0)
                        fprintf(stderr, "[launch-t] slow prop_check %.3f ms: n_nodes=%d uid=%zu cached-uid=%zu changed=%d (%lld of %lld calls)\n",
                                (lt2 - lt1) / 1e3, cgraph->n_nodes, (size_t) cgraph->uid, (size_t) graph->uid, (int) properties_changed, slow, tot);
                }
            }

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call. Batched graphs
                // (n_tokens > 1, capturable with ALPHA_BATCH_GRAPH=2) need 4: an MTP verify width recurs
                // many times, a repeated prompt length only a few, and capturing the latter costs more than
                // the rare replay saves.
                // ALPHA_GRAPH_WARMUP overrides the batched count. use_cuda_graph starts false and needs
                // `warmup_needed` consecutive property-stable calls; any property change resets warmup_hits
                // to 0 (and post-warmup clears warmup_complete). Deeper MTP drafts toggle properties more
                // often, so a high requirement is met less often. Defaults: 4 batched / 1 unbatched.
                static const int warmup_batched = []() {
                    const char * e = getenv("ALPHA_GRAPH_WARMUP");
                    const int v = e ? atoi(e) : 4;
                    return v < 1 ? 1 : v;
                }();
                const int warmup_needed = (cgraph->n_nodes > 0 && cgraph->nodes[0]->ne[1] > 1) ? warmup_batched : 1;
                if (!properties_changed) {
                    graph->warmup_hits++;
                } else {
                    graph->warmup_hits = 0;
                }
                if (!properties_changed && graph->warmup_hits >= warmup_needed) {
                    graph->warmup_complete = true;
                    graph->warmup_hits = 0;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or not enough identical calls - execute directly
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH


    // ALPHA_GRAPH_STATS=1: tally of how each backend graph call ran (direct launch / captured /
    // replayed), one stderr line per 1024 calls. Useful for MTP-vs-plain comparisons, since MTP
    // passes replay graphs less often.
    static const bool graph_stats = alpha_env_on("ALPHA_GRAPH_STATS");
    // ALPHA_GRAPH_DUMP=N: once, list the first N nodes (op, name, ne, src0 type) of the first graph with > 1000
    // nodes (the per-layer launch inventory)
    {
        static const int dump_n = []() { const char * e = getenv("ALPHA_GRAPH_DUMP"); return e ? atoi(e) : 0; }();
        static bool dumped = false;
        if (dump_n > 0 && !dumped && cgraph->n_nodes > 1000) {
            dumped = true;
            for (int i = 0; i < cgraph->n_nodes && i < dump_n; i++) {
                const ggml_tensor * t = cgraph->nodes[i];
                fprintf(stderr, "[graph-dump] %4d %-14s %-30s ne=[%lld,%lld,%lld] %s src0=%s(%s)\n",
                        i, ggml_op_name(t->op), t->name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                        ggml_type_name(t->type), t->src[0] ? t->src[0]->name : "-", t->src[0] ? ggml_type_name(t->src[0]->type) : "-");
            }
        }
    }
    if (graph_stats) {
        static int64_t n_calls = 0, n_direct = 0, n_capture = 0, n_replay = 0;
        n_calls++;
        if (!use_cuda_graph) { n_direct++; } else if (cuda_graph_update_required) { n_capture++; } else { n_replay++; }
        {
            alpha_graph_key_stats & ks = g_alpha_graph_keys[graph_key];
            ks.n_nodes = cgraph->n_nodes;
            ks.ne1 = cgraph->n_nodes > 0 ? (long long) cgraph->nodes[cgraph->n_nodes - 1]->ne[1] : -1;
            if (!use_cuda_graph) { ks.direct++; } else if (cuda_graph_update_required) { ks.captured++; } else { ks.replayed++; }
        }
        if ((n_calls & 1023) == 0) {
            std::string vet;
            for (std::map<int, long long>::const_iterator it = g_alpha_graph_veto.begin(); it != g_alpha_graph_veto.end(); ++it) {
                char b[48]; snprintf(b, sizeof(b), " L%d=%lld", it->first, it->second); vet += b;
            }
            fprintf(stderr, "[graph-stats] vetoes:%s\n", vet.empty() ? " none" : vet.c_str());
            for (std::map<const void *, alpha_graph_key_stats>::const_iterator it = g_alpha_graph_keys.begin(); it != g_alpha_graph_keys.end(); ++it) {
                fprintf(stderr, "[graph-stats]   key=%p n_nodes=%d out_cols=%lld direct=%lld captured=%lld replayed=%lld\n",
                        it->first, it->second.n_nodes, it->second.ne1, it->second.direct, it->second.captured, it->second.replayed);
            }
            fprintf(stderr, "[graph-stats] calls=%lld direct=%lld captured=%lld replayed=%lld cache=%zu\n",
                    (long long) n_calls, (long long) n_direct, (long long) n_capture, (long long) n_replay, cuda_ctx->cuda_graphs.size());
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    ggml_cuda_dre_graph_end(*cuda_ctx, cgraph);

    if (alpha_op_prof_on()) alpha_op_prof_flush(cuda_ctx->stream());

    // Signal new-token boundary for pcache offset table clearing
    g_pcache_new_token = true;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) {
        return;
    }

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemGetInfo(free, total));

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx->device));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__)

}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // split buffers can only be used with GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda_split(op->src[i]->buffer->buft)) {
                return false;
            }
        }
    }

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
        case GGML_OP_DSV4_HC_EXPAND:
            return op->src[0] && op->src[0]->type == GGML_TYPE_F32; // CUDA kernels for F32
        case GGML_OP_HC_COMBINE:
        case GGML_OP_HC_MIX_TAIL:
            return true;
        case GGML_OP_HC_MIX_FUSED:
            return ggml_cuda_hc_mix_fused_supported(op);
        case GGML_OP_DSV4_ROPE_TAIL:
            return false;
        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
            return false; // CPU-only for now
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->buffer && ggml_backend_buft_is_cuda_split(a->buffer->buft)) {
                    if (a->ne[2] > 1 || a->ne[3] > 1) {
                        return false;
                    }
                    // for small weight matrices the active device can end up without any rows, don't use row split in those cases
                    // this avoids some edge cases (and the performance would not be good anyways)
                    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) a->buffer->buft->context;
                    int64_t row_low;
                    int64_t row_high;
                    get_row_split(&row_low, &row_high, a, buft_ctx->tensor_split, dev_ctx->device);
                    if (row_low == row_high) {
                        return false;
                    }
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    // ik-quant types. Backbone MUL_MAT routes to the cuBLAS dequant path
                    // (ggml_cuda_op_mul_mat, ik excluded from MMVQ/MMQ); expert MUL_MAT_ID decodes
                    // from the fused-infer pool via ggml_cuda_mul_mat_id_ik_pool (device dequant ->
                    // cuBLAS GEMV). Without this the scheduler falls the ops back to CPU.
                    case GGML_TYPE_IQ4_K:
                    case GGML_TYPE_IQ6_K:
                    case GGML_TYPE_IQ4_KS:
                    case GGML_TYPE_IQ5_KS:
                    case GGML_TYPE_IQ3_KS:
                    case GGML_TYPE_IQ1_KT:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_XS:   // device token_embd gather (getrows.cu)
                        return op->src[0]->type != GGML_TYPE_IQ4_XS || op->src[0]->ne[0] % QK_K == 0;
                    case GGML_TYPE_IQ3_S:    // device token_embd gather (getrows.cu, whole superblocks only)
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                       op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                       op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                       op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                // I32 rides the F32 kernels as 4-byte words (QSA's chunked top-k indices)
                if (src0_type == GGML_TYPE_I32) return op->src[1]->type == GGML_TYPE_I32;
                return src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SSM_SCAN: {
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
            if (op->src[1]) {   // ALPHA_QSA_TOPK_EXPAND
                return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1]->type == GGML_TYPE_I32 &&
                       (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[2]->nb[0] == ggml_type_size(op->src[2]->type);
            }
            // fall through
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (((ggml_backend_buft_is_cuda(buft) || ggml_backend_buft_is_cuda_split(buft)) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft)));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // Always offload MUL_MAT_ID to the GPU: the persistent expert cache handles DMA
    // (GPU->GPU for cached experts, CPU->GPU only on miss).
    if (op->op == GGML_OP_MUL_MAT_ID) {
        return true;
    }

    // Multi-user offload (ALPHA_DENSE_BATCH_MIN): pull an offloaded dense-FFN weight (ffn_gate/up/down.weight
    // only, not attention/DeltaNet, so recurrent-layer op placement and context init are undisturbed) to CUDA
    // when the co-batch width is >= ALPHA_DENSE_BATCH_MIN. Wide batch -> GPU (weight DMA'd once per forward,
    // batched GEMM over N tokens). Narrow batch -> falls through to the min_batch=32 check -> FFN stays on CPU.
    if (g_alpha_dense_batch_min > 0 && op->op == GGML_OP_MUL_MAT &&
        op->src[0] && op->src[0]->buffer &&
        ggml_backend_buffer_is_host(op->src[0]->buffer) && op->src[0]->name[0]) {
        const char * nm = op->src[0]->name;
        if (strstr(nm, "ffn_gate.weight") || strstr(nm, "ffn_up.weight") ||
            strstr(nm, "ffn_down.weight")) {
            return get_op_batch_size(op) >= g_alpha_dense_batch_min;
        }
    }

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

// Persistent GPU expert cache for MoE weight offloading: a cudaMalloc pool, read in place through
// per-expert offset tables on pool-direct decode, otherwise copied D2D from the pool to input_cpy.
struct cuda_expert_persist_cache {
    // --- D2D pool with LRU eviction ---
    void *   pool      = nullptr;
    size_t   pool_size = 0;
    size_t   used      = 0;   // high-water mark for linear allocation

    // key = hash(weight_base_ptr, expert_id) -> pool entry
    struct pool_entry {
        size_t   offset;       // byte offset in pool
        size_t   size;         // expert_size (for LRU same-size eviction)
        uint64_t last_access;  // monotonic counter
        uint32_t hits;         // ALPHA_EVICT_POLICY=1: reuses since admission (freq/age score)
    };
    std::unordered_map<uint64_t, pool_entry> entries;
    uint64_t lru_counter = 0;
    int64_t  lru_evictions = 0;

    // Sampled (lazy) eviction: per-size-class key index for O(K) random sampling
    // instead of an O(n) same-size victim scan. entries is an unordered_map (no random
    // indexing), so we keep a vector of keys per size and sample from it (Redis-style
    // approximate LRU). Stale keys (already evicted) are swap-removed lazily on sample.
    std::unordered_map<size_t, std::vector<uint64_t>> size_ring;
    uint64_t rng_state = 0x9e3779b97f4a7c15ULL;  // xorshift; seeded, deterministic
    inline uint64_t rng() {
        uint64_t x = rng_state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        return rng_state = x;
    }
    uint64_t admission_ctr = 0;   // counts full-pool insert attempts (admission filter)
    // Entries accessed at/after this floor were touched this forward pass; a kernel
    // may still read them via an offset table, so eviction must never pick them.
    // Snapshotted from lru_counter at pass begin (sched hook, before any copies).
    uint64_t pool_pass_floor = 0;

    // Frequency-ranked hot set (ALPHA_HOTSET). Expert access is skewed (a minority of experts
    // takes most reads), and LRU ranks by recency, which re-misses a frequently used expert that
    // was briefly idle. Access frequency is profiled for the first N passes, then the most frequent
    // experts are pinned (by cumulative size up to a fraction of the pool): they bypass the admission
    // filter and are never evicted, and LRU manages the remaining pool space.
    struct freq_ent { uint32_t count; uint32_t size; };
    std::unordered_map<uint64_t, freq_ent> access_freq;  // profiling only, freed after
    std::unordered_set<uint64_t> hot_protected;          // pinned keys (never evicted)
    // Default on: raises the hit rate with diverse traffic; neutral when the working set already
    // fits the pool. ALPHA_HOTSET=0 disables.
    bool     hotset_enabled   = true;    // ALPHA_HOTSET (0 disables)
    bool     hotset_finalized = false;
    int      hotset_profile_passes = 2000;
    // Pin up to this fraction of the pool by bytes (ALPHA_HOTSET_FRAC). Too large a fraction leaves
    // little room after any workload shift, since a hot set profiled on one kind of prompt cannot be
    // displaced by another.
    float    hotset_budget_frac = 0.35f;
    int64_t  hot_hits = 0;               // accesses to protected experts (instrumentation)
    int64_t  hot_evicted = 0;            // protected slots surrendered because nothing else was
                                         // evictable (workload shifted away from the hot set)
    // Windowed hit-rate (steady-state, not cumulative)
    int64_t  win_access = 0, win_hits_snap = 0, win_miss_snap = 0, win_cold_snap = 0;

    // Adaptive hot set (ALPHA_HOTSET_ADAPTIVE, default off). The one-shot hot set above is profiled on
    // the first N passes and then frozen, so during a long generation the routing drifts away from it
    // and the hit rate decays. Adaptive mode keeps access_freq live and periodically re-ranks: pin the
    // currently hottest experts and un-pin ones that cooled off (they become LRU-evictable). Counts are
    // aged each window (x decay) so recent accesses dominate.
    bool     hotset_adaptive        = false; // ALPHA_HOTSET_ADAPTIVE
    // Aggressive re-pinning (short period, strong decay) churns the pinned set and thrashes the pool;
    // the defaults track drift without churn.
    int      hotset_adaptive_period = 1500;  // re-pin every N passes (ALPHA_HOTSET_ADAPTIVE_PERIOD)
    float    hotset_adaptive_decay  = 0.9f;  // age factor per window (ALPHA_HOTSET_ADAPTIVE_DECAY)
    uint64_t hotset_last_repin      = 0;     // pass ctr of the last re-pin
    int64_t  hotset_repins          = 0;     // instrumentation

    // Adaptive: decay every profiling count (recent-weighted window) and drop entries that
    // reach zero, so access_freq stays bounded and tracks the current hot distribution.
    void age_access_freq() {
        for (auto it = access_freq.begin(); it != access_freq.end(); ) {
            uint32_t c = (uint32_t)((float)it->second.count * hotset_adaptive_decay);
            if (c == 0) it = access_freq.erase(it);
            else { it->second.count = c; ++it; }
        }
    }
    // Adaptive re-pin: rebuild hot_protected from the current access frequencies (pin the
    // hottest up to the byte budget), then age so the next window re-measures fresh.
    // Newly-hot keys get a protected slot on their next insert; cooled keys fall to LRU.
    void repin_hotset() {
        std::vector<std::pair<uint64_t, freq_ent>> v(access_freq.begin(), access_freq.end());
        std::sort(v.begin(), v.end(),
                  [](const auto & a, const auto & b){ return a.second.count > b.second.count; });
        const size_t budget = (size_t)(hotset_budget_frac * (float)pool_size);
        std::unordered_set<uint64_t> new_hot;
        size_t used_b = 0;
        for (auto & [k, fe] : v) {
            if (fe.size == 0) continue;
            if (used_b + fe.size > budget) break;
            new_hot.insert(k);
            used_b += fe.size;
        }
        hot_protected.swap(new_hot);
        hotset_repins++;
        age_access_freq();
    }

    void finalize_hotset() {
        // Rank keys by access count (desc); pin until cumulative size hits the budget.
        std::vector<std::pair<uint64_t, freq_ent>> v(access_freq.begin(), access_freq.end());
        std::sort(v.begin(), v.end(),
                  [](const auto & a, const auto & b){ return a.second.count > b.second.count; });
        const size_t budget = (size_t)(hotset_budget_frac * (float)pool_size);
        size_t used_b = 0;
        for (auto & [k, fe] : v) {
            if (used_b + fe.size > budget) break;
            hot_protected.insert(k);
            used_b += fe.size;
        }
        if (!hotset_adaptive) {
            access_freq.clear();   // one-shot mode: free the profiling map for good
            access_freq.rehash(0);
        } else {
            age_access_freq();     // adaptive: keep counting, but start the next window fresh-ish
        }
        hotset_finalized = true;
        // Reset the hit/miss instrumentation so the periodic log reports the post-finalize
        // hit rate (the cumulative counter is dominated by the cold profiling phase).
        hits = 0; misses = 0;
        fprintf(stderr, "[expert-cache] hot set: pinned %zu experts (%.0f MB, %.0f%% of pool) "
                "from %zu profiled — hit-rate counter reset for clean post-finalize read\n",
                hot_protected.size(), (float)used_b/(1024*1024),
                100.0f*used_b/(float)pool_size, v.size());
    }

    // Get byte offset for a cached entry (used by offset table builder)
    size_t get_offset(uint64_t k) const {
        auto it = entries.find(k);
        return (it != entries.end()) ? it->second.offset : 0;
    }

    int64_t  hits   = 0;
    int64_t  misses = 0;

    // --- DMA/compute overlap state ---
    // Stream 1 handles H2D DMA for cache misses, overlapping with stream 0 compute (attention).
    // After all experts are copied, flush() makes stream 0 wait for stream 1 via event.
    cudaStream_t dma_stream     = nullptr;   // stream 1 for async H2D
    cudaEvent_t  dma_done_event = nullptr;   // recorded on dma_stream after all H2D
    cudaEvent_t  d2d_done_event = nullptr;   // recorded on compute_stream after deferred D2D
    bool         dma_pending    = false;     // true if any H2D was issued this batch

    // Tier 2: CPU pinned staging buffer (3-tier cache: GPU > CPU pinned > SSD/mmap).
    // Instead of cudaHostRegister on entire mmap'd tensors (which can exceed RAM), a fixed pinned
    // buffer is sized to available RAM. Experts are copied from mmap (tier 3) -> pinned buffer
    // (tier 2) -> GPU pool (tier 1).
    void *   pinned_stage      = nullptr;   // cudaMallocHost'd staging buffer
    size_t   pinned_stage_size = 0;         // total size
    size_t   pinned_stage_used = 0;

    struct staged_entry {
        size_t   offset;          // byte offset in pinned_stage
        size_t   size;            // expert_size
        uint64_t last_access;     // monotonic counter for LRU
    };
    std::unordered_map<uint64_t, staged_entry> staged_entries;  // key = hash(cpu_src, eid)
    uint64_t staged_access_ctr = 0;
    bool     staging_enabled   = false;

    // Staging eviction safety: a slot may have an in-flight async H2D reading it (queued on
    // dma_stream behind GPU kernels). Overwriting it with a host memcpy before that H2D executes
    // would feed the wrong expert's bytes to the GPU. Entries with last_access <= staging_safe_floor
    // have no in-flight reads. The floor advances when dma_done_event (recorded after each flush's
    // H2D batch) is observed complete, since every staged read issued before that record has then
    // executed.
    uint64_t staging_safe_floor       = 0;  // evict only entries at/below this access ctr
    uint64_t staging_ctr_at_dma_record = 0; // staged_access_ctr snapshot at last event record
    int64_t  stage_calls              = 0;  // total stage_expert invocations
    int64_t  stage_evictions          = 0;  // evict-restage count (churn limiter input)

    // Legacy: track directly-pinned host buffers (for models that fit in RAM)
    std::unordered_set<const void *> pinned_buffers;
    std::unordered_set<const void *> host_registered;  // tensors actually cudaHostRegister'd (ALPHA_TENSOR_PIN)
    std::unordered_set<const void *> pin_failed;

    // Pool pre-fill: track tensors that have been bulk-loaded at startup
    std::unordered_set<const void *> prefilled_tensors;

    // Init staging buffer from available system RAM
    void init_staging() {
#ifdef __linux__
        // Read available RAM from /proc/meminfo
        size_t avail_ram = 0;
        FILE * f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                size_t val;
                if (sscanf(line, "MemAvailable: %zu kB", &val) == 1) {
                    avail_ram = val * 1024;
                    break;
                }
            }
            fclose(f);
        }
        if (avail_ram == 0) return;

        // Staging buffer: sole pinned H2D source for expert misses. It should hold the diverse working
        // set (VRAM pool + staging = the fast tier), otherwise misses thrash and fall back to serial
        // direct-mmap H2D (SSD-bound). Default 15GB: bigger staging pins RAM that the model's page cache
        // needs, so on a model larger than RAM it gains nothing (the staging then re-populates from the
        // evicted model on SSD). ALPHA_STAGING_GB overrides; raise it when the model fits RAM with headroom.
        const size_t reserve = 6ULL * 1024 * 1024 * 1024;
        size_t max_staging = 15ULL * 1024 * 1024 * 1024;
        if (const char * e = getenv("ALPHA_STAGING_GB")) {
            int v = atoi(e);
            if (v > 0) max_staging = (size_t)v * 1024ULL * 1024 * 1024;
        }
        size_t budget = (avail_ram > reserve) ? avail_ram - reserve : 0;
        if (budget > max_staging) budget = max_staging;
        if (budget < 512ULL * 1024 * 1024) return;  // too small to bother

        // Try progressively smaller allocations until one succeeds.
        // After tensor pinning, contiguous free RAM may be limited.
        for (int attempt = 0; attempt < 4 && !staging_enabled; attempt++) {
            size_t try_size = budget >> attempt;  // full, half, quarter, eighth
            if (try_size < 512ULL * 1024 * 1024) break;
            cudaError_t err = cudaMallocHost(&pinned_stage, try_size);
            if (err == cudaSuccess) {
                pinned_stage_size = try_size;
                staging_enabled = true;
                fprintf(stderr, "[expert-cache] Tier 2 staging: %.1f GB pinned buffer (%.0f GB RAM available)\n",
                        (float)try_size / (1024*1024*1024), (float)avail_ram / (1024*1024*1024));
            } else {
                (void)cudaGetLastError();
            }
        }
        if (!staging_enabled) {
            fprintf(stderr, "[expert-cache] Tier 2 staging: allocation failed, using direct mmap\n");
        }
#endif
    }

    // Stage an expert: copy from mmap -> pinned buffer. Returns pinned address or nullptr.
    // Slots never change size (same-size reuse only): shrinking pinned_stage_used on the linear
    // allocator would make new appends overwrite live entries at the tail.
    void * stage_expert(const void * mmap_src, size_t offset, size_t expert_size, uint64_t key) {
        if (!staging_enabled) return nullptr;
        stage_calls++;

        // Already staged -> bump LRU, return pinned address
        auto it = staged_entries.find(key);
        if (it != staged_entries.end()) {
            it->second.last_access = ++staged_access_ctr;
            return (char *)pinned_stage + it->second.offset;
        }

        size_t slot = SIZE_MAX;

        // 1) Carve a fresh slot from the linear tail
        if (pinned_stage_used + expert_size <= pinned_stage_size) {
            slot = pinned_stage_used;
            pinned_stage_used += expert_size;
        }

        // 2) Full: evict the LRU same-size entry, but only one that provably has
        //    no in-flight GPU read (last_access <= staging_safe_floor).
        if (slot == SIZE_MAX) {
            // Churn limiter: if the working set greatly exceeds staging capacity,
            // constant evict+memcpy costs more than it saves. Cap evict-restages
            // to 1/DIV of stage calls (ALPHA_STAGE_EVICT_DIV, default 8; lower =
            // more aggressive staging re-population).
            static int64_t evict_div = []() {
                const char * e = getenv("ALPHA_STAGE_EVICT_DIV");
                int v = e ? atoi(e) : 8;
                return (int64_t)(v < 1 ? 1 : (v > 64 ? 64 : v));
            }();
            if (stage_evictions * evict_div > stage_calls) return nullptr;

            // Advance the safe floor: if the last recorded DMA batch has completed,
            // every staged read issued before that record has executed.
            if (dma_done_event && cudaEventQuery(dma_done_event) == cudaSuccess) {
                staging_safe_floor = staging_ctr_at_dma_record;
            }

            uint64_t oldest_access = UINT64_MAX;
            uint64_t oldest_key    = 0;
            bool     found         = false;
            for (auto & [k, e] : staged_entries) {
                if (e.size != expert_size) continue;               // same-size slots only
                if (e.last_access > staging_safe_floor) continue;  // may have in-flight H2D
                if (e.last_access < oldest_access) {
                    oldest_access = e.last_access;
                    oldest_key    = k;
                    found         = true;
                }
            }
            if (!found) return nullptr;  // caller falls back to direct mmap H2D

            slot = staged_entries[oldest_key].offset;
            staged_entries.erase(oldest_key);
            stage_evictions++;
        }

        // Timed: this memcpy is on the flush's critical path, and no GPU-side change (gather kernel,
        // batching) can remove it, so its cost shows whether raising pin coverage would help.
        {
            const auto t0 = std::chrono::steady_clock::now();
            memcpy((char *)pinned_stage + slot, (const char *)mmap_src + offset, expert_size);
            stage_memcpy_ns += (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
        }
        staged_entries[key] = {slot, expert_size, ++staged_access_ctr};
        return (char *)pinned_stage + slot;
    }

    // --- Hybrid CPU/GPU miss tracking ---
    // Strategy: batch MMVQ runs for all experts (hits correct, CPU-half ~0).
    // A CPU thread computes the output for CPU-half misses in parallel.
    // After MMVQ: overwrite CPU-half output positions in dst with CPU results.
    //
    // Per-tensor tracking (keyed by gpu_dst = input_cpy->data):
    struct tensor_miss_info {
        const void * host_base    = nullptr;  // cpu_src (host weight tensor base)
        void *       gpu_dst      = nullptr;  // input_cpy->data for this tensor
        size_t       expert_size  = 0;        // bytes per expert slot (nb02)
        size_t       nb01         = 0;        // row stride in host weights
        int          block_size   = 0;        // ggml_type_size (authoritative offset stride)
        int64_t      n_expert     = 0;
        bool         flushed      = false;    // true after flush processed this tensor
        std::vector<int64_t> all_miss_ids;    // accumulated during copy phase
        std::vector<int64_t> cpu_miss_ids;    // second half (assigned at flush)
    };
    std::unordered_map<void *, tensor_miss_info> per_tensor_misses;
    bool token_has_misses = false;

    // CPU worker state: persistent thread for miss expert computation
    struct cpu_expert_work {
        int64_t      expert_id;
        const void * host_weights;    // host_base + expert_id * expert_size
        size_t       nb01;            // row stride in quantized weights
        float *      dst_ptr;         // where to write result in dst (GPU, H2D target)
        const float* src1_ptr;        // activation (CPU copy)
        int          n_tokens;        // tokens routed to this expert
        int          ne0;             // output dimension
        int          ne10;            // input dimension (hidden dim)
    };
    std::vector<cpu_expert_work>  cpu_work_items;
    std::vector<float>            cpu_activations;  // D2H'd activations for CPU experts
    std::vector<float>            cpu_results;      // CPU-computed outputs (uploaded to GPU)
    std::atomic<bool>             cpu_done{false};
    std::thread                   cpu_thread;

    void clear_miss_tracking() {
        per_tensor_misses.clear();
        token_has_misses = false;
        cpu_work_items.clear();
        cpu_activations.clear();
        cpu_results.clear();
        cpu_done.store(false);
    }

    // --- Adaptive insert state ---
    struct freq_entry {
        int32_t      count       = 0;
        const void * cpu_src     = nullptr;   // host weight tensor base (for bulk insert)
        int64_t      expert_id   = -1;
        size_t       expert_size = 0;
        // Quant block size of this expert's own tensor, captured during observation while
        // g_pcache_cur_block_size still refers to it. bulk_insert_hot runs inside one tensor's
        // copy but inserts experts belonging to all of them, so the global "current" value would
        // give the wrong stride on any model with mixed expert quants (e.g. iq3_xxs gate/up = 98 B,
        // iq4_xs down = 136 B), and the kernel would read a valid, in-bounds, but wrong pool slot.
        uint32_t     block_size  = 0;   // 0 = unknown, fall back to inference
    };
    std::unordered_map<uint64_t, freq_entry> freq;

    int64_t  total_calls      = 0;
    bool     warmup_done      = false;
    bool     immediate_cache  = false;  // true when pool fits all experts (no observation needed)
    int32_t  insert_threshold = 0;
    float    dma_cost_us      = 42.0f;  // CPU->GPU per expert (from pcie_bw_gbps)
    float    gpu_copy_us      = 1.7f;   // GPU->GPU per expert

    // Warmup = 64 tokens x ~960 expert copies/token (8 active x 120 tensors)
    // Only used when pool_size < total_expert_bytes (need frequency gating).
    // When pool fits everything, immediate_cache=true skips warmup entirely.
    static constexpr int EXPERTS_PER_TOKEN_EST = 960;
    static constexpr int WARMUP_TOKENS         = 64;
    static constexpr int64_t WARMUP_CALLS      = (int64_t)WARMUP_TOKENS * EXPERTS_PER_TOKEN_EST;

    static uint64_t key(const void * base, int64_t eid) {
        uint64_t h = (uint64_t)(uintptr_t)base;
        h ^= (uint64_t)eid * 0x9E3779B97F4A7C15ULL;
        return h;
    }

    // Pool VRAM preallocated at model load time (ggml_cuda_expert_pcache_prealloc). With a lazy pool
    // init, a long first request lets prefill compute buffers grab VRAM first and the pool ends up much
    // smaller for the rest of the run. Preallocating right after context creation makes the pool size
    // deterministic; a later UBBoost buffer grow simply fails and rolls back to chunked prefill.
    void * prealloc_base  = nullptr;
    size_t prealloc_bytes = 0;

    void init(size_t budget, size_t expert_size_hint) {
        if (prealloc_base && prealloc_bytes >= budget) {
            pool = prealloc_base;   // adopt the load-time allocation (full size)
            budget = prealloc_bytes;
        } else {
            CUDA_CHECK(cudaMalloc(&pool, budget));
        }
        pool_size = budget;
        // Carve a scratch ring from the pool tail: transient landing slots for decode misses that
        // the admission filter rejects. The offset table points kernels at these slots, so every
        // decode tensor is table-mode (input_cpy never written on decode), which keeps kernel params
        // uniform and makes CUDA graphs safe.
        // Sized as a slot count (~one pass of non-admitted misses) so it scales with the model's
        // expert size, clamped to [64MB, budget/8]. A smaller ring returns pool hit coverage while
        // event-guarded wraps stay cheap. ALPHA_SCRATCH_MB overrides.
        const size_t SCRATCH_SLOTS_TARGET = 280;
        size_t ring = SCRATCH_SLOTS_TARGET * std::max(expert_size_hint, (size_t)(256 * 1024));
        if (const char * e = getenv("ALPHA_SCRATCH_MB")) {
            int v = atoi(e);
            if (v >= 64 && v <= 8192) ring = (size_t)v * 1024 * 1024;
        }
        ring = std::max(ring, (size_t)64 * 1024 * 1024);
        ring = std::min(ring, budget / 8);
        scratch_size = ring;
        scratch_base = budget - scratch_size;
        pool_size    = budget - scratch_size;  // insert() capacity excludes the ring
        // Hybrid: reserve a small never-recycled zone at the ring tail for
        // zero-expert images (a few expert sizes x ~2MB each)
        zero_zone_size = std::min(scratch_size / 8, (size_t)32 * 1024 * 1024);
        scratch_size  -= zero_zone_size;
        fprintf(stderr, "[expert-cache] persistent GPU pool: %.0f MB (+%.0f MB miss scratch ring)\n",
            (float)pool_size / (1024 * 1024), (float)scratch_size / (1024 * 1024));
    }

    // --- Decode-miss scratch ring (inside the pool allocation, after pool_size) ---
    size_t scratch_base = 0;   // byte offset of ring start within the pool allocation
    size_t scratch_size = 0;   // ring bytes
    size_t scratch_off  = 0;   // bump pointer within the ring
    cudaEvent_t scratch_wrap_event = nullptr;
    // epoch ring: armed at pass_begin, consumed by the first scratch_alloc of the pass
    bool    scratch_reset_pending = false;
    int64_t scratch_declined      = 0;   // reported in [pcache-ring]; the ring always wraps, so it stays 0
    int64_t scratch_epoch_resets  = 0;   // pass-boundary ring recycles

    // --- Hybrid zero-expert zone (tail of the original ring, never recycled) ---
    // CPU-half experts get offset-table entries pointing at an all-zero expert image
    // (zero Q4_K/Q5_K/Q6_K blocks dequantize to 0), so batch MMVQ contributes exactly
    // 0 for them and the CPU correction overwrites their dst rows afterward.
    size_t zero_zone_off  = 0;
    size_t zero_zone_size = 0;
    std::unordered_map<size_t, int64_t> zero_slot_by_size;  // expert_size -> absolute byte offset

    // Absolute pool-byte-offset of the zero expert for this size (memset once), or -1.
    int64_t zero_slot(size_t size, size_t type_align, cudaStream_t stream) {
        auto it = zero_slot_by_size.find(size);
        if (it != zero_slot_by_size.end()) return it->second;
        size_t abs = scratch_base + scratch_size + zero_zone_off;
        if (type_align > 1) abs = (abs + type_align - 1) / type_align * type_align;
        if (abs + size > scratch_base + scratch_size + zero_zone_size) return -1;
        CUDA_CHECK(cudaMemsetAsync((char *)pool + abs, 0, size, stream));
        zero_zone_off = (abs + size) - (scratch_base + scratch_size);
        zero_slot_by_size[size] = (int64_t)abs;
        return (int64_t)abs;
    }

    // Miss-path instrumentation: where does miss traffic flow?
    // src: pinned staging (async on dma stream) vs pageable mmap (compute stream, serial)
    // dst: persistent pool slot (admitted) vs transient scratch ring slot
    // miss_src_staged is the sum of the cudaHostRegister'd single-hop DMA and the staging path, which
    // have very different costs (the latter pays a synchronous host memcpy in stage_expert), so the two
    // are also counted separately.
    int64_t miss_src_staged  = 0;   // = pinned + stagecopy (back-compat total)
    int64_t miss_src_pinned  = 0;   // single-hop DMA from a cudaHostRegister'd expert tensor
    int64_t miss_src_stagecp = 0;   // served via the pinned staging buffer
    int64_t stage_memcpy_ns  = 0;   // wall time inside stage_expert's memcpy (critical path)

    // ALPHA_PCACHE_DMA_STATS: instantaneous H2D rate vs duty cycle.
    // A wall-clock average of expert traffic cannot distinguish "the link is slow" from "the link is
    // idle most of the layer". These timing-enabled events bracket each flush's copy burst on
    // h2d_stream, so
    //   inst_gbps = bytes / event-elapsed   (how fast the link actually runs)
    //   duty      = event-elapsed / decode wall (how much of the time it runs at all)
    // Queried lazily (never blocks). 64 slots: the overlap probe below pairs a fence-ring entry with
    // the fetch slot open at the time, so a fetch slot must not be recycled before that fence entry drains.
    enum { FETCH_RING = 64 };     // enum, not static const int: no out-of-class definition needed
    cudaEvent_t fetch_beg[FETCH_RING] = {};
    cudaEvent_t fetch_end[FETCH_RING] = {};
    int64_t     fetch_ev_bytes[FETCH_RING] = {};
    int         fetch_head = 0, fetch_tail = 0;   // ring: [tail, head) awaiting query
    int64_t     fetch_burst_bytes = 0;            // bytes issued in the burst being timed
    bool        fetch_open = false;
    double      fetch_ms_total = 0.0;
    int64_t     fetch_bytes_total = 0, fetch_bursts = 0;
    // H2D bytes that cross PCIe but are NOT inside the timed bracket, so they must never be
    // added to fetch_burst_bytes (they would corrupt inst) yet must not vanish from the budget
    // either: pool pre-fill and the panic fallback (offstream), and speculative prefetch (pf).
    int64_t     offstream_bytes = 0, pf_bytes = 0;

    void fetch_timing_drain() {                   // non-blocking: harvest completed pairs
        while (fetch_tail != fetch_head) {
            const int i = fetch_tail % FETCH_RING;
            if (cudaEventQuery(fetch_end[i]) != cudaSuccess) return;   // still in flight
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, fetch_beg[i], fetch_end[i]) == cudaSuccess) {
                fetch_ms_total    += ms;
                fetch_bytes_total += fetch_ev_bytes[i];
                fetch_bursts++;
            }
            fetch_tail++;
        }
    }
    void fetch_timing_begin(cudaStream_t s) {
        if (fetch_head - fetch_tail >= FETCH_RING) return;   // ring full -> skip this burst
        const int i = fetch_head % FETCH_RING;
        if (!fetch_beg[i]) {                                  // timing enabled (unlike
            cudaEventCreate(&fetch_beg[i]);                   // every other event here, which
            cudaEventCreate(&fetch_end[i]);                   // uses cudaEventDisableTiming)
        }
        cudaEventRecord(fetch_beg[i], s);
        fetch_burst_bytes = 0;
        fetch_open = true;
    }
    void fetch_timing_end(cudaStream_t s) {
        if (!fetch_open) return;
        fetch_open = false;
        if (fetch_burst_bytes == 0) return;                   // nothing issued -> drop the pair
        const int i = fetch_head % FETCH_RING;
        cudaEventRecord(fetch_end[i], s);
        fetch_ev_bytes[i] = fetch_burst_bytes;
        fetch_head++;
    }

    // GPU stall: how long compute_stream sat blocked waiting on expert bytes.
    // A cudaStreamWaitEvent cannot be timed directly, but it can be bracketed on the waiting stream:
    // fence_beg executes when compute drains down to the wait, and fence_end executes only once the
    // wait has cleared. elapsed(beg,end) is the blocked interval plus ~1-2us of event-record overhead.
    // Duty cycle alone does not show whether compute is serialized behind the link. Split by size:
    //   > STALL_EPS_MS => data stall: compute really is waiting on PCIe  => pipeline deeper
    //   ~ 0            => bytes had already landed; pure fence cost      => fuse fences
    static constexpr double STALL_EPS_MS = 0.005;   // 5us: floor for two event records
    // Must hold a whole token's fences (two per MoE tensor: exit + drain), otherwise the ring fills
    // early in a sampled token and the rest are skipped, biasing the stall estimate.
    enum { FENCE_RING = 256 };
    cudaEvent_t fence_beg[FENCE_RING] = {};
    cudaEvent_t fence_end[FENCE_RING] = {};
    uint8_t     fence_is_drain[FENCE_RING] = {};
    int         fence_head = 0, fence_tail = 0;
    double      stall_data_ms = 0.0, stall_ovh_ms = 0.0;
    int64_t     stall_data_n  = 0,   stall_ovh_n  = 0;
    int64_t     fence_exit_n  = 0,   fence_drain_n = 0;

    // Overlap probe: when does compute arrive, relative to the burst it waits on? STALL contains the
    // transport time it is exposed to, so it cannot distinguish:
    //   (a) the burst starts early and compute arrives near its end  -> STALL is a small tail,
    //       transport is already well hidden, and the fix is "move fewer bytes";
    //   (b) the burst and compute arrival are simultaneous           -> nothing overlaps at all,
    //       and the fix is "give compute something to do during the burst" (or issue earlier).
    // overlap = elapsed(fetch_beg, fence_beg): how long the burst had been running when compute
    // reached the fence. ~0 => (b) lock-step. Large positive => (a). Negative => compute was already
    // waiting before the burst started, i.e. the host is the pacer, not the link.
    int64_t     fence_fetch_h[FENCE_RING] = {};   // fetch_head at pairing time, -1 = unpaired
    double      overlap_ms_total = 0.0;
    int64_t     overlap_n = 0, overlap_neg_n = 0;
    // Dead wait: compute blocked while the link was idle. STALL says "compute was blocked" but not
    // whether anything was in flight while it waited:
    //   blocked with transfer    -> transport-bound. Fewer bytes / more overlap helps.
    //   blocked without one      -> dead wait. The bytes were not in flight; fewer bytes would
    //                               change nothing. Look at issue latency, event propagation,
    //                               or another serializer.
    // Per paired sample:  dead = stall - (burst - overlap)
    // i.e. anything the GPU waited beyond the burst's own remaining time. Needs no new events:
    // burst = elapsed(fetch_beg, fetch_end), overlap = elapsed(fetch_beg, fence_beg),
    // stall = elapsed(fence_beg, fence_end), all already recorded.
    double      dead_ms_total = 0.0;
    int64_t     dead_n = 0;

    void fence_timing_drain() {                   // non-blocking, same idiom as fetch_timing_drain
        while (fence_tail != fence_head) {
            const int i = fence_tail % FENCE_RING;
            if (cudaEventQuery(fence_end[i]) != cudaSuccess) return;   // still in flight
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, fence_beg[i], fence_end[i]) == cudaSuccess && ms >= 0.0f) {
                if (ms > STALL_EPS_MS) { stall_data_ms += ms; stall_data_n++; }
                else                   { stall_ovh_ms  += ms; stall_ovh_n++;  }
            }
            // Overlap: how far into its burst was the link when compute reached this fence?
            // Only valid while the paired fetch slot has not been recycled (fetch_head advances
            // once per closed burst; the slot dies at h + FETCH_RING).
            const int64_t h = fence_fetch_h[i];
            if (h >= 0 && fetch_head < h + FETCH_RING) {
                const int fi = (int)(h % FETCH_RING);
                float oms = 0.0f;
                if (cudaEventElapsedTime(&oms, fetch_beg[fi], fence_beg[i]) == cudaSuccess) {
                    overlap_ms_total += oms;          // signed: negative = compute waiting first
                    overlap_n++;
                    if (oms < 0.0f) overlap_neg_n++;
                    // Dead wait. fetch_end is recorded later in program order than fence_end, so
                    // query it before use; a not-yet-complete event would give a garbage elapsed.
                    float bms = 0.0f;
                    if (cudaEventQuery(fetch_end[fi]) == cudaSuccess &&
                        cudaEventElapsedTime(&bms, fetch_beg[fi], fetch_end[fi]) == cudaSuccess && bms > 0.0f) {
                        const double remaining = (double)bms - (double)oms;   // burst left when compute arrived
                        const double dead     = (double)ms - (remaining > 0.0 ? remaining : 0.0);
                        if (dead > 0.0) dead_ms_total += dead;                // only the excess counts
                        dead_n++;
                    }
                }
            }
            fence_tail++;
        }
    }

    // The forward fence, in one place. Records dma_done on the H2D stream and makes `compute`
    // wait on it, while timing how long the wait actually blocks. Both fence sites (entry drain
    // and flush exit) route through here, since instrumenting only the exit would miss stalls.
    // fence_wait_on brackets a wait on an already-recorded event (LAYER_FUSE=2 sibling waits on
    // its per-member completion). It must be instrumented like every other fence, otherwise the
    // wait would be missing from the STALL figure.
    void fence_wait_on(cudaStream_t compute, cudaEvent_t ev, bool is_drain) {
        if (!ev) return;
        if (is_drain) fence_drain_n++; else fence_exit_n++;
        const bool tm = g_pcache_dma_stats_now && (fence_head - fence_tail < FENCE_RING);
        int i = 0;
        if (tm) {
            i = fence_head % FENCE_RING;
            if (!fence_beg[i]) { cudaEventCreate(&fence_beg[i]); cudaEventCreate(&fence_end[i]); }
            CUDA_CHECK(cudaEventRecord(fence_beg[i], compute));
        }
        CUDA_CHECK(cudaStreamWaitEvent(compute, ev, 0));
        if (tm) {
            CUDA_CHECK(cudaEventRecord(fence_end[i], compute));
            fence_is_drain[i] = is_drain ? 1 : 0;
            fence_fetch_h[i]  = -1;      // not paired to a burst we opened
            fence_head++;
        }
    }

    void fence_wait(cudaStream_t compute, cudaStream_t h2d, bool is_drain) {
        if (is_drain) fence_drain_n++; else fence_exit_n++;
        CUDA_CHECK(cudaEventRecord(dma_done_event, h2d));
        staging_ctr_at_dma_record = staged_access_ctr;   // staged-read safety snapshot
        const bool tm = g_pcache_dma_stats_now && (fence_head - fence_tail < FENCE_RING);
        int i = 0;
        if (tm) {
            i = fence_head % FENCE_RING;
            if (!fence_beg[i]) {                          // timing enabled (unlike dma_done_event)
                cudaEventCreate(&fence_beg[i]);
                cudaEventCreate(&fence_end[i]);
            }
            CUDA_CHECK(cudaEventRecord(fence_beg[i], compute));
        }
        CUDA_CHECK(cudaStreamWaitEvent(compute, dma_done_event, 0));
        if (tm) {
            CUDA_CHECK(cudaEventRecord(fence_end[i], compute));
            fence_is_drain[i] = is_drain ? 1 : 0;
            // Pair with the burst this fence is waiting on; only meaningful for the exit fence
            // (the entry drain runs before fetch_timing_begin, so no burst is open yet).
            fence_fetch_h[i] = (!is_drain && fetch_open) ? (int64_t)fetch_head : -1;
            fence_head++;
        }
    }
    int64_t miss_src_mmap    = 0;
    int64_t miss_dst_pool    = 0;
    int64_t miss_dst_scratch = 0;
    int64_t miss_dst_cpu     = 0;   // hybrid: misses computed on the CPU (no H2D at all)
    int64_t cold_skipped     = 0;   // cold-skip: cold low-weight experts dropped (no H2D)
    int64_t scratch_wraps    = 0;

    // --- Per-token routed-expert trace (ALPHA_TRACE_IDS, decode only) ---
    struct prefetch_tensor_info {
        std::vector<int32_t> curr_used;      // experts used this pass (written out at flush)
        std::vector<uint8_t> curr_hit;       // trace annotation: pool-resident at request time
    };
    std::unordered_map<const void *, prefetch_tensor_info> prefetch_info;
    // Cross-layer prefetch counters, still reported in the miss-flow and [pf] lines; nothing
    // prefetches any more, so they stay 0.
    int64_t prefetch_issued  = 0;
    int64_t prefetch_skipped = 0;
    int64_t prefetch_used    = 0;

    // Allocate a transient slot for a decode miss. Returns the absolute byte offset
    // within the pool allocation (so table entries divide by type_size uniformly),
    // or SIZE_MAX if the ring is disabled.
    // Wrap safety: on wrap, older slots may still be read by kernels enqueued earlier
    // this pass. Record an event on the compute stream (captures every kernel enqueued
    // so far) and make the DMA stream wait, so the overwrite H2D cannot execute until
    // those readers are done. Cross-pass reuse needs no guard (host syncs on logits).
    size_t scratch_alloc(size_t size, size_t type_align, cudaStream_t compute_stream) {
        if (scratch_size == 0 || size > scratch_size) return SIZE_MAX;
        // Pass-boundary recycle. Every kernel of the previous pass is enqueued by the time the new
        // pass's first input copy runs, so an event recorded here covers every reader of every ring
        // slot. This is the only point at which recycling is sound.
        if (g_pcache_scratch_epoch && scratch_reset_pending) {
            if (scratch_off != 0) {
                if (!scratch_wrap_event) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&scratch_wrap_event, cudaEventDisableTiming));
                }
                CUDA_CHECK(cudaEventRecord(scratch_wrap_event, compute_stream));
                if (dma_stream) CUDA_CHECK(cudaStreamWaitEvent(dma_stream, scratch_wrap_event, 0));
                scratch_epoch_resets++;
                scratch_wraps++;
            }
            scratch_off = 0;
            scratch_reset_pending = false;
        }
        size_t abs = scratch_base + scratch_off;
        if (type_align > 1) abs = (abs + type_align - 1) / type_align * type_align;
        if (abs + size > scratch_base + scratch_size) {
            // Wrap: guard in-flight readers, then restart at ring base
            if (!scratch_wrap_event) {
                CUDA_CHECK(cudaEventCreateWithFlags(&scratch_wrap_event, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(scratch_wrap_event, compute_stream));
            if (dma_stream) {
                CUDA_CHECK(cudaStreamWaitEvent(dma_stream, scratch_wrap_event, 0));
            }
            scratch_wraps++;
            abs = scratch_base;
            if (type_align > 1) abs = (abs + type_align - 1) / type_align * type_align;
        }
        scratch_off = (abs + size) - scratch_base;
        return abs;
    }

    void init_dma_stream(ggml_backend_cuda_context * ctx) {
        if (!dma_stream) {
            dma_stream = ctx->stream(ctx->device, 1);  // stream 1 for H2D DMA
            CUDA_CHECK(cudaEventCreateWithFlags(&dma_done_event, cudaEventDisableTiming));
        }
    }

    // Pin host buffer for true async DMA on non-default streams.
    size_t total_pinned_bytes = 0;
    size_t pin_budget = 0;   // max bytes to pin (0 = unlimited until first failure)
    bool   pin_closed = false;  // live MemAvailable floor hit -> no more pins this run

    // Pin host buffer for async DMA. Try direct cudaHostRegister first (fastest:
    // single-hop async DMA). If the pin fails or the budget is exceeded, fall back to the tier 2
    // staging buffer. The budget prevents pinning more than the system can spare.
    void ensure_pinned(const void * cpu_src, size_t total_size) {
        if (pinned_buffers.count(cpu_src)) return;
        // Direct expert-tensor pinning, default on (see g_pcache_tensor_pin): a pinned tensor lets a
        // miss DMA single-hop from pinned RAM to VRAM, with no staging memcpy and no pageable-sync
        // fallback. Budget-capped; beyond the budget the cold tail falls back to staging.
        if (!g_pcache_tensor_pin) {
            pinned_buffers.insert(cpu_src);   // staging-only (ALPHA_TENSOR_PIN=0)
            if (!staging_enabled) init_staging();
            return;
        }
        // Pin budget exceeded -> fall back to staging for the remaining (cold-tail) tensors.
        static size_t eff_budget = []() -> size_t {
            const char * e = getenv("ALPHA_PIN_BUDGET_GB");
            return e && atoi(e) > 0 ? (size_t)atoi(e) * 1024ULL*1024*1024 : 0;
        }();
        size_t budget = eff_budget ? eff_budget : pin_budget;
        if (pin_closed || (budget > 0 && total_pinned_bytes + total_size > budget)) {
            pinned_buffers.insert(cpu_src);
            if (!staging_enabled) init_staging();
            return;
        }
        // Live safety floor: the one-shot budget is computed from MemAvailable, which counts
        // reclaimable page cache, and pinning locks those pages, so a budget that looked safe can
        // starve userspace. MemAvailable is re-read before every pin; once it drops below the floor,
        // pinning stops for good and the tail is served from staging. ALPHA_PIN_FLOOR_GB overrides (min 4).
#ifdef __linux__
        {
            static size_t floor_bytes = []() -> size_t {
                const char * e = getenv("ALPHA_PIN_FLOOR_GB");
                long v = e ? atol(e) : 10;
                if (v < 4) v = 4;
                return (size_t)v * 1024ULL*1024*1024;
            }();
            size_t avail_now = 0;
            FILE * f = fopen("/proc/meminfo", "r");
            if (f) {
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    size_t val;
                    if (sscanf(line, "MemAvailable: %zu kB", &val) == 1) { avail_now = val * 1024; break; }
                }
                fclose(f);
            }
            if (avail_now > 0 && avail_now < floor_bytes + total_size) {
                pin_closed = true;
                fprintf(stderr, "[expert-cache] pin STOPPED at %.1f GB: MemAvailable %.1f GB < floor %.1f GB "
                        "(remaining tensors → staging)\n",
                        (float)total_pinned_bytes/(1024*1024*1024),
                        (float)avail_now/(1024*1024*1024), (float)floor_bytes/(1024*1024*1024));
                pinned_buffers.insert(cpu_src);
                if (!staging_enabled) init_staging();
                return;
            }
        }
#endif
        cudaError_t err = cudaHostRegister(const_cast<void *>(cpu_src), total_size,
                                           cudaHostRegisterPortable | cudaHostRegisterReadOnly);
        pinned_buffers.insert(cpu_src);
        if (err == cudaSuccess) {
            host_registered.insert(cpu_src);   // miss path DMAs single-hop from this tensor
            g_inl_host_registered = &host_registered;   // CPU-inline read-class: pinned lookup
            total_pinned_bytes += total_size;
            static int64_t plog = 0;
            if (++plog <= 4 || (plog & 31) == 0)
                fprintf(stderr, "[expert-cache] tensor-pinned %.0f MB (total %.1f GB)\n",
                        (float)total_size/(1024*1024), (float)total_pinned_bytes/(1024*1024*1024));
        } else {
            (void)cudaGetLastError();
            if (!staging_enabled) init_staging();  // pin failed -> staging fallback
        }
    }

    // Check if a buffer is actually pinned (safe for stream 1 async DMA)
    // Returns true if async DMA on stream 1 is safe for this buffer.
    // True when: directly pinned (cudaHostRegister) OR staging buffer available.
    bool is_pinned(const void * cpu_src) const {
        // Directly pinned tensor -> its pages are DMA-safe on the async stream.
        if (host_registered.count(cpu_src)) return true;
        return false;
    }

    // Record completion event on DMA stream, to be waited on before compute
    void mark_dma_pending() {
        if (!dma_pending && dma_stream) {
            dma_pending = true;
        }
    }

    // Called after all expert copies: record event on stream 1, make stream 0 wait.
    // Ensures all H2D DMA (stream 1) completes before graph launch reads input_cpy.
    // D2D copies (stream 0) already ran in parallel with H2D; that is the overlap.
    void flush(cudaStream_t compute_stream) {
        if (dma_pending && dma_stream) {
            fence_wait(compute_stream, dma_stream, /*is_drain=*/true);
            dma_pending = false;
        }
    }

    void * lookup(uint64_t k) {
        auto it = entries.find(k);
        if (it == entries.end()) return nullptr;
        it->second.last_access = ++lru_counter;
        if (it->second.hits < 0xffffffffu) it->second.hits++;   // freq input for eviction scoring
        return (char *)pool + it->second.offset;
    }

    void * insert(uint64_t k, size_t size, size_t type_align = 0) {
        // Try linear allocation first (fast path)
        size_t aligned_used = used;
        if (type_align > 1) {
            aligned_used = (used + type_align - 1) / type_align * type_align;
        }
        if (aligned_used + size <= pool_size) {
            size_t off = aligned_used;
            used = aligned_used + size;
            entries[k] = {off, size, ++lru_counter, 1};
            size_ring[size].push_back(k);
            return (char *)pool + off;
        }

        // Admission filter: when the working set exceeds the pool, evict-on-every-miss is pure LRU
        // thrash: each insert evicts an entry that will itself miss again soon, and every eviction
        // costs an extra copy. Damped admission (only 1-in-2^N misses may evict) lets the pool
        // converge on the most frequently missed experts instead of cycling.
        // ALPHA_POOL_ADMIT=N sets the shift (0 = admit all = plain LRU).
        static uint32_t admit_mask = []() {
            const char * e = getenv("ALPHA_POOL_ADMIT");
            int shift = e ? atoi(e) : 5;  // default: 1-in-32
            if (shift < 0)  shift = 0;
            if (shift > 10) shift = 10;
            return (uint32_t)((1u << shift) - 1);
        }();
        // Hot-set (ALPHA_HOTSET): a protected (frequency-pinned) expert always gets a
        // pool slot: it bypasses the admission filter and evicts a non-protected
        // victim. This is what pins the hot core; LRU handles the rest of the pool.
        const bool is_protected = hotset_finalized && hot_protected.count(k) > 0;

        if (!is_protected) {
            if ((++admission_ctr & admit_mask) != 0) return nullptr;
        }

        // Pool full: evict a same-size victim and reuse its slot. Same-size is required
        // because quant types have different expert sizes (Q4_K gate/up vs Q5_K down).
        // Sampled (lazy) eviction: draw K keys from size_ring[size] and evict the oldest
        // by last_access (Redis-style approximate LRU, O(K) not O(n)). The O(n) scan
        // (ALPHA_EVICT_SAMPLE_K=0) is far too slow on the flush path with frequent admits.
        static const int sample_k = []() {
            const char * e = getenv("ALPHA_EVICT_SAMPLE_K");
            int v = e ? atoi(e) : 24;
            return v < 0 ? 0 : (v > 256 ? 256 : v);
        }();

        uint64_t oldest_key = 0;
        uint64_t oldest_access = UINT64_MAX;
        // Fallback victim among protected entries. Protection is a preference, not an absolute:
        // if every same-size candidate is hot-set-protected and none were evicted, the expert would
        // never be admitted to the persistent pool, a permanent lockout whenever the workload shifts
        // away from whatever the hot set profiled on. Evicting the LRU protected entry as a last
        // resort keeps the hot set's benefit under a stable workload while letting a genuinely new
        // working set take hold.
        uint64_t prot_key = 0;
        uint64_t prot_access = UINT64_MAX;
        // ALPHA_EVICT_POLICY=1 (freq/age) victim candidate
        uint64_t best_key = 0, best_freq = 1, best_age = 1;

        if (sample_k > 0) {
            auto rit = size_ring.find(size);
            if (rit != size_ring.end()) {
                auto & ring = rit->second;
                int seen = 0;
                int budget = sample_k * 4;   // resample budget for stale/protected skips
                while (seen < sample_k && budget-- > 0 && !ring.empty()) {
                    size_t idx = (size_t)(rng() % ring.size());
                    uint64_t ck = ring[idx];
                    auto eit = entries.find(ck);
                    // Only dead keys may be swap-removed. Dropping a live-but-protected (or
                    // touched-this-pass) entry from the ring would drain it toward empty, and
                    // eviction would stop finding any victim at all.
                    const bool dead = (eit == entries.end()) || (eit->second.size != size);
                    if (dead) {
                        ring[idx] = ring.back();
                        ring.pop_back();
                        continue;
                    }
                    if (eit->second.last_access >= pool_pass_floor) { seen++; continue; }  // in use this pass
                    if (hotset_finalized && hot_protected.count(ck) > 0) {                 // keep as fallback
                        if (eit->second.last_access < prot_access) {
                            prot_access = eit->second.last_access;
                            prot_key    = ck;
                        }
                        seen++;
                        continue;
                    }
                    seen++;
                    // ALPHA_EVICT_POLICY=1: evict the lowest frequency-per-age instead of simply
                    // the oldest (closer to Belady OPT than sampled LRU or LFU on recorded traces).
                    // Compare a/b < c/d as a*d < c*b to stay in integers; age is at least 1.
                    if (g_pcache_evict_policy == 1) {
                        const uint64_t age = (lru_counter > eit->second.last_access)
                                           ? (lru_counter - eit->second.last_access) : 1;
                        const uint64_t f   = eit->second.hits ? eit->second.hits : 1;
                        if (best_key == 0 || alpha_mul_lt(f, best_age, best_freq, age)) {
                            best_freq = f; best_age = age; best_key = ck;
                        }
                        continue;
                    }
                    if (eit->second.last_access < oldest_access) {
                        oldest_access = eit->second.last_access;
                        oldest_key = ck;
                    }
                }
            }
        } else {
            // O(n) scan (ALPHA_EVICT_SAMPLE_K=0).
            for (auto & [ek, e] : entries) {
                if (e.size != size) continue;
                if (e.last_access >= pool_pass_floor) continue;
                if (hotset_finalized && hot_protected.count(ek)) {   // fallback victim, see above
                    if (e.last_access < prot_access) { prot_access = e.last_access; prot_key = ek; }
                    continue;
                }
                if (e.last_access < oldest_access) {
                    oldest_access = e.last_access;
                    oldest_key = ek;
                }
            }
        }
        if (g_pcache_evict_policy == 1 && best_key != 0) {
            oldest_key = best_key; oldest_access = 0;   // freq/age winner
        }
        if (oldest_access == UINT64_MAX) {
            // Nothing unprotected to evict. Rather than refuse (which sends the expert to scratch
            // forever and locks the new working set out of the pool), give up the LRU protected
            // slot and drop its protection; the hot set no longer reflects the workload.
            // TODO: the damper at the top has already consumed a slot of the 1-in-32 budget before
            // this scan runs, so a failure here lowers the real admit rate below 1/32 by the
            // scan-failure fraction. Moving the decrement to the successful return would fix it,
            // but changes admits/token.
            if (prot_access == UINT64_MAX) { return nullptr; }   // nothing evictable
            oldest_key    = prot_key;
            oldest_access = prot_access;
            hot_protected.erase(oldest_key);
            hot_evicted++;
        }

        // Reuse the evicted entry's slot; register the new key in the size ring.
        // Past pool-fill this is the only path that grants a slot, so admits == evictions.
        size_t reuse_offset = entries[oldest_key].offset;
        entries.erase(oldest_key);
        entries[k] = {reuse_offset, size, ++lru_counter, 1};
        size_ring[size].push_back(k);
        lru_evictions++;
        return (char *)pool + reuse_offset;
    }

    // Compute adaptive threshold from observed frequency distribution.
    // Sets insert_threshold so only the hottest experts that fit in the pool get cached.
    void compute_threshold(size_t default_expert_size) {
        size_t max_slots = pool_size / std::max(default_expert_size, (size_t)1);

        // Sort observed experts by frequency (descending)
        std::vector<int32_t> counts;
        counts.reserve(freq.size());
        for (auto & [k, fe] : freq) {
            counts.push_back(fe.count);
        }
        std::sort(counts.begin(), counts.end(), std::greater<int32_t>());

        if (counts.size() <= max_slots) {
            insert_threshold = 1;  // everything fits
        } else {
            insert_threshold = counts[max_slots];
            if (insert_threshold < 2) insert_threshold = 2;  // must be seen 2+ times
        }

        fprintf(stderr, "[expert-cache] adaptive: threshold=%d, %zu unique experts, %zu pool slots, "
                "DMA=%.1fus, D2D=%.1fus\n",
                insert_threshold, freq.size(), max_slots, dma_cost_us, gpu_copy_us);
    }

    // Bulk-insert the hottest experts after warmup using stored cpu_src pointers.
    void bulk_insert_hot(size_t default_expert_size, cudaStream_t stream) {
        // Collect candidates above threshold, sorted by frequency
        struct candidate { int32_t count; uint64_t k; };
        std::vector<candidate> cands;
        for (auto & [k, fe] : freq) {
            if (fe.count >= insert_threshold && entries.find(k) == entries.end()) {
                cands.push_back({fe.count, k});
            }
        }
        std::sort(cands.begin(), cands.end(),
                  [](const candidate & a, const candidate & b) { return a.count > b.count; });

        int inserted = 0;
        for (auto & c : cands) {
            auto & fe = freq[c.k];
            if (!fe.cpu_src || fe.expert_id < 0) continue;

            size_t sz = fe.expert_size > 0 ? fe.expert_size : default_expert_size;
            // Offset-table stride: the expert's own captured block size, not pcache_block_size(sz),
            // which resolves against g_pcache_cur_block_size, i.e. the tensor whose copy happens to be
            // running this bulk insert, the wrong stride for other tensors' experts on a mixed-quant model.
            size_t ta = fe.block_size > 0 ? (size_t) fe.block_size : pcache_block_size(sz);
            void * slot = insert(c.k, sz, ta);
            if (!slot) break;  // pool full

            const size_t offset = fe.expert_id * sz;
            CUDA_CHECK(cudaMemcpyAsync(
                slot, (const char *)fe.cpu_src + offset, sz,
                cudaMemcpyHostToDevice, stream));
            inserted++;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, "[expert-cache] bulk insert: %d experts (%.1f MB), pool %.1f / %.1f MB\n",
                inserted, (float)inserted * default_expert_size / (1024*1024),
                (float)used / (1024*1024), (float)pool_size / (1024*1024));

        // Clear cpu_src pointers (no longer needed)
        for (auto & [k, fe] : freq) { fe.cpu_src = nullptr; }
    }
};

static cuda_expert_persist_cache g_expert_pcache;

// VRAM headroom the expert pool leaves free (for the graph's compute buffers). Default 1GB.
// ALPHA_POOL_RESERVE_GB raises it, which is needed to boot large models on a small card: if the pool
// takes free-1GB, too little may remain for the graph's largest compute tensor, and the first decode
// runs out of memory. Both the prealloc and lazy-init sites honor this.
// Dynamic reserve override: the server computes the reserve from the actual model/KV/compute-buffer
// picture (it holds the params) and sets it here before the pool claim. 0 = fall back to the static
// default. Over-reserving starves the pool (low hit rate) and under-reserving runs a big model out of
// memory. The env ALPHA_POOL_RESERVE_GB still overrides everything.
static std::atomic<size_t> g_pool_reserve_override{0};
GGML_BACKEND_API_C void ggml_cuda_set_pool_reserve_bytes(size_t bytes) { g_pool_reserve_override.store(bytes); }

static size_t alpha_pool_reserve_bytes() {
    const char * e = getenv("ALPHA_POOL_RESERVE_GB");
    if (e) { long gb = atol(e); return (size_t)(gb < 1 ? 1 : gb) * 1024ULL * 1024 * 1024; }
    size_t ov = g_pool_reserve_override.load();
    if (ov >= 256ULL*1024*1024) return ov;   // server-computed dynamic reserve (floored for safety)
    return 1024ULL * 1024 * 1024;             // static default when no dynamic value is set
}

// Called by the server right after model+context creation (fused-infer only).
// Claims the expert-pool VRAM while the allocation picture is deterministic
// (weights + KV + configured-ubatch compute buffers); otherwise a long first request could let
// UBBoost/prefill buffers grab this VRAM before the lazily-initialized pool, leaving it much
// smaller. The lazy init in pcache_copy adopts this allocation and carves the scratch ring once
// the real expert size is known.
// ALPHA_DRE: after a prefill borrowed VRAM, drop every captured CUDA graph (they bake temp-pool addresses),
// then unmap the idle temp pools the big micro-batches grew. Returns bytes freed.
size_t ggml_cuda_trim_pools(ggml_backend_cuda_context & ctx) {
    CUDA_CHECK(cudaDeviceSynchronize());
    ctx.cuda_graphs.clear();
    size_t freed = 0;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        for (int s = 0; s < GGML_CUDA_MAX_STREAMS; s++) {
            if (ctx.pools[d][s]) freed += ctx.pools[d][s]->trim();
        }
    }
    return freed;
}

// ALPHA_DRE: hand the load-time pool allocation to the device-resolved-experts module (dre.cu); the
// expert cache is not registered in that mode, so nothing else will adopt it.
GGML_BACKEND_API_C void * ggml_cuda_expert_pool_take(size_t * bytes) {
    void * b = g_expert_pcache.prealloc_base;
    *bytes = g_expert_pcache.prealloc_bytes;
    g_expert_pcache.prealloc_base  = nullptr;
    g_expert_pcache.prealloc_bytes = 0;
    return b;
}

GGML_BACKEND_API_C void ggml_cuda_expert_pcache_prealloc(void) {
    if (g_expert_pcache.pool || g_expert_pcache.prealloc_base) return;
    ggml_cuda_set_device(0);
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) { (void)cudaGetLastError(); return; }
    const size_t reserve = alpha_pool_reserve_bytes();  // same headroom as the lazy init
    size_t budget = (free_mem > reserve) ? free_mem - reserve : free_mem / 2;
    if (budget < 512ULL * 1024 * 1024) return;     // not worth claiming
    void * base = nullptr;
    cudaError_t err = cudaMalloc(&base, budget);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        // Retry at 7/8 steps until it fits: a partial pool still beats a starved one.
        for (int i = 0; i < 3 && err != cudaSuccess; i++) {
            budget = budget / 8 * 7;
            err = cudaMalloc(&base, budget);
            if (err != cudaSuccess) (void)cudaGetLastError();
        }
        if (err != cudaSuccess) return;
    }
    g_expert_pcache.prealloc_base  = base;
    g_expert_pcache.prealloc_bytes = budget;
    fprintf(stderr, "[expert-cache] pool VRAM preallocated at load: %.0f MB (init-order fix)\n",
            (float)budget / (1024 * 1024));
}

// Forward declaration: the seq-stream module is defined below (after pass_begin), so pass_begin
// references it through this thin wrapper.
static void alpha_seq_maybe_start();

GGML_BACKEND_API_C void ggml_cuda_expert_pcache_pass_begin(void) {
    g_expert_pcache.pool_pass_floor = g_expert_pcache.lru_counter + 1;
    // Arm the epoch ring reset. Deferred to the first scratch_alloc of the new pass because that
    // is where a compute stream handle is available, and by then the previous pass is fully
    // enqueued (so the barrier event covers all of its ring readers).
    g_expert_pcache.scratch_reset_pending = true;
    g_pcache_pass_ctr++;

    // Duty-cycle denominator + per-token sampling gate (ALPHA_PCACHE_DMA_STATS).
    // Wall time between consecutive pass_begin calls is the duty-cycle denominator.
    // ggml_cuda_expert_pcache_set_ntokens is called from ggml-backend.cpp after pass_begin, so
    // g_pcache_batch_ntokens read here still describes the pass that just ended, which is the
    // "was that a decode token?" test needed. (The denominator includes the sampling/detokenize/HTTP
    // gap so that duty x R_pcie reconciles against the reported tok/s.)
    // Decode-pass clock: one increment per decode pass.
    if (g_pcache_batch_ntokens == 1) g_pcache_decode_passes++;

    if (g_pcache_dma_stats) {
        const int64_t now = ggml_time_us();
        if (g_pcache_stats_prev_us && g_pcache_batch_ntokens == 1) {
            g_pcache_decode_wall_us += now - g_pcache_stats_prev_us;
            // g_pcache_dma_stats_now still holds the flag for the pass that just ended, so this
            // counts sampled decode tokens exactly, with no assumption about prefill interleave.
            if (g_pcache_dma_stats_now) g_pcache_sampled_passes++;
        }
        g_pcache_stats_prev_us = now;
        g_pcache_dma_stats_now = (g_pcache_pass_ctr % g_pcache_dma_stats_n) == 0;
    }

    // After the first forward has registered every layer's expert band, sort by file offset
    // and start the sequential prefetch thread (idempotent; env-gated).
    if (g_pcache_pass_ctr >= 2) alpha_seq_maybe_start();
    // CPU-inline: per-token admission cap + telemetry summary. The damped insert() filter
    // does the real rate limiting; this cap only bounds the worst case
    // (ALPHA_INLINE_ADMIT, default 64/token).
    if (g_pcache_cpu_inline) {
        static const int inl_admit_cap = []() {
            const char * e = getenv("ALPHA_INLINE_ADMIT");
            int v = e ? atoi(e) : 64;
            return v < 0 ? 0 : (v > 512 ? 512 : v);
        }();
        g_pcache_warm_left = inl_admit_cap;
        // Predictive prefetch: WILLNEED the previous token's inline miss set now, so
        // the SSD readahead overlaps this token's CPU compute.
        inl_prefetch_token_boundary();
        if ((g_pcache_pass_ctr & 255) == 0 && g_pcache_pass_ctr > 0) {
            const double n = 256.0;
            const int64_t pn = g_inl_pin_n.exchange(0, std::memory_order_relaxed);
            const int64_t cn = g_inl_pc_n.exchange(0, std::memory_order_relaxed);
            const int64_t sn = g_inl_ssd_n.exchange(0, std::memory_order_relaxed);
            const int64_t su = g_inl_ssd_us.exchange(0, std::memory_order_relaxed);
            const int64_t pf = g_inl_pf_issued; g_inl_pf_issued = 0;
            const int64_t fn  = g_inline_fence_n.exchange(0, std::memory_order_relaxed);
            const int64_t f_wait = fn & 0xffffffff;           // fences that blocked on compute (imbalance)
            const int64_t f_free = (fn >> 32) & 0xffffffff;   // fences where worker already done (pure overhead)
            // Roofline: is the CPU expert pipe compute-bound or RAM-bound? achieved = the bytes a
            // perfectly amortized kernel must read, over the time actually spent; compare with the
            // machine's RAM bandwidth. amort = pairs/groups = the reuse on offer.
            const int64_t cpu_us_v = g_inline_cpu_us.exchange(0, std::memory_order_relaxed);
            const int64_t grp_v    = g_inline_groups.exchange(0, std::memory_order_relaxed);
            const int64_t wb_v     = g_inline_wbytes.exchange(0, std::memory_order_relaxed);
            const int64_t exp_v    = g_inline_experts.load(std::memory_order_relaxed);
            fprintf(stderr, "[cpu-roofline] pairs/tok=%.1f groups/tok=%.1f amort=%.2fx | "
                    "wbytes/tok=%.1fMB cpu=%.2fms -> achieved=%.1f GB/s (RAM ~50) => %s\n",
                    (double)exp_v / n, (double)grp_v / n,
                    grp_v > 0 ? (double)exp_v / (double)grp_v : 0.0,
                    (double)wb_v / n / 1e6, (double)cpu_us_v / 1e3 / n,
                    cpu_us_v > 0 ? (double)wb_v / (double)cpu_us_v / 1e3 : 0.0,
                    (cpu_us_v > 0 && (double)wb_v / (double)cpu_us_v / 1e3 < 25.0)
                        ? "COMPUTE-BOUND (amortizing the Q4_K row decode is worth it)"
                        : "RAM-BOUND (no headroom from amortization)");
            {   // which serial phase is the Amdahl ceiling?
                const int64_t qv = g_inline_quant_us.exchange(0, std::memory_order_relaxed);
                const int64_t cv = g_inline_class_us.exchange(0, std::memory_order_relaxed);
                fprintf(stderr, "[cpu-serial] quant=%.2fms/tok (%.0f%% of cpu) classify=%.2fms/tok "
                        "(%.0f%%) parallel_dot=%.2fms/tok (%.0f%%)\n",
                        (double)qv / 1e3 / n, cpu_us_v > 0 ? 100.0*qv/cpu_us_v : 0.0,
                        (double)cv / 1e3 / n, cpu_us_v > 0 ? 100.0*cv/cpu_us_v : 0.0,
                        (double)(cpu_us_v - qv - cv) / 1e3 / n,
                        cpu_us_v > 0 ? 100.0*(cpu_us_v - qv - cv)/cpu_us_v : 0.0);
            }
            g_inline_cpu_us.fetch_add(cpu_us_v, std::memory_order_relaxed);  // restore for the line below
            fprintf(stderr, "[cpu-inline] pass=%lld per-token: cpu=%.2fms stall=%.2fms experts=%.1f | "
                    "read-class/tok pinned=%.1f pagecache=%.1f ssd=%.1f (ssd %.2fms/tok, %.0fus/read) | prefetch=%.1f/tok "
                    "| fences/tok=%.1f wait=%.1f free=%.1f (%.0f%% blocked-on-compute)\n",
                    (long long)g_pcache_pass_ctr,
                    (double)g_inline_cpu_us.exchange(0, std::memory_order_relaxed) / 1e3 / n,
                    (double)g_inline_stall_us.exchange(0, std::memory_order_relaxed) / 1e3 / n,
                    (double)g_inline_experts.exchange(0, std::memory_order_relaxed) / n,
                    (double)pn / n, (double)cn / n, (double)sn / n,
                    (double)su / 1e3 / n, sn > 0 ? (double)su / sn : 0.0,
                    (double)pf / n,
                    (double)(f_wait + f_free) / n, (double)f_wait / n, (double)f_free / n,
                    (f_wait + f_free) > 0 ? 100.0 * (double)f_wait / (double)(f_wait + f_free) : 0.0);
        }
    }
    // [link]/[fence]/[roof]: duty cycle and GPU stall, windowed. Reported at the pass boundary
    // as deltas over the window, normalized per decode token, so they compose directly with tok/s.
    if (g_pcache_dma_stats && g_pcache_pass_ctr > 0 &&
        (g_pcache_pass_ctr % g_pcache_stats_every == 0 ||
         (g_pcache_pass_ctr <= 128 && (g_pcache_pass_ctr & (g_pcache_pass_ctr - 1)) == 0))) {
        auto & C = g_expert_pcache;
        C.fetch_timing_drain();
        C.fence_timing_drain();
        static double  p_fetch_ms = 0.0, p_stall_d_ms = 0.0, p_stall_o_ms = 0.0;
        static int64_t p_bytes = 0, p_bursts = 0, p_pf = 0, p_off = 0;
        static int64_t p_sd_n = 0, p_so_n = 0, p_fx_n = 0, p_fd_n = 0;
        static int64_t p_hits = 0, p_miss = 0, p_wall_us = 0, p_passes = 0, p_samp = 0;
        static int64_t p_pf_i = 0, p_pf_u = 0, p_pf_s = 0;
        static double  p_ov_ms = 0.0; static int64_t p_ov_n = 0, p_ov_neg = 0;
        static double  p_dead_ms = 0.0; static int64_t p_dead_n = 0;

        const double  d_fetch_ms = C.fetch_ms_total   - p_fetch_ms;
        const int64_t d_bytes    = C.fetch_bytes_total - p_bytes;
        const int64_t d_bursts   = C.fetch_bursts      - p_bursts;
        const int64_t d_pf       = C.pf_bytes          - p_pf;
        const int64_t d_off      = C.offstream_bytes   - p_off;
        const double  d_sd_ms    = C.stall_data_ms     - p_stall_d_ms;
        const double  d_so_ms    = C.stall_ovh_ms      - p_stall_o_ms;
        const int64_t d_sd_n     = C.stall_data_n      - p_sd_n;
        const int64_t d_so_n     = C.stall_ovh_n       - p_so_n;
        const int64_t d_fx       = C.fence_exit_n      - p_fx_n;
        const int64_t d_fd       = C.fence_drain_n     - p_fd_n;
        const int64_t d_hits     = C.hits              - p_hits;
        const int64_t d_miss     = C.misses            - p_miss;
        const int64_t d_wall_us  = g_pcache_decode_wall_us - p_wall_us;
        const int64_t d_passes   = g_pcache_decode_passes  - p_passes;

        if (d_passes > 0) {
            const double tok       = (double)d_passes;
            const double wall_ms   = (double)d_wall_us / 1e3 / tok;
            // Stall/fetch events are only recorded on sampled tokens, so scale them back up to
            // a per-token figure by the sampled-token count, not by the whole window.
            const int64_t samp_n   = g_pcache_sampled_passes - p_samp;
            const double samp      = (double)(samp_n > 0 ? samp_n : 1);
            const double busy_ms   = d_fetch_ms / samp;
            const double stall_ms  = (d_sd_ms + d_so_ms) / samp;
            const double duty      = wall_ms > 0 ? 100.0 * busy_ms / wall_ms : 0.0;
            const double inst      = d_fetch_ms > 0 ? (double)d_bytes / 1e9 / (d_fetch_ms / 1e3) : 0.0;
            const double hitpct    = (d_hits + d_miss) > 0 ? 100.0 * d_hits / (d_hits + d_miss) : 0.0;
            fprintf(stderr,
                "[link]  pass=%lld win=%lldtok sampled=%lldtok wall=%.2fms/tok | h2d busy=%.2fms/tok "
                "DUTY=%.1f%%%s | bytes/tok=%.1fMB (pf=%.1f off=%.1f) inst=%.2fGB/s | bursts=%.1f/tok avg=%.0fKB\n",
                (long long)g_pcache_pass_ctr, (long long)d_passes, (long long)samp_n,
                wall_ms, busy_ms, duty,
                // Duty > 100% is impossible and would mean the sampled-token scaling is wrong (an
                // event-derived quantity divided by the wrong denominator); flag it in the line.
                duty > 100.5 ? " <<INVALID: scaling bug, do not quote>>" : "",
                (double)d_bytes / 1e6 / samp, (double)d_pf / 1e6 / tok, (double)d_off / 1e6 / tok,
                inst, (double)d_bursts / samp,
                d_bursts > 0 ? (double)d_bytes / d_bursts / 1024.0 : 0.0);
            fprintf(stderr,
                "[fence] fences/tok=%.1f (exit=%.1f drain=%.1f) | STALL=%.2fms/tok = %.1f%% of token"
                " | data-stall n=%.1f %.2fms | overhead-only n=%.1f %.2fms | hit=%.1f%% win\n",
                (double)(d_fx + d_fd) / tok, (double)d_fx / tok, (double)d_fd / tok,
                stall_ms, wall_ms > 0 ? 100.0 * stall_ms / wall_ms : 0.0,
                (double)d_sd_n / samp, d_sd_ms / samp,
                (double)d_so_n / samp, d_so_ms / samp, hitpct);
            // Overlap: shows which restructure is worth building. STALL alone cannot, because
            // STALL contains the transport it is exposed to.
            {
                const int64_t d_ovn  = C.overlap_n     - p_ov_n;
                const double  d_ovms = C.overlap_ms_total - p_ov_ms;
                const int64_t d_ovneg= C.overlap_neg_n - p_ov_neg;
                const double  burst_us = d_bursts > 0 ? (d_fetch_ms * 1e3 / (double)d_bursts) : 0.0;
                const double  arr_us   = d_ovn > 0 ? (d_ovms * 1e3 / (double)d_ovn) : 0.0;
                const char *  verdict =
                    d_ovn == 0            ? "no samples" :
                    arr_us < 0.0          ? "HOST-PACED: compute waited BEFORE the burst started -> fix the issue path, not the overlap" :
                    (burst_us > 0.0 && arr_us < 0.15 * burst_us)
                                          ? "LOCK-STEP: burst and compute arrival are simultaneous -> nothing overlaps; give compute work during the burst" :
                    (burst_us > 0.0 && arr_us > 0.60 * burst_us)
                                          ? "WELL-OVERLAPPED: stall is only the burst tail -> move FEWER bytes" :
                                            "PARTIAL overlap";
                // Dead = blocked with the link idle. Separates "waiting on bytes in flight"
                // (transport-bound; fewer bytes helps) from "waiting on nothing" (fewer bytes
                // would change nothing).
                const int64_t d_dn   = C.dead_n        - p_dead_n;
                const double  d_dms  = C.dead_ms_total - p_dead_ms;
                const double  dead_us = d_dn > 0 ? (d_dms * 1e3 / (double)d_dn) : 0.0;
                const double  stall_us_ea = d_sd_n + d_so_n > 0
                                          ? ((d_sd_ms + d_so_ms) * 1e3 / (double)(d_sd_n + d_so_n)) : 0.0;
                fprintf(stderr,
                    "[ovlp]  compute arrives %.1fus into a %.1fus burst (%.0f%% of it already done)"
                    " | n=%lld neg=%lld | %s\n",
                    arr_us, burst_us, burst_us > 0.0 ? 100.0 * arr_us / burst_us : 0.0,
                    (long long)d_ovn, (long long)d_ovneg, verdict);
                fprintf(stderr,
                    "[dead]  blocked-with-link-IDLE %.1fus/fence of %.1fus stall (%.0f%%) | n=%lld"
                    " | %s\n",
                    dead_us, stall_us_ea,
                    stall_us_ea > 0.0 ? 100.0 * dead_us / stall_us_ea : 0.0, (long long)d_dn,
                    d_dn == 0                    ? "no samples" :
                    dead_us < 0.15 * stall_us_ea ? "TRANSPORT-BOUND: the wait is bytes in flight -> fewer bytes / more overlap"
                                                 : "DEAD WAIT DOMINATES: blocked on nothing -> fewer bytes will NOT help, find the other serializer");
            }
            // Prefetch counters, windowed (the power-of-two block in pcache_copy stops emitting
            // during warmup, before prefetch has built its layer chain).
            fprintf(stderr,
                "[pf]    issued=%.1f/tok used=%.1f/tok skip=%.1f/tok | adopt=%.0f%% | pf bytes=%.1fMB/tok"
                " (%.0f%% of all h2d)\n",
                (double)(g_expert_pcache.prefetch_issued  - p_pf_i) / tok,
                (double)(g_expert_pcache.prefetch_used    - p_pf_u) / tok,
                (double)(g_expert_pcache.prefetch_skipped - p_pf_s) / tok,
                (g_expert_pcache.prefetch_issued - p_pf_i) > 0
                    ? 100.0 * (double)(g_expert_pcache.prefetch_used - p_pf_u)
                            / (double)(g_expert_pcache.prefetch_issued - p_pf_i) : 0.0,
                (double)d_pf / 1e6 / tok,
                // Scale-match before dividing: pf_bytes accumulates on every token, while
                // fetch_burst_bytes only on sampled ones.
                d_bytes > 0 ? 100.0 * ((double)d_pf / tok) / ((double)d_bytes / samp) : 0.0);
            // Derived arithmetic, not measurement: the rest of the token is real compute, not idle
            // capacity.
            const double zs = wall_ms - stall_ms;
            fprintf(stderr,
                "[roof]  zero-stall bound=%.2fms/tok -> %.1f tok/s (%+.1f%%) | link floor=%.2fms/tok"
                " -> %.1f tok/s at DUTY=100%% [derived, not measured]\n",
                zs, zs > 0 ? 1e3 / zs : 0.0,
                zs > 0 && wall_ms > 0 ? 100.0 * (wall_ms / zs - 1.0) : 0.0,
                busy_ms, busy_ms > 0 ? 1e3 / busy_ms : 0.0);
        }
        p_fetch_ms = C.fetch_ms_total; p_bytes = C.fetch_bytes_total; p_bursts = C.fetch_bursts;
        p_pf = C.pf_bytes; p_off = C.offstream_bytes;
        p_stall_d_ms = C.stall_data_ms; p_stall_o_ms = C.stall_ovh_ms;
        p_sd_n = C.stall_data_n; p_so_n = C.stall_ovh_n;
        p_fx_n = C.fence_exit_n; p_fd_n = C.fence_drain_n;
        p_hits = C.hits; p_miss = C.misses;
        p_wall_us = g_pcache_decode_wall_us; p_passes = g_pcache_decode_passes;
        p_samp = g_pcache_sampled_passes;
        p_pf_i = g_expert_pcache.prefetch_issued;
        p_pf_u = g_expert_pcache.prefetch_used;
        p_pf_s = g_expert_pcache.prefetch_skipped;
        p_ov_ms = C.overlap_ms_total; p_ov_n = C.overlap_n; p_ov_neg = C.overlap_neg_n;
        p_dead_ms = C.dead_ms_total; p_dead_n = C.dead_n;
    }

    // Hot-set: end the profiling window and pin the frequency-ranked hot core.
    if (g_expert_pcache.hotset_enabled && !g_expert_pcache.hotset_finalized &&
        !g_expert_pcache.access_freq.empty() &&
        g_pcache_pass_ctr >= (uint64_t)g_expert_pcache.hotset_profile_passes) {
        g_expert_pcache.finalize_hotset();
        g_expert_pcache.hotset_last_repin = g_pcache_pass_ctr;
    }
    // Adaptive hot set: after finalize, periodically re-rank the pinned core from live
    // access frequency so the pool tracks routing drift during long generations (otherwise the
    // frozen set decays). Cheap: one partial sort of the (aged, bounded) access_freq every N
    // passes on the pass-begin thread.
    else if (g_expert_pcache.hotset_enabled && g_expert_pcache.hotset_finalized &&
             g_expert_pcache.hotset_adaptive && !g_expert_pcache.access_freq.empty() &&
             g_pcache_pass_ctr - g_expert_pcache.hotset_last_repin
                 >= (uint64_t)g_expert_pcache.hotset_adaptive_period) {
        g_expert_pcache.repin_hotset();
        g_expert_pcache.hotset_last_repin = g_pcache_pass_ctr;
    }
}

// Extern-C callable from ggml-backend.cpp via function pointer.
// Two-phase adaptive: observe frequency during warmup, then selectively cache.
// Cache hits: GPU->GPU on stream 0 (compute stream), in-line with the graph.
// Cache misses: H2D on stream 1 (DMA stream), overlapping with attention on stream 0.
// Caller must invoke ggml_cuda_expert_pcache_flush() after all copies to sync streams.
// Track batch boundaries: clear miss tracking when a new copy batch starts.
// A "new batch" = first pcache_copy call after the previous token's graph_compute finished.
// Detected by: flush sets batch_flushed=true, next pcache_copy clears and resets.
static bool g_pcache_batch_flushed = false;

// Sequential expert-band prefetcher (ALPHA_SEQ_STREAM, default off).
// For MoE models whose experts exceed RAM, multi-user decode reads close to the whole expert set per
// step. Random mmap faults are far slower than reading each layer's expert tensor as a contiguous
// block in file order on NVMe. A background thread posix_madvise(WILLNEED)s a sliding window of
// upcoming expert bands (in file/layer order) ahead of the decode's layer cursor, so the per-layer
// stage_expert memcpy hits warm pages. The multi-user decode_batch path already provides independent
// users and layer-order expert dedup; this module adds the sequential SSD read. Each expert tensor
// (input->data == mmap_base+file_offset, all n_expert experts contiguous, stride nb[2]) is one band.
struct alpha_seq_band { const void * base; size_t bytes; };
static const bool   g_seq_stream = []{ const char * e = getenv("ALPHA_SEQ_STREAM"); return e && atoi(e) != 0; }();
static const size_t g_seq_window_bytes = []{ const char * e = getenv("ALPHA_SEQ_WINDOW_GB");
    long gb = e ? atol(e) : 24; if (gb <= 0) gb = 24; return (size_t)gb * ((size_t)1 << 30); }();
static std::vector<alpha_seq_band>          g_seq_bands;
static std::unordered_map<const void*, int> g_seq_idx;      // band base -> index (post-sort)
static std::mutex                           g_seq_mtx;
static std::atomic<int>                      g_seq_cursor{0}; // band index the decode is consuming
static std::atomic<bool>                     g_seq_sorted{false};
static std::atomic<bool>                     g_seq_run{false};
static std::atomic<long long>                g_seq_pf_bytes{0};
static std::atomic<long long>                g_seq_pf_ns{0};

static void alpha_seq_worker() {
#ifdef __linux__
    int       last_cur   = -1;
    int       keepalive  = 0;
    long long sweeps     = 0;
    while (g_seq_run.load(std::memory_order_relaxed)) {
        if (!g_seq_sorted.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        const int n   = (int)g_seq_bands.size();
        const int cur = g_seq_cursor.load(std::memory_order_relaxed);
        // Only re-advise when the decode cursor moves (a stuck cursor must not hot-loop),
        // plus a periodic keepalive (~200 ms) to re-prime the window against page eviction.
        if (n > 0 && (cur != last_cur || ++keepalive >= 40)) {
            last_cur = cur; keepalive = 0; sweeps++;
            size_t ahead = 0;
            for (int k = 0; k < n && ahead < g_seq_window_bytes; k++) {
                const alpha_seq_band & b = g_seq_bands[(cur + k) % n];
                if (!b.base || !b.bytes) continue;
                posix_madvise((void *)b.base, b.bytes, POSIX_MADV_WILLNEED);
                ahead += b.bytes;
            }
            g_seq_pf_bytes.fetch_add((long long)ahead, std::memory_order_relaxed);
            if ((sweeps & 1023) == 1)
                fprintf(stderr, "[seq-stream] cursor=%d/%d window=%.0fGB sweeps=%lld\n",
                        cur, n, g_seq_window_bytes / 1e9, sweeps);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
#endif
}

// Collect a unique expert band during the first forward; after finalize, update the cursor.
static inline void alpha_seq_note(const void * base, size_t bytes) {
    if (!g_seq_stream || !base || !bytes) return;
    if (g_seq_sorted.load(std::memory_order_acquire)) {
        auto it = g_seq_idx.find(base);
        if (it != g_seq_idx.end()) g_seq_cursor.store(it->second, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lk(g_seq_mtx);
    for (const auto & b : g_seq_bands) if (b.base == base) return;   // dedup (small n)
    g_seq_bands.push_back({ base, bytes });
}

// Sort the collected bands by file offset (== base) and launch the prefetch thread once.
static void alpha_seq_finalize_and_start() {
    if (!g_seq_stream || g_seq_sorted.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(g_seq_mtx);
    if (g_seq_bands.empty() || g_seq_sorted.load(std::memory_order_relaxed)) return;
    std::sort(g_seq_bands.begin(), g_seq_bands.end(),
              [](const alpha_seq_band & a, const alpha_seq_band & b){ return a.base < b.base; });
    g_seq_idx.clear();
    for (int i = 0; i < (int)g_seq_bands.size(); i++) g_seq_idx[g_seq_bands[i].base] = i;
    size_t total = 0; for (const auto & b : g_seq_bands) total += b.bytes;
    fprintf(stderr, "[seq-stream] %zu expert bands, %.1f GB total, window %.0f GB -> prefetch thread up\n",
            g_seq_bands.size(), total / 1e9, g_seq_window_bytes / 1e9);
    g_seq_sorted.store(true, std::memory_order_release);
    g_seq_run.store(true, std::memory_order_relaxed);
    std::thread(alpha_seq_worker).detach();
}

// Thin wrapper called from pass_begin (forward-declared above it). No-op unless enabled.
static void alpha_seq_maybe_start() {
    if (g_seq_stream) alpha_seq_finalize_and_start();
}

// Layer-fused expert fetch (ALPHA_LAYER_FUSE, default 2).
// The flush runs once per expert tensor (ggml-backend.cpp calls it after each split input), so each
// layer has three fetch+fence points per token, each moving only a few experts. The three expert
// tensors of one layer are routed by the same ids, so at the first tensor of a layer the experts the
// other two will need are already known, and all of them can be fetched in that one flush.
//
// Only legal on the pool-direct path: there the destination is the pool and the consumer address is
// published through the per-tensor offset table (keyed by cpu_src), so a sibling's experts can be
// placed without knowing its gpu_dst (which does not exist yet). Prefill / hybrid / ik / panic keep
// the per-tensor behavior.
//
// Ordering is inherited: the fused H2Ds go on h2d_stream before this flush's dma_done_event record,
// and the siblings' matmuls are enqueued on compute_stream later (their splits come later in the
// scheduler loop), so the surviving fence already covers them.
// Placements must be applied in ascending expert id so they land on the same ring offsets that
// one-by-one fetching would produce, adopted experts must still update freq[] for the admission
// filter, and a tensor whose experts were all adopted must still publish its offset table.
//
// ALPHA_LAYER_FUSE: 0 = off | 1 = fused DMA, one fence | 2 = fused DMA, per-tensor fences (default)
// Mode 1 batches the DMA but keeps one fence covering all three expert tensors, so the gate's kernel
// still waits for down's bytes; burst size is not the limiter, so it gains nothing. Mode 2 fences
// each tensor against only its own bytes: the gate fence is recorded before the sibling fetch, so
// gate compute starts after ~1/3 of the layer's bytes and the up/down transfers run underneath it.
// dma_stream is in-order, so a per-member event recorded after that member's copies completes exactly
// when that member's bytes have landed. No kernel change and no CUDA-graph risk. At 100% hit rate the
// per-member events are pure (small) cost. ALPHA_LAYER_FUSE=0 restores per-tensor fetching.
static const int g_pcache_layer_fuse = []() {
    const char * e = getenv("ALPHA_LAYER_FUSE");
    const int v = e ? atoi(e) : 2;
    return v < 0 ? 0 : (v > 2 ? 2 : v);
}();

struct pcache_layer_group {
    enum { MAXM = 8 };
    const void * members[MAXM]     = {};
    size_t       expert_size[MAXM] = {};
    size_t       type_size[MAXM]   = {};
    int          n_members = 0;
    bool         complete  = false;   // set when the next layer starts (all members seen)
};
// key = cpu_src of the layer's first expert tensor. cpu_src is the host weight pointer: stable for
// the process lifetime (gpu_dst is not, since ggml-alloc reuses it, so it must never be a key).
static std::unordered_map<const void *, pcache_layer_group> g_pcache_groups;
static const void * g_pcache_group_key    = nullptr;  // layer currently being walked
static bool         g_pcache_group_isnew  = false;    // next copy's tensor opens a new layer
static const void * g_pcache_fused_key    = nullptr;  // layer already fused in this pass
// (cpu_src,eid) -> pool byte offset placed by the fused fetch, consumed by the sibling's copy.
static std::unordered_map<uint64_t, size_t> g_pcache_pass_placements;
static int64_t g_pcache_fuse_issued = 0, g_pcache_fuse_adopted = 0, g_pcache_fuse_layers = 0;
// Mode 2 (per-tensor fences): one completion event per fused member, recorded on h2d_stream right
// after that member's copies. dma_stream is in-order, so the event fires exactly when that member's
// bytes have landed, letting each sibling's kernel wait on its own bytes instead of the whole
// layer's. Cleared as they are consumed; any left over are harmless (the next layer's fused fetch
// overwrites the slot, and a stale wait would only over-synchronize, never under-).
static cudaEvent_t  g_pcache_fuse_ev[pcache_layer_group::MAXM]     = {};
static const void * g_pcache_fuse_ev_src[pcache_layer_group::MAXM] = {};
static int64_t g_pcache_fuse_ev_waits = 0;   // telemetry: how often a sibling fenced on its own bytes

// Called from ggml-backend at each expert tensor, with is_new_group=1 on the layer's first tensor
// (derived from the graph: ids_tensor != prev_ids_tensor). Graph-derived, not guessed.
GGML_BACKEND_API_C void ggml_cuda_expert_pcache_set_group(const void * ids_tensor, int is_new_group) {
    (void) ids_tensor;
    if (!g_pcache_layer_fuse || !is_new_group) return;
    if (g_pcache_group_key) {                       // close the layer we were walking
        auto it = g_pcache_groups.find(g_pcache_group_key);
        if (it != g_pcache_groups.end() && it->second.n_members > 0) it->second.complete = true;
    }
    // Placements are consumed by siblings within the same layer; drop them at the boundary so a
    // wrapped scratch slot can never be adopted by a later layer.
    g_pcache_pass_placements.clear();
    g_pcache_group_key   = nullptr;
    g_pcache_fused_key   = nullptr;   // must clear, or the same layer next token would not re-fuse
    g_pcache_group_isnew = true;
}

GGML_BACKEND_API_C bool ggml_cuda_expert_pcache_copy(
        void * backend_ptr,      // ggml_backend_t (for stream)
        void * gpu_dst,          // input_cpy->data  (GPU)
        const void * cpu_src,    // input->data       (CPU)
        size_t expert_size,      // bytes per expert slot (nb[2])
        int64_t n_expert,        // ne[2]
        int64_t expert_id,       // which expert
        size_t copy_size) {      // expert_size + padding

    ggml_backend_t backend = (ggml_backend_t)backend_ptr;
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)backend->context;
    cudaStream_t compute_stream = ctx->stream();  // stream 0

    // Register this expert tensor's contiguous band (all n_expert experts) for the sequential
    // prefetcher, and advance its layer cursor as the decode consumes layers.
    // Record this tensor's authoritative block size while g_pcache_cur_block_size still refers
    // to this tensor, the only point at which that is guaranteed.
    // First write wins: g_pcache_cur_block_size is authoritative only for the tensor currently being
    // processed, and the divisibility guard below is ambiguous (iq3_xxs=98=2*7^2, so an expert size
    // that is a multiple of 98 is also a multiple of 34 and 18). With last-write-wins a later call
    // carrying a stale global that happens to divide would overwrite this tensor's block size, and
    // pcache_block_size_for() would hand the wrong stride to the prefetch insert sites, putting
    // wrong-but-in-bounds experts in the pool. emplace() keeps the value recorded on this tensor's
    // first copy call. (Comparing the map against pcache_block_size() cannot detect a bad entry, since
    // both use the same ambiguous guess.)
    if (g_pcache_cur_block_size > 0 && expert_size % (size_t) g_pcache_cur_block_size == 0) {
        g_pcache_bs_by_src.emplace(cpu_src, (uint32_t) g_pcache_cur_block_size);
    }

    alpha_seq_note(cpu_src, (size_t)n_expert * expert_size);

    // Layer-fuse bookkeeping: learn which expert tensors belong to this layer, in visitation
    // order. Pass 1 only learns (fusion stays off until a group is complete), so the first token
    // of any new graph shape behaves like baseline.
    if (g_pcache_layer_fuse) {
        if (g_pcache_group_isnew) {
            g_pcache_group_isnew = false;
            g_pcache_group_key   = cpu_src;          // this tensor opens the layer
        }
        if (g_pcache_group_key) {
            auto & grp = g_pcache_groups[g_pcache_group_key];
            if (!grp.complete) {
                bool known = false;
                for (int i = 0; i < grp.n_members; i++) if (grp.members[i] == cpu_src) { known = true; break; }
                if (!known && grp.n_members < pcache_layer_group::MAXM) {
                    const int i = grp.n_members++;
                    grp.members[i]     = cpu_src;
                    grp.expert_size[i] = expert_size;
                    grp.type_size[i]   = pcache_block_size(expert_size);
                }
            } else if (grp.members[0] != g_pcache_group_key) {
                grp.complete = false;                // structure changed underneath us -> relearn
            }
        }
    }

    // New token boundary: reset offset table state from previous forward pass.
    // Bump LRU counter by a large epoch gap so current-token hits are always newer
    // than any previous-token entries. This prevents LRU eviction from reusing a pool
    // slot that was HIT earlier in the same token (deferred D2D race).
    if (g_pcache_new_token) {
        g_pcache_offset_active = false;
        g_pcache_new_token = false;
        g_expert_pcache.lru_counter += 1000000;
    }

    // Clear per-tensor state from previous flush.
    // active_tables and offset_active accumulate across tensors within one forward pass.
    if (g_pcache_batch_flushed) {
        g_expert_pcache.clear_miss_tracking();
        g_pcache_has_misses = false;
        g_pcache_cpu_misses.clear();
        g_pcache_tensor_build.clear();
        g_pcache_batch_flushed = false;
    }

    // Zero the full input_cpy expert region once per tensor-eval before its experts are written (see
    // g_pcache_input_cpy_needs_zero), so the multi-token MMQ kernel (need_check=false) reads 0 for
    // unwritten/unused expert slots. Ordering: this memset is queued on compute_stream during the copy
    // phase, ahead of the flush's deferred-D2D hits (compute_stream) and ahead of the flush's
    // reverse-sync event, so the miss H2Ds (dma_stream) also wait for it.
    // Single-token decode is pool-direct (never reads input_cpy), so it is skipped. The flag is armed
    // at each mul_mat_id entry (consumed by the next tensor's first copy).
    // Fires for multi-token/prefill (MMQ reads input_cpy) and for the ik-quant path: ik is forced
    // non-pool-direct and reads input_cpy on both prefill and decode, so its unwritten expert slots
    // must be zeroed every decode token too. Safe because the ik flush always populates input_cpy
    // (unlike K-quant pool-direct decode, which populates only the pool, so zeroing there would blank it).
    if ((g_pcache_batch_ntokens >= 2 || g_pcache_batch_src_is_ik)
            && g_pcache_input_cpy_needs_zero && gpu_dst && expert_size > 0) {
        CUDA_CHECK(cudaMemsetAsync(gpu_dst, 0, (size_t)n_expert * expert_size, compute_stream));
        g_pcache_input_cpy_needs_zero = false;
    }

    // Lazy init of the D2D pool
    if (!g_expert_pcache.pool) {
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        const size_t reserve = alpha_pool_reserve_bytes();
        size_t budget = (free_mem > reserve) ? free_mem - reserve : free_mem / 2;
        if (budget < 512ULL * 1024 * 1024) budget = 512ULL * 1024 * 1024;

        // expert_size of the first-seen tensor is the sizing hint for the scratch ring (all expert
        // tensors are within ~2x of each other across quant types, good enough for a clamped heuristic).
        fprintf(stderr, "[expert-cache] using D2D pool\n");
        g_expert_pcache.init(budget, expert_size);
        g_expert_pcache.init_dma_stream(ctx);

        // Hot-set config (default on; ALPHA_HOTSET=0 disables): frequency-pin hot experts.
        if (const char * e = getenv("ALPHA_HOTSET")) {
            g_expert_pcache.hotset_enabled = atoi(e) != 0;
        }
        if (const char * e = getenv("ALPHA_HOTSET_PASSES")) {
            int v = atoi(e); if (v >= 100) g_expert_pcache.hotset_profile_passes = v;
        }
        if (const char * e = getenv("ALPHA_HOTSET_FRAC")) {
            float v = atof(e); if (v > 0.1f && v <= 0.95f) g_expert_pcache.hotset_budget_frac = v;
        }
        // Adaptive hot set (default off): keep re-ranking the pinned core so it tracks
        // routing drift on long generations instead of freezing after one profiling window.
        if (const char * e = getenv("ALPHA_HOTSET_ADAPTIVE")) {
            g_expert_pcache.hotset_adaptive = atoi(e) != 0;
        }
        if (const char * e = getenv("ALPHA_HOTSET_ADAPTIVE_PERIOD")) {
            int v = atoi(e); if (v >= 32) g_expert_pcache.hotset_adaptive_period = v;
        }
        if (const char * e = getenv("ALPHA_HOTSET_ADAPTIVE_DECAY")) {
            float v = atof(e); if (v > 0.0f && v < 1.0f) g_expert_pcache.hotset_adaptive_decay = v;
        }
        if (g_expert_pcache.hotset_enabled) {
            fprintf(stderr, "[expert-cache] hot set ENABLED: profile %d passes, pin %.0f%% of pool%s\n",
                    g_expert_pcache.hotset_profile_passes, 100.0f*g_expert_pcache.hotset_budget_frac,
                    g_expert_pcache.hotset_adaptive ? " [ADAPTIVE]" : "");
            if (g_expert_pcache.hotset_adaptive) {
                fprintf(stderr, "[expert-cache] hot set ADAPTIVE: re-pin every %d passes, age x%.2f\n",
                        g_expert_pcache.hotset_adaptive_period, g_expert_pcache.hotset_adaptive_decay);
            }
        }

        // Order matters: compute the pin budget from MemAvailable before any staging allocation.
        // A big pinned staging buffer allocated up front would shrink MemAvailable and collapse the
        // pin budget, leaving most tensors unpinned.
        // With tensor-pin on (default), staging is on-demand only (ensure_pinned inits it when the
        // pin budget is exhausted or a pin fails; it serves the unpinned tail).
        // With ALPHA_TENSOR_PIN=0, staging is the sole pinned H2D source: allocate eagerly.
        if (!g_pcache_tensor_pin && !g_expert_pcache.staging_enabled) {
            g_expert_pcache.init_staging();
        }

        // Pin budget: min(MemAvailable - 8GB, 50% of MemTotal)
        if (g_expert_pcache.pin_budget == 0) {
#ifdef __linux__
            size_t avail_ram = 0;
            FILE * f = fopen("/proc/meminfo", "r");
            if (f) {
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    size_t val;
                    if (sscanf(line, "MemAvailable: %zu kB", &val) == 1) {
                        avail_ram = val * 1024;
                        break;
                    }
                }
                fclose(f);
            }
            if (avail_ram > 0) {
                // Also read total RAM for the cap
                size_t total_ram = 0;
                f = fopen("/proc/meminfo", "r");
                if (f) {
                    char line2[256];
                    while (fgets(line2, sizeof(line2), f)) {
                        size_t val2;
                        if (sscanf(line2, "MemTotal: %zu kB", &val2) == 1) {
                            total_ram = val2 * 1024;
                            break;
                        }
                    }
                    fclose(f);
                }
                // Pin budget: min(available - 8GB, 50% of total RAM).
                // MemAvailable counts reclaimable page cache and pinning locks it, so the one-shot
                // budget can over-commit and starve userspace. The 50% cap plus the live MemAvailable
                // floor in ensure_pinned (pin_closed) keep userspace alive; the tail beyond the stop
                // point is served by on-demand staging.
                const size_t headroom = 8ULL * 1024 * 1024 * 1024;
                size_t budget_avail = (avail_ram > headroom) ? avail_ram - headroom : avail_ram / 2;
                size_t budget_cap = total_ram > 0 ? total_ram / 2 : budget_avail;
                g_expert_pcache.pin_budget = std::min(budget_avail, budget_cap);
                fprintf(stderr, "[expert-cache] pin budget: %.0f GB (%.0f GB available)\n",
                        (float)g_expert_pcache.pin_budget / (1024*1024*1024),
                        (float)avail_ram / (1024*1024*1024));
            }
#endif
        }

        // DMA cost model from measured PCIe bandwidth
        if (ctx->pcie_bw_gbps > 0) {
            g_expert_pcache.dma_cost_us = (float)expert_size / (ctx->pcie_bw_gbps * 1e3f);
            g_expert_pcache.gpu_copy_us = (float)expert_size / (300.0f * 1e3f);
        }

        // Immediate cache mode: cache on first miss, skipping the 64-token observation delay.
        // Token 1 = all misses (slow), later tokens mostly hit.
        // The pool may fill before all unique experts are seen; insert() then returns nullptr
        // and overflow experts fall back to direct H2D.
        // Frequency gating (warmup) is only needed when the pool is very small relative to the
        // expected unique experts. With pools holding 2+ tokens' worth of experts (~2000+ slots),
        // immediate caching warms up faster.
        size_t max_slots = g_expert_pcache.pool_size / std::max(expert_size, (size_t)1);
        if (max_slots >= (size_t)(2 * cuda_expert_persist_cache::EXPERTS_PER_TOKEN_EST)) {
            g_expert_pcache.immediate_cache = true;
            g_expert_pcache.warmup_done = true;
            g_expert_pcache.insert_threshold = 1;  // cache everything on first miss
            fprintf(stderr, "[expert-cache] immediate mode: %zu slots >> %d experts/token, no warmup delay\n",
                    max_slots, cuda_expert_persist_cache::EXPERTS_PER_TOKEN_EST);
        }
    }

    // Pin the host expert buffer on first encounter (enables true async DMA on stream 1)
    const size_t total_expert_bytes = (size_t)n_expert * expert_size;
    g_expert_pcache.ensure_pinned(cpu_src, total_expert_bytes);

    // Select DMA stream: use stream 1 if buffer is pinned, else fall back to stream 0
    const bool use_dma_stream = g_expert_pcache.is_pinned(cpu_src) && g_expert_pcache.dma_stream;
    cudaStream_t h2d_stream = use_dma_stream ? g_expert_pcache.dma_stream : compute_stream;

    const size_t offset = expert_id * expert_size;

    // Pool pre-fill: on first encounter of each tensor, bulk H2D experts.
    // Instead of starting cold (0% hits on token 1), pre-fill the pool with expert data
    // from every weight tensor. Limited to 50% of the pool to leave room for runtime caching.
    // This spreads coverage across all layers (vs filling the first few layers completely).
    if (g_expert_pcache.pool &&
        g_expert_pcache.prefilled_tensors.find(cpu_src) == g_expert_pcache.prefilled_tensors.end()) {
        g_expert_pcache.prefilled_tensors.insert(cpu_src);
        const size_t prefill_limit = g_expert_pcache.pool_size / 2;  // reserve 50% for runtime
        if (g_expert_pcache.used < prefill_limit) {
            // Offset-table stride: authoritative block size (IQ-safe), inference fallback
            size_t fill_align = pcache_block_size(expert_size);
            // Budget per tensor: remaining prefill space / estimated remaining tensors
            // Estimate: ~144 weight tensors for MoE (48 layers x 3 types), be conservative
            size_t remaining_space = prefill_limit - g_expert_pcache.used;
            size_t n_tensors_seen = g_expert_pcache.prefilled_tensors.size();
            size_t est_remaining_tensors = std::max((size_t)1, (size_t)144 - n_tensors_seen);
            size_t budget_per_tensor = remaining_space / est_remaining_tensors;
            int max_experts = (int)std::min((size_t)n_expert, budget_per_tensor / std::max(expert_size, (size_t)1));
            if (max_experts < 1) max_experts = 1;

            int prefilled = 0;
            for (int64_t eid = 0; eid < n_expert && prefilled < max_experts; eid++) {
                uint64_t ek = cuda_expert_persist_cache::key(cpu_src, eid);
                if (g_expert_pcache.lookup(ek)) continue;
                void * slot = g_expert_pcache.insert(ek, expert_size, fill_align);
                if (!slot) break;  // pool full
                // H2D to GPU pool directly from mmap (staging adds memcpy overhead here).
                // Issued from pcache_copy, i.e. outside any fetch_timing bracket -> offstream.
                const void * fill_src = (const char *)cpu_src + eid * expert_size;
                g_expert_pcache.offstream_bytes += (int64_t)expert_size;
                CUDA_CHECK(cudaMemcpyAsync(
                    slot, fill_src, expert_size,
                    cudaMemcpyHostToDevice, h2d_stream));
                if (use_dma_stream) g_expert_pcache.mark_dma_pending();
                prefilled++;
            }
            if (prefilled > 0) {
                static int64_t fill_log_count = 0;
                if (++fill_log_count <= 5 || (fill_log_count & 31) == 0) {
                    fprintf(stderr, "[expert-cache] pre-filled %d/%lld experts (budget %d, pool %.0f/%.0f MB)\n",
                            prefilled, (long long)n_expert, max_experts,
                            (float)g_expert_pcache.used / (1024*1024),
                            (float)g_expert_pcache.pool_size / (1024*1024));
                }
            }
        }
    }

    const uint64_t k = cuda_expert_persist_cache::key(cpu_src, expert_id);

    // Windowed hit rate: the current steady-state rate over the last N accesses (the cumulative
    // counter is dominated by the cold-start phase). Uses cumulative snapshots; runs on every access.
    // ALPHA_CACHE_WINDOW sets the window length; a few thousand lookups (about 10 tokens on a 40-layer
    // top-8 model) traces hit% against the token index within one generation.
    static const int64_t win_len = []() -> int64_t {
        const char * s = getenv("ALPHA_CACHE_WINDOW");
        const int64_t v = s ? strtoll(s, nullptr, 10) : 0;
        return v > 0 ? v : 65536;
    }();
    if (++g_expert_pcache.win_access >= win_len) {
        int64_t h = g_expert_pcache.hits  - g_expert_pcache.win_hits_snap;
        int64_t m = g_expert_pcache.misses - g_expert_pcache.win_miss_snap;
        int64_t c = g_expert_pcache.cold_skipped - g_expert_pcache.win_cold_snap;  // cold-skip drops this window
        int64_t t = h + m;
        if (t > 0) {
            // cold_skip%% = fraction of this window's misses that cold-skip dropped (no DMA);
            // DMA'd misses = m - c.
            fprintf(stderr, "[expert-cache] WINDOW hit %.1f%% (%lld/%lld), cold_skip=%lld (%.1f%% of miss), pool %.0f MB, evict=%lld%s\n",
                    100.0*h/t, (long long)h, (long long)t,
                    (long long)c, m > 0 ? 100.0*c/m : 0.0,
                    (float)g_expert_pcache.used/(1024*1024),
                    (long long)g_expert_pcache.lru_evictions,
                    g_expert_pcache.hotset_finalized ? ", hotset=active" :
                        (g_expert_pcache.hotset_enabled ? ", hotset=profiling" : ""));
            // Pool-direct gate diagnostic: how many decode passes had pool_direct
            // disabled by the >=800-miss use_hybrid gate, and miss-count stats.
            if (g_dbg_decode_passes > 0) {
                fprintf(stderr, "[expert-cache]   pool-direct: OFF on %lld/%lld decode passes (>=%d misses), "
                        "misses/pass avg=%.0f max=%lld\n",
                        (long long)g_dbg_hybrid_passes, (long long)g_dbg_decode_passes,
                        g_pcache_hybrid_miss_threshold,
                        (double)g_dbg_misses_sum/g_dbg_decode_passes, (long long)g_dbg_misses_max);
            }
            g_dbg_decode_passes = 0; g_dbg_hybrid_passes = 0;
            g_dbg_misses_sum = 0; g_dbg_misses_max = 0;
        }
        g_expert_pcache.win_access = 0;
        g_expert_pcache.win_hits_snap = g_expert_pcache.hits;
        g_expert_pcache.win_miss_snap = g_expert_pcache.misses;
        g_expert_pcache.win_cold_snap = g_expert_pcache.cold_skipped;
    }

    // Hot-set profiling: count every access (hit or miss) during the profiling window.
    // Bounded (first N passes), so the hot-path map insert cost is one-time warmup only.
    // In adaptive mode counting continues after finalize so the periodic re-pin sees the
    // current hot distribution.
    if (g_expert_pcache.hotset_enabled &&
        (!g_expert_pcache.hotset_finalized || g_expert_pcache.hotset_adaptive)) {
        auto & fe = g_expert_pcache.access_freq[k];
        fe.count++;
        fe.size = (uint32_t)expert_size;
    }
    if (g_expert_pcache.hotset_finalized && g_expert_pcache.hot_protected.count(k)) {
        g_expert_pcache.hot_hits++;
    }

    // Routed-expert trace bookkeeping (ALPHA_TRACE_IDS, decode only).
    if (g_pcache_batch_ntokens == 1 && g_pcache_ids_trace) {
        auto & pti = g_expert_pcache.prefetch_info[cpu_src];
        pti.curr_used.push_back((int32_t)expert_id);
        // Trace annotation: pool-resident at request time? (non-mutating peek; the
        // real lookup below bumps LRU). H=0 misses define the PCIe-cost stream.
        pti.curr_hit.push_back(g_expert_pcache.entries.count(k) ? 1 : 0);
    }

    // --- D2D cache lookup ---
    void * cached = g_expert_pcache.lookup(k);
    if (cached) {
        // HIT: defer D2D, batched in flush for reduced API overhead.
        // Use expert_size (not copy_size): the pool stores exactly expert_size per entry.
        // Race safety: flush adds a reverse sync (compute->h2d) after the deferred D2D,
        // preventing next tensor's H2D from racing with this tensor's D2D on
        // ggml-alloc-reused compute buffer addresses.
        auto & toff = g_pcache_tensor_build[gpu_dst];
        // full width from the start: the kernel indexes this by raw expert id, so a table
        // shorter than n_expert would let an id past the high-water mark read off the end
        toff.ensure_experts(n_expert);
        toff.set_offset(expert_id, g_expert_pcache.get_offset(k));
        toff.deferred.push_back({(char *)gpu_dst + offset, cached, expert_size});
        if (!toff.cpu_src) {
            toff.cpu_src = cpu_src;
            toff.type_size = pcache_block_size(expert_size);  // IQ-safe stride
        }
        g_expert_pcache.hits++;
        return true;
    }

    // Layer-fuse adoption: this layer's first tensor already fetched this tensor's copy of the expert
    // into the pool/scratch and published the byte offset. Resolve from that placement: write the
    // table entry and return without recording a miss, so this tensor's flush finds nothing to do and
    // takes the early return (no second H2D, no second fence). Not counted as a cache hit: it is the
    // same token's own prefetch, and counting it would inflate the hit rate.
    // The gate must match the fused fetch's own gate (pool-direct single-token decode). A placement
    // adopted on a path that reads input_cpy would be wrong: the bytes live in the pool, and nothing
    // would write them into input_cpy.
    if (g_pcache_layer_fuse && g_pcache_batch_ntokens == 1 && !g_pcache_pass_placements.empty()) {
        auto pit = g_pcache_pass_placements.find(k);
        if (pit != g_pcache_pass_placements.end()) {
            auto & toff = g_pcache_tensor_build[gpu_dst];
            toff.ensure_experts(n_expert);
            toff.set_offset(expert_id, (int64_t)pit->second);
            if (!toff.cpu_src) {
                toff.cpu_src   = cpu_src;
                toff.type_size = pcache_block_size(expert_size);
            }
            // Account it exactly as the demand path would have: freq[] is the input to the damped
            // admission filter, and skipping it would make the pool fill with the wrong experts.
            // Counting it as a miss also keeps the hit-rate metric comparable; this expert would
            // have been fetched either way, only by the layer's first tensor.
            auto & afe = g_expert_pcache.freq[k];
            afe.count++;
            g_expert_pcache.total_calls++;
            g_expert_pcache.misses++;
            toff.n_adopted++;          // so the flush publishes this tensor's offset table
            g_pcache_fuse_adopted++;
            return true;
        }
    }

    // MISS: track frequency (only on misses, to avoid hash map overhead on the hot path)
    auto & fe = g_expert_pcache.freq[k];
    fe.count++;
    g_expert_pcache.total_calls++;
    g_expert_pcache.misses++;

    if (!g_expert_pcache.warmup_done) {
        // Phase 1: observation, record metadata, no inserts
        if (!fe.cpu_src) {
            fe.cpu_src     = cpu_src;
            fe.expert_id   = expert_id;
            fe.expert_size = expert_size;
            // authoritative here (inside this tensor's own copy); see block_size decl
            fe.block_size  = (uint32_t) pcache_block_size(expert_size);
        }

        // Check warmup completion. ALPHA_WARMUP_CALLS overrides the WARMUP_CALLS threshold.
        // Also finalize on the first decode token after a prefill (g_pcache_seen_prefill), so short
        // prompt + short generation, which never reaches WARMUP_CALLS, does not decode entirely in the
        // warmup/input_cpy regime (which produces NaN for gemma-4 MoE).
        static const int64_t warmup_calls_thr = []() {
            const char * e = getenv("ALPHA_WARMUP_CALLS");
            return e ? (int64_t)atoll(e) : (int64_t)cuda_expert_persist_cache::WARMUP_CALLS;
        }();
        static const bool no_prefill_warmup = alpha_env_on("ALPHA_NO_PREFILL_WARMUP");
        const bool prefill_ended = !no_prefill_warmup && g_pcache_batch_ntokens == 1 && g_pcache_seen_prefill;
        if (g_expert_pcache.total_calls >= warmup_calls_thr ||
            (prefill_ended && g_expert_pcache.total_calls > 0)) {
            g_expert_pcache.compute_threshold(expert_size);
            // Bulk insert uses compute stream (one-time cost, correctness over speed)
            g_expert_pcache.bulk_insert_hot(expert_size, compute_stream);
            g_expert_pcache.warmup_done = true;

            // Re-check: this expert may have been bulk-inserted
            cached = g_expert_pcache.lookup(k);
            if (cached) {
                CUDA_CHECK(cudaMemcpyAsync(
                    (char *)gpu_dst + offset, cached, copy_size,
                    cudaMemcpyDeviceToDevice, compute_stream));
                g_expert_pcache.misses--;
                g_expert_pcache.hits++;
                return true;
            }
        }

        // Warmup miss: deferred; flush() handles H2D or zero based on threshold
        return true;
    }

    // --- Record miss for hybrid CPU/GPU dispatch ---
    // Track per-tensor so mul_mat_id knows which experts to CPU-correct after batch MMVQ.
    {
        auto & tmi = g_expert_pcache.per_tensor_misses[gpu_dst];
        if (!tmi.host_base) {
            tmi.host_base   = cpu_src;
            tmi.gpu_dst     = gpu_dst;
            tmi.expert_size = expert_size;
            tmi.n_expert    = n_expert;
            tmi.block_size  = (int)pcache_block_size(expert_size);  // IQ-safe, per-tensor
        }
        // Make the offset table's stride authoritative for miss-only tensors too (flush
        // reads toff.type_size; without a hit it would otherwise infer and mis-size IQ).
        {
            auto & toff_bs = g_pcache_tensor_build[gpu_dst];
            if (toff_bs.type_size == 0 && tmi.block_size > 0) toff_bs.type_size = tmi.block_size;
        }
        tmi.all_miss_ids.push_back(expert_id);
        g_expert_pcache.token_has_misses = true;
        g_pcache_has_misses = true;
        // Mark this tensor as having a miss; the offset table cannot be used
        g_pcache_tensor_build[gpu_dst].has_miss = true;
    }

    // Miss H2D + pool insert both happen in flush (single writer for input_cpy):
    //   - one H2D per miss, instead of an immediate H2D here + re-H2D in flush + host H2D
    //     for the pool insert
    //   - flush's reverse sync (compute->dma event) runs before the miss H2Ds, so a
    //     dma-stream write to a ggml-alloc-reused input_cpy address cannot race a prior
    //     tensor's kernel or deferred D2D
    //   - flush's forward sync (dma->compute event) runs after, so this tensor's kernel
    //     sees fully-written input_cpy

    // Periodic logging (cumulative, capped)
    {
        int64_t total = g_expert_pcache.hits + g_expert_pcache.misses;
        if (total > 0 && (total & (total - 1)) == 0 && total <= 131072) {
            fprintf(stderr, "[expert-cache] %lld hits / %lld total (%.1f%%), pool %.0f / %.0f MB, "
                    "threshold=%d, unique=%zu, evictions=%lld, dma_stream=%s, offset_table=%s\n",
                    (long long)g_expert_pcache.hits, (long long)total,
                    100.0 * g_expert_pcache.hits / total,
                    (float)g_expert_pcache.used / (1024*1024),
                    (float)g_expert_pcache.pool_size / (1024*1024),
                    g_expert_pcache.insert_threshold,
                    g_expert_pcache.freq.size(),
                    (long long)g_expert_pcache.lru_evictions,
                    use_dma_stream ? "active" : "fallback",
                    g_pcache_offset_active ? "active" : "off");
            if (g_expert_pcache.hotset_enabled) {
                fprintf(stderr, "[expert-cache] hot set: %s, pinned=%zu, protected-accesses=%lld\n",
                        g_expert_pcache.hotset_finalized ? "active" : "profiling",
                        g_expert_pcache.hot_protected.size(),
                        (long long)g_expert_pcache.hot_hits);
                if (g_expert_pcache.hot_evicted > 0) {
                    fprintf(stderr, "[expert-cache] hot set: %lld protected slots surrendered "
                            "(workload shifted away from the profiled set)\n",
                            (long long)g_expert_pcache.hot_evicted);
                }
            }
            if (g_pcache_layer_fuse) {
                fprintf(stderr, "[expert-cache] layer-fuse: layers=%lld issued=%lld adopted=%lld "
                        "(groups learned=%zu)\n",
                        (long long)g_pcache_fuse_layers, (long long)g_pcache_fuse_issued,
                        (long long)g_pcache_fuse_adopted, g_pcache_groups.size());
            }
            if (g_pcache_dma_stats) {
                g_expert_pcache.fetch_timing_drain();
                const double ms = g_expert_pcache.fetch_ms_total;
                const double gb = (double)g_expert_pcache.fetch_bytes_total / 1e9;
                fprintf(stderr, "[expert-cache] H2D timing: bursts=%lld bytes=%.2fGB busy=%.1fms "
                        "-> inst=%.1f GB/s | avg burst %.0fKB in %.0fus "
                        "(PCIe4 x8 ceiling ~12 GB/s; low inst => slow transfer, high inst => idle link)\n",
                        (long long)g_expert_pcache.fetch_bursts, gb, ms,
                        ms > 0 ? gb / (ms / 1e3) : 0.0,
                        g_expert_pcache.fetch_bursts > 0
                            ? (double)g_expert_pcache.fetch_bytes_total / g_expert_pcache.fetch_bursts / 1024.0 : 0.0,
                        g_expert_pcache.fetch_bursts > 0 ? ms * 1e3 / g_expert_pcache.fetch_bursts : 0.0);
            }
            fprintf(stderr, "[expert-cache] miss flow: src staged=%lld (pinned=%lld stagecp=%lld "
                    "memcpy=%.1fms) mmap=%lld | "
                    "dst pool=%lld scratch=%lld cpu=%lld cold_skip=%lld | wraps=%lld | stage calls=%lld evict=%lld | "
                    "prefetch issued=%lld used=%lld skip=%lld\n",
                    (long long)g_expert_pcache.miss_src_staged,
                    (long long)g_expert_pcache.miss_src_pinned,
                    (long long)g_expert_pcache.miss_src_stagecp,
                    (double)g_expert_pcache.stage_memcpy_ns / 1e6,
                    (long long)g_expert_pcache.miss_src_mmap,
                    (long long)g_expert_pcache.miss_dst_pool,
                    (long long)g_expert_pcache.miss_dst_scratch,
                    (long long)g_expert_pcache.miss_dst_cpu,
                    (long long)g_expert_pcache.cold_skipped,
                    (long long)g_expert_pcache.scratch_wraps,
                    (long long)g_expert_pcache.stage_calls,
                    (long long)g_expert_pcache.stage_evictions,
                    (long long)g_expert_pcache.prefetch_issued,
                    (long long)g_expert_pcache.prefetch_used,
                    (long long)g_expert_pcache.prefetch_skipped);
            // Ring-epoch telemetry.
            fprintf(stderr, "[pcache-ring] epoch=%d resets=%lld declined=%lld wraps=%lld\n",
                    (int)g_pcache_scratch_epoch,
                    (long long)g_expert_pcache.scratch_epoch_resets,
                    (long long)g_expert_pcache.scratch_declined,
                    (long long)g_expert_pcache.scratch_wraps);
        }
    }

    return true;
}

// Flush: sync DMA stream + batch H2D for GPU-half misses.
// Called after the expert copy loop, before graph compute launches.
//
// When misses exist: splits each tensor's misses into GPU-half (H2D now) and CPU-half
// (computed by CPU thread in parallel with graph execution). Only waits for GPU-half DMA.
GGML_BACKEND_API_C void ggml_cuda_expert_pcache_flush(void * backend_ptr) {
    ggml_backend_t backend = (ggml_backend_t)backend_ptr;
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)backend->context;
    cudaStream_t compute_stream = ctx->stream();

    // Sync any pending DMA from cache inserts (background warming)
    if (g_expert_pcache.dma_pending) {
        g_expert_pcache.flush(compute_stream);
    }

    // Decode passes (n_tokens==1, non-hybrid) run in pool-direct mode: every used
    // expert, hit or miss, resolves to an offset inside the pool allocation, the
    // full per-expert table is uploaded, and the MMVQ kernel reads the pool directly.
    // input_cpy is never written on decode, so:
    //   - the ggml-alloc address-reuse race (deferred D2D) cannot occur
    //     -> no reverse sync needed on decode
    //   - kernel params are uniform across decode passes (vx=pool, d_offsets=d_buf)
    //     -> CUDA graphs are safe (table contents refresh host-side every pass)
    static const int HYBRID_MISS_THRESHOLD = g_pcache_hybrid_miss_threshold;
    int total_misses = 0;
    for (auto & [gpu_dst, tmi] : g_expert_pcache.per_tensor_misses) {
        total_misses += (int)tmi.all_miss_ids.size();
    }
    const bool use_hybrid  = (total_misses >= HYBRID_MISS_THRESHOLD);
    if (g_pcache_batch_ntokens == 1 && g_expert_pcache.warmup_done) {
        g_dbg_decode_passes++;
        if (use_hybrid) g_dbg_hybrid_passes++;
        g_dbg_misses_sum += total_misses;
        if (total_misses > g_dbg_misses_max) g_dbg_misses_max = total_misses;
    }
    // Pool-direct requires: single token (ne2==1 -> MMVQ guaranteed; the per-type
    // mmid_max_batch minimum is 4, so 1 always qualifies), quantized src0 (F16/BF16
    // route to mul_mat_vec_f which has no table support and reads input_cpy), no
    // hybrid CPU correction, and warmup complete (warmup misses bypass per-tensor
    // tracking, so their table entries would be unresolved).
    // ALPHA_PD_MMVQ=1 (window patch-decode): route 2..8-token passes pool-direct via the
    // batched MMVQ MoE kernel (mul_mat_vec_q_moe has full expert_offsets support): no
    // input_cpy write, no deferred-D2D reshuffle, no reverse sync. This is the kernel family
    // the dispatch already uses for M <= 8, unlike the MMQ route. Makes the window verify pass
    // fast on all-hit/pool-landed experts.
    static const bool pd_mmvq = alpha_env_on("ALPHA_PD_MMVQ");
    const bool mt_enabled = pd_mmvq;
    const bool mmq_consumes = pd_mmvq;   // batched MMVQ consumes the tables for M<=8
    const bool pool_direct_mt = mt_enabled
                          && g_pcache_batch_ntokens >= 2
                          && g_pcache_batch_ntokens <= g_pcache_mt_max
                          && g_pcache_batch_src_quantized
                          && !use_hybrid
                          && g_expert_pcache.warmup_done
                          && mmq_consumes;
    // ALPHA_PD_TELEM: engagement telemetry. Every gate below can silently disarm this path, so
    // print which gate failed.
    if (getenv("ALPHA_PD_TELEM")) {
        static int pd_shown = 0;
        if (pd_shown < 12) { pd_shown++;
            fprintf(stderr, "[pd-telem] ntok=%d ENGAGED=%d | mt_enabled=%d ntok>=2:%d ntok<=max(%d):%d "
                            "quantized=%d !hybrid=%d warmup_done=%d mmq_consumes=%d",
                    (int) g_pcache_batch_ntokens, (int) pool_direct_mt, (int) mt_enabled,
                    (int) (g_pcache_batch_ntokens >= 2), g_pcache_mt_max,
                    (int) (g_pcache_batch_ntokens <= g_pcache_mt_max),
                    (int) g_pcache_batch_src_quantized, (int) !use_hybrid,
                    (int) g_expert_pcache.warmup_done, (int) mmq_consumes);
            fputc(10, stderr);
        }
    }
    // ik-quant experts are decoded by ggml_cuda_mul_mat_id_ik_pool. Force non-pool-direct so the
    // flush always populates input_cpy (deferred D2D for pool hits + H2D for misses) at the natural
    // full-tensor stride. The ik decode then reads input_cpy directly, independent of the
    // scratch-ring/panic offset-table machinery, whose miss offsets are unreliable at very low hit
    // rates. Correct but slow; a device-side trellis MMVQ reading the pool table would be faster.
    // ALPHA_NO_POOL_DIRECT: diagnostic, force the coherent non-pool-direct (input_cpy)
    // path even on single-token decode, without engaging the CPU-hybrid correction.
    static const bool g_no_pool_direct = alpha_env_on("ALPHA_NO_POOL_DIRECT");
    const bool pool_direct = !g_no_pool_direct
                          && !g_pcache_batch_src_is_ik
                          && (pool_direct_mt
                          || ((g_pcache_batch_ntokens == 1)
                          && g_pcache_batch_src_quantized
                          && !use_hybrid
                          && g_expert_pcache.warmup_done));

    cudaStream_t h2d_stream = g_expert_pcache.dma_stream ? g_expert_pcache.dma_stream : compute_stream;

    // Upload a tensor's per-expert block-offset table and publish it for the host
    // MMVQ launch (via the extra fn -> per-pass tensor map).
    auto upload_table = [&](auto & toff) {
        // The kernels index this by the raw routed expert id, so it must cover every expert the
        // model has.
        const int64_t n_tbl = toff.n_experts();
        if (n_tbl <= 0) return;
        int64_t * & d_buf = g_pcache_d_offsets_per_tensor[toff.cpu_src];
        size_t    & d_cap = g_pcache_d_offsets_cap[toff.cpu_src];
        if (!d_buf || d_cap < (size_t) n_tbl) {
            if (d_buf) CUDA_CHECK(cudaFree(d_buf));
            CUDA_CHECK(cudaMalloc(&d_buf, n_tbl * sizeof(int64_t)));
            d_cap = (size_t) n_tbl;
        }
        std::vector<int64_t> block_offsets(n_tbl);
        for (int64_t i = 0; i < n_tbl; i++) {
            int64_t bo = toff.pool_byte_offsets[i];
            // The kernel indexes by block, so this divide must be exact. A byte offset that is not
            // a multiple of this tensor's type_size would truncate to a wrong but in-bounds slot.
            // Counted rather than assumed.
            if (g_pcache_bs_telem && bo > 0 && toff.type_size > 0 &&
                (bo % (int64_t) toff.type_size) != 0) {
                static int shown = 0; static int64_t nbad = 0;
                nbad++;
                if (shown < 10) { shown++;
                    fprintf(stderr, "[pcache-align] BAD bo=%lld %% type_size=%zu = %lld  (expert %lld, src=%p, n_bad=%lld)",
                            (long long) bo, toff.type_size, (long long)(bo % (int64_t) toff.type_size),
                            (long long) i, toff.cpu_src, (long long) nbad);
                    fputc(10, stderr);
                }
            }
            block_offsets[i] = (bo > 0) ? (int64_t)(bo / (int64_t)toff.type_size) : 0;
            // the kernel truncates this to int (mmvq.cu); a pool large enough to overflow it
            // would silently wrap into a wild address, so fail loudly instead
            GGML_ASSERT(block_offsets[i] <= (int64_t) INT_MAX && "pcache offset table overflows int");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_buf, block_offsets.data(), n_tbl * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, compute_stream));
        pcache_offset_info * & info = g_pcache_info_per_tensor[toff.cpu_src];
        if (!info) info = new pcache_offset_info;
        info->pool_base = g_expert_pcache.pool;
        info->d_offsets = d_buf;
        g_pcache_last_all_hit  = true;   // semantics: table valid for this tensor
        g_pcache_last_info_ptr = info;
        g_pcache_offset_active = true;
    };

    // Routed-expert trace: run at every flush exit for the tensor just flushed.
    auto write_ids_trace = [&]() {
        if (g_pcache_batch_ntokens != 1) return;  // decode passes only
        if (!g_pcache_ids_trace) return;          // fully inert by default
        const void * src = nullptr;
        for (auto & [gd, toff] : g_pcache_tensor_build) {
            if (toff.cpu_src) { src = toff.cpu_src; break; }
        }
        if (!src) return;
        auto & self = g_expert_pcache.prefetch_info[src];
        // Ordered per-token top-k trace (ALPHA_TRACE_IDS): input for the offline
        // spec-verify M-distribution analysis (per draft length K).
        // Annotated format: "<id>:H" per expert (H=1 pool-hit at request time,
        // H=0 miss); the H=0 stream is the real PCIe cost.
        if (!self.curr_used.empty()) {
            const bool annotated = self.curr_hit.size() == self.curr_used.size();
            fprintf(g_pcache_ids_trace, "%lld %p", (long long)g_pcache_pass_ctr, src);
            for (size_t i = 0; i < self.curr_used.size(); i++) {
                if (annotated) {
                    fprintf(g_pcache_ids_trace, " %d:%d", self.curr_used[i], (int)self.curr_hit[i]);
                } else {
                    fprintf(g_pcache_ids_trace, " %d", self.curr_used[i]);
                }
            }
            fputc('\n', g_pcache_ids_trace);
        }

        self.curr_used.clear();
        self.curr_hit.clear();
    };

    // --- Offset table resolution (hits) ---
    g_pcache_last_all_hit = false;
    g_pcache_last_offset  = { nullptr, nullptr };
    if (!g_pcache_tensor_build.empty()) {
        for (auto & [gpu_dst, toff] : g_pcache_tensor_build) {
            if (!toff.has_miss && (!toff.deferred.empty() || toff.n_adopted > 0) &&
                toff.cpu_src && toff.type_size > 0) {
                // All experts were cache hits.
                // Pool-direct decode: the kernel reads the pool via the table, no D2D.
                // Multi-token staging (pool_direct_mt): the table is built for MMQ, but
                // input_cpy is kept authoritative via the deferred D2D, so output is unchanged.
                if (!pool_direct) {
                    for (auto & dc : toff.deferred) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            dc.dst, dc.src, dc.size,
                            cudaMemcpyDeviceToDevice, compute_stream));
                    }
                }
                upload_table(toff);
            } else if (toff.has_miss && !pool_direct) {
                // Mixed hit/miss on a multi-token pass (or staging): retroactive D2D
                // for the deferred hits (misses get input_cpy H2D below).
                for (auto & dc : toff.deferred) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        dc.dst, dc.src, dc.size,
                        cudaMemcpyDeviceToDevice, compute_stream));
                }
                g_pcache_last_all_hit = false;
                g_pcache_last_offset  = { nullptr, nullptr };
            }
            // has_miss && pool_direct: table uploaded after miss resolution below.
        }

        // Reverse sync: needed whenever input_cpy is written (multi-token legacy and
        // multi-token staging). ggml-alloc reuses input_cpy addresses across tensors,
        // so a later tensor's dma H2D could race an earlier tensor's D2D/kernel at the
        // same address. Single-token pool-direct never writes input_cpy, so it is not needed.
        if (!pool_direct && g_expert_pcache.dma_stream) {
            if (!g_expert_pcache.d2d_done_event) {
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &g_expert_pcache.d2d_done_event, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(g_expert_pcache.d2d_done_event, compute_stream));
            CUDA_CHECK(cudaStreamWaitEvent(
                g_expert_pcache.dma_stream, g_expert_pcache.d2d_done_event, 0));
        }
    }

    // ALPHA_MT_DEBUG: report the multi-token pool-direct publication state
    if (pool_direct_mt && getenv("ALPHA_MT_DEBUG")) {
        static int64_t mtdbg = 0;
        if ((mtdbg++ & 255) == 0) {
            size_t defsz = 0, nhit = 0, nmiss = 0;
            for (auto & [gd, toff] : g_pcache_tensor_build) {
                defsz += toff.deferred.size();
                if (toff.has_miss) nmiss++; else nhit++;
            }
            fprintf(stderr, "[mt-dbg] ntok=%lld pd=%d pd_mt=%d mmq_cons=%d hasmiss=%d off_active=%d "
                    "ntensors=%zu (hit=%zu miss=%zu) deferred_total=%zu\n",
                    (long long)g_pcache_batch_ntokens, (int)pool_direct, (int)pool_direct_mt,
                    (int)mmq_consumes, (int)g_expert_pcache.token_has_misses, (int)g_pcache_offset_active,
                    g_pcache_tensor_build.size(), nhit, nmiss, defsz);
        }
    }

    // LAYER_FUSE=2: a sibling whose experts were all pre-placed by the gate's fused fetch has no
    // misses of its own, so it would otherwise fall straight through with no fence. In mode 2 the
    // post-fused fence covers only the gate, so each sibling must wait on its own completion event
    // here. The wait is nearly free when the bytes already landed under the gate's kernel.
    if (g_pcache_layer_fuse == 2 && g_expert_pcache.dma_stream) {
        const void * self_src = nullptr;   // same resolution write_ids_trace uses
        for (auto & [gd, toff] : g_pcache_tensor_build) {
            if (toff.cpu_src) { self_src = toff.cpu_src; break; }
        }
        for (int m = 1; m < pcache_layer_group::MAXM; m++) {
            if (self_src && g_pcache_fuse_ev_src[m] == self_src && g_pcache_fuse_ev[m]) {
                // Instrumented (see fence_wait_on): counted as an exit fence for STALL.
                g_expert_pcache.fence_wait_on(compute_stream, g_pcache_fuse_ev[m], /*is_drain=*/false);
                g_pcache_fuse_ev_src[m] = nullptr;      // consumed
                g_pcache_fuse_ev_waits++;
                break;
            }
        }
    }

    // If no misses this token: nothing more to do (the ids trace still runs)
    if (!g_expert_pcache.token_has_misses) {
        write_ids_trace();
        g_pcache_batch_flushed = true;
        return;
    }

    // Clear forward-visible CPU miss map (will be repopulated below)
    g_pcache_cpu_misses.clear();

    bool panic_fallback = false;      // pool-direct miss couldn't be placed -> input_cpy fallback
    bool hybrid_cpu_assigned = false; // hybrid: some misses assigned to CPU compute
    bool self_fenced = false;         // LAYER_FUSE=2: this tensor already fenced on its own bytes

    // Bracket this flush's copy burst (see fetch_timing_*). Gated on _now, not on the bool:
    // fetch timing must sample the same tokens as the fence timing, or the per-token scaling
    // would divide link bytes collected on every token by the sampled-token count.
    if (g_pcache_dma_stats_now) {
        g_expert_pcache.fetch_timing_drain();
        g_expert_pcache.fence_timing_drain();   // per-flush drain: the ring must not fill
        g_expert_pcache.fetch_timing_begin(h2d_stream);
    }

    for (auto & [gpu_dst, tmi] : g_expert_pcache.per_tensor_misses) {
        if (tmi.flushed) continue;
        tmi.flushed = true;

        int n_misses = (int)tmi.all_miss_ids.size();

        if (pool_direct) {
            // Pool-direct decode miss: land the expert inside the pool allocation and
            // point the offset table at it. Admitted misses take a persistent pool
            // slot (stays cached); the rest take a transient scratch-ring slot.
            // One H2D per miss, straight to its final location: no input_cpy write,
            // no insert-D2D.
            auto & toff = g_pcache_tensor_build[gpu_dst];
            if (!toff.cpu_src) toff.cpu_src = tmi.host_base;
            if (toff.type_size == 0) {
                // Authoritative per-tensor block size (IQ-safe); inference only as last resort
                toff.type_size = tmi.block_size > 0 ? (size_t)tmi.block_size
                                                    : pcache_block_size(tmi.expert_size);
            }
            size_t insert_align = toff.type_size;
            bool table_ok = (toff.type_size > 0);

            // CPU-inline: the CPU-taken misses are computed on the CPU (weights read from RAM)
            // instead of DMA'd over PCIe. Their table entries point at the zero-expert image so
            // batch MMVQ contributes 0; the mul_mat_id correction overwrites their dst rows.
            // By default all misses go to the CPU (zero-slot, no H2D);
            // ALPHA_INLINE_SPLIT<100 hands the leading (100-SPLIT)% to the GPU via the normal
            // fetch/H2D path (run_batch_mmvq computes them) and keeps only the tail SPLIT% on
            // the CPU worker.
            // ALPHA_CPU_INLINE_MT lifts the !pool_direct_mt veto for the CPU-inline share:
            // the correction works off per-expert zero-slot table entries, which are just
            // as valid when MMQ reads the pool for a multi-token batch. Gated to batches
            // the dispatch can enumerate (PCACHE_INL_MAX pairs): the assign side and the
            // compute side must agree, or a CPU-assigned expert would contribute zero and
            // silently corrupt output.
            const bool inline_mt_ok = g_pcache_inline_mt &&
                                      g_pcache_ids_ne0 > 0 && g_pcache_batch_ntokens > 0 &&
                                      g_pcache_ids_ne1 == g_pcache_batch_ntokens &&
                                      (int64_t)g_pcache_batch_ntokens * g_pcache_ids_ne0 <= PCACHE_INL_MAX;
            const bool inline_take_ok = g_pcache_cpu_inline && table_ok &&
                                        (!pool_direct_mt || inline_mt_ok);
            // Batch-aware split. A missed expert DMA'd to the GPU serves one token at single-user
            // decode but several at batch, so the PCIe cost per token drops at batch while the
            // CPU cost per token drops less. ALPHA_INLINE_SPLIT governs single-token decode;
            // ALPHA_INLINE_SPLIT_MT governs multi-token.
            // Note that ALPHA_CPU_INLINE is the master switch and defaults off (see
            // g_pcache_cpu_inline). ALPHA_CPU_INLINE_MT only lifts the multi-token veto and does not
            // enable the pipe: with ALPHA_CPU_INLINE unset, cpu_take stays 0 at batch regardless of
            // the MT knobs. Check engagement with ALPHA_INLINE_MT_DEBUG=1 when tuning these.
            // cpu_take = n_misses*split/100 uses integer division, so with few misses per pass a
            // small split floors to zero.
            //
            // Multi-token share: miss-density gated, default 100. The sign of the benefit flips with
            // miss density, not bandwidth: with a starved pool (many misses per pass) the CPU share
            // helps, while with a pool that fits the working set there is nothing to offload and the
            // fence/coordination cost is pure overhead. The rate-based autogate above cannot see
            // this. The CPU side does amortize at batch (several tokens per expert group).
            // Gate: engage the CPU share only when this pass has misses worth offloading.
            static const int split_mt = []() {
                const char * e = getenv("ALPHA_INLINE_SPLIT_MT");
                int v = e ? atoi(e) : 100;
                return v < 0 ? 0 : (v > 100 ? 100 : v);
            }();
            // Threshold (default 8): low enough to engage when the pool is starved, high enough to
            // stay off when the pool fits and to skip low-miss passes where the fence costs more
            // than the offload saves. Much higher values give up useful work.
            static const int split_mt_min_miss = []() {      // ALPHA_INLINE_SPLIT_MT_MIN_MISS
                const char * e = getenv("ALPHA_INLINE_SPLIT_MT_MIN_MISS");
                int v = e ? atoi(e) : 8;
                return v < 0 ? 0 : v;
            }();
            const int split_mt_eff = (n_misses >= split_mt_min_miss) ? split_mt : 0;
            const int eff_split = (g_pcache_batch_ntokens >= 2)
                                ? split_mt_eff
                                : alpha_resolve_split_for(n_misses, tmi.expert_size);  // measured-throughput split (opt-in)
            const int cpu_take = inline_take_ok
                               ? (int)((int64_t)n_misses * eff_split / 100) : 0;
            if (g_inline_mt_dbg) {
                static int64_t asg_dbg = 0;
                if ((asg_dbg++ & 1023) == 0) {
                    fprintf(stderr, "[inline-mt/assign] ntok=%d ids_ne0=%d ids_ne1=%d pd_mt=%d "
                            "inline_mt_ok=%d table_ok=%d n_misses=%d cpu_take=%d\n",
                            g_pcache_batch_ntokens, g_pcache_ids_ne0, g_pcache_ids_ne1,
                            (int)pool_direct_mt, (int)inline_mt_ok, (int)table_ok,
                            n_misses, cpu_take);
                }
            }

            for (int i = 0; i < n_misses && table_ok; i++) {
                int64_t eid = tmi.all_miss_ids[i];
                size_t offset = eid * tmi.expert_size;
                uint64_t key = cuda_expert_persist_cache::key(tmi.host_base, eid);

                if (cpu_take > 0 && i >= n_misses - cpu_take) {
                    int64_t z = g_expert_pcache.zero_slot(tmi.expert_size, insert_align, compute_stream);
                    if (z >= 0) {
                        toff.set_offset(eid, z);
                        tmi.cpu_miss_ids.push_back(eid);
                        g_expert_pcache.miss_dst_cpu++;
                        // Route every CPU-taken miss through the pool's own admission policy
                        // (damped/ranked, as plain decode does), and DMA the expert
                        // in only when a slot is granted. A fixed per-token forced budget warms the
                        // pool far more slowly than plain decode. The damped filter self-limits DMA
                        // volume to plain decode's admitted rate; a per-pass safety cap bounds the
                        // worst case.
                        if (g_pcache_cpu_inline && g_pcache_warm_left > 0 &&
                            g_expert_pcache.dma_stream) {
                            void * pslot = g_expert_pcache.insert(key, tmi.expert_size, insert_align);
                            if (pslot) {
                                const void * wsrc = nullptr;
                                if (g_expert_pcache.host_registered.count(tmi.host_base)) {
                                    wsrc = (const char *)tmi.host_base + offset;
                                } else if (g_expert_pcache.staging_enabled) {
                                    wsrc = g_expert_pcache.stage_expert(tmi.host_base, offset, tmi.expert_size, key);
                                }
                                if (wsrc) {
                                    g_expert_pcache.fetch_burst_bytes += (int64_t)tmi.expert_size;
                                    CUDA_CHECK(cudaMemcpyAsync(pslot, wsrc, tmi.expert_size,
                                                               cudaMemcpyHostToDevice, g_expert_pcache.dma_stream));
                                    g_expert_pcache.mark_dma_pending();
                                    g_pcache_warm_left--;
                                }
                            }
                        }
                        continue;  // no VRAM slot, no H2D on the kernel path: the CPU computes this one
                    }
                    // zero zone exhausted -> fall through to the normal GPU path
                }

                // Resolve destination: pool slot (admitted) or scratch ring slot
                size_t dst_byte_off;
                void * slot = g_expert_pcache.insert(key, tmi.expert_size, insert_align);
                if (slot) {
                    dst_byte_off = (size_t)((char *)slot - (char *)g_expert_pcache.pool);
                    g_expert_pcache.miss_dst_pool++;
                } else {
                    dst_byte_off = g_expert_pcache.scratch_alloc(
                        tmi.expert_size, insert_align, compute_stream);
                    if (dst_byte_off == SIZE_MAX) { table_ok = false; break; }
                    g_expert_pcache.miss_dst_scratch++;
                }
                toff.set_offset(eid, (int64_t)dst_byte_off);

                // H2D source, fastest first:
                //  (1) tensor cudaHostRegister'd (ALPHA_TENSOR_PIN) -> single-hop async DMA
                //      straight from the pinned expert tensor, no staging memcpy.
                //  (2) else pinned staging buffer (memcpy -> async DMA).
                //  (3) else pageable mmap on the compute stream (serial).
                const void * h2d_src = nullptr;
                cudaStream_t s;
                if (g_expert_pcache.host_registered.count(tmi.host_base) && g_expert_pcache.dma_stream) {
                    h2d_src = (const char *)tmi.host_base + offset;
                    s = h2d_stream;
                    g_expert_pcache.mark_dma_pending();
                    g_expert_pcache.miss_src_staged++;
                    g_expert_pcache.miss_src_pinned++;
                } else {
                    if (g_expert_pcache.staging_enabled) {
                        h2d_src = g_expert_pcache.stage_expert(
                            tmi.host_base, offset, tmi.expert_size, key);
                    }
                    if (h2d_src && g_expert_pcache.dma_stream) {
                        s = h2d_stream;
                        g_expert_pcache.mark_dma_pending();
                        g_expert_pcache.miss_src_staged++;
                        g_expert_pcache.miss_src_stagecp++;
                    } else {
                        if (!h2d_src) h2d_src = (const char *)tmi.host_base + offset;
                        s = compute_stream;
                        g_expert_pcache.miss_src_mmap++;
                    }
                }
                if (s == h2d_stream) g_expert_pcache.fetch_burst_bytes += (int64_t)tmi.expert_size;
                CUDA_CHECK(cudaMemcpyAsync(
                    (char *)g_expert_pcache.pool + dst_byte_off,
                    h2d_src, tmi.expert_size,
                    cudaMemcpyHostToDevice, s));
                // No input_cpy write: the kernel reads the pool directly via the table.
            }

            if (table_ok) {
                // Hybrid: publish the CPU-half list for the mul_mat_id correction
                if (!tmi.cpu_miss_ids.empty()) {
                    g_pcache_cpu_misses[(void *)gpu_dst] = {
                        tmi.host_base, tmi.expert_size, tmi.cpu_miss_ids
                    };
                    hybrid_cpu_assigned = true;
                }
                upload_table(toff);
            } else {
                // Panic path (should not happen: the scratch ring always fits an expert).
                // Fall back to input_cpy H2D for all misses of this tensor and force
                // CUDA graphs off this pass; a baked table-mode graph must not replay
                // against a tensor that reverted to input_cpy mode.
                // Disarm the offset table: upload_table() is the only writer of info->d_offsets, and
                // it is not otherwise reset. Skipping upload_table on this path would leave the
                // tensor advertising the table it uploaded on a previous pass, whose ring offsets the
                // pass-epoch reset has since recycled, so a pool-direct kernel would read another
                // pass's slots.
                {
                    auto it_info = g_pcache_info_per_tensor.find(toff.cpu_src);
                    if (it_info != g_pcache_info_per_tensor.end() && it_info->second) {
                        it_info->second->d_offsets = nullptr;
                    }
                }
                g_pcache_offset_active  = false;
                g_pcache_last_info_ptr  = nullptr;
                g_pcache_last_all_hit   = false;
                panic_fallback = true;
                for (int i = 0; i < n_misses; i++) {
                    int64_t eid = tmi.all_miss_ids[i];
                    size_t offset = eid * tmi.expert_size;
                    // On compute_stream, so not part of inst, but it is real link traffic
                    // and it blocks compute directly by being on that stream.
                    g_expert_pcache.offstream_bytes += (int64_t)tmi.expert_size;
                    CUDA_CHECK(cudaMemcpyAsync(
                        (char *)gpu_dst + offset,
                        (const char *)tmi.host_base + offset,
                        tmi.expert_size, cudaMemcpyHostToDevice, compute_stream));
                }
                // Also replay the deferred hit D2Ds so input_cpy is complete
                for (auto & dc : toff.deferred) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        dc.dst, dc.src, dc.size,
                        cudaMemcpyDeviceToDevice, compute_stream));
                }
            }
        } else if (!use_hybrid) {
            // Multi-token DMA-only: single H2D per miss to input_cpy. Source priority:
            //   1. Tensor cudaHostRegister'd (default) -> single-hop async DMA from the
            //      pinned expert tensor itself, no staging memcpy
            //   2. Tier 2 staging (pinned RAM) -> async DMA on dma_stream
            //   3. Fallback -> direct mmap on the compute stream (ordering vs reused addresses)
            // The reverse sync above ordered dma_stream after all prior compute work.
            for (int i = 0; i < n_misses; i++) {
                int64_t eid = tmi.all_miss_ids[i];
                size_t offset = eid * tmi.expert_size;
                uint64_t key = cuda_expert_persist_cache::key(tmi.host_base, eid);

                const void * h2d_src = nullptr;
                cudaStream_t s;
                if (g_expert_pcache.host_registered.count(tmi.host_base) && g_expert_pcache.dma_stream) {
                    h2d_src = (const char *)tmi.host_base + offset;
                    s = h2d_stream;  // pinned tensor source -> true async DMA
                    g_expert_pcache.mark_dma_pending();
                } else {
                    if (g_expert_pcache.staging_enabled) {
                        h2d_src = g_expert_pcache.stage_expert(
                            tmi.host_base, offset, tmi.expert_size, key);
                    }
                    if (h2d_src && g_expert_pcache.dma_stream) {
                        s = h2d_stream;  // pinned staging source -> true async DMA
                        g_expert_pcache.mark_dma_pending();
                    } else {
                        if (!h2d_src) h2d_src = (const char *)tmi.host_base + offset;
                        s = compute_stream;
                    }
                }
                if (s == h2d_stream) g_expert_pcache.fetch_burst_bytes += (int64_t)tmi.expert_size;
                CUDA_CHECK(cudaMemcpyAsync(
                    (char *)gpu_dst + offset,
                    h2d_src, tmi.expert_size,
                    cudaMemcpyHostToDevice, s));

                // Pool insert: D2D from the input_cpy bytes just written (same stream ->
                // ordered after the H2D).
                auto fit = g_expert_pcache.freq.find(key);
                if (fit != g_expert_pcache.freq.end() &&
                    fit->second.count >= g_expert_pcache.insert_threshold) {
                    size_t insert_align = tmi.block_size > 0 ? (size_t)tmi.block_size
                                                             : pcache_block_size(tmi.expert_size);
                    void * slot = g_expert_pcache.insert(key, tmi.expert_size, insert_align);
                    if (slot) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            slot, (const char *)gpu_dst + offset, tmi.expert_size,
                            cudaMemcpyDeviceToDevice, s));
                    }
                }
            }
        } else {
            // Hybrid: first half -> GPU (H2D), second half -> CPU (zero + CPU correction)
            int gpu_count = (n_misses + 1) / 2;

            for (int i = 0; i < gpu_count; i++) {
                int64_t eid = tmi.all_miss_ids[i];
                size_t offset = eid * tmi.expert_size;
                g_expert_pcache.fetch_burst_bytes += (int64_t)tmi.expert_size;
                CUDA_CHECK(cudaMemcpyAsync(
                    (char *)gpu_dst + offset,
                    (const char *)tmi.host_base + offset,
                    tmi.expert_size,
                    cudaMemcpyHostToDevice, h2d_stream));
            }

            for (int i = gpu_count; i < n_misses; i++) {
                int64_t eid = tmi.all_miss_ids[i];
                tmi.cpu_miss_ids.push_back(eid);
                // Zero input_cpy for the CPU half so batch MMVQ produces ~0 (overwritten later)
                size_t offset = eid * tmi.expert_size;
                CUDA_CHECK(cudaMemsetAsync(
                    (char *)gpu_dst + offset, 0, tmi.expert_size, compute_stream));
            }

            if (!tmi.cpu_miss_ids.empty()) {
                g_pcache_cpu_misses[(void *)gpu_dst] = {
                    tmi.host_base, tmi.expert_size, tmi.cpu_miss_ids
                };
            }
        }
    }

    // has_misses gates CUDA graphs. DMA-only/pool-direct misses keep graphs enabled
    // (all copies are host-issued before graph launch). Hybrid needs graphs off (CPU
    // correction is mid-graph host work). Panic fallback forces graphs off (mode-flip).
    g_pcache_has_misses = use_hybrid || panic_fallback || hybrid_cpu_assigned;

    // Layer-fused sibling fetch (ALPHA_LAYER_FUSE).
    // Runs after this tensor's own miss loop, so the sequence of insert() calls per token is the
    // same as without fusion (gate's experts, then up's, then down's), which keeps the
    // admission-damping phase, the LRU order and therefore the pool contents unchanged.
    // Decode + pool-direct only; anything else keeps the per-tensor path.
    if (g_pcache_layer_fuse && pool_direct && g_pcache_batch_ntokens == 1 &&
        g_expert_pcache.warmup_done && g_pcache_group_key &&
        g_pcache_fused_key != g_pcache_group_key) {
        auto git = g_pcache_groups.find(g_pcache_group_key);
        if (git != g_pcache_groups.end() && git->second.complete && git->second.n_members > 1) {
            const pcache_layer_group & grp = git->second;
            g_pcache_fused_key = g_pcache_group_key;   // one fused fetch per layer per pass
            g_pcache_fuse_layers++;
            // Unique routed experts for this layer (decode: ids is one row of ne0 slots).
            int32_t uniq[64]; int n_uniq = 0;
            for (int s = 0; s < g_pcache_ids_ne0 && n_uniq < 64; s++) {
                const size_t idx = (size_t)s * g_pcache_ids_s0;
                if (idx >= g_pcache_batch_ids.size()) break;
                const int32_t e = g_pcache_batch_ids[idx];
                if (e < 0) continue;
                bool dup = false;
                for (int u = 0; u < n_uniq; u++) if (uniq[u] == e) { dup = true; break; }
                if (!dup) uniq[n_uniq++] = e;
            }
            // Place in ascending expert-id order, per member, exactly as the demand path would:
            // the backend's copy loop walks `for eid in 0..n_expert` against the used_ids bitset,
            // so a tensor's misses are recorded ascending. The routing array is in top-k order, so
            // iterating it verbatim would allocate ring slots in a different sequence and change
            // which offsets each expert lands on. Sorting makes the fused placement match fetching
            // them one at a time.
            for (int a = 1; a < n_uniq; a++) {
                const int32_t v = uniq[a];
                int b = a - 1;
                while (b >= 0 && uniq[b] > v) { uniq[b + 1] = uniq[b]; b--; }
                uniq[b + 1] = v;
            }
            // Mode 2: fence this tensor against only its own bytes, before issuing the siblings'.
            // Everything enqueued on h2d_stream up to here is this tensor's demand misses, so the
            // event covers exactly them. The sibling copies issued below then transfer underneath
            // this tensor's kernel instead of in front of it. (Mode 1 records its fence after the
            // loop, so gate compute waits on down's bytes.)
            if (g_pcache_layer_fuse == 2 && g_expert_pcache.dma_stream) {
                g_expert_pcache.fence_wait(compute_stream, h2d_stream, /*is_drain=*/false);
                g_expert_pcache.dma_pending = false;
                self_fenced = true;
            }
            // members[0] is the tensor just flushed; its experts are already resolved.
            for (int m = 1; m < grp.n_members; m++) {
                const void * msrc = grp.members[m];
                const size_t esz  = grp.expert_size[m];
                const size_t tsz  = grp.type_size[m];
                if (!msrc || esz == 0) continue;
                int m_issued = 0;
                for (int u = 0; u < n_uniq; u++) {
                    const int64_t  eid = uniq[u];
                    const uint64_t mk  = cuda_expert_persist_cache::key(msrc, eid);
                    if (g_expert_pcache.entries.count(mk)) continue;      // resident (peek, no LRU bump)
                    if (g_pcache_pass_placements.count(mk))  continue;    // already placed this layer
                    // Source: the same 3-tier ladder as the demand path. Tier 3 (pageable mmap)
                    // is skipped rather than fetched early: it would have to go on the compute
                    // stream, which is the serialization this is trying to remove.
                    const void * src = nullptr;
                    const size_t off = (size_t)eid * esz;
                    if (g_expert_pcache.host_registered.count(msrc) && g_expert_pcache.dma_stream) {
                        src = (const char *)msrc + off;
                    } else if (g_expert_pcache.staging_enabled) {
                        src = g_expert_pcache.stage_expert(msrc, off, esz, mk);
                    }
                    if (!src || !g_expert_pcache.dma_stream) continue;
                    // Destination: unchanged insert()-then-scratch policy.
                    size_t dst_off;
                    void * slot = g_expert_pcache.insert(mk, esz, tsz);
                    if (slot) {
                        dst_off = (size_t)((char *)slot - (char *)g_expert_pcache.pool);
                        g_expert_pcache.miss_dst_pool++;
                    } else {
                        dst_off = g_expert_pcache.scratch_alloc(esz, tsz, compute_stream);
                        if (dst_off == SIZE_MAX) break;      // ring exhausted -> leave to demand path
                        g_expert_pcache.miss_dst_scratch++;
                    }
                    g_expert_pcache.fetch_burst_bytes += (int64_t)esz;
                    CUDA_CHECK(cudaMemcpyAsync((char *)g_expert_pcache.pool + dst_off, src, esz,
                                               cudaMemcpyHostToDevice, h2d_stream));
                    // Mode 2: do not arm dma_pending. These copies are covered by the per-member
                    // event each sibling waits on. Arming it would make the next flush's entry drain
                    // wait on the whole layer's outstanding DMA, the coarse wait the per-member events
                    // replace. Mode 1 still arms it (its single fence is the only thing covering
                    // these bytes).
                    if (g_pcache_layer_fuse != 2) g_expert_pcache.mark_dma_pending();
                    g_pcache_pass_placements[mk] = dst_off;
                    g_pcache_fuse_issued++;
                    m_issued++;
                }
                // Mode 2: publish this member's completion. h2d_stream is in-order, so this fires
                // exactly when member m's bytes have landed, not the whole layer's.
                if (g_pcache_layer_fuse == 2 && m_issued > 0 && m < pcache_layer_group::MAXM) {
                    if (!g_pcache_fuse_ev[m]) {
                        CUDA_CHECK(cudaEventCreateWithFlags(&g_pcache_fuse_ev[m], cudaEventDisableTiming));
                    }
                    CUDA_CHECK(cudaEventRecord(g_pcache_fuse_ev[m], h2d_stream));
                    g_pcache_fuse_ev_src[m] = msrc;
                }
            }
        }
    }

    // Sync H2D: record event on stream 1, stream 0 waits. Unconditional (not gated on
    // dma_pending): the hybrid GPU-half H2D below never calls mark_dma_pending(), so it
    // depends on this fence always firing. See fence_wait() for the stall instrumentation.
    if (g_expert_pcache.dma_stream && !self_fenced) {
        g_expert_pcache.fence_wait(compute_stream, h2d_stream, /*is_drain=*/false);
        // This fence covers everything issued above it, so the flag it leaves behind is stale and
        // the next flush's entry drain would re-fence copies already ordered here. Clearing it is
        // safe because what the drain genuinely protects re-arms the flag after this point: the
        // pool pre-fill in pcache_copy marks it on a pass whose flush may take the no-miss early
        // return and skip this fence entirely, so the drain still fires exactly when it is needed.
        // The fence itself stays unconditional: the hybrid GPU-half H2D never calls
        // mark_dma_pending() and depends on it.
        g_expert_pcache.dma_pending = false;
    }

    write_ids_trace();

    // Close the timing bracket: the fence is an event record on the same stream and costs
    // nothing on the stream timeline.
    if (g_pcache_dma_stats_now) g_expert_pcache.fetch_timing_end(h2d_stream);

    // Mark batch as flushed; the next pcache_copy call clears state for the new token
    g_pcache_batch_flushed = true;
}

// Get per-tensor offset info for last flushed tensor.
// Returns pointer to persistent pcache_offset_info { pool_base, d_offsets }.
// The struct is heap-allocated per-cpu_src, survives across forward passes.
GGML_BACKEND_API_C void * ggml_cuda_expert_pcache_get_extra(void) {
    return g_pcache_last_all_hit ? (void *)g_pcache_last_info_ptr : nullptr;
}

// Offset table lookup: no longer used (replaced by src0->extra mechanism).
// Kept for API compatibility but always returns false.
GGML_BACKEND_API_C bool ggml_cuda_expert_pcache_get_offsets(
        const void * src0_data, size_t type_size,
        void ** pool_base_out, int64_t ** d_offsets_out,
        cudaStream_t stream) {
    (void)src0_data; (void)type_size; (void)pool_base_out; (void)d_offsets_out; (void)stream;
    return false;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_PEER_MAX_BATCH_SIZE
        features.push_back({ "PEER_MAX_BATCH_SIZE", STRINGIFY(GGML_CUDA_PEER_MAX_BATCH_SIZE) });
    #endif

    #ifdef GGML_CUDA_FA_ALL_QUANTS
        features.push_back({ "FA_ALL_QUANTS", "1" });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_cuda_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_cuda_info().device_count; i++) {
                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);

                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
                dev_ctx->description = prop.name;

                char pci_bus_id[16] = {};
                snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
                dev_ctx->pci_bus_id = pci_bus_id;
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)

// Async transfer helpers

static cudaStream_t g_xfr_s = nullptr;

GGML_BACKEND_API_C void _acdi(void) {
    if (!g_xfr_s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_xfr_s, cudaStreamNonBlocking));
    }
}

// Async H2D copy, does not sync. Caller must call _acds() when done.
GGML_BACKEND_API_C void _acda(
        struct ggml_tensor * dst, const void * src,
        size_t offset, size_t size) {
    if (!g_xfr_s) _acdi();
    CUDA_CHECK(cudaMemcpyAsync(
        (char *)dst->data + offset, src, size,
        cudaMemcpyHostToDevice, g_xfr_s));
}

// Sync all pending async DMAs. Call before GPU uses the transferred data.
GGML_BACKEND_API_C void _acds(void) {
    if (g_xfr_s) {
        CUDA_CHECK(cudaStreamSynchronize(g_xfr_s));
    }
}

// (g_cmp_s defined earlier, auto-detected during first MUL_MAT_ID dispatch)

// Non-blocking barrier: the compute stream waits for DMA via an event dependency.
// Called from the eval callback (outside CUDA graph capture).
// (g_xfr_e defined earlier)
GGML_BACKEND_API_C void _acdb(void) {
    if (!g_xfr_s) return;
    if (!g_xfr_e) {
        CUDA_CHECK(cudaEventCreateWithFlags(&g_xfr_e, cudaEventDisableTiming));
    }
    // Record event after all pending DMA ops
    CUDA_CHECK(cudaEventRecord(g_xfr_e, g_xfr_s));
    if (g_cmp_s) {
        // Check if the compute stream is being graph-captured (no cross-stream deps during capture)
        cudaStreamCaptureStatus capture_status;
        cudaStreamIsCapturing(g_cmp_s, &capture_status);
        if (capture_status == cudaStreamCaptureStatusNone) {
            // Stream not captured: insert a GPU-side wait (non-blocking for the host)
            CUDA_CHECK(cudaStreamWaitEvent(g_cmp_s, g_xfr_e, 0));
        } else {
            // Stream is being captured: fall back to host sync
            CUDA_CHECK(cudaStreamSynchronize(g_xfr_s));
        }
    } else {
        // Compute stream not yet known: fall back to host sync
        CUDA_CHECK(cudaStreamSynchronize(g_xfr_s));
    }
}

// Pinned staging: allocate page-locked host memory for truly async DMA.
// cudaMemcpyAsync from pageable (mmap'd) memory is effectively synchronous: CUDA copies to an
// internal pinned buffer before the DMA, blocking the CPU. With explicit pinned staging, the CPU
// memcpy is fast and the GPU DMA is truly non-blocking.
GGML_BACKEND_API_C void * _acpa(size_t size) {
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    return (err == cudaSuccess) ? ptr : nullptr;
}

GGML_BACKEND_API_C void _acpf(void * ptr) {
    if (ptr) cudaFreeHost(ptr);
}

// Staged async H2D: memcpy pageable->pinned (CPU), then cudaMemcpyAsync pinned->GPU (truly async).
GGML_BACKEND_API_C void _acdas(
        struct ggml_tensor * dst, const void * src,
        size_t offset, size_t size, void * pinned_buf) {
    if (!g_xfr_s) _acdi();
    memcpy(pinned_buf, src, size);
    CUDA_CHECK(cudaMemcpyAsync(
        (char *)dst->data + offset, pinned_buf, size,
        cudaMemcpyHostToDevice, g_xfr_s));
}

// Multi-user expert cache API
GGML_BACKEND_API_C void ggml_cuda_set_expert_user(int user_id) {
    if (g_cuda_ctx_for_expert) {
        g_cuda_ctx_for_expert->active_user_id = user_id;
    }
}

GGML_BACKEND_API_C void ggml_cuda_init_expert_users(int n_users) {
    if (g_cuda_ctx_for_expert) {
        g_cuda_ctx_for_expert->n_expert_users = n_users;
        // Partitions will be initialized when expert_cache_budget is set (first host expert encounter)
    }
}

// Called during backend init to capture the context pointer
GGML_BACKEND_API_C void ggml_cuda_set_expert_ctx(void * ctx) {
    g_cuda_ctx_for_expert = (ggml_backend_cuda_context *)ctx;
}
