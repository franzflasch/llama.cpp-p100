// included by common.cpp

// Freed large blocks are kept for the next checkpoint (up to 4 blocks): checkpoints
// of one slot all have the same size, and a fresh 150 MiB block pays ~38k first-touch page faults and the
// kernel's zero fill (~0.1 s on a host whose RAM is mostly page cache) where a recycled one pays nothing.
namespace {
struct ckpt_pool {
    std::mutex mtx;
    std::unordered_map<void *, size_t> live;            // large blocks handed out: ptr -> bytes
    std::unordered_multimap<size_t, void *> free_blocks; // bytes -> ptr
    size_t cap = 4;
    // After a block is handed out and none of its size is left, fault in a spare one
    // on a background thread. With many checkpoints per slot (-ctxcp 32) nothing is evicted for 32 requests, so every
    // checkpoint was a fresh block and each short chat turn paid its ~0.1-0.16 s of page faults and zero fill.
    bool refilling = false;
};
ckpt_pool & get_ckpt_pool() {
    static ckpt_pool pool;
    return pool;
}

// Every pointer handed to common_ckpt_free comes from common_ckpt_alloc, so one free covers them
// all. MSVC has no aligned_alloc, and _aligned_malloc cannot mix with malloc/free, so on MSVC the
// small path is aligned too. Linux keeps malloc for small blocks and free for both.
void * ckpt_raw_alloc(size_t n) {
#ifdef _MSC_VER
    return _aligned_malloc(n, 64);
#else
    return malloc(n);
#endif
}
void * ckpt_raw_aligned(size_t alignment, size_t size) {
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}
void ckpt_raw_free(void * p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
}
}

void * common_ckpt_alloc(size_t n) {
    constexpr size_t huge = (size_t) 2 << 20;
    if (n < 2*huge) {
        void * p = ckpt_raw_alloc(n);
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return p;
    }
    const size_t sz = (n + huge - 1)/huge*huge;
    auto & pool = get_ckpt_pool();
    std::lock_guard<std::mutex> lock(pool.mtx);
    // keep one pre-faulted block of this size ready for the next checkpoint (pool.mtx held)
    auto refill = [&pool, sz]() {
        if (pool.refilling || pool.free_blocks.count(sz) > 0 || pool.free_blocks.size() >= pool.cap) {
            return;
        }
        pool.refilling = true;
        std::thread([sz]() {
            void * q = ckpt_raw_aligned((size_t) 2 << 20, sz);
            if (q != nullptr) {
                memset(q, 0, sz);   // take the page faults here, off the request's path
            }
            auto & pl = get_ckpt_pool();
            std::lock_guard<std::mutex> lk(pl.mtx);
            pl.refilling = false;
            if (q != nullptr) {
                if (pl.free_blocks.size() < pl.cap) {
                    pl.free_blocks.emplace(sz, q);
                } else {
                    ckpt_raw_free(q);
                }
            }
        }).detach();
    };
    auto it = pool.free_blocks.find(sz);
    if (it != pool.free_blocks.end()) {
        void * p = it->second;
        pool.free_blocks.erase(it);
        pool.live[p] = sz;
        refill();
        return p;
    }
    void * p = ckpt_raw_aligned(huge, sz);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    refill();
    pool.live[p] = sz;
    return p;
}

void common_ckpt_free(void * p) {
    if (p == nullptr) {
        return;
    }
    auto & pool = get_ckpt_pool();
    {
        std::lock_guard<std::mutex> lock(pool.mtx);
        auto it = pool.live.find(p);
        if (it != pool.live.end()) {
            const size_t sz = it->second;
            pool.live.erase(it);
            if (pool.free_blocks.size() < pool.cap) {
                pool.free_blocks.emplace(sz, p);
                return;
            }
        }
    }
    ckpt_raw_free(p);
}
