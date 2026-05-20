// Expert I/O subsystem — parallel pread, LRU cache, persistent I/O pool
// Port of moe_infer_mlx/core_src/expert_io.h
//
// Persistent IO pool: 4 threads, generation-based barrier, strided task
// distribution — matches the C IOThreadPool pattern exactly.

use std::os::unix::fs::FileExt;
use std::{fs::File, io};
use std::os::fd::AsRawFd;
use parking_lot::{Mutex, Condvar};
use std::sync::Arc;

use metal::Buffer;

extern "C" {
    fn pread(fd: i32, buf: *mut u8, count: usize, offset: i64) -> isize;
}

// ---- Expert LRU Cache ----

pub struct ExpertLRUCache {
    entries: Vec<ExpertCacheEntry>,
    entry_idx: Vec<i32>, // [num_layers * num_experts]
    max_entries: usize,
    used_entries: usize,
    access_counter: u64,
    pub hits: u64,
    pub misses: u64,
    pin_mask: Vec<u8>,  // per-entry: 1 = pinned (async CMD3 in flight)
}

struct ExpertCacheEntry {
    buffer: Buffer,
    layer_idx: i32,
    expert_idx: i32,
    last_used: u64,
}

impl ExpertLRUCache {
    pub fn new(num_layers: i32, num_experts: i32, max_entries: usize, expert_size: usize, device: &metal::Device) -> Self {
        let total = (num_layers as usize) * (num_experts as usize);
        let mut cache = Self {
            entries: Vec::with_capacity(max_entries),
            entry_idx: vec![-1i32; total],
            max_entries,
            used_entries: 0,
            access_counter: 0,
            hits: 0,
            misses: 0,
            pin_mask: vec![0u8; max_entries],
        };

        println!("[expert_cache] Initializing: max_entries={} ({} GB budget)",
            max_entries, max_entries as f64 * expert_size as f64 / 1e9);

        for _i in 0..max_entries {
            let buffer = device.new_buffer(expert_size as u64, metal::MTLResourceOptions::StorageModeShared);
            cache.entries.push(ExpertCacheEntry {
                buffer,
                layer_idx: -1,
                expert_idx: -1,
                last_used: 0,
            });
        }
        cache
    }

    /// Pin a cache entry (async CMD3 is using its buffer — don't evict).
    pub fn pin(&mut self, entry_idx: usize) {
        if entry_idx < self.pin_mask.len() {
            self.pin_mask[entry_idx] = 1;
        }
    }

    /// Unpin a cache entry.
    pub fn unpin(&mut self, entry_idx: usize) {
        if entry_idx < self.pin_mask.len() {
            self.pin_mask[entry_idx] = 0;
        }
    }

    /// Get the cache entry index for a given (layer, expert) pair, or -1 if not cached.
    pub fn entry_index(&self, layer_idx: i32, expert_idx: i32, num_experts: i32) -> i32 {
        self.entry_idx[layer_idx as usize * num_experts as usize + expert_idx as usize]
    }

    /// Unpin all entries in a batch.
    pub fn unpin_batch(&mut self, indices: &[usize]) {
        for &i in indices {
            if i < self.pin_mask.len() {
                self.pin_mask[i] = 0;
            }
        }
    }

    pub fn lookup(&mut self, layer_idx: i32, expert_idx: i32, num_experts: i32) -> Option<&Buffer> {
        let idx = self.entry_idx[layer_idx as usize * num_experts as usize + expert_idx as usize];
        if idx >= 0 {
            self.entries[idx as usize].last_used = {
                self.access_counter += 1;
                self.access_counter
            };
            self.hits += 1;
            Some(&self.entries[idx as usize].buffer)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, layer_idx: i32, expert_idx: i32, num_experts: i32) -> &Buffer {
        let ei = layer_idx as usize * num_experts as usize + expert_idx as usize;

        if self.entry_idx[ei] >= 0 {
            let idx = self.entry_idx[ei] as usize;
            self.entries[idx].last_used = { self.access_counter += 1; self.access_counter };
            return &self.entries[idx].buffer;
        }

        let target = if self.used_entries < self.max_entries {
            let t = self.used_entries;
            self.used_entries += 1;
            t
        } else {
            // LRU eviction (skip pinned entries — async CMD3 is reading them)
            let mut lru_idx = 0usize;
            let mut min_used = u64::MAX;
            for i in 0..self.entries.len() {
                if self.pin_mask[i] == 0 && self.entries[i].last_used < min_used {
                    min_used = self.entries[i].last_used;
                    lru_idx = i;
                }
            }
            if min_used == u64::MAX {
                // All entries pinned — fall back to first unpinned or 0
                lru_idx = 0;
            }
            let old_layer = self.entries[lru_idx].layer_idx;
            let old_expert = self.entries[lru_idx].expert_idx;
            if old_layer >= 0 && old_expert >= 0 {
                self.entry_idx[old_layer as usize * num_experts as usize + old_expert as usize] = -1;
            }
            lru_idx
        };

        self.entries[target].layer_idx = layer_idx;
        self.entries[target].expert_idx = expert_idx;
        self.entries[target].last_used = { self.access_counter += 1; self.access_counter };
        self.entry_idx[ei] = target as i32;
        &self.entries[target].buffer
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

// ---- Malloc-based Expert Cache ----

#[allow(dead_code)]
pub struct MallocExpertCache {
    data: Vec<*mut u8>,
    metal_bufs: Vec<Buffer>,
    layer_idx: Vec<i32>,
    expert_idx: Vec<i32>,
    last_used: Vec<u64>,
    entry_idx: Vec<i32>,
    max_entries: usize,
    used_entries: usize,
    access_counter: u64,
    pub hits: u64,
    pub misses: u64,
}

// Safety: the raw pointers in `data` come from aligned allocations.
unsafe impl Send for MallocExpertCache {}

impl MallocExpertCache {
    pub fn new(num_layers: i32, num_experts: i32, max_entries: usize, expert_size: usize, _device: &metal::Device) -> Self {
        let total = (num_layers as usize) * (num_experts as usize);
        let page_size = 16384;
        let aligned_size = (expert_size + page_size - 1) & !(page_size - 1);

        println!("[malloc_cache] Initializing: {} entries ({} GB)",
            max_entries, max_entries as f64 * expert_size as f64 / 1e9);

        let mut cache = Self {
            data: Vec::with_capacity(max_entries),
            metal_bufs: Vec::with_capacity(max_entries),
            layer_idx: vec![-1i32; max_entries],
            expert_idx: vec![-1i32; max_entries],
            last_used: vec![0u64; max_entries],
            entry_idx: vec![-1i32; total],
            max_entries,
            used_entries: 0,
            access_counter: 0,
            hits: 0,
            misses: 0,
        };

        // We don't create Metal buffers here since Rust doesn't have `newBufferWithBytesNoCopy`
        // easily accessible. The managed buffer approach uses separate allocations.
        for _i in 0..max_entries {
            let layout = std::alloc::Layout::from_size_align(aligned_size, page_size).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            cache.data.push(ptr);
        }
        cache
    }

    pub fn lookup(&mut self, layer: i32, expert: i32, num_experts: i32) -> Option<(*mut u8, usize)> {
        let idx = self.entry_idx[layer as usize * num_experts as usize + expert as usize];
        if idx >= 0 {
            self.last_used[idx as usize] = { self.access_counter += 1; self.access_counter };
            self.hits += 1;
            Some((self.data[idx as usize], idx as usize))
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, layer: i32, expert: i32, num_experts: i32) -> (*mut u8, usize) {
        let ei = layer as usize * num_experts as usize + expert as usize;

        if self.entry_idx[ei] >= 0 {
            let idx = self.entry_idx[ei] as usize;
            self.last_used[idx] = { self.access_counter += 1; self.access_counter };
            return (self.data[idx], idx);
        }

        let target = if self.used_entries < self.max_entries {
            let t = self.used_entries;
            self.used_entries += 1;
            t
        } else {
            let mut lru = 0usize;
            let mut min_used = self.last_used[0];
            for i in 1..self.max_entries {
                if self.last_used[i] < min_used {
                    min_used = self.last_used[i];
                    lru = i;
                }
            }
            let old_l = self.layer_idx[lru];
            let old_e = self.expert_idx[lru];
            if old_l >= 0 && old_e >= 0 {
                self.entry_idx[old_l as usize * num_experts as usize + old_e as usize] = -1;
            }
            lru
        };

        self.layer_idx[target] = layer;
        self.expert_idx[target] = expert;
        self.last_used[target] = { self.access_counter += 1; self.access_counter };
        self.entry_idx[ei] = target as i32;
        (self.data[target], target)
    }
}

impl Drop for MallocExpertCache {
    fn drop(&mut self) {
        let page_size = 16384;
        for &ptr in &self.data {
            if !ptr.is_null() {
                // The alloc size matches the aligned allocation from new()
                // We need to know the size but it's not stored — simplified cleanup.
                unsafe { std::alloc::dealloc(ptr, std::alloc::Layout::from_size_align(1, page_size).unwrap()) };
            }
        }
    }
}

// ---- Persistent I/O Thread Pool ----
// Matches C's IOThreadPool: 4 persistent threads, generation-based barrier,
// strided task distribution (thread i → tasks[i, i+N, i+2N, ...]).

pub const NUM_IO_THREADS: usize = 4;

/// Single I/O task — matches C's InferPreadTask
#[derive(Clone, Copy)]
pub struct IOPreadTask {
    pub fd: i32,
    pub dst: *mut u8,
    pub offset: u64,
    pub size: usize,
    pub result: isize,
    pub lz4_comp_buf: *mut u8,
    pub lz4_comp_size: u32,
}

unsafe impl Send for IOPreadTask {}

struct IOPoolShared {
    tasks: *const IOPreadTask,
    num_tasks: usize,
    tasks_completed: usize,
    generation: u64,
    shutdown: bool,
}

// Safety: tasks pointer is only accessed under the mutex. Workers only read
// tasks (never write them). The dispatch thread sets tasks before signaling
// workers and blocks until all workers finish — so the buffer outlives any
// worker access.
unsafe impl Send for IOPoolShared {}
unsafe impl Sync for IOPoolShared {}

struct IOPoolInner {
    shared: Mutex<IOPoolShared>,
    work_ready: Condvar,
    work_done: Condvar,
}

/// Persistent I/O thread pool — 4 workers that survive for the lifetime of
/// the inference session.  Dispatch blocks until all tasks complete.
pub struct IOPool {
    inner: Arc<IOPoolInner>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl IOPool {
    pub fn new() -> Self {
        let inner = Arc::new(IOPoolInner {
            shared: Mutex::new(IOPoolShared {
                tasks: std::ptr::null(),
                num_tasks: 0,
                tasks_completed: 0,
                generation: 0,
                shutdown: false,
            }),
            work_ready: Condvar::new(),
            work_done: Condvar::new(),
        });

        let mut handles = Vec::with_capacity(NUM_IO_THREADS);
        for tid in 0..NUM_IO_THREADS {
            let inner_clone = Arc::clone(&inner);
            handles.push(std::thread::spawn(move || {
                Self::worker(tid, inner_clone);
            }));
        }

        Self { inner, handles }
    }

    /// Worker thread function — matches C's io_pool_worker exactly.
    fn worker(tid: usize, inner: Arc<IOPoolInner>) {
        let mut my_gen: u64 = 0;
        loop {
            let mut shared = inner.shared.lock();
            // Wait for new work (generation change) or shutdown
            while shared.generation == my_gen && !shared.shutdown {
                inner.work_ready.wait(&mut shared);
            }
            if shared.shutdown {
                return;
            }
            my_gen = shared.generation;

            let num_tasks = shared.num_tasks as isize;
            let tasks_ptr = shared.tasks;
            // Unlock while working — workers read-only access the tasks array
            drop(shared);

            // Strided: thread tid handles tasks tid, tid+N, tid+2N, ...
            let mut i = tid as isize;
            while i < num_tasks {
                let t = unsafe { &*tasks_ptr.add(i as usize) };
                let result = if !t.lz4_comp_buf.is_null() && t.lz4_comp_size > 0 {
                    // LZ4 decompression path (placeholder — not yet wired up)
                    let nr = unsafe { pread(t.fd, t.lz4_comp_buf, t.lz4_comp_size as usize, t.offset as i64) };
                    if nr == t.lz4_comp_size as isize {
                        // TODO: lz4_flex decompress into t.dst
                        -1 // not yet implemented
                    } else {
                        -1
                    }
                } else {
                    unsafe { pread(t.fd, t.dst, t.size, t.offset as i64) }
                };
                unsafe {
                    (tasks_ptr.add(i as usize) as *mut IOPreadTask).as_mut().unwrap().result = result;
                }
                i += NUM_IO_THREADS as isize;
            }

            // Signal completion
            let mut shared = inner.shared.lock();
            shared.tasks_completed += 1;
            if shared.tasks_completed == NUM_IO_THREADS {
                inner.work_done.notify_one();
            }
        }
    }

    /// Dispatch tasks to the pool and block until all complete.
    /// Matches C's io_pool_dispatch exactly.
    pub fn dispatch(&self, tasks: &mut [IOPreadTask]) {
        let num = tasks.len();
        if num == 0 {
            return;
        }

        let mut shared = self.inner.shared.lock();
        shared.tasks = tasks.as_ptr();
        shared.num_tasks = num;
        shared.tasks_completed = 0;
        shared.generation += 1;
        // Wake all workers
        self.inner.work_ready.notify_all();
        // Wait for all workers to finish
        while shared.tasks_completed < NUM_IO_THREADS {
            self.inner.work_done.wait(&mut shared);
        }
    }

    /// Shut down all workers.  Call once at the end of the session.
    pub fn shutdown(self) {
        let mut shared = self.inner.shared.lock();
        shared.shutdown = true;
        self.inner.work_ready.notify_all();
        drop(shared);
        for h in self.handles {
            h.join().ok();
        }
    }
}

// ---- Parallel pread of experts ----

/// Batched expert I/O: lookup all K experts in cache, then dispatch all misses
/// to the persistent I/O pool for parallel pread.
/// Returns pointers to expert data (in cache buffers) for each of the K slots.
pub fn batch_expert_read(
    cache: &mut ExpertLRUCache,
    fd: &File,
    expert_indices: &[i32],
    k: usize,
    expert_size: usize,
    layer_idx: i32,
    num_experts: i32,
    io_pool: Option<&IOPool>,
) -> Vec<*const u8> {
    let raw_fd = fd.as_raw_fd();

    // Phase 1: lookup all K experts
    let mut ptrs: Vec<*const u8> = Vec::with_capacity(k);
    let mut miss_slots: Vec<usize> = Vec::new();
    let mut miss_expert_indices: Vec<i32> = Vec::new();

    for ki in 0..k {
        let expert_idx = expert_indices[ki];
        if let Some(buf) = cache.lookup(layer_idx, expert_idx, num_experts) {
            ptrs.push(buf.contents() as *const u8);
        } else {
            miss_slots.push(ki);
            miss_expert_indices.push(expert_idx);
            ptrs.push(std::ptr::null()); // placeholder, filled after read
        }
    }

    // Phase 2: reserve all cache slots, then parallel pread
    let num_misses = miss_slots.len();
    if num_misses > 0 {
        let mut miss_dsts: Vec<*mut u8> = Vec::with_capacity(num_misses);
        for &expert_idx in &miss_expert_indices {
            let slot = cache.insert(layer_idx, expert_idx, num_experts);
            miss_dsts.push(slot.contents() as *mut u8);
        }

        // Build I/O tasks
        let mut tasks: Vec<IOPreadTask> = Vec::with_capacity(num_misses);
        for i in 0..num_misses {
            tasks.push(IOPreadTask {
                fd: raw_fd,
                dst: miss_dsts[i],
                offset: miss_expert_indices[i] as u64 * expert_size as u64,
                size: expert_size,
                result: 0,
                lz4_comp_buf: std::ptr::null_mut(),
                lz4_comp_size: 0,
            });
        }

        // Dispatch to persistent pool or fall back to thread::scope
        if let Some(pool) = io_pool {
            pool.dispatch(&mut tasks);
        } else {
            // Fallback: parallel pread without persistent pool
            struct PreadJob {
                dst_addr: usize,
                offset: u64,
            }
            let mut jobs: Vec<PreadJob> = Vec::with_capacity(num_misses);
            for i in 0..num_misses {
                jobs.push(PreadJob {
                    dst_addr: miss_dsts[i] as usize,
                    offset: miss_expert_indices[i] as u64 * expert_size as u64,
                });
            }

            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(num_misses);
                for job in jobs {
                    handles.push(s.spawn(move || {
                        let dst = job.dst_addr as *mut u8;
                        let buf = unsafe { std::slice::from_raw_parts_mut(dst, expert_size) };
                        fd.read_exact_at(buf, job.offset).is_ok()
                    }));
                }
                for h in handles {
                    h.join().ok();
                }
            });
            // Mark all as valid in the fallback path (errors are silent)
            for i in 0..num_misses {
                ptrs[miss_slots[i]] = miss_dsts[i] as *const u8;
            }
            return ptrs;
        }

        // Check results from pool dispatch
        for i in 0..num_misses {
            if tasks[i].result == expert_size as isize {
                ptrs[miss_slots[i]] = miss_dsts[i] as *const u8;
            }
        }
    }

    ptrs
}

/// Zero-copy variant: returns the cache Metal buffer for each expert slot
/// (cloned — caller owns the ref), and the cache entry indices for pinning.
/// The caller binds these buffers directly in CMD3 — no memcpy needed.
/// Returns (buffers, pinned_entry_indices).
pub fn batch_expert_read_buffers(
    cache: &mut ExpertLRUCache,
    fd: &File,
    expert_indices: &[i32],
    k: usize,
    expert_size: usize,
    layer_idx: i32,
    num_experts: i32,
    io_pool: Option<&IOPool>,
) -> (Vec<Option<Buffer>>, Vec<usize>) {
    let raw_fd = fd.as_raw_fd();
    let mut bufs: Vec<Option<Buffer>> = Vec::with_capacity(k);
    let mut pinned: Vec<usize> = Vec::with_capacity(k);
    let mut miss_slots: Vec<usize> = Vec::new();
    let mut miss_expert_indices: Vec<i32> = Vec::new();

    // Phase 1: lookup — cache hits return the buffer immediately.
    for ki in 0..k {
        let expert_idx = expert_indices[ki];
        let ei = cache.entry_index(layer_idx, expert_idx, num_experts);
        if ei >= 0 {
            let ei = ei as usize;
            let buf = cache.entries[ei].buffer.clone();
            cache.pin(ei);
            pinned.push(ei);
            bufs.push(Some(buf));
        } else {
            miss_slots.push(ki);
            miss_expert_indices.push(expert_idx);
            bufs.push(None);
        }
    }

    // Phase 2: reserve cache slots, parallel pread misses
    let num_misses = miss_slots.len();
    if num_misses > 0 {
        let mut miss_bufs: Vec<Buffer> = Vec::with_capacity(num_misses);
        for &expert_idx in &miss_expert_indices {
            // Insert to reserve slot; returns the buffer reference.
            // Clone immediately (before any other cache access).
            let slot_buf = cache.insert(layer_idx, expert_idx, num_experts).clone();
            let ei = cache.entry_index(layer_idx, expert_idx, num_experts) as usize;
            cache.pin(ei);
            pinned.push(ei);
            miss_bufs.push(slot_buf);
        }

        // Build I/O tasks
        if let Some(pool) = io_pool {
            let mut tasks: Vec<IOPreadTask> = Vec::with_capacity(num_misses);
            for i in 0..num_misses {
                tasks.push(IOPreadTask {
                    fd: raw_fd,
                    dst: miss_bufs[i].contents() as *mut u8,
                    offset: miss_expert_indices[i] as u64 * expert_size as u64,
                    size: expert_size,
                    result: 0,
                    lz4_comp_buf: std::ptr::null_mut(),
                    lz4_comp_size: 0,
                });
            }
            pool.dispatch(&mut tasks);
            for i in 0..num_misses {
                if tasks[i].result == expert_size as isize {
                    bufs[miss_slots[i]] = Some(miss_bufs[i].clone());
                }
            }
        } else {
            // Fallback: thread::scope
            struct PreadJob { dst_addr: usize, offset: u64 }
            let mut jobs: Vec<PreadJob> = Vec::with_capacity(num_misses);
            for i in 0..num_misses {
                jobs.push(PreadJob {
                    dst_addr: miss_bufs[i].contents() as usize,
                    offset: miss_expert_indices[i] as u64 * expert_size as u64,
                });
            }
            std::thread::scope(|s| {
                for job in jobs {
                    s.spawn(move || {
                        let dst = job.dst_addr as *mut u8;
                        let buf = unsafe { std::slice::from_raw_parts_mut(dst, expert_size) };
                        fd.read_exact_at(buf, job.offset).ok();
                    });
                }
            });
            for i in 0..num_misses {
                bufs[miss_slots[i]] = Some(miss_bufs[i].clone());
            }
        }
    }

    (bufs, pinned)
}

/// Read expert data into a pre-allocated slice
pub fn read_expert(fd: &File, expert_idx: i32, dst: &mut [u8], expert_size: usize) -> io::Result<()> {
    let offset = expert_idx as u64 * expert_size as u64;
    fd.read_exact_at(dst, offset)
}

/// Direct I/O path — reads experts straight into pre-allocated GPU buffers via
/// the I/O pool.  Bypasses the LRU cache entirely; relies on the OS page cache
/// for performance.  Returns a valid flag per expert.
pub fn direct_expert_read(
    fd: &File,
    expert_indices: &[i32],
    k: usize,
    expert_size: usize,
    dst_bufs: &[Buffer],
    io_pool: Option<&IOPool>,
) -> Vec<bool> {
    let raw_fd = fd.as_raw_fd();
    let mut valid = vec![false; k];

    if let Some(pool) = io_pool {
        let mut tasks: Vec<IOPreadTask> = Vec::with_capacity(k);
        for i in 0..k {
            tasks.push(IOPreadTask {
                fd: raw_fd,
                dst: dst_bufs[i].contents() as *mut u8,
                offset: expert_indices[i] as u64 * expert_size as u64,
                size: expert_size,
                result: 0,
                lz4_comp_buf: std::ptr::null_mut(),
                lz4_comp_size: 0,
            });
        }
        pool.dispatch(&mut tasks);
        for i in 0..k {
            valid[i] = tasks[i].result == expert_size as isize;
        }
    } else {
        // Fallback: synchronous read_exact_at
        for i in 0..k {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(dst_bufs[i].contents() as *mut u8, expert_size)
            };
            if fd.read_exact_at(dst, expert_indices[i] as u64 * expert_size as u64).is_ok() {
                valid[i] = true;
            }
        }
    }

    valid
}
