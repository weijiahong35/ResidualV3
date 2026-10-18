#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

/* Runtime ABI only: use the same libcudart as this PyTorch environment. */
extern int cudaSetDevice(int);
extern int cudaMalloc(void **, size_t);
extern const char *cudaGetErrorString(int);

typedef struct Block {
    size_t offset, size;
    int used;
    struct Block *prev, *next;
} Block;
static void *base = NULL;
static size_t capacity = 0, current = 0, peak = 0;
static uint64_t allocations = 0;
static Block *head = NULL;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static void fail(const char *reason, size_t requested) {
    fprintf(stderr, "ARENA_FAILURE %s requested=%zu current=%zu peak=%zu capacity=%zu\n",
            reason, requested, current, peak, capacity);
    fflush(stderr);
    /* A pilot must stop rather than return a NULL storage or corrupt a tensor. */
    _exit(75);
}

int arena_init(size_t bytes) {
    pthread_mutex_lock(&mutex);
    if (base) { pthread_mutex_unlock(&mutex); return -1; }
    int error = cudaSetDevice(0);
    if (!error) error = cudaMalloc(&base, bytes);
    if (error) {
        fprintf(stderr, "ARENA_INIT_FAILURE %s\n", cudaGetErrorString(error));
        pthread_mutex_unlock(&mutex);
        return error;
    }
    capacity = bytes;
    head = calloc(1, sizeof(Block));
    if (!head) fail("host_metadata", sizeof(Block));
    head->size = bytes;
    pthread_mutex_unlock(&mutex);
    return 0;
}

void *arena_alloc(ssize_t bytes, int device, void *stream) {
    if (device != 0 || stream != NULL) fail("requires_one_device_default_stream", bytes);
    if (bytes <= 0) return NULL;
    size_t size = ((size_t)bytes + 511) & ~(size_t)511;
    pthread_mutex_lock(&mutex);
    Block *best = NULL;
    for (Block *b = head; b; b = b->next)
        if (!b->used && b->size >= size && (!best || b->size < best->size)) best = b;
    if (!best) fail("fixed_workspace_exhausted_or_fragmented", size);
    if (best->size > size && size >= 64 * 1024 * 1024) {
        /* Keep large tensors at the high end, small temporaries at the low end. */
        Block *used = calloc(1, sizeof(Block));
        if (!used) fail("host_metadata", sizeof(Block));
        used->offset = best->offset + best->size - size; used->size = size;
        used->prev = best; used->next = best->next;
        if (used->next) used->next->prev = used;
        best->next = used; best->size -= size; best = used;
    } else if (best->size > size) {
        Block *tail = calloc(1, sizeof(Block));
        if (!tail) fail("host_metadata", sizeof(Block));
        tail->offset = best->offset + size;
        tail->size = best->size - size;
        tail->prev = best; tail->next = best->next;
        if (tail->next) tail->next->prev = tail;
        best->next = tail; best->size = size;
    }
    best->used = 1; current += size;
    if (current > peak) peak = current;
    allocations++;
    void *result = (char *)base + best->offset;
    pthread_mutex_unlock(&mutex);
    return result;
}

void arena_free(void *ptr, size_t bytes, int device, void *stream) {
    if (!ptr) return;
    if (device != 0 || stream != NULL) fail("requires_one_device_default_stream", bytes);
    pthread_mutex_lock(&mutex);
    size_t offset = (char *)ptr - (char *)base;
    Block *b = head;
    while (b && b->offset != offset) b = b->next;
    if (!b || !b->used) fail("invalid_or_double_free", bytes);
    current -= b->size; b->used = 0;
    if (b->next && !b->next->used) {
        Block *next = b->next;
        b->size += next->size; b->next = next->next;
        if (b->next) b->next->prev = b;
        free(next);
    }
    if (b->prev && !b->prev->used) {
        Block *prev = b->prev;
        prev->size += b->size; prev->next = b->next;
        if (prev->next) prev->next->prev = prev;
        free(b);
    }
    pthread_mutex_unlock(&mutex);
}

int arena_owns(void *ptr) {
    uintptr_t p = (uintptr_t)ptr, start = (uintptr_t)base;
    return base && p >= start && p < start + capacity;
}
size_t arena_capacity(void) { return capacity; }
size_t arena_current(void) { return current; }
size_t arena_peak(void) { return peak; }
uint64_t arena_allocations(void) { return allocations; }

