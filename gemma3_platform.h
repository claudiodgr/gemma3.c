/*
 * gemma3_platform.h - Platform abstraction layer
 *
 * Provides cross-platform APIs for:
 * - Memory mapping (mmap/MapViewOfFile)
 * - File operations
 * - Threading primitives
 * - CPU detection
 */

#ifndef GEMMA3_PLATFORM_H
#define GEMMA3_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Platform Detection
 * ========================================================================== */

#if defined(_WIN32) || defined(_WIN64)
    #define GEMMA3_WINDOWS 1
    #define GEMMA3_POSIX 0
#elif defined(__APPLE__) && defined(__MACH__)
    #define GEMMA3_WINDOWS 0
    #define GEMMA3_POSIX 1
    #define GEMMA3_MACOS 1
#elif defined(__linux__)
    #define GEMMA3_WINDOWS 0
    #define GEMMA3_POSIX 1
    #define GEMMA3_LINUX 1
#else
    #define GEMMA3_WINDOWS 0
    #define GEMMA3_POSIX 1
#endif

/* ============================================================================
 * Memory Mapping
 * ========================================================================== */

/**
 * Opaque handle for memory-mapped files
 */
typedef struct gemma3_mmap_handle gemma3_mmap_handle;

/**
 * Map a file into memory for reading
 *
 * @param path File path to map
 * @param size Output: size of the mapped region
 * @return Pointer to mapped memory, or NULL on failure
 */
void *gemma3_mmap_file(const char *path, size_t *size);

/**
 * Unmap a previously mapped file
 *
 * @param ptr Pointer returned by gemma3_mmap_file
 * @param size Size of the mapped region
 */
void gemma3_munmap_file(void *ptr, size_t size);

/**
 * Advise the OS about memory access patterns (hint, may be no-op)
 *
 * @param ptr Mapped memory region
 * @param size Size of the region
 * @param sequential 1 for sequential access, 0 for random
 */
void gemma3_madvise(void *ptr, size_t size, int sequential);

/* ============================================================================
 * File Operations
 * ========================================================================== */

/**
 * Get the size of a file
 *
 * @param path File path
 * @return File size in bytes, or -1 on error
 */
int64_t gemma3_file_size(const char *path);

/**
 * Check if a file exists
 *
 * @param path File path
 * @return 1 if exists, 0 otherwise
 */
int gemma3_file_exists(const char *path);

/**
 * Check if a path is a directory
 *
 * @param path Path to check
 * @return 1 if directory, 0 otherwise
 */
int gemma3_is_directory(const char *path);

/**
 * Join two path components
 *
 * @param base Base path
 * @param name Name to append
 * @param out Output buffer
 * @param out_size Size of output buffer
 * @return Length of result, or -1 on error
 */
int gemma3_path_join(const char *base, const char *name, char *out, size_t out_size);

/**
 * Get the directory separator for the current platform
 */
char gemma3_path_separator(void);

/* ============================================================================
 * Directory Iteration
 * ========================================================================== */

/**
 * Opaque handle for directory iteration
 */
typedef struct gemma3_dir gemma3_dir;

/**
 * Directory entry information
 */
typedef struct {
    char name[256];
    int is_directory;
    size_t size;
} gemma3_dirent;

/**
 * Open a directory for iteration
 *
 * @param path Directory path
 * @return Directory handle, or NULL on failure
 */
gemma3_dir *gemma3_opendir(const char *path);

/**
 * Read next directory entry
 *
 * @param dir Directory handle
 * @param entry Output entry structure
 * @return 1 if entry was read, 0 if end of directory, -1 on error
 */
int gemma3_readdir(gemma3_dir *dir, gemma3_dirent *entry);

/**
 * Close directory handle
 */
void gemma3_closedir(gemma3_dir *dir);

/* ============================================================================
 * CPU Information
 * ========================================================================== */

/**
 * Get the number of available CPU cores
 *
 * @return Number of cores, or 1 if detection fails
 */
int gemma3_cpu_count(void);

/**
 * Check for AVX2 support
 *
 * @return 1 if AVX2 is available, 0 otherwise
 */
int gemma3_has_avx2(void);

/**
 * Check for AVX-512 support
 *
 * @return 1 if AVX-512 is available, 0 otherwise
 */
int gemma3_has_avx512(void);

/**
 * Check for ARM NEON support
 *
 * @return 1 if NEON is available, 0 otherwise
 */
int gemma3_has_neon(void);

/* ============================================================================
 * Time Functions
 * ========================================================================== */

/**
 * Get current time in milliseconds (monotonic clock)
 *
 * @return Time in milliseconds
 */
uint64_t gemma3_time_ms(void);

/**
 * Get current time in microseconds (monotonic clock)
 *
 * @return Time in microseconds
 */
uint64_t gemma3_time_us(void);

/* ============================================================================
 * Thread-Local Storage
 * ========================================================================== */

#if GEMMA3_WINDOWS
    #define GEMMA3_THREAD_LOCAL __declspec(thread)
#else
    #define GEMMA3_THREAD_LOCAL __thread
#endif

/* ============================================================================
 * Compiler Intrinsics
 * ========================================================================== */

/* Memory copy that works across compilers */
#if defined(_MSC_VER)
    #include <string.h>
    #define gemma3_memcpy(dst, src, n) memcpy(dst, src, n)
#else
    #define gemma3_memcpy(dst, src, n) __builtin_memcpy(dst, src, n)
#endif

/* Likely/unlikely branch hints */
#if defined(__GNUC__) || defined(__clang__)
    #define GEMMA3_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define GEMMA3_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define GEMMA3_LIKELY(x)   (x)
    #define GEMMA3_UNLIKELY(x) (x)
#endif

/* Alignment */
#if defined(_MSC_VER)
    #define GEMMA3_ALIGN(n) __declspec(align(n))
#else
    #define GEMMA3_ALIGN(n) __attribute__((aligned(n)))
#endif

/* Restrict pointer (C99) */
#if defined(_MSC_VER)
    #define GEMMA3_RESTRICT __restrict
#else
    #define GEMMA3_RESTRICT restrict
#endif

/* Inline */
#if defined(_MSC_VER)
    #define GEMMA3_INLINE __forceinline
#else
    #define GEMMA3_INLINE static inline __attribute__((always_inline))
#endif

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_PLATFORM_H */
