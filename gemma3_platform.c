/*
 * gemma3_platform.c - Platform abstraction layer implementation
 *
 * Implements cross-platform APIs for memory mapping, file operations,
 * threading primitives, and CPU detection.
 */

#include "gemma3_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Windows Implementation
 * ========================================================================== */

#if GEMMA3_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

/* Structure to track mapped file handles */
typedef struct gemma3_mmap_info {
    HANDLE file_handle;
    HANDLE mapping_handle;
    void *ptr;
    size_t size;
} gemma3_mmap_info;

/* Store mapping info (simple linked list for now) */
typedef struct mmap_node {
    gemma3_mmap_info info;
    struct mmap_node *next;
} mmap_node;

static mmap_node *g_mmap_list = NULL;
static CRITICAL_SECTION g_mmap_lock;
static int g_mmap_lock_init = 0;

static void mmap_lock_init(void) {
    if (!g_mmap_lock_init) {
        InitializeCriticalSection(&g_mmap_lock);
        g_mmap_lock_init = 1;
    }
}

static void mmap_register(gemma3_mmap_info *info) {
    mmap_lock_init();
    EnterCriticalSection(&g_mmap_lock);

    mmap_node *node = (mmap_node *)malloc(sizeof(mmap_node));
    if (node) {
        node->info = *info;
        node->next = g_mmap_list;
        g_mmap_list = node;
    }

    LeaveCriticalSection(&g_mmap_lock);
}

static int mmap_unregister(void *ptr, gemma3_mmap_info *out_info) {
    mmap_lock_init();
    EnterCriticalSection(&g_mmap_lock);

    mmap_node **pp = &g_mmap_list;
    while (*pp) {
        if ((*pp)->info.ptr == ptr) {
            mmap_node *node = *pp;
            *out_info = node->info;
            *pp = node->next;
            free(node);
            LeaveCriticalSection(&g_mmap_lock);
            return 1;
        }
        pp = &(*pp)->next;
    }

    LeaveCriticalSection(&g_mmap_lock);
    return 0;
}

void *gemma3_mmap_file(const char *path, size_t *size) {
    HANDLE file_handle = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (file_handle == INVALID_HANDLE_VALUE) {
        return NULL;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle, &file_size)) {
        CloseHandle(file_handle);
        return NULL;
    }

    HANDLE mapping_handle = CreateFileMappingA(
        file_handle,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    if (mapping_handle == NULL) {
        CloseHandle(file_handle);
        return NULL;
    }

    void *ptr = MapViewOfFile(
        mapping_handle,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (ptr == NULL) {
        CloseHandle(mapping_handle);
        CloseHandle(file_handle);
        return NULL;
    }

    *size = (size_t)file_size.QuadPart;

    /* Register mapping for cleanup */
    gemma3_mmap_info info = {
        .file_handle = file_handle,
        .mapping_handle = mapping_handle,
        .ptr = ptr,
        .size = *size
    };
    mmap_register(&info);

    return ptr;
}

void gemma3_munmap_file(void *ptr, size_t size) {
    (void)size; /* Size tracked internally */

    gemma3_mmap_info info;
    if (mmap_unregister(ptr, &info)) {
        UnmapViewOfFile(info.ptr);
        CloseHandle(info.mapping_handle);
        CloseHandle(info.file_handle);
    }
}

void gemma3_madvise(void *ptr, size_t size, int sequential) {
    (void)ptr;
    (void)size;
    (void)sequential;
    /* Windows prefetch hint - optional */
    /* Could use PrefetchVirtualMemory on Windows 8+ */
}

int64_t gemma3_file_size(const char *path) {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &data)) {
        return -1;
    }
    LARGE_INTEGER size;
    size.HighPart = data.nFileSizeHigh;
    size.LowPart = data.nFileSizeLow;
    return size.QuadPart;
}

int gemma3_file_exists(const char *path) {
    DWORD attrs = GetFileAttributesA(path);
    return attrs != INVALID_FILE_ATTRIBUTES;
}

int gemma3_is_directory(const char *path) {
    DWORD attrs = GetFileAttributesA(path);
    if (attrs == INVALID_FILE_ATTRIBUTES) return 0;
    return (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

int gemma3_path_join(const char *base, const char *name, char *out, size_t out_size) {
    int len = snprintf(out, out_size, "%s\\%s", base, name);
    return (len < 0 || (size_t)len >= out_size) ? -1 : len;
}

char gemma3_path_separator(void) {
    return '\\';
}

/* Directory iteration */
struct gemma3_dir {
    HANDLE handle;
    WIN32_FIND_DATAA find_data;
    int first;
};

gemma3_dir *gemma3_opendir(const char *path) {
    char search_path[MAX_PATH];
    snprintf(search_path, sizeof(search_path), "%s\\*", path);

    gemma3_dir *dir = (gemma3_dir *)malloc(sizeof(gemma3_dir));
    if (!dir) return NULL;

    dir->handle = FindFirstFileA(search_path, &dir->find_data);
    if (dir->handle == INVALID_HANDLE_VALUE) {
        free(dir);
        return NULL;
    }

    dir->first = 1;
    return dir;
}

int gemma3_readdir(gemma3_dir *dir, gemma3_dirent *entry) {
    if (!dir || dir->handle == INVALID_HANDLE_VALUE) return -1;

    while (1) {
        if (!dir->first) {
            if (!FindNextFileA(dir->handle, &dir->find_data)) {
                return 0; /* End of directory */
            }
        }
        dir->first = 0;

        /* Skip . and .. */
        if (strcmp(dir->find_data.cFileName, ".") == 0 ||
            strcmp(dir->find_data.cFileName, "..") == 0) {
            continue;
        }

        strncpy(entry->name, dir->find_data.cFileName, sizeof(entry->name) - 1);
        entry->name[sizeof(entry->name) - 1] = '\0';
        entry->is_directory = (dir->find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

        LARGE_INTEGER size;
        size.HighPart = dir->find_data.nFileSizeHigh;
        size.LowPart = dir->find_data.nFileSizeLow;
        entry->size = (size_t)size.QuadPart;

        return 1;
    }
}

void gemma3_closedir(gemma3_dir *dir) {
    if (dir) {
        if (dir->handle != INVALID_HANDLE_VALUE) {
            FindClose(dir->handle);
        }
        free(dir);
    }
}

int gemma3_cpu_count(void) {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (int)sysinfo.dwNumberOfProcessors;
}

int gemma3_has_avx2(void) {
    /* Use CPUID to check for AVX2 */
    int cpuinfo[4] = {0};
#if defined(_MSC_VER)
    __cpuidex(cpuinfo, 7, 0);
    return (cpuinfo[1] & (1 << 5)) != 0; /* AVX2 is bit 5 of EBX */
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(
        "cpuid"
        : "=a"(cpuinfo[0]), "=b"(cpuinfo[1]), "=c"(cpuinfo[2]), "=d"(cpuinfo[3])
        : "a"(7), "c"(0)
    );
    return (cpuinfo[1] & (1 << 5)) != 0;
#else
    return 0;
#endif
}

int gemma3_has_avx512(void) {
    int cpuinfo[4] = {0};
#if defined(_MSC_VER)
    __cpuidex(cpuinfo, 7, 0);
    return (cpuinfo[1] & (1 << 16)) != 0; /* AVX-512F is bit 16 of EBX */
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(
        "cpuid"
        : "=a"(cpuinfo[0]), "=b"(cpuinfo[1]), "=c"(cpuinfo[2]), "=d"(cpuinfo[3])
        : "a"(7), "c"(0)
    );
    return (cpuinfo[1] & (1 << 16)) != 0;
#else
    return 0;
#endif
}

int gemma3_has_neon(void) {
    return 0; /* NEON is ARM-only */
}

uint64_t gemma3_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (uint64_t)(count.QuadPart * 1000 / freq.QuadPart);
}

uint64_t gemma3_time_us(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (uint64_t)(count.QuadPart * 1000000 / freq.QuadPart);
}

/* ============================================================================
 * POSIX Implementation (Linux, macOS)
 * ========================================================================== */

#else /* POSIX */

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <time.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach_time.h>
#endif

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

void *gemma3_mmap_file(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    *size = st.st_size;

    void *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* Can close fd after mmap */

    if (ptr == MAP_FAILED) {
        return NULL;
    }

    return ptr;
}

void gemma3_munmap_file(void *ptr, size_t size) {
    if (ptr && ptr != MAP_FAILED) {
        munmap(ptr, size);
    }
}

void gemma3_madvise(void *ptr, size_t size, int sequential) {
#ifdef MADV_SEQUENTIAL
    if (ptr && size > 0) {
        int advice = sequential ? MADV_SEQUENTIAL : MADV_RANDOM;
        madvise(ptr, size, advice);
    }
#else
    (void)ptr;
    (void)size;
    (void)sequential;
#endif
}

int64_t gemma3_file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) < 0) {
        return -1;
    }
    return st.st_size;
}

int gemma3_file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

int gemma3_is_directory(const char *path) {
    struct stat st;
    if (stat(path, &st) < 0) return 0;
    return S_ISDIR(st.st_mode);
}

int gemma3_path_join(const char *base, const char *name, char *out, size_t out_size) {
    int len = snprintf(out, out_size, "%s/%s", base, name);
    return (len < 0 || (size_t)len >= out_size) ? -1 : len;
}

char gemma3_path_separator(void) {
    return '/';
}

/* Directory iteration */
struct gemma3_dir {
    DIR *handle;
    char path[1024];
};

gemma3_dir *gemma3_opendir(const char *path) {
    gemma3_dir *dir = (gemma3_dir *)malloc(sizeof(gemma3_dir));
    if (!dir) return NULL;

    dir->handle = opendir(path);
    if (!dir->handle) {
        free(dir);
        return NULL;
    }

    strncpy(dir->path, path, sizeof(dir->path) - 1);
    dir->path[sizeof(dir->path) - 1] = '\0';

    return dir;
}

int gemma3_readdir(gemma3_dir *dir, gemma3_dirent *entry) {
    if (!dir || !dir->handle) return -1;

    struct dirent *d;
    while ((d = readdir(dir->handle)) != NULL) {
        /* Skip . and .. */
        if (strcmp(d->d_name, ".") == 0 || strcmp(d->d_name, "..") == 0) {
            continue;
        }

        strncpy(entry->name, d->d_name, sizeof(entry->name) - 1);
        entry->name[sizeof(entry->name) - 1] = '\0';

        /* Get file info */
        char full_path[1280];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir->path, d->d_name);

        struct stat st;
        if (stat(full_path, &st) == 0) {
            entry->is_directory = S_ISDIR(st.st_mode);
            entry->size = st.st_size;
        } else {
            entry->is_directory = 0;
            entry->size = 0;
        }

        return 1;
    }

    return 0; /* End of directory */
}

void gemma3_closedir(gemma3_dir *dir) {
    if (dir) {
        if (dir->handle) {
            closedir(dir->handle);
        }
        free(dir);
    }
}

int gemma3_cpu_count(void) {
#ifdef __APPLE__
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.ncpu", &count, &size, NULL, 0) == 0) {
        return count > 0 ? count : 1;
    }
    return 1;
#elif defined(__linux__)
    int count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? count : 1;
#else
    return 1;
#endif
}

int gemma3_has_avx2(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
#endif
    return 0;
}

int gemma3_has_avx512(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;
    }
#endif
    return 0;
}

int gemma3_has_neon(void) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return 1;
#else
    return 0;
#endif
}

uint64_t gemma3_time_ms(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t time = mach_absolute_time();
    return time * timebase.numer / timebase.denom / 1000000;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#endif
}

uint64_t gemma3_time_us(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t time = mach_absolute_time();
    return time * timebase.numer / timebase.denom / 1000;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

#endif /* POSIX */
