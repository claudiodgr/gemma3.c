/*
 * gemma3_threads.c - Cross-platform thread pool implementation
 *
 * Supports:
 * - POSIX pthreads (Linux, macOS)
 * - Windows native threads
 */

#include "gemma3_threads.h"
#include "gemma3_platform.h"
#include <stdlib.h>

/* ============================================================================
 * Windows Implementation
 * ========================================================================== */

#if GEMMA3_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

struct gemma3_thread_pool {
    int num_threads;
    HANDLE *threads;
    CRITICAL_SECTION mutex;
    CONDITION_VARIABLE cond_start;
    CONDITION_VARIABLE cond_done;
    volatile gemma3_task_fn current_fn;
    volatile void *current_arg;
    volatile int tasks_remaining;
    volatile int generation;
    volatile int shutdown;
};

typedef struct {
    gemma3_thread_pool *pool;
    int thread_idx;
} worker_arg;

static DWORD WINAPI worker_func(LPVOID param) {
    worker_arg *wa = (worker_arg *)param;
    gemma3_thread_pool *pool = wa->pool;
    int idx = wa->thread_idx;
    free(wa);

    int last_gen = 0;
    while (1) {
        EnterCriticalSection(&pool->mutex);
        while (pool->generation == last_gen && !pool->shutdown) {
            SleepConditionVariableCS(&pool->cond_start, &pool->mutex, INFINITE);
        }
        if (pool->shutdown) {
            LeaveCriticalSection(&pool->mutex);
            break;
        }
        last_gen = pool->generation;
        gemma3_task_fn fn = pool->current_fn;
        void *arg = (void *)pool->current_arg;
        int nt = pool->num_threads;
        LeaveCriticalSection(&pool->mutex);

        if (fn) {
            fn(arg, idx, nt);
        }

        EnterCriticalSection(&pool->mutex);
        pool->tasks_remaining--;
        if (pool->tasks_remaining == 0) {
            WakeConditionVariable(&pool->cond_done);
        }
        LeaveCriticalSection(&pool->mutex);
    }
    return 0;
}

gemma3_thread_pool *gemma3_thread_pool_create(int num_threads) {
    if (num_threads <= 0) {
        num_threads = gemma3_cpu_count();
    }

    gemma3_thread_pool *pool = (gemma3_thread_pool *)calloc(1, sizeof(gemma3_thread_pool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->shutdown = 0;
    pool->generation = 0;
    pool->tasks_remaining = 0;
    pool->current_fn = NULL;
    pool->current_arg = NULL;

    InitializeCriticalSection(&pool->mutex);
    InitializeConditionVariable(&pool->cond_start);
    InitializeConditionVariable(&pool->cond_done);

    pool->threads = (HANDLE *)calloc(num_threads, sizeof(HANDLE));
    if (!pool->threads) {
        gemma3_thread_pool_destroy(pool);
        return NULL;
    }

    for (int i = 0; i < num_threads; i++) {
        worker_arg *wa = (worker_arg *)malloc(sizeof(worker_arg));
        if (!wa) {
            pool->num_threads = i;
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
        wa->pool = pool;
        wa->thread_idx = i;

        pool->threads[i] = CreateThread(NULL, 0, worker_func, wa, 0, NULL);
        if (pool->threads[i] == NULL) {
            free(wa);
            pool->num_threads = i;
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

void gemma3_thread_pool_destroy(gemma3_thread_pool *pool) {
    if (!pool) return;

    EnterCriticalSection(&pool->mutex);
    pool->shutdown = 1;
    WakeAllConditionVariable(&pool->cond_start);
    LeaveCriticalSection(&pool->mutex);

    if (pool->threads) {
        for (int i = 0; i < pool->num_threads; i++) {
            if (pool->threads[i]) {
                WaitForSingleObject(pool->threads[i], INFINITE);
                CloseHandle(pool->threads[i]);
            }
        }
        free(pool->threads);
    }

    DeleteCriticalSection(&pool->mutex);
    free(pool);
}

int gemma3_thread_pool_size(const gemma3_thread_pool *pool) {
    return pool ? pool->num_threads : 0;
}

void gemma3_thread_pool_run(gemma3_thread_pool *pool, gemma3_task_fn fn, void *arg) {
    if (!pool || !fn) return;

    EnterCriticalSection(&pool->mutex);
    pool->current_fn = fn;
    pool->current_arg = arg;
    pool->tasks_remaining = pool->num_threads;
    pool->generation++;
    WakeAllConditionVariable(&pool->cond_start);

    /* Wait for all workers to finish */
    while (pool->tasks_remaining > 0) {
        SleepConditionVariableCS(&pool->cond_done, &pool->mutex, INFINITE);
    }
    pool->current_fn = NULL;
    LeaveCriticalSection(&pool->mutex);
}

/* ============================================================================
 * POSIX Implementation (Linux, macOS)
 * ========================================================================== */

#else /* POSIX */

#include <pthread.h>

struct gemma3_thread_pool {
    int num_threads;
    pthread_t *threads;
    pthread_mutex_t mutex;
    pthread_cond_t cond_start;
    pthread_cond_t cond_done;
    volatile gemma3_task_fn current_fn;
    volatile void *current_arg;
    volatile int tasks_remaining;
    volatile int generation;  /* Incremented each run to avoid spurious wakeups */
    volatile int shutdown;
};

typedef struct {
    gemma3_thread_pool *pool;
    int thread_idx;
} worker_arg;

static void *worker_func(void *param) {
    worker_arg *wa = (worker_arg *)param;
    gemma3_thread_pool *pool = wa->pool;
    int idx = wa->thread_idx;
    free(wa);

    int last_gen = 0;
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->generation == last_gen && !pool->shutdown) {
            pthread_cond_wait(&pool->cond_start, &pool->mutex);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        last_gen = pool->generation;
        gemma3_task_fn fn = pool->current_fn;
        void *arg = (void *)pool->current_arg;
        int nt = pool->num_threads;
        pthread_mutex_unlock(&pool->mutex);

        if (fn) {
            fn(arg, idx, nt);
        }

        pthread_mutex_lock(&pool->mutex);
        pool->tasks_remaining--;
        if (pool->tasks_remaining == 0) {
            pthread_cond_signal(&pool->cond_done);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

gemma3_thread_pool *gemma3_thread_pool_create(int num_threads) {
    if (num_threads <= 0) {
        num_threads = gemma3_cpu_count();
    }

    gemma3_thread_pool *pool = (gemma3_thread_pool *)calloc(1, sizeof(gemma3_thread_pool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->shutdown = 0;
    pool->generation = 0;
    pool->tasks_remaining = 0;
    pool->current_fn = NULL;
    pool->current_arg = NULL;

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_start, NULL);
    pthread_cond_init(&pool->cond_done, NULL);

    pool->threads = (pthread_t *)calloc(num_threads, sizeof(pthread_t));
    if (!pool->threads) {
        gemma3_thread_pool_destroy(pool);
        return NULL;
    }

    for (int i = 0; i < num_threads; i++) {
        worker_arg *wa = (worker_arg *)malloc(sizeof(worker_arg));
        if (!wa) {
            pool->num_threads = i;
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
        wa->pool = pool;
        wa->thread_idx = i;
        if (pthread_create(&pool->threads[i], NULL, worker_func, wa) != 0) {
            free(wa);
            pool->num_threads = i;
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

void gemma3_thread_pool_destroy(gemma3_thread_pool *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond_start);
    pthread_mutex_unlock(&pool->mutex);

    if (pool->threads) {
        for (int i = 0; i < pool->num_threads; i++) {
            pthread_join(pool->threads[i], NULL);
        }
        free(pool->threads);
    }

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond_start);
    pthread_cond_destroy(&pool->cond_done);
    free(pool);
}

int gemma3_thread_pool_size(const gemma3_thread_pool *pool) {
    return pool ? pool->num_threads : 0;
}

void gemma3_thread_pool_run(gemma3_thread_pool *pool, gemma3_task_fn fn, void *arg) {
    if (!pool || !fn) return;

    pthread_mutex_lock(&pool->mutex);
    pool->current_fn = fn;
    pool->current_arg = arg;
    pool->tasks_remaining = pool->num_threads;
    pool->generation++;
    pthread_cond_broadcast(&pool->cond_start);

    /* Wait for all workers to finish */
    while (pool->tasks_remaining > 0) {
        pthread_cond_wait(&pool->cond_done, &pool->mutex);
    }
    pool->current_fn = NULL;
    pthread_mutex_unlock(&pool->mutex);
}

#endif /* POSIX */
