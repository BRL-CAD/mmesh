/* *****************************************************************************
 *
 * Copyright (c) 2009-2023 Alexis Naveros.
 * Portions developed under contract to the SURVICE Engineering Company.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * *****************************************************************************
 */
/**
 * @file
 *
 * Thread interface, POSIX threads and Windows threads implementations.
 */


#ifndef MMTHREAD_H
#define MMTHREAD_H


////


#if !MT_PTHREADS && !MT_WINTHREADS && !MT_NOTHREADS
 #if defined(__linux__) || defined(__gnu_linux__) || defined(__linux) || defined(__linux) || defined(__APPLE__) || defined(__unix__) || defined(__unix) || defined(unix)
  #undef MT_PTHREADS
  #define MT_PTHREADS (1)
 #elif defined(_WIN64) || defined(__WIN64__) || defined(WIN64) || defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
  #undef MT_WINTHREADS
  #define MT_WINTHREADS (1)
 #else
  #undef MT_NOTHREADS
  #define MT_NOTHREADS (1)
  #warning We have no threading implementation for the platform.
 #endif
#endif

#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
 #define MT_GNUC (1)
#elif defined(_MSC_VER)
 #define MT_MSVC (1)
#else
 #if MT_WINTHREADS
  #error Can not target Windows threads with an unrecognized compiler! Need atomics and stuff.
 #endif
#endif


/////


#define MT_THREAD_FLAGS_JOINABLE (0x1)

#ifdef MT_DEBUG
 #ifndef MT_DEBUG_WARNING()
  #define MT_DEBUG_WARNING() ({printf( "\nMMTHREAD WARNING : %s %s %d\n", __FILE__, __FUNCTION__, __LINE__ ); fflush( stdout );})
 #endif
#endif


////


////


#if MT_PTHREADS


/* POSIX threads */

 #ifndef _GNU_SOURCE
  #define _GNU_SOURCE
 #endif
 #include <pthread.h>
 #include <unistd.h>
 #include <sched.h>
 #include <limits.h>
 #include <sys/time.h>


static inline void mtYield()
{
  sched_yield();
  return;
}



typedef struct
{
  pthread_t pthread;
} mtThread;

static inline void mtThreadCreate( mtThread *thread, void *(*threadmain)( void *value ), void *value, int flags )
{
  pthread_attr_t attr;

  pthread_attr_init( &attr );
  if( flags & MT_THREAD_FLAGS_JOINABLE )
    pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
  else
    pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_DETACHED );
  pthread_create( &thread->pthread, &attr, threadmain, value );
  pthread_attr_destroy( &attr );

  return;
}

static inline void mtThreadExit()
{
  pthread_exit( 0 );
  return;
}

static inline void mtThreadJoin( mtThread *thread )
{
  void *ret;
  pthread_join( thread->pthread, &ret );
  return;
}

static inline mtThread mtThreadSelf()
{
  mtThread thread;
  thread.pthread = pthread_self();
  return thread;
}

static inline int mtThreadCmpEqual( mtThread *thread0, mtThread *thread1 )
{
  return ( pthread_equal( thread0->pthread, thread1->pthread ) != 0 ? 1 : 0 );
}

static inline size_t mtGetMinimumStackSize()
{
 #ifdef PTHREAD_STACK_MIN
  return PTHREAD_STACK_MIN;
 #elif defined(MM_H)
  return ( mmcontext.pagesize ? 4*mmcontext.pagesize : 16384 );
 #else
  return 16384;
 #endif
}


////


typedef struct
{
 #ifdef MT_DEBUG
  unsigned char *function;
  unsigned char *file;
  int line;
 #endif
  pthread_mutex_t pmutex;
} mtMutex;

static inline void mtMutexInit( mtMutex *mutex )
{
  pthread_mutex_init( &mutex->pmutex, 0 );
  return;
}

static inline void mtMutexDestroy( mtMutex *mutex )
{
  pthread_mutex_destroy( &mutex->pmutex );
  return;
}

 #ifdef MT_DEBUG

  #define mtMutexLock(a) mtMutexLockDebug(a,__FUNCTION__,__FILE__,__LINE__)
  #define mtMutexUnlock(a) mtMutexUnlockDebug(a,__FUNCTION__,__FILE__,__LINE__)
  #define mtMutexTryLock(a) mtMutexTryLockDebug(a,__FUNCTION__,__FILE__,__LINE__)

static inline void mtMutexLockDebug( mtMutex *mutex, const unsigned char *function, const unsigned char *file, const int line )
{
  printf( "Mutex lock :    Thread %d on %p from %s() in %s:%d\n", (int)pthread_self(), mutex, function, file, line );
  fflush( stdout );
  if( pthread_mutex_trylock( &mutex->pmutex ) )
  {
    printf( "    Mutex %p locked by %s() in %s:%d\n", mutex, mutex->function, mutex->file, mutex->line );
    fflush( stdout );
    if( pthread_mutex_lock( &mutex->pmutex ) )
      MT_DEBUG_WARNING();
  }
  mutex->function = (unsigned char *)function;
  mutex->file = (unsigned char *)file;
  mutex->line = line;
  return;
}

static inline void mtMutexUnlockDebug( mtMutex *mutex, const unsigned char *function, const unsigned char *file, const int line )
{
  mutex->function = (unsigned char *)function;
  mutex->file = (unsigned char *)file;
  mutex->line = line;
  printf( "Mutex Unlock :  Thread %d on %p from %s() in %s:%d\n", (int)pthread_self(), mutex, function, file, line );
  fflush( stdout );
  if( pthread_mutex_unlock( &mutex->pmutex ) )
    MT_DEBUG_WARNING();
  return;
}

static inline int mtMutexTryLockDebug( mtMutex *mutex, const unsigned char *function, const unsigned char *file, const int line )
{
  printf( "Mutex Trylock :  Thread %d on %p from %s() in %s:%d\n", (int)pthread_self(), mutex, function, file, line );
  fflush( stdout );
  if( !( pthread_mutex_trylock( &mutex->pmutex ) ) )
  {
    mutex->function = (unsigned char *)function;
    mutex->file = (unsigned char *)file;
    mutex->line = line;
    return 1;
  }
  else
    return 0;
}

 #else

static inline void mtMutexLock( mtMutex *mutex )
{
  pthread_mutex_lock( &mutex->pmutex );
  return;
}

static inline void mtMutexUnlock( mtMutex *mutex )
{
  pthread_mutex_unlock( &mutex->pmutex );
  return;
}

static inline int mtMutexTryLock( mtMutex *mutex )
{
  return !( pthread_mutex_trylock( &mutex->pmutex ) );
}

 #endif


////


typedef struct
{
  pthread_cond_t pcond;
} mtSignal;

static inline void mtSignalInit( mtSignal *signal )
{
  pthread_cond_init( &signal->pcond, 0 );
  return;
}

static inline void mtSignalDestroy( mtSignal *signal )
{
  pthread_cond_destroy( &signal->pcond );
  return;
}

static inline void mtSignalWake( mtSignal *signal )
{
 #ifdef MT_DEBUG
  if( pthread_cond_signal( &signal->pcond ) )
    MT_DEBUG_WARNING();
 #else
  pthread_cond_signal( &signal->pcond );
 #endif
  return;
}

static inline void mtSignalBroadcast( mtSignal *signal )
{
 #ifdef MT_DEBUG
  if( pthread_cond_broadcast( &signal->pcond ) )
    MT_DEBUG_WARNING();
 #else
  pthread_cond_broadcast( &signal->pcond );
 #endif
  return;
}

static inline void mtSignalWait( mtSignal *signal, mtMutex *mutex )
{
 #ifdef MT_DEBUG
  if( pthread_cond_wait( &signal->pcond, &mutex->pmutex ) )
    MT_DEBUG_WARNING();
 #else
  pthread_cond_wait( &signal->pcond, &mutex->pmutex );
 #endif
  return;
}

static inline int mtSignalWaitTimeout( mtSignal *signal, mtMutex *mutex, long milliseconds )
{
  unsigned long long microsecs;
  struct timespec ts;
  struct timeval tp;
  gettimeofday( &tp, NULL );
  ts.tv_sec = tp.tv_sec + ( milliseconds / 1000 );
  microsecs = tp.tv_usec + ( ( milliseconds % 1000 ) * 1000 );
  if( microsecs >= 1000000 )
  {
    ts.tv_sec++;
    microsecs -= 1000000;
  }
  ts.tv_nsec = microsecs * 1000;
  return ( pthread_cond_timedwait( &signal->pcond, &mutex->pmutex, &ts ) == 0 );
}

static inline unsigned long long mtSignalTime( long milliseconds )
{
  unsigned long long millitime;
  struct timeval lntime;
  gettimeofday( &lntime, 0 );
  millitime = ( (unsigned long long)lntime.tv_sec * 1000 ) + ( (unsigned long long)lntime.tv_usec / 1000 ) + (unsigned long long)milliseconds;
  return millitime;
}

static inline int mtSignalWaitTime( mtSignal *signal, mtMutex *mutex, unsigned long long millitime )
{
  struct timespec ts;
  ts.tv_sec = millitime / 1000;
  ts.tv_nsec = ( millitime % 1000 ) * 1000000;
  return ( pthread_cond_timedwait( &signal->pcond, &mutex->pmutex, &ts ) == 0 );
}

 #ifdef MT_DEBUG
  #define MT_MUTEX_INITIALIZER { 0, 0, 0, PTHREAD_MUTEX_INITIALIZER }
 #else
  #define MT_MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER }
 #endif


#elif MT_WINTHREADS


/* Windows threads */

 #define WIN32_LEAN_AND_MEAN
 #include <windows.h>
 #if MT_MSVC
  #include <intrin.h>
 #endif

static inline void mtYield()
{
  SwitchToThread();
  return;
}


typedef struct
{
  HANDLE winthread;
  DWORD threadidentifier;
} mtThread;

typedef struct
{
  void *(*threadmain)( void *value );
  void *value;
} mtWinThreadLaunch;

static DWORD WINAPI mtWinThreadMain( void *launchdata )
{
  mtWinThreadLaunch launch;
  launch = *(mtWinThreadLaunch *)launchdata;
  free( (void *)launchdata );
  launch.threadmain( launch.value );
  return 0;
}

static inline void mtThreadCreate( mtThread *thread, void *(*threadmain)( void *value ), void *value, int flags )
{
  mtWinThreadLaunch *launch;
  launch = (mtWinThreadLaunch *)malloc( sizeof(mtWinThreadLaunch) );
  launch->threadmain = threadmain;
  launch->value = value;
  thread->winthread = CreateThread( (LPSECURITY_ATTRIBUTES)0, 0, mtWinThreadMain, (void *)launch, 0, &thread->threadidentifier );
  if( !( flags & MT_THREAD_FLAGS_JOINABLE ) )
  {
    CloseHandle( thread->winthread );
    thread->winthread = 0;
  }
  return;
}

static inline void mtThreadExit()
{
  ExitThread( 0 );
  return;
}

static inline void mtThreadJoin( mtThread *thread )
{
  if( thread->winthread )
  {
    WaitForSingleObject( thread->winthread, INFINITE );
    CloseHandle( thread->winthread );
    thread->winthread = 0;
  }
  return;
}

static inline mtThread mtThreadSelf()
{
  mtThread thread;
  thread.winthread = 0;
  thread.threadidentifier = GetCurrentThreadId();
  return thread;
}

static inline int mtThreadCmpEqual( mtThread *thread0, mtThread *thread1 )
{
  return ( thread0->threadidentifier == thread1->threadidentifier );
}

static inline size_t mtGetMinimumStackSize()
{
  return 16384;
}


////


typedef struct
{
  CRITICAL_SECTION critsection;
} mtMutex;

static inline void mtMutexInit( mtMutex *mutex )
{
  InitializeCriticalSection( &mutex->critsection );
  return;
}

static inline void mtMutexDestroy( mtMutex *mutex )
{
  DeleteCriticalSection( &mutex->critsection );
  return;
}


static inline void mtMutexLock( mtMutex *mutex )
{
  EnterCriticalSection( &mutex->critsection );
  return;
}

static inline void mtMutexUnlock( mtMutex *mutex )
{
  LeaveCriticalSection( &mutex->critsection );
  return;
}

static inline int mtMutexTryLock( mtMutex *mutex )
{
  return TryEnterCriticalSection( &mutex->critsection );
}


////


typedef struct
{
  volatile long waitcount;
  /* Signal:0, broadcast:1 */
  HANDLE events[2];
} mtSignal;

static inline void mtSignalInit( mtSignal *signal )
{
  signal->waitcount = 0;
  signal->events[0] = CreateEvent( 0, 0, 0, 0 );
  signal->events[1] = CreateEvent( 0, 1, 0, 0 );
  return;
}

static inline void mtSignalDestroy( mtSignal *signal )
{
  CloseHandle( signal->events[0] );
  CloseHandle( signal->events[1] );
  return;
}

static inline void mtSignalWake( mtSignal *signal )
{
  if( signal->waitcount )
    SetEvent( signal->events[0] );
  return;
}

static inline void mtSignalBroadcast( mtSignal *signal )
{
  if( signal->waitcount )
    SetEvent( signal->events[1] );
  return;
}

static inline void mtSignalWait( mtSignal *signal, mtMutex *mutex )
{
  int waitresult;
  long remwaitcount;

#if MT_GNUC
  __sync_add_and_fetch( &signal->waitcount, (long)1 );
#elif MT_MSVC
  _InterlockedIncrement( &signal->waitcount );
#endif
  LeaveCriticalSection( &mutex->critsection );
  waitresult = WaitForMultipleObjects( 2, signal->events, FALSE, INFINITE );
  EnterCriticalSection( &mutex->critsection );
#if MT_GNUC
  remwaitcount = __sync_sub_and_fetch( &signal->waitcount, (long)1 );
#elif MT_MSVC
  remwaitcount = _InterlockedDecrement( &signal->waitcount );
#endif
  if( ( waitresult == ( WAIT_OBJECT_0 + 1 ) ) && !remwaitcount )
    ResetEvent( signal->events[1] );
  return;
}

static inline int mtSignalWaitTimeout( mtSignal *signal, mtMutex *mutex, long milliseconds )
{
  int waitresult;
  long remwaitcount;

#if MT_GNUC
  __sync_add_and_fetch( &signal->waitcount, (long)1 );
#elif MT_MSVC
  _InterlockedIncrement( &signal->waitcount );
#endif
  LeaveCriticalSection( &mutex->critsection );
  waitresult = WaitForMultipleObjects( 2, signal->events, FALSE, milliseconds );
  EnterCriticalSection( &mutex->critsection );
#if MT_GNUC
  remwaitcount = __sync_sub_and_fetch( &signal->waitcount, (long)1 );
#elif MT_MSVC
  remwaitcount = _InterlockedDecrement( &signal->waitcount );
#endif
  if( ( waitresult == ( WAIT_OBJECT_0 + 1 ) ) && !remwaitcount )
    ResetEvent( signal->events[1] );
  return ( waitresult != WAIT_TIMEOUT );
}

static inline int mtGetTimeOfDay( struct timeval *tv )
{
 #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
  FILETIME ft;
  unsigned long long curtime;
  if( tv )
  {
    GetSystemTimeAsFileTime( &ft );
    curtime = ( ( ( (unsigned long long)ft.dwHighDateTime << 32 ) | (unsigned long long)ft.dwLowDateTime ) / 10 ) - DELTA_EPOCH_IN_MICROSECS;
    tv->tv_sec = (long)( curtime / 1000000UL );
    tv->tv_usec = (long)( curtime % 1000000UL );
  }
  return 0;
}

static inline unsigned long long mtSignalTime( long milliseconds )
{
  unsigned long long millitime;
  struct timeval lntime;
  mtGetTimeOfDay( &lntime );
  millitime = ( (unsigned long long)lntime.tv_sec * 1000 ) + ( (unsigned long long)lntime.tv_usec / 1000 ) + (unsigned long long)milliseconds;
  return millitime;
}

static inline int mtSignalWaitTime( mtSignal *signal, mtMutex *mutex, unsigned long long millitime )
{
  unsigned long long currenttime;
  long timeout;
  struct timeval lntime;
  mtGetTimeOfDay( &lntime );
  currenttime = ( (unsigned long long)lntime.tv_sec * 1000 ) + ( (unsigned long long)lntime.tv_usec / 1000 );
  timeout = (long)( millitime - currenttime );
  if( timeout < 0 )
    timeout = 0;
  return mtSignalWaitTimeout( signal, mutex, timeout );
}


#else


/* No threads */

 #ifndef MT_DISABLED
  #define MT_DISABLED (1)
 #endif

typedef struct { } mtMutex;
typedef struct { } mtSpin;
typedef struct { } mtSignal;

 #define mtMutexInit(a)
 #define mtMutexDestroy(a)
 #define mtMutexLock(a)
 #define mtMutexUnlock(a)
 #define mtMutexTryLock(a)

 #define mtSpinInit(a)
 #define mtSpinDestroy(a)
 #define mtSpinLock(a)
 #define mtSpinUnlock(a)
 #define mtSpinTryLock(a)

 #define mtSignalInit(a)
 #define mtSignalDestroy(a)
 #define mtSignalWake(a)
 #define mtSignalBroadcast(a)
 #define mtSignalMutexWait(a,b)
 #define mtSignalTime(a) (0)
 #define mtSignalWaitTime(a) (1)


#endif


////


/* Implemented in mmthread.c */

typedef struct
{
  mtMutex mutex;
  mtSignal signal;
  int resetcount;
  volatile int index;
  volatile int count[2];
  /* Global lock stuff */
  volatile int lockflag;
  volatile int lockcount;
  mtSignal locksignal;
  mtSignal lockwakesignal;
} mtSleepBarrier;

#define MT_SLEEP_BARRIER_LOCK_READY(barrier) (((barrier)->count[(barrier)->index])-(((barrier)->lockcount))==1)

/* Initialize sleep barrier, count specifies count of threads before barrier is released for all threads */
void mtSleepBarrierInit( mtSleepBarrier *barrier, int count );

/* Destroy barrier */
void mtSleepBarrierDestroy( mtSleepBarrier *barrier );

/* Wait on barrier until 'count' threads are sleeping on it */
int mtSleepBarrierSync( mtSleepBarrier *barrier );

/* Wait on barrier until 'count' threads are sleeping on it, return 0 on failure if timeout reached */
int mtSleepBarrierSyncTimeout( mtSleepBarrier *barrier, long milliseconds );

/* Check if any thread requested a global barrier lock, sleep until fully locked if so */
void mtSleepBarrierCheckGlobal( mtSleepBarrier *barrier );

/* Acquire global barrier lock, all other threads will be in barrier */
void mtSleepBarrierLockGlobal( mtSleepBarrier *barrier );

/* Release an acquired global barrier lock */
void mtSleepBarrierUnlockGlobal( mtSleepBarrier *barrier );



////


#if MT_GNUC


/* Spin locks, GNUC/clang/ICC implementation */

typedef struct mtSpin
{
  volatile unsigned int atomicspin;
} mtSpin;

static inline void mtSpinInit( mtSpin *spin )
{
  __asm__ __volatile__( "":::"memory" );
  spin->atomicspin = 0;
  /* Insert store-load memory barrier */
 #if defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || defined(__amd64)
  __asm__ __volatile__( "lock ; addl $0,(%%rsp)"::: "memory", "cc" );
 #elif defined(__i386__) || defined(__i386)  || defined(i386)
  __asm__ __volatile__( "lock ; addl $0,(%%esp)"::: "memory", "cc" );
 #elif defined(__ARM_ARCH_8__)
  __asm__ __volatile__( "dmb st":::"memory" );
 #else
  __sync_synchronize();
 #endif
  return;
}

static inline void mtSpinDestroy( mtSpin *spin )
{
  return;
}

static inline void mtSpinLock( mtSpin *spin )
{
  for( ; __sync_val_compare_and_swap( &spin->atomicspin, 0x0, 0x1 ) != 0x0 ; )
  {
    for( ; spin->atomicspin != 0x0 ; )
    {
 #if defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || defined(__amd64) || defined(__i386__) || defined(__i386)  || defined(i386)
      __asm__ __volatile__( "rep ; nop" :::"memory" );
 #else
      __asm__ __volatile__( "":::"memory" );
 #endif
    }
  }
  /* Insert store-load memory barrier */
  /* NOT NEEDED ~ Implicit in __sync_val_compare_and_swap() */
  return;
}

static inline void mtSpinUnlock( mtSpin *spin )
{
  /* Insert load-store memory barrier */
 #if defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || defined(__amd64)
  /* Not needed on arch */
 #elif defined(__i386__) || defined(__i386)  || defined(i386)
  /* Not needed on arch */
 #elif defined(__ARM_ARCH_8__)
  __asm__ __volatile__( "dmb ld":::"memory" );
 #else
  __sync_synchronize();
 #endif
  __asm__ __volatile__( "":::"memory" );
  spin->atomicspin = 0;
  return;
}

static inline int mtSpinTryLock( mtSpin *spin )
{
  int retval;
  retval = ( __sync_val_compare_and_swap( &spin->atomicspin, 0x0, 0x1 ) == 0x0 );
  /* Insert store-load memory barrier */
  /* NOT NEEDED ~ Implicit in __sync_val_compare_and_swap() */
  return retval;
}


#elif MT_MSVC


/* Spin locks, MSVC implementation */

typedef struct mtSpin
{
  volatile long atomicspin;
} mtSpin;

static inline void mtSpinInit( mtSpin *spin )
{
  _ReadWriteBarrier();
  spin->atomicspin = 0;
  /* Insert store-load memory barrier */
#if ( defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) )
  _mm_mfence();
#elif _WIN32_WINNT >= 0x600
  MemoryBarrier();
#endif
  return;
}

static inline void mtSpinDestroy( mtSpin *spin )
{
  return;
}

static inline void mtSpinLock( mtSpin *spin )
{
  for( ; _InterlockedCompareExchange( &spin->atomicspin, 0x1, 0x0 ) != 0x0 ; )
  {
    for( ; spin->atomicspin != 0x0 ; )
    {
      YieldProcessor();
      _ReadWriteBarrier();
    }
  }
  /* Insert store-load memory barrier */
  /* NOT NEEDED ~ Implicit in _InterlockedCompareExchange() */
  return;
}

static inline void mtSpinUnlock( mtSpin *spin )
{
  /* Insert load-store memory barrier */
#if defined(i386) || defined(__i386) || defined(__i386__) || defined(__i386) || defined(__IA32__) || defined(_M_IX86) || defined(__X86__) || defined(_X86_) || defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
  /* x86/AMD64 don't normally reorder loads ~ different rules apply to WC/WT/UC memory */
#elif _WIN32_WINNT >= 0x600
  MemoryBarrier();
#endif
  _ReadWriteBarrier();
  spin->atomicspin = 0;
  return;
}

static inline int mtSpinTryLock( mtSpin *spin )
{
  int retval;
  retval = ( _InterlockedCompareExchange( &spin->atomicspin, 0x1, 0x0 ) == 0x0 );
  /* Insert store-load memory barrier */
  /* NOT NEEDED ~ Implicit in _InterlockedCompareExchange() */
  return retval;
}


#else


typedef struct mtMutex mtSpin;

 #define mtSpinInit(a) mtMutexInit(a)
 #define mtSpinDestroy(a) mtMutexDestroy(a)
 #define mtSpinLock(a) mtMutexLock(a)
 #define mtSpinUnlock(a) mtMutexUnlock(a)
 #define mtSpinTryLock(a) mtMutexTryLock(a)


#endif


////


#if MT_GNUC


typedef struct
{
  volatile int counter __attribute__((aligned(64)));
  volatile unsigned int state __attribute__((aligned(64)));
  volatile unsigned int stateref __attribute__((aligned(64)));
  int resetvalue __attribute__((aligned(64)));
} mtBarrier;

static inline void mtBarrierBuild( mtBarrier *barrier, int waitcount )
{
  barrier->counter = waitcount;
  barrier->state = 0;
  barrier->stateref = 0;
  barrier->resetvalue = waitcount;
  __asm__ __volatile__( "":::"memory" );
  return;
}

static inline unsigned int mtBarrierWait( mtBarrier *barrier, unsigned int spinwaitcounter )
{
  unsigned int stateref, nextstateref;

  stateref = barrier->stateref;
  if( __sync_sub_and_fetch( &barrier->counter, 1 ) == 0 )
  {
    nextstateref = stateref + 1;
    /* Barrier is cleared, that was the last thread waiting on it */
    barrier->stateref = nextstateref;
    __sync_add_and_fetch( &barrier->counter, barrier->resetvalue );
    /* Ensure stateref and counter are updated and visible before releasing all spinning threads */
    __sync_synchronize();
    barrier->state = nextstateref;
  }
  else
  {
    /* Spin-wait */
    for( ; ( barrier->state == stateref ) && ( spinwaitcounter ) ; spinwaitcounter-- )
    {
 #if defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || defined(__amd64) || defined(__i386__) || defined(__i386)  || defined(i386)
      __asm__ __volatile__( "rep ; nop" :::"memory" );
 #else
      __asm__ __volatile__( "":::"memory" );
 #endif
    }
    /* Spin-wait counter exhausted, yield thread */
    if( !spinwaitcounter )
    {
      __asm__ __volatile__( "":::"memory" );
      for( ; barrier->state == stateref ; )
        mtYield();
    }
  }

  return spinwaitcounter;
}


#elif MT_MSVC


typedef struct
{
  __declspec(align(64)) volatile long counter;
  __declspec(align(64)) volatile unsigned long state;
  __declspec(align(64)) volatile unsigned long stateref;
  __declspec(align(64)) int resetvalue;
} mtBarrier;

static inline void mtBarrierBuild( mtBarrier *barrier, int waitcount )
{
  barrier->counter = waitcount;
  barrier->state = 0;
  barrier->stateref = 0;
  barrier->resetvalue = waitcount;
  _ReadWriteBarrier();
  return;
}

static inline unsigned int mtBarrierWait( mtBarrier *barrier, unsigned int spinwaitcounter )
{
  unsigned int stateref, nextstateref;

  stateref = barrier->stateref;
  if( _InterlockedDecrement( &barrier->counter ) == 0 )
  {
    nextstateref = stateref + 1;
    /* Barrier is cleared, that was the last thread waiting on it */
    barrier->stateref = nextstateref;
    _InterlockedExchangeAdd( &barrier->counter, barrier->resetvalue );
    /* Ensure stateref and counter are updated and visible before releasing all spinning threads */
    _ReadWriteBarrier();
    barrier->state = nextstateref;
  }
  else
  {
    /* Spin-wait */
    for( ; ( barrier->state == stateref ) && ( spinwaitcounter ) ; spinwaitcounter-- )
    {
      YieldProcessor();
      _ReadWriteBarrier();
    }
    /* Spin-wait counter exhausted, yield thread */
    if( !spinwaitcounter )
    {
      _ReadWriteBarrier();
      for( ; barrier->state == stateref ; )
        mtYield();
    }
  }

  return spinwaitcounter;
}


#endif


////


/* Functions below require mmcore stuff */
#include "mm.h"


typedef struct
{
  void *arg;
  int cpulock;
  int cpulockcount;
  int nodelock;
  void *(*threadmain)( void *value );
} mtThreadLaunchState;

static MM_NOINLINE void *mtThreadLaunchJump( void *p )
{
  mtThreadLaunchState *launch;
  /* Waste a page of stack _after_ locking the thread on CPU/node, so that the stack is allocated on the proper NUMA node */
  volatile char stackpage[4096];
  void *arg;
  void *(*threadmain)( void *value );

  launch = p;
  arg = launch->arg;
  threadmain = launch->threadmain;
  free( launch );
  return threadmain( arg );
}

static void *mtThreadLaunch( void *p )
{
  mtThreadLaunchState *launch;

  launch = p;
  if( launch->cpulock != -1 )
  {
    if( launch->cpulockcount <= 1 )
      mmBindThreadToCpu( launch->cpulock );
    else
      mmBindThreadToCpuGroup( launch->cpulock, launch->cpulockcount );
  }
  else if( launch->nodelock != -1 )
    mmBindThreadToNode( launch->nodelock );

  /* Migrate the stack to the proper NUMA node */
#if MT_PTHREADS && MM_LINUX
/*
  pthread_attr_t attr;
  void *stackstart;
  size_t stacksize;
  if( ( pthread_getattr_np( pthread_self(), &attr ) == 0 ) && ( pthread_attr_getstack( &attr, &stackstart, &stacksize ) == 0 ) )
    mmNumaMigrate( nodeindex, stackstart, stacksize );
*/
#endif

  return mtThreadLaunchJump( p );
}


static inline void mtThreadCreateOnCpu( mtThread *thread, void *(*threadmain)( void *value ), void *value, int flags, int cpulock )
{
  mtThreadLaunchState *launch;
  launch = malloc( sizeof(mtThreadLaunchState) );
  launch->arg = value;
  launch->cpulock = cpulock;
  launch->cpulockcount = 1;
  launch->nodelock = -1;
  launch->threadmain = threadmain;
  mtThreadCreate( thread, mtThreadLaunch, launch, flags );
  return;
}

static inline void mtThreadCreateOnCpuGroup( mtThread *thread, void *(*threadmain)( void *value ), void *value, int flags, int cpulock, int cpulockcount )
{
  mtThreadLaunchState *launch;
  launch = malloc( sizeof(mtThreadLaunchState) );
  launch->arg = value;
  launch->cpulock = cpulock;
  launch->cpulockcount = cpulockcount;
  launch->nodelock = -1;
  launch->threadmain = threadmain;
  mtThreadCreate( thread, mtThreadLaunch, launch, flags );
  return;
}

static inline void mtThreadCreateOnNode( mtThread *thread, void *(*threadmain)( void *value ), void *value, int flags, int nodelock )
{
  mtThreadLaunchState *launch;
  launch = malloc( sizeof(mtThreadLaunchState) );
  launch->arg = value;
  launch->cpulock = -1;
  launch->cpulockcount = 1;
  launch->nodelock = nodelock;
  launch->threadmain = threadmain;
  mtThreadCreate( thread, mtThreadLaunch, launch, flags );
  return;
}


////


#endif


