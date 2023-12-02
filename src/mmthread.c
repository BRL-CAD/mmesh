/* *****************************************************************************
 *
 * Copyright (c) 2007-2023 Alexis Naveros.
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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "mm.h"
#include "mmthread.h"


void mtSleepBarrierInit( mtSleepBarrier *barrier, int count )
{
  mtMutexInit( &barrier->mutex );
  mtSignalInit( &barrier->signal );
  barrier->resetcount = count;
  barrier->index = 0;
  barrier->count[0] = count;
  barrier->count[1] = count;
  barrier->lockflag = 0;
  barrier->lockcount = 0;
  mtSignalInit( &barrier->locksignal );
  mtSignalInit( &barrier->lockwakesignal );
  return;
}

void mtSleepBarrierDestroy( mtSleepBarrier *barrier )
{
  mtMutexDestroy( &barrier->mutex );
  mtSignalDestroy( &barrier->signal );
  mtSignalDestroy( &barrier->locksignal );
  mtSignalDestroy( &barrier->lockwakesignal );
  return;
}

int mtSleepBarrierSync( mtSleepBarrier *barrier )
{
  int index, ret;
  mtMutexLock( &barrier->mutex );
  index = barrier->index;
  ret = 0;
  if( !( --barrier->count[index] ) )
  {
    ret = 1;
    mtSignalBroadcast( &barrier->signal );
    index ^= 1;
    barrier->index = index;
    barrier->count[index] = barrier->resetcount;
  }
  else
  {
    if( barrier->lockflag )
    {
      if( MT_SLEEP_BARRIER_LOCK_READY(barrier) )
        mtSignalBroadcast( &barrier->locksignal );
    }
    for( ; barrier->count[index] ; )
      mtSignalWait( &barrier->signal, &barrier->mutex );
  }
  mtMutexUnlock( &barrier->mutex );
  return ret;
}

int mtSleepBarrierSyncTimeout( mtSleepBarrier *barrier, long milliseconds )
{
  int index, ret;
  mtMutexLock( &barrier->mutex );
  index = barrier->index;
  ret = 0;
  if( !( --barrier->count[index] ) )
  {
    ret = 1;
    mtSignalBroadcast( &barrier->signal );
    index ^= 1;
    barrier->index = index;
    barrier->count[index] = barrier->resetcount;
  }
  else
  {
    if( barrier->lockflag )
    {
      if( MT_SLEEP_BARRIER_LOCK_READY(barrier) )
        mtSignalBroadcast( &barrier->locksignal );
    }
    mtSignalWaitTimeout( &barrier->signal, &barrier->mutex, milliseconds );
    if( !( barrier->count[index] ) )
      ret = 1;
    else
      barrier->count[index]++;
  }
  mtMutexUnlock( &barrier->mutex );
  return ret;
}


/* Check if the barrier requires a global lock */
void mtSleepBarrierCheckGlobal( mtSleepBarrier *barrier )
{
  if( barrier->lockflag )
  {
    mtMutexLock( &barrier->mutex );
    if( barrier->lockflag )
    {
      barrier->lockcount++;
      if( MT_SLEEP_BARRIER_LOCK_READY(barrier) )
        mtSignalBroadcast( &barrier->locksignal );
      for( ; barrier->lockflag ; )
        mtSignalWait( &barrier->lockwakesignal, &barrier->mutex );
      barrier->lockcount--;
    }
    mtMutexUnlock( &barrier->mutex );
  }
  return;
}

/* Acquire global lock, all threads must be in barrier */
void mtSleepBarrierLockGlobal( mtSleepBarrier *barrier )
{
  mtMutexLock( &barrier->mutex );
  while( barrier->lockflag )
  {
    barrier->lockcount++;
    if( MT_SLEEP_BARRIER_LOCK_READY(barrier) )
      mtSignalBroadcast( &barrier->locksignal );
    for( ; barrier->lockflag ; )
      mtSignalWait( &barrier->lockwakesignal, &barrier->mutex );
    barrier->lockcount--;
  }
  barrier->lockflag = 1;
  while( !MT_SLEEP_BARRIER_LOCK_READY(barrier) )
    mtSignalWait( &barrier->locksignal, &barrier->mutex );
  mtMutexUnlock( &barrier->mutex );
  return;
}

/* Release global lock */
void mtSleepBarrierUnlockGlobal( mtSleepBarrier *barrier )
{
  mtMutexLock( &barrier->mutex );
  barrier->lockflag = 0;
  mtSignalBroadcast( &barrier->lockwakesignal );
  mtMutexUnlock( &barrier->mutex );
  return;
}

