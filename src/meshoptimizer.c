/* *****************************************************************************
 *
 * Copyright (c) 2012-2023 Alexis Naveros.
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
#include <math.h>
#include <float.h>

#include "cc.h"
#include "cchash.h"
#include "mmthread.h"

#include "mmatomic.h"

#include "meshoptimizer.h"



#if MM_ATOMIC_SUPPORT
 #define MO_CONFIG_ATOMIC_SUPPORT (1)
#else
 #define MO_CONFIG_ATOMIC_SUPPORT (0)
#endif



#define MO_DEBUG (0)
#define MO_DEBUG_TIME (0)

#define MO_DEBUG_PROGRESS (1)



#if MO_DEBUG && !defined(_MSC_VER)
 #define MO_ERROR(s,f,...) ({fprintf(stderr,s,__VA_ARGS__);if(f) exit(1);})
#else
 #define MO_ERROR(s,f,...) fprintf(stderr,s,__VA_ARGS__)
#endif


#ifdef CPUCONF_CORES_COUNT
 #define MO_THREAD_COUNT_DEFAULT CPUCONF_CORES_COUNT
#else
 #define MO_THREAD_COUNT_DEFAULT (4)
#endif
#define MO_THREAD_COUNT_MAX (64)

#define MO_TRIANGLE_PER_THREAD_MINIMUM (1024)


////


typedef int moi;
typedef float mof;

#define MO_VERTEX_CACHE_SIZE_MAX (512+1)
#define MO_TRIREFSCORE_COUNT (128)
/* Must be power of two */
#define MO_CACHE_HASH_SIZE_MAX (1024)


////


typedef struct
{
#if MO_CONFIG_ATOMIC_SUPPORT
  mmAtomic32 atomicowner;
  mmAtomic32 atomictrirefcount;
#else
  int owner;
  mtSpin ownerspinlock;
  moi trirefcount;
#endif
  moi trirefbase;
  moi redirectindex;
} moVertex;

typedef struct
{
  moi v[3];
#if MO_CONFIG_ATOMIC_SUPPORT
  mmAtomic32 atomictrinext;
#else
  moi trinext;
  mtSpin spinlock;
#endif
} moTriangle;

#define MO_TRINEXT_ENDOFLIST (-1)
#define MO_TRINEXT_PENDING (-2)

typedef struct
{
  moi vertexindex;
  int cacheorder;
} moCacheEntry;

typedef struct
{
  int threadid;

  moi tricount;
  moi trifirst;
  moi trilast;

  int32_t hashsize;
  int32_t hashmask;
  moCacheEntry cachehash[MO_CACHE_HASH_SIZE_MAX];

  ccQuickRandState32 randstate;

#if MO_DEBUG_PROGRESS
  int64_t timestart;
  int64_t timelast;
#endif
} moThreadData;

typedef struct
{
  int threadid;
  moMesh *mesh;
  moThreadData *tdata;
  moi trifirst;
} moThreadInit;

struct moMesh
{
  int threadcount;
  uint32_t operationflags;

  void *indices;
  size_t indicesstride;
  void (*indicesUserToNative)( moi *dst, void *src );
  void (*indicesNativeToUser)( void *dst, moi *src );

  moVertex *vertexlist;
  moi vertexcount;

  moTriangle *trilist;
  moi tricount;

  moi *trireflist;
  moi trirefcount;

  /* Vertex shuffle */
  void *shuffleopaquepointer;
  void (*shufflecallback)( void *opaquepointer, long newvertexindex, long oldvertexindex );

  /* Score tables */
  mof lookaheadfactor;
  int vertexcachesize;
  mof cachescore[MO_VERTEX_CACHE_SIZE_MAX];
  mof trirefscore[MO_TRIREFSCORE_COUNT];

  /* Synchronization stuff */
  mtSleepBarrier workbarrier;

  /* Finish status tracking */
  int finishcount;
  mtMutex finishmutex;
  mtSignal finishsignal;

  moThreadInit threadinit[MO_THREAD_COUNT_MAX];
};


////


static void moIndicesInt8ToNative( moi *dst, void *src )
{
  uint8_t *s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void moIndicesInt16ToNative( moi *dst, void *src )
{
  uint16_t *s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void moIndicesInt32ToNative( moi *dst, void *src )
{
  uint32_t *s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void moIndicesInt64ToNative( moi *dst, void *src )
{
  uint64_t *s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}


static void moIndicesNativeToInt8( void *dst, moi *src )
{
  uint8_t *d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void moIndicesNativeToInt16( void *dst, moi *src )
{
  uint16_t *d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void moIndicesNativeToInt32( void *dst, moi *src )
{
  uint32_t *d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void moIndicesNativeToInt64( void *dst, moi *src )
{
  uint64_t *d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}


////


/* Mesh init step 1, initialize vertices, threaded */
static void moMeshInitVertices( moMesh *mesh, moThreadData *tdata, int threadcount )
{
  int vertexindex, vertexindexmax, vertexperthread;
  moVertex *vertex;

  vertexperthread = ( mesh->vertexcount / threadcount ) + 1;
  vertexindex = tdata->threadid * vertexperthread;
  vertexindexmax = vertexindex + vertexperthread;
  if( vertexindexmax > mesh->vertexcount )
    vertexindexmax = mesh->vertexcount;

  vertex = &mesh->vertexlist[vertexindex];
  for( ; vertexindex < vertexindexmax ; vertexindex++, vertex++ )
  {
#if MO_CONFIG_ATOMIC_SUPPORT
    mmAtomicWrite32( &vertex->atomicowner, -1 );
    mmAtomicWrite32( &vertex->atomictrirefcount, 0 );
#else
    vertex->owner = -1;
    mtSpinInit( &vertex->ownerspinlock );
    vertex->trirefcount = 0;
#endif
    vertex->redirectindex = -1;
  }

  return;
}


/* Mesh init step 2, initialize triangles, threaded */
static void moMeshInitTriangles( moMesh *mesh, moThreadData *tdata, int threadcount )
{
  int i, triperthread, triindex, triindexmax;
  void *indices;
  moTriangle *tri;
  moVertex *vertex;

  triperthread = ( mesh->tricount / threadcount ) + 1;
  triindex = tdata->threadid * triperthread;
  triindexmax = triindex + triperthread;
  if( triindexmax > mesh->tricount )
    triindexmax = mesh->tricount;

  /* Initialize triangles */
  indices = ADDRESS( mesh->indices, triindex * mesh->indicesstride );
  tri = &mesh->trilist[triindex];
  for( ; triindex < triindexmax ; triindex++, indices = ADDRESS( indices, mesh->indicesstride ), tri++ )
  {
    mesh->indicesUserToNative( tri->v, indices );
#if MO_CONFIG_ATOMIC_SUPPORT
    mmAtomicWrite32( &tri->atomictrinext, MO_TRINEXT_PENDING );
#else
    tri->trinext = MO_TRINEXT_PENDING;
    mtSpinInit( &tri->spinlock );
#endif
    for( i = 0 ; i < 3 ; i++ )
    {
      vertex = &mesh->vertexlist[ tri->v[i] ];
#if MO_CONFIG_ATOMIC_SUPPORT
      mmAtomicAdd32( &vertex->atomictrirefcount, 1 );
#else
      mtSpinLock( &vertex->ownerspinlock );
      vertex->trirefcount++;
      mtSpinUnlock( &vertex->ownerspinlock );
#endif
    }
  }

  return;
}


/* Mesh init step 3, initialize vertex trirefbase, compute vertex scores, NOT threaded */
static void moMeshInitTrirefs( moMesh *mesh )
{
  moi vertexindex, trirefcount;
  moVertex *vertex;

  /* Compute base of vertex triangle references */
  trirefcount = 0;
  vertex = mesh->vertexlist;
  for( vertexindex = 0 ; vertexindex < mesh->vertexcount ; vertexindex++, vertex++ )
  {
#if MO_CONFIG_ATOMIC_SUPPORT
    trirefcount += mmAtomicRead32( &vertex->atomictrirefcount );
#else
    trirefcount += vertex->trirefcount;
#endif
    vertex->trirefbase = trirefcount;
  }
  mesh->trirefcount = trirefcount;

  return;
}


/* Mesh init step 4, store vertex trirefs, compute triangle scores, return highest score triindex, threaded */
static moi moMeshBuildTrirefs( moMesh *mesh, moThreadData *tdata, int threadcount )
{
  int i, triperthread, triindex, triindexmax, besttriindex;
  moi trirefcount;
  mof score, bestscore;
  moTriangle *tri;
  moVertex *vertex;

  triperthread = ( mesh->tricount / threadcount ) + 1;
  triindex = tdata->threadid * triperthread;
  triindexmax = triindex + triperthread;
  if( triindexmax > mesh->tricount )
    triindexmax = mesh->tricount;

  /* Store vertex triangle references */
  bestscore = 0.0f;
  besttriindex = -1;
  tri = &mesh->trilist[triindex];
  for( ; triindex < triindexmax ; triindex++, tri++ )
  {
    score = 0.0f;
    for( i = 0 ; i < 3 ; i++ )
    {
      vertex = &mesh->vertexlist[ tri->v[i] ];
#if MO_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &vertex->atomicowner, -1, tdata->threadid );
      mesh->trireflist[ --vertex->trirefbase ] = triindex;
      mmAtomicWrite32( &vertex->atomicowner, -1 );
      trirefcount = mmAtomicRead32( &vertex->atomictrirefcount );
#else
      mtSpinLock( &vertex->ownerspinlock );
      mesh->trireflist[ --vertex->trirefbase ] = triindex;
      mtSpinUnlock( &vertex->ownerspinlock );
      trirefcount = vertex->trirefcount;
#endif
      if( trirefcount < MO_TRIREFSCORE_COUNT )
        score += mesh->trirefscore[ trirefcount ];
    }
    if( score > bestscore )
    {
      bestscore = score;
      besttriindex = triindex;
    }
  }

  return besttriindex;
}


////


static inline uint32_t moCacheHashKey( moi index )
{
  uint32_t hashkey;
/*
  hashkey  = index;
  hashkey += hashkey << 10;
  hashkey ^= hashkey >> 6;
  hashkey += hashkey << 6;
  hashkey ^= hashkey >> 11;
*/
  hashkey = ccHash32Int32( index );
  return hashkey;
}

static void moCacheInit( moMesh *mesh, moThreadData *tdata, int minimumcount )
{
  int entryindex;
  moCacheEntry *cache;
  tdata->hashsize = ccPow2Round32( minimumcount );
  if( tdata->hashsize > MO_CACHE_HASH_SIZE_MAX )
    tdata->hashsize = MO_CACHE_HASH_SIZE_MAX;
  tdata->hashmask = tdata->hashsize - 1;
  for( entryindex = 0 ; entryindex < tdata->hashsize ; entryindex++ )
  {
    cache = &tdata->cachehash[entryindex];
    cache->vertexindex = -1;
    cache->cacheorder = 0;
  }
  return;
}

static int moCacheGetOrder( moMesh *mesh, moThreadData *tdata, moi vertexindex )
{
  int cacheorder;
  uint32_t hashkey;
  moCacheEntry *cache;
  cacheorder = mesh->vertexcachesize;
  hashkey = moCacheHashKey( vertexindex ) & tdata->hashmask;
  for( ; ; hashkey = ( hashkey + 1 ) & tdata->hashmask )
  {
    cache = &tdata->cachehash[hashkey];
    if( cache->vertexindex == -1 )
      break;
    if( cache->vertexindex == vertexindex )
    {
      cacheorder = cache->cacheorder;
      break;
    }
  }
  return cacheorder;
}

static int moCacheSetOrder( moMesh *mesh, moThreadData *tdata, moi vertexindex, int cacheorder )
{
  int retcacheorder;
  uint32_t hashkey;
  moCacheEntry *cache;
  hashkey = moCacheHashKey( vertexindex ) & tdata->hashmask;
  for( ; ; hashkey = ( hashkey + 1 ) & tdata->hashmask )
  {
    cache = &tdata->cachehash[hashkey];
    if( cache->vertexindex == -1 )
    {
      cache->vertexindex = vertexindex;
      retcacheorder = -1;
      cache->cacheorder = cacheorder;
      break;
    }
    if( cache->vertexindex == vertexindex )
    {
      retcacheorder = cache->cacheorder;
      cache->cacheorder = cacheorder;
      break;
    }
  }
  return retcacheorder;
}

static void moCacheDelete( moMesh *mesh, moThreadData *tdata, moi vertexindex )
{
  uint32_t hashkey, delbase, targetkey, sourcekey, targetindex, sourceindex;
  moCacheEntry *cache, *cachesource, *cachetarget;
  hashkey = moCacheHashKey( vertexindex ) & tdata->hashmask;
  for( ; ; hashkey = ( hashkey + 1 ) & tdata->hashmask )
  {
    cache = &tdata->cachehash[hashkey];
    if( cache->vertexindex == -1 )
    {
      MO_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
      return;
    }
    if( cache->vertexindex == vertexindex )
      break;
  }
  for( delbase = hashkey ; ; )
  {
    delbase = ( delbase - 1 ) & tdata->hashmask;
    if( tdata->cachehash[delbase].vertexindex == -1 )
      break;
  }
  delbase = ( delbase + 1 ) & tdata->hashmask;
  for( ; ; )
  {
    targetkey = hashkey;
    cachetarget = 0;
    for( sourceindex = hashkey ; ; )
    {
      sourceindex = ( sourceindex + 1 ) & tdata->hashmask;
      cachesource = &tdata->cachehash[sourceindex];
      if( cachesource->vertexindex == -1 )
        break;
      sourcekey = moCacheHashKey( cachesource->vertexindex ) & tdata->hashmask;
      /* Check if moving the entry backwards is allowed without breaking chain */
      if( targetkey >= delbase )
      {
        if( ( sourcekey < delbase ) || ( sourcekey > targetkey ) )
          continue;
      }
      else if( ( sourcekey > targetkey ) && ( sourcekey < delbase ) )
        continue;
      cachetarget = cachesource;
      targetkey = sourcekey;
      targetindex = sourceindex;
    }
    cache = &tdata->cachehash[hashkey];
    if( !( cachetarget ) )
    {
      cache->vertexindex = -1;
      break;
    }
    cache->vertexindex = cachetarget->vertexindex;
    cache->cacheorder = cachetarget->cacheorder;
    hashkey = targetindex;
  }
  return;
}


////


static mof moTriangleScore( moMesh *mesh, moThreadData *tdata, moTriangle *tri )
{
  int axisindex, cacheorder, trirefcount;
#ifndef MO_CONFIG_ATOMIC_SUPPORT
  int32_t owner;
#endif
  moi vertexindex;
  mof score;
  moVertex *vertex;

  score = 0.0f;
  for( axisindex = 0 ; axisindex < 3 ; axisindex++ )
  {
    vertexindex = tri->v[axisindex];
    cacheorder = moCacheGetOrder( mesh, tdata, vertexindex );
    score += mesh->cachescore[ cacheorder ];
    vertex = &mesh->vertexlist[ vertexindex ];
#if MO_CONFIG_ATOMIC_SUPPORT
    trirefcount = mmAtomicRead32( &vertex->atomictrirefcount );
#else
    mtSpinLock( &vertex->ownerspinlock );
    owner = vertex->owner;
    trirefcount = vertex->trirefcount;
    mtSpinUnlock( &vertex->ownerspinlock );
    if( ( owner != tdata->threadid ) && ( owner != -1 ) )
      continue;
#endif
    if( trirefcount < MO_TRIREFSCORE_COUNT )
      score += mesh->trirefscore[ trirefcount ];
  }

  return score;
}


static mof moTriangleNextScore( moMesh *mesh, moThreadData *tdata, moTriangle *tri, moi *prevtriv )
{
  int axisindex, cacheorder, trirefcount;
#ifndef MO_CONFIG_ATOMIC_SUPPORT
  int32_t owner;
#endif
  moi vertexindex;
  mof score;
  moVertex *vertex;

  score = 0.0f;
  for( axisindex = 0 ; axisindex < 3 ; axisindex++ )
  {
    vertexindex = tri->v[axisindex];
    cacheorder = 0;
    if( ( vertexindex != prevtriv[0] ) && ( vertexindex != prevtriv[1] ) && ( vertexindex != prevtriv[2] ) )
    {
      cacheorder = moCacheGetOrder( mesh, tdata, vertexindex ) + 3;
      if( cacheorder > mesh->vertexcachesize )
        cacheorder = mesh->vertexcachesize;
    }
    score += mesh->cachescore[ cacheorder ];
    vertex = &mesh->vertexlist[ vertexindex ];
#if MO_CONFIG_ATOMIC_SUPPORT
    trirefcount = mmAtomicRead32( &vertex->atomictrirefcount );
#else
    mtSpinLock( &vertex->ownerspinlock );
    owner = vertex->owner;
    trirefcount = vertex->trirefcount;
    mtSpinUnlock( &vertex->ownerspinlock );
    if( ( owner != tdata->threadid ) && ( owner != -1 ) )
      continue;
#endif
    if( trirefcount < MO_TRIREFSCORE_COUNT )
      score += mesh->trirefscore[ trirefcount ];
  }

  return score;
}


static mof moLookAheadScore( moMesh *mesh, moThreadData *tdata, moTriangle *tri )
{
  int axisindex, cacheorder, trirefindex, trirefcount;
  moi vertexindex, triindex;
  mof score, bestscore;
  moi *trireflist;
  moVertex *vertex;
  moTriangle *tricheck;

  bestscore = 0.0f;
  for( axisindex = 0 ; axisindex < 3 ; axisindex++ )
  {
    vertexindex = tri->v[axisindex];
    vertex = &mesh->vertexlist[ vertexindex ];
    trireflist = &mesh->trireflist[vertex->trirefbase];
#if MO_CONFIG_ATOMIC_SUPPORT
    mmAtomicSpin32( &vertex->atomicowner, -1, tdata->threadid );
    trirefcount = mmAtomicRead32( &vertex->atomictrirefcount );
#else
    mtSpinLock( &vertex->ownerspinlock );
    vertex->owner = tdata->threadid;
    mtSpinUnlock( &vertex->ownerspinlock );
    trirefcount = vertex->trirefcount;
#endif
    for( trirefindex = 0 ; trirefindex < trirefcount ; trirefindex++ )
    {
      triindex = trireflist[trirefindex];
      tricheck = &mesh->trilist[triindex];
      if( tricheck == tri )
        continue;
      score = moTriangleNextScore( mesh, tdata, tricheck, tri->v );
      if( score > bestscore )
        bestscore = score;
    }
#if MO_CONFIG_ATOMIC_SUPPORT
    mmAtomicWrite32( &vertex->atomicowner, -1 );
#else
    mtSpinLock( &vertex->ownerspinlock );
    vertex->owner = -1;
    mtSpinUnlock( &vertex->ownerspinlock );
#endif
    cacheorder = moCacheGetOrder( mesh, tdata, vertexindex );
    score += mesh->cachescore[ cacheorder ];
    vertex = &mesh->vertexlist[ vertexindex ];
  }

  return bestscore;
}


static void moDetachTriangle( moMesh *mesh, moThreadData *tdata, moi detachtriindex )
{
  int axisindex, trirefindex, trirefcount;
  moi vertexindex, triindex;
  moi *trireflist;
  moVertex *vertex;
  moTriangle *tri;

  tri = &mesh->trilist[detachtriindex];
  for( axisindex = 0 ; axisindex < 3 ; axisindex++ )
  {
    /* Adjust triref lists and count for the 3 vertices */
    vertexindex = tri->v[axisindex];
    vertex = &mesh->vertexlist[vertexindex];
#if MO_CONFIG_ATOMIC_SUPPORT
    mmAtomicSpin32( &vertex->atomicowner, -1, tdata->threadid );
#else
    mtSpinLock( &vertex->ownerspinlock );
    vertex->owner = tdata->threadid;
#endif
    trireflist = &mesh->trireflist[vertex->trirefbase];
    for( trirefindex = 0 ; ; trirefindex++ )
    {
#if MO_DEBUG
 #if MO_CONFIG_ATOMIC_SUPPORT
      if( trirefindex >= mmAtomicRead32( &vertex->atomictrirefcount ) )
        MO_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
 #else
      if( trirefindex >= vertex->trirefcount )
        MO_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
 #endif
#endif
      triindex = trireflist[trirefindex];
      if( triindex != detachtriindex )
        continue;
#if MO_CONFIG_ATOMIC_SUPPORT
      trirefcount = mmAtomicRead32( &vertex->atomictrirefcount ) - 1;
      mmAtomicWrite32( &vertex->atomictrirefcount, trirefcount );
#else
      trirefcount = vertex->trirefcount - 1;
      vertex->trirefcount = trirefcount;
#endif
      for( ; trirefindex < trirefcount ; trirefindex++ )
        trireflist[trirefindex] = trireflist[trirefindex+1];
      break;
    }
#if MO_CONFIG_ATOMIC_SUPPORT
    mmAtomicWrite32( &vertex->atomicowner, -1 );
#else
    vertex->owner = -1;
    mtSpinUnlock( &vertex->ownerspinlock );
#endif
  }

  return;
}

static moi moFindSeedTriangle( moMesh *mesh, moThreadData *tdata )
{
  int testcount;
  moi seedindex, triindex, triindexstart, triindexend;
  mof score, bestscore;
  moTriangle *tri;
#ifndef MO_CONFIG_ATOMIC_SUPPORT
  moi trinext;
#endif

  testcount = 16384;
  if( mesh->operationflags & MO_FLAGS_FAST_SEED_SELECT )
    testcount = 256;
  if( testcount > mesh->tricount )
    testcount = mesh->tricount;

  triindexstart = ccQuickRand32( &tdata->randstate ) % mesh->tricount;
  triindexend = mesh->tricount;
  bestscore = 0;
  seedindex = -1;
  for( ; ; )
  {
    tri = &mesh->trilist[triindexstart];
    for( triindex = triindexstart ; triindex < triindexend ; triindex++, tri++ )
    {
      if( ( --testcount < 0 ) && ( seedindex != -1 ) )
        goto done;
#if MO_CONFIG_ATOMIC_SUPPORT
      if( mmAtomicRead32( &tri->atomictrinext ) != MO_TRINEXT_PENDING )
        continue;
#else
      mtSpinLock( &tri->spinlock );
      trinext = tri->trinext;
      mtSpinUnlock( &tri->spinlock );
      if( trinext != MO_TRINEXT_PENDING )
        continue;
#endif
      score = moTriangleScore( mesh, tdata, tri );
      if( score > bestscore )
      {
        bestscore = score;
        seedindex = triindex;
      }
    }
    if( !( triindexstart ) )
      break;
    triindexend = triindexstart;
    triindexstart = 0;
  }

  done:

  return seedindex;
}


////


static moi moFindNextStep( moMesh *mesh, moThreadData *tdata )
{
  int trirefindex, trirefcount, cacheordercap;
  uint32_t hashkey;
  moi triindex, besttriindex;
  mof score, bestscore;
  moi *trireflist;
  moCacheEntry *cache;
  moVertex *vertex;
  moTriangle *tri;
#ifndef MO_CONFIG_ATOMIC_SUPPORT
  moi trinext;
#endif

  cacheordercap = mesh->vertexcachesize;
  if( mesh->operationflags & MO_FLAGS_ENABLE_LAZY_SEARCH )
    cacheordercap = 4;

  bestscore = 0.0f;
  besttriindex = -1;
  for( ; ; )
  {
    cache = tdata->cachehash;
    for( hashkey = 0 ; hashkey < tdata->hashsize ; hashkey++, cache++ )
    {
      if( cache->vertexindex == -1 )
        continue;
      if( ( cacheordercap < mesh->vertexcachesize ) && ( moCacheGetOrder( mesh, tdata, cache->vertexindex ) > cacheordercap ) )
        continue;
      vertex = &mesh->vertexlist[cache->vertexindex];
      trireflist = &mesh->trireflist[vertex->trirefbase];
#if MO_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &vertex->atomicowner, -1, tdata->threadid );
      trirefcount = mmAtomicRead32( &vertex->atomictrirefcount );
#else
      mtSpinLock( &vertex->ownerspinlock );
      vertex->owner = tdata->threadid;
      mtSpinUnlock( &vertex->ownerspinlock );
      trirefcount = vertex->trirefcount;
#endif
      for( trirefindex = 0 ; trirefindex < trirefcount ; trirefindex++ )
      {
        triindex = trireflist[trirefindex];
        tri = &mesh->trilist[triindex];
#if MO_CONFIG_ATOMIC_SUPPORT
        if( mmAtomicRead32( &tri->atomictrinext ) != MO_TRINEXT_PENDING )
          continue;
#else
        mtSpinLock( &tri->spinlock );
        trinext = tri->trinext;
        mtSpinUnlock( &tri->spinlock );
        if( trinext != MO_TRINEXT_PENDING )
          continue;
#endif
        score = moTriangleScore( mesh, tdata, tri );
        if( score < bestscore )
          continue;
        bestscore = score;
        besttriindex = triindex;
      }
#if MO_CONFIG_ATOMIC_SUPPORT
      mmAtomicWrite32( &vertex->atomicowner, -1 );
#else
      mtSpinLock( &vertex->ownerspinlock );
      vertex->owner = -1;
      mtSpinUnlock( &vertex->ownerspinlock );
#endif
    }
    if( besttriindex != -1 )
      break;
    if( cacheordercap >= mesh->vertexcachesize )
      return -1;
    cacheordercap <<= 1;
  }

  /* Add score bonus for triangle strip continuity ? */
#if MO_DEBUG
  tri = &mesh->trilist[besttriindex];
  printf( "  Tri %d : %d,%d,%d ( %f )\n", besttriindex, tri->v[0], tri->v[1], tri->v[2], bestscore );
#endif

  return besttriindex;
}


#define MO_LOOK_AHEAD_BEST_BUFFER_SIZE (4)

typedef struct
{
  moi triindex;
  mof score;
} moScoreEntry;

static inline void moBufferRegisterScore( moScoreEntry *entry, moi triindex, mof score )
{
  int bufferindex, targetindex;
  if( score < entry[0].score )
    return;
  targetindex = 0;
  for( bufferindex = 1 ; bufferindex < MO_LOOK_AHEAD_BEST_BUFFER_SIZE ; bufferindex++ )
  {
    if( score < entry[bufferindex].score )
      break;
    if( triindex == entry[bufferindex].triindex )
      return;
    targetindex = bufferindex;
  }
  for( bufferindex = 0 ; bufferindex < targetindex ; bufferindex++ )
    entry[bufferindex+0] = entry[bufferindex+1];
  entry[targetindex].triindex = triindex;
  entry[targetindex].score = score;
  return;
}

static moi moFindNextStepLookAhead( moMesh *mesh, moThreadData *tdata )
{
  int trirefindex, trirefcount, cacheordercap;
  uint32_t hashkey;
  moi triindex, besttriindex;
  mof score, nextscore, bestscore;
  moi *trireflist;
  moCacheEntry *cache;
  moVertex *vertex;
  moTriangle *tri;
  moScoreEntry scorebuffer[MO_LOOK_AHEAD_BEST_BUFFER_SIZE];
  moScoreEntry *entry, *entrynext;
#ifndef MO_CONFIG_ATOMIC_SUPPORT
  moi trinext;
#endif

  for( entry = &scorebuffer[MO_LOOK_AHEAD_BEST_BUFFER_SIZE-1] ; entry >= scorebuffer ; entry-- )
  {
    entry->triindex = -1;
    entry->score = 0.0f;
  }

  cacheordercap = mesh->vertexcachesize;
  if( mesh->operationflags & MO_FLAGS_ENABLE_LAZY_SEARCH )
    cacheordercap = 4;

  bestscore = 0.0f;
  besttriindex = -1;
  for( ; ; )
  {
    cache = tdata->cachehash;
    for( hashkey = 0 ; hashkey < tdata->hashsize ; hashkey++, cache++ )
    {
      if( cache->vertexindex == -1 )
        continue;
      if( ( cacheordercap < mesh->vertexcachesize ) && ( moCacheGetOrder( mesh, tdata, cache->vertexindex ) > cacheordercap ) )
        continue;
      vertex = &mesh->vertexlist[cache->vertexindex];
      trireflist = &mesh->trireflist[vertex->trirefbase];
#if MO_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &vertex->atomicowner, -1, tdata->threadid );
      trirefcount = mmAtomicRead32( &vertex->atomictrirefcount );
#else
      mtSpinLock( &vertex->ownerspinlock );
      vertex->owner = tdata->threadid;
      mtSpinUnlock( &vertex->ownerspinlock );
      trirefcount = vertex->trirefcount;
#endif
      for( trirefindex = 0 ; trirefindex < trirefcount ; trirefindex++ )
      {
        triindex = trireflist[trirefindex];
        tri = &mesh->trilist[triindex];
#if MO_CONFIG_ATOMIC_SUPPORT
        if( mmAtomicRead32( &tri->atomictrinext ) != MO_TRINEXT_PENDING )
          continue;
#else
        mtSpinLock( &tri->spinlock );
        trinext = tri->trinext;
        mtSpinUnlock( &tri->spinlock );
        if( trinext != MO_TRINEXT_PENDING )
          continue;
#endif
        score = moTriangleScore( mesh, tdata, tri );
        moBufferRegisterScore( scorebuffer, triindex, score );
      }
#if MO_CONFIG_ATOMIC_SUPPORT
      mmAtomicWrite32( &vertex->atomicowner, -1 );
#else
      mtSpinLock( &vertex->ownerspinlock );
      vertex->owner = -1;
      mtSpinUnlock( &vertex->ownerspinlock );
#endif
    }
    if( scorebuffer[MO_LOOK_AHEAD_BEST_BUFFER_SIZE-1].triindex != -1 )
      break;
    if( cacheordercap >= mesh->vertexcachesize )
      return -1;
    cacheordercap <<= 1;
  }

  bestscore = 0.0f;
  besttriindex = -1;
  for( entry = &scorebuffer[MO_LOOK_AHEAD_BEST_BUFFER_SIZE-1] ; entry >= scorebuffer ; entry-- )
  {
#if MO_DEBUG
    printf( "Result[%d] = %d : %f\n", (int)(entry-scorebuffer), entry->triindex, entry->score );
#endif
    if( entry->triindex == -1 )
      break;
    tri = &mesh->trilist[entry->triindex];
    nextscore = moLookAheadScore( mesh, tdata, tri );
    for( entrynext = &scorebuffer[MO_LOOK_AHEAD_BEST_BUFFER_SIZE-1] ; entrynext >= scorebuffer ; entrynext-- )
    {
      if( entrynext->triindex == -1 )
        break;
      if( entrynext == entry )
        continue;
      score = moTriangleNextScore( mesh, tdata, &mesh->trilist[entrynext->triindex], tri->v );
      if( score > nextscore )
        nextscore = score;
    }
    score = entry->score + ( mesh->lookaheadfactor * nextscore );
    if( score < bestscore )
      continue;
    bestscore = score;
    besttriindex = entry->triindex;
  }

  /* Add score bonus for triangle strip continuity ? */
#if MO_DEBUG
  tri = &mesh->trilist[besttriindex];
  printf( "  Tri %d : %d,%d,%d ( %f )\n", besttriindex, tri->v[0], tri->v[1], tri->v[2], bestscore );
#endif

  return besttriindex;
}


////


static void moRebuildMesh( moMesh *mesh, moThreadData *tdata, moi seedindex )
{
  int axisindex, cacheorder, cacheorderaddglobal;
  uint32_t hashkey;
  moi besttriindex;
  moCacheEntry *cache;
  moTriangle *tri, *trilast;
  int cacheorderadd[MO_VERTEX_CACHE_SIZE_MAX];
  moi delindex, delcount;
  moi delbuffer[8];
  moi (*findnextstep)( moMesh *mesh, moThreadData *tdata );
#ifndef MO_CONFIG_ATOMIC_SUPPORT
  moi trinext;
#endif

  for( ; ; )
  {
    if( seedindex == -1 )
    {
      /* Exhaustive search for a new seed triangle */
      seedindex = moFindSeedTriangle( mesh, tdata );
      if( seedindex == -1 )
        return;
    }
    tri = &mesh->trilist[seedindex];

#if MO_DEBUG
    printf( "Seed %d : %d,%d,%d\n", seedindex, tri->v[0], tri->v[1], tri->v[2] );
#endif

    /* Add triangle to list */
#if MO_CONFIG_ATOMIC_SUPPORT
    if( mmAtomicCmpReplace32( &tri->atomictrinext, MO_TRINEXT_PENDING, MO_TRINEXT_ENDOFLIST ) )
      break;
#else
    mtSpinLock( &tri->spinlock );
    trinext = tri->trinext;
    if( tri->trinext == MO_TRINEXT_PENDING )
      tri->trinext = MO_TRINEXT_ENDOFLIST;
    mtSpinUnlock( &tri->spinlock );
    if( trinext == MO_TRINEXT_PENDING )
      break;
#endif
    seedindex = -1;
  }
  tdata->trifirst = seedindex;
  tdata->trilast = seedindex;

  /* Adjust triref lists and count for the 3 vertices */
  moDetachTriangle( mesh, tdata, seedindex );

  /* Adjust cache */
  moCacheSetOrder( mesh, tdata, tri->v[0], 0 );
  moCacheSetOrder( mesh, tdata, tri->v[1], 0 );
  moCacheSetOrder( mesh, tdata, tri->v[2], 0 );

  /* Select step pipeline */
  findnextstep = moFindNextStepLookAhead;
  if( mesh->operationflags & MO_FLAGS_DISABLE_LOOK_AHEAD )
    findnextstep = moFindNextStep;

  for( ; ; )
  {
    /* Find highest score triangle of all the triangles linked to vertices present in cache */
    besttriindex = findnextstep( mesh, tdata );

    if( besttriindex == -1 )
    {
#if MO_DEBUG
      printf( "Strip Dead End\n" );
#endif
      /* Exhaustive search for a new seed triangle */
      besttriindex = moFindSeedTriangle( mesh, tdata );
      if( besttriindex == -1 )
        return;
#if MO_DEBUG
      printf( "Seed %d\n", besttriindex );
#endif
    }

    /* If triangle has already been added by the time we got here, we have to start over... */
    tri = &mesh->trilist[besttriindex];
#if MO_CONFIG_ATOMIC_SUPPORT
    if( !( mmAtomicCmpReplace32( &tri->atomictrinext, MO_TRINEXT_PENDING, MO_TRINEXT_ENDOFLIST ) ) )
      continue;
#else
    mtSpinLock( &tri->spinlock );
    trinext = tri->trinext;
    if( tri->trinext == MO_TRINEXT_PENDING )
      tri->trinext = MO_TRINEXT_ENDOFLIST;
    mtSpinUnlock( &tri->spinlock );
    if( trinext != MO_TRINEXT_PENDING )
      continue;
#endif

    /* Add triangle to list */
    trilast = &mesh->trilist[tdata->trilast];
#if MO_CONFIG_ATOMIC_SUPPORT
    mmAtomicWrite32( &trilast->atomictrinext, besttriindex );
#else
    mtSpinLock( &tri->spinlock );
    trilast->trinext = besttriindex;
    mtSpinUnlock( &tri->spinlock );
#endif
    tdata->trilast = besttriindex;
    tdata->tricount++;

#if MO_DEBUG_PROGRESS
    if( ( ccGetMillisecondsTime() - tdata->timelast ) > 5000 )
    {
      tdata->timelast = ccGetMillisecondsTime();
      printf( "Mesh optimizer time elapsed: %.2f seconds ~ thread %d ~ %d/%d triangles processed\n", 0.001f * (double)( ccGetMillisecondsTime() - tdata->timestart ), tdata->threadid, tdata->tricount, mesh->tricount );
      fflush( stdout );
    }
#endif

    /* Adjust triref lists and count for the 3 vertices */
    moDetachTriangle( mesh, tdata, besttriindex );

    /* Build the shift table */
    cacheorderaddglobal = 3;
    for( cacheorder = 0 ; cacheorder < mesh->vertexcachesize ; cacheorder++ )
      cacheorderadd[cacheorder] = 0;
    for( axisindex = 0 ; axisindex < 3 ; axisindex++ )
    {
      cacheorder = moCacheGetOrder( mesh, tdata, tri->v[axisindex] );
      if( cacheorder != -1 )
      {
        cacheorderaddglobal--;
        for( ; cacheorder >= 0 ; cacheorder-- )
          cacheorderadd[cacheorder]++;
      }
    }

    /* Update cache entries */
    delcount = 0;
    cache = tdata->cachehash;
    for( hashkey = 0 ; hashkey < tdata->hashsize ; hashkey++, cache++ )
    {
      if( cache->vertexindex == -1 )
        continue;
      cache->cacheorder += cacheorderaddglobal + cacheorderadd[cache->cacheorder];
      if( cache->cacheorder >= mesh->vertexcachesize )
        delbuffer[ delcount++ ] = cache->vertexindex;
    }
    for( delindex = 0 ; delindex < delcount ; delindex++ )
      moCacheDelete( mesh, tdata, delbuffer[delindex] );

    /* Set new cache entries */
    for( axisindex = 0 ; axisindex < 3 ; axisindex++ )
      moCacheSetOrder( mesh, tdata, tri->v[axisindex], 0 );
  }

  return;
}


////


static void *moThreadMain( void *value )
{
  moi seedindex;
  moMesh *mesh;
  moThreadInit *tinit;
  moThreadData tdata;

  tinit = value;
  mesh = tinit->mesh;
  tinit->tdata = &tdata;

#if MO_DEBUG_PROGRESS
  tdata.timestart = ccGetMillisecondsTime();
  tdata.timelast = tdata.timestart;
#endif

  tdata.threadid = tinit->threadid;
  moCacheInit( mesh, &tdata, mesh->vertexcachesize + ( mesh->vertexcachesize >> 1 ) + 3 );
  tdata.trifirst = MO_TRINEXT_ENDOFLIST;
  tdata.trilast = -1;
  tdata.tricount = 0;
#if 0
  ccQuickRand32Seed( &tdata.randstate, (int)((uintptr_t)&tdata) + tdata.threadid );
#else
  ccQuickRand32Seed( &tdata.randstate, tdata.threadid );
#endif

  /* Step 1 */
  moMeshInitVertices( mesh, &tdata, mesh->threadcount );
  mtSleepBarrierSync( &mesh->workbarrier );

  /* Step 2 */
  moMeshInitTriangles( mesh, &tdata, mesh->threadcount );
  mtSleepBarrierSync( &mesh->workbarrier );

  /* Step 3 is done by a single thread */
  if( tdata.threadid == 0 )
    moMeshInitTrirefs( mesh );
  mtSleepBarrierSync( &mesh->workbarrier );

  /* Step 4 */
  seedindex = moMeshBuildTrirefs( mesh, &tdata, mesh->threadcount );
  mtSleepBarrierSync( &mesh->workbarrier );

  /* Step 5, threads rebuild the mesh */
  moRebuildMesh( mesh, &tdata, seedindex );
  mtSleepBarrierSync( &mesh->workbarrier );

  /* Set the linked list's first item for main thread to access */
  tinit->trifirst = tdata.trifirst;

  /* Send finish signal */
  mtMutexLock( &mesh->finishmutex );
  mesh->finishcount--;
  if( mesh->finishcount == 0 )
    mtSignalBroadcast( &mesh->finishsignal );
  mtMutexUnlock( &mesh->finishmutex );

  return 0;
}


////


static void moWriteIndices( moMesh *mesh, moThreadInit *threadinit )
{
  int threadindex;
  moi triindex, trinext;
  moThreadInit *tinit;
  moTriangle *tri;
  void *indices;

  indices = mesh->indices;
  for( threadindex = 0 ; threadindex < mesh->threadcount ; threadindex++ )
  {
    tinit = &threadinit[threadindex];
    for( triindex = tinit->trifirst ; triindex !=  MO_TRINEXT_ENDOFLIST ; triindex = trinext )
    {
      tri = &mesh->trilist[triindex];
#if MO_CONFIG_ATOMIC_SUPPORT
      trinext = mmAtomicRead32( &tri->atomictrinext );
#else
      trinext = tri->trinext;
#endif

#if MO_DEBUG
      printf( "Tri %d,%d,%d\n", tri->v[0], tri->v[1], tri->v[2] );
#endif

      mesh->indicesNativeToUser( indices, tri->v );
      indices = ADDRESS( indices, mesh->indicesstride );
    }
  }

  return;
}


static void moBuildRedirection( moMesh *mesh, moThreadInit *threadinit )
{
  int threadindex, redirectindex;
  moi triindex, trinext;
  moThreadInit *tinit;
  moVertex *vertex;
  moTriangle *tri;

  redirectindex = 0;
  for( threadindex = 0 ; threadindex < mesh->threadcount ; threadindex++ )
  {
    tinit = &threadinit[threadindex];
    for( triindex = tinit->trifirst ; triindex != MO_TRINEXT_ENDOFLIST ; triindex = trinext )
    {
      tri = &mesh->trilist[triindex];
#if MO_CONFIG_ATOMIC_SUPPORT
      trinext = mmAtomicRead32( &tri->atomictrinext );
#else
      trinext = tri->trinext;
#endif
      vertex = &mesh->vertexlist[tri->v[0]];
      if( vertex->redirectindex < 0 )
      {
        mesh->shufflecallback( mesh->shuffleopaquepointer, redirectindex, tri->v[0] );
        vertex->redirectindex = redirectindex++;
      }
      vertex = &mesh->vertexlist[tri->v[1]];
      if( vertex->redirectindex < 0 )
      {
        mesh->shufflecallback( mesh->shuffleopaquepointer, redirectindex, tri->v[1] );
        vertex->redirectindex = redirectindex++;
      }
      vertex = &mesh->vertexlist[tri->v[2]];
      if( vertex->redirectindex < 0 )
      {
        mesh->shufflecallback( mesh->shuffleopaquepointer, redirectindex, tri->v[2] );
        vertex->redirectindex = redirectindex++;
      }
    }
  }

  return;
}


static void moWriteRedirectIndices( moMesh *mesh, moThreadInit *threadinit )
{
  int threadindex;
  moi triindices[3];
  moi triindex, trinext;
  moThreadInit *tinit;
  moVertex *vertex;
  moTriangle *tri;
  void *indices;

  indices = mesh->indices;
  for( threadindex = 0 ; threadindex < mesh->threadcount ; threadindex++ )
  {
    tinit = &threadinit[threadindex];
    for( triindex = tinit->trifirst ; triindex !=  MO_TRINEXT_ENDOFLIST ; triindex = trinext )
    {
      tri = &mesh->trilist[triindex];
#if MO_CONFIG_ATOMIC_SUPPORT
      trinext = mmAtomicRead32( &tri->atomictrinext );
#else
      trinext = tri->trinext;
#endif
      vertex = &mesh->vertexlist[tri->v[0]];
      triindices[0] = vertex->redirectindex;
      vertex = &mesh->vertexlist[tri->v[1]];
      triindices[1] = vertex->redirectindex;
      vertex = &mesh->vertexlist[tri->v[2]];
      triindices[2] = vertex->redirectindex;
      mesh->indicesNativeToUser( indices, triindices );
      indices = ADDRESS( indices, mesh->indicesstride );
    }
  }

  return;
}


////


/* Initialize state to optimize the mesh specified */
moMesh *moMeshOptimizationInit( size_t vertexcount, size_t tricount, void *indices, int indiceswidth, size_t indicesstride, void (*shufflecallback)( void *opaquepointer, long newvertexindex, long oldvertexindex ), void *shuffleopaquepointer, int vertexcachesize, int threadcount, int flags )
{
  int a, maxthreadcount;
  mof factor, cachefactor;
  moMesh *mesh;

  if( tricount < 3 )
    return 0;
  if( threadcount <= 0 )
    return 0;

  mesh = malloc( sizeof(moMesh) );
  memset( mesh, 0, sizeof(moMesh) );
  maxthreadcount = tricount / MO_TRIANGLE_PER_THREAD_MINIMUM;
  if( threadcount <= 0 )
    threadcount = MO_THREAD_COUNT_DEFAULT;
  if( threadcount > maxthreadcount )
    threadcount = maxthreadcount;
  if( threadcount > MO_THREAD_COUNT_MAX )
    threadcount = MO_THREAD_COUNT_MAX;

  mesh->vertexcount = vertexcount;
  mesh->tricount = tricount;
  mesh->indices = indices;
  switch( indiceswidth )
  {
    case sizeof(uint8_t):
      mesh->indicesUserToNative = moIndicesInt8ToNative;
      mesh->indicesNativeToUser = moIndicesNativeToInt8;
      break;
    case sizeof(uint16_t):
      mesh->indicesUserToNative = moIndicesInt16ToNative;
      mesh->indicesNativeToUser = moIndicesNativeToInt16;
      break;
    case sizeof(uint32_t):
      mesh->indicesUserToNative = moIndicesInt32ToNative;
      mesh->indicesNativeToUser = moIndicesNativeToInt32;
      break;
    case sizeof(uint64_t):
      mesh->indicesUserToNative = moIndicesInt64ToNative;
      mesh->indicesNativeToUser = moIndicesNativeToInt64;
      break;
    default:
      free( mesh );
      return 0;
  }
  mesh->indicesstride = indicesstride;
  mesh->shuffleopaquepointer = shuffleopaquepointer;
  mesh->shufflecallback = shufflecallback;
  mesh->threadcount = threadcount;
  mesh->operationflags = flags;

  /* Synchronization */
  mtSleepBarrierInit( &mesh->workbarrier, threadcount );

  /* Finish status tracking */
  mesh->finishcount = threadcount;
  mtMutexInit( &mesh->finishmutex );
  mtSignalInit( &mesh->finishsignal );

#if MO_DEBUG
  printf( "Mesh Optimization ; %d vertices ; %d triangles\n", (int)vertexcount, (int)tricount );
#endif

  /* Score look-up tables */
  mesh->lookaheadfactor = 0.5f;
  if( vertexcachesize < 12 )
    vertexcachesize = 12;
  else if( vertexcachesize > MO_VERTEX_CACHE_SIZE_MAX-1 )
    vertexcachesize = MO_VERTEX_CACHE_SIZE_MAX-1;
  mesh->vertexcachesize = vertexcachesize;
  mesh->cachescore[0] = 0.85f;
  mesh->cachescore[1] = 0.85f;
  mesh->cachescore[2] = 0.85f;
  if( mesh->operationflags & MO_FLAGS_FIXED_CACHE_SIZE )
  {
    mesh->cachescore[0] = 0.75f;
    mesh->cachescore[1] = 0.75f;
    mesh->cachescore[2] = 0.75f;
  }
  factor = 1.0f / (mof)( mesh->vertexcachesize - 3 );
  if( mesh->operationflags & MO_FLAGS_FIXED_CACHE_SIZE )
    factor *= 0.25f;
  for( a = 3 ; a < mesh->vertexcachesize ; a++ )
  {
    cachefactor = 1.0f - ( (mof)(a-3) * factor );
    cachefactor = sqrtf( cachefactor * cachefactor * cachefactor );
    mesh->cachescore[a] = cachefactor;
  }
  mesh->cachescore[mesh->vertexcachesize] = 0.0f;
  if( mesh->operationflags & MO_FLAGS_FIXED_CACHE_SIZE )
  {
    if( mesh->vertexcachesize > 1 )
      mesh->cachescore[mesh->vertexcachesize-1] *= 0.50f;
    if( mesh->vertexcachesize > 2 )
      mesh->cachescore[mesh->vertexcachesize-2] *= 0.75f;
  }
  mesh->trirefscore[0] = -256.0f;
  for( a = 1 ; a < MO_TRIREFSCORE_COUNT ; a++ )
    mesh->trirefscore[a] = 2.0f / sqrtf( (float)a );

  /* Allocation */
  mesh->vertexlist = malloc( mesh->vertexcount * sizeof(moVertex) );
  mesh->trilist = malloc( mesh->tricount * sizeof(moTriangle) );
  mesh->trirefcount = 0;
  mesh->trireflist = malloc( 3 * mesh->tricount * sizeof(moi) );

  return mesh;
}

/* Perform the work for specified thread, must be called synchronously for all threadcount (they wait for each other) */
void moMeshOptimizationThread( moMesh *mesh, int threadindex )
{
  moThreadInit *tinit;
  if( (unsigned)threadindex < mesh->threadcount )
  {
    tinit = &mesh->threadinit[threadindex];
    tinit->threadid = threadindex;
    tinit->mesh = mesh;
    moThreadMain( tinit );
  }
  return;
}

/* Wait until the work has completed */
void moMeshOptimizationEnd( moMesh *mesh )
{
  /* Wait for all threads to be done */
  mtMutexLock( &mesh->finishmutex );
  while( mesh->finishcount )
    mtSignalWait( &mesh->finishsignal, &mesh->finishmutex );
  mtMutexUnlock( &mesh->finishmutex );

  /* Read the linked list of each thread and rebuild the new indices */
  if( !( mesh->shufflecallback ) )
    moWriteIndices( mesh, mesh->threadinit );
  else
  {
    moBuildRedirection( mesh, mesh->threadinit );
    moWriteRedirectIndices( mesh, mesh->threadinit );
  }

  /* Free all global data */
  free( mesh->vertexlist );
  free( mesh->trilist );
  free( mesh->trireflist );
  mtSleepBarrierDestroy( &mesh->workbarrier );
  mtMutexDestroy( &mesh->finishmutex );
  mtSignalDestroy( &mesh->finishsignal );
  free( mesh );

  return;
}



////


typedef struct
{
  moMesh *mesh;
  int threadindex;
} moThreadLaunch;

static void *moThreadLaunchMain( void *value )
{
  moThreadLaunch *threadlaunch;
  threadlaunch = (moThreadLaunch *)value;
  moMeshOptimizationThread( threadlaunch->mesh, threadlaunch->threadindex );
  return 0;
}

int moOptimizeMesh( size_t vertexcount, size_t tricount, void *indices, int indiceswidth, size_t indicesstride, void (*shufflecallback)( void *opaquepointer, long newvertexindex, long oldvertexindex ), void *shuffleopaquepointer, int vertexcachesize, int threadcount, int flags )
{
  int threadindex, maxthreadcount;
  moMesh *mesh;
  mtThread thread[MO_THREAD_COUNT_MAX];
  moThreadLaunch threadlaunch[MO_THREAD_COUNT_MAX];

#if MO_DEBUG_TIME
  long long msecs;
  msecs = ccGetMillisecondsTime();
#endif

  maxthreadcount = tricount / MO_TRIANGLE_PER_THREAD_MINIMUM;
  if( threadcount <= 0 )
    threadcount = MO_THREAD_COUNT_DEFAULT;
  if( threadcount > maxthreadcount )
    threadcount = maxthreadcount;
  if( threadcount > MO_THREAD_COUNT_MAX )
    threadcount = MO_THREAD_COUNT_MAX;

  mesh = moMeshOptimizationInit( vertexcount, tricount, indices, indiceswidth, indicesstride, shufflecallback, shuffleopaquepointer, vertexcachesize, threadcount, flags );
  if( !mesh )
    return 0;

  if( threadcount >= 2 )
  {
    /* Launch threads! */
    for( threadindex = 0 ; threadindex < threadcount ; threadindex++ )
    {
      threadlaunch[threadindex].mesh = mesh;
      threadlaunch[threadindex].threadindex = threadindex;
      mtThreadCreate( &thread[threadindex], moThreadLaunchMain, &threadlaunch[threadindex], MT_THREAD_FLAGS_JOINABLE );
    }
    moMeshOptimizationEnd( mesh );
    for( threadindex = 0 ; threadindex < threadcount ; threadindex++ )
      mtThreadJoin( &thread[threadindex] );
  }
  else
  {
    moMeshOptimizationThread( mesh, 0 );
    moMeshOptimizationEnd( mesh );
  }

#if MO_DEBUG_TIME
  msecs = ccGetMillisecondsTime() - msecs;
  printf( "Mesh Optimization : %lld msecs\n", (long long)msecs );
#endif

  return 1;
}


////


#define MO_EVAL_VERTEX_CACHE_MAX (256)

static int moEvalCacheInsert( moi *vertexcache, int vertexcachesize, moi vertexindex )
{
  int cacheindex, pushlimit, cachemiss;
  pushlimit = vertexcachesize-1;
  cachemiss = 1;
  for( cacheindex = 0 ; cacheindex < vertexcachesize ; cacheindex++ )
  {
    if( vertexcache[cacheindex] != vertexindex )
      continue;
    pushlimit = cacheindex;
    cachemiss = 0;
    break;
  }
  for( cacheindex = pushlimit-1 ; cacheindex >= 0 ; cacheindex-- )
    vertexcache[cacheindex+1] = vertexcache[cacheindex+0];
  vertexcache[0] = vertexindex;
  return cachemiss;
}

/*
Returns the ACMR (Average Cache Miss Rate) for the mesh.
ACMR is the sum of vertex cache miss divided by the number of triangles in the mesh. 
*/
double moEvaluateMesh( size_t tricount, void *indices, int indiceswidth, size_t indicesstride, int vertexcachesize, int flags )
{
  int cacheindex, cachemiss;
  moi triindex;
  void (*indicesUserToNative)( moi *dst, void *src );
  moi vertexcache[MO_EVAL_VERTEX_CACHE_MAX];
  moi triv[3];

  switch( indiceswidth )
  {
    case sizeof(uint8_t):
      indicesUserToNative = moIndicesInt8ToNative;
      break;
    case sizeof(uint16_t):
      indicesUserToNative = moIndicesInt16ToNative;
      break;
    case sizeof(uint32_t):
      indicesUserToNative = moIndicesInt32ToNative;
      break;
    case sizeof(uint64_t):
      indicesUserToNative = moIndicesInt64ToNative;
      break;
    default:
      return 0;
  }

  if( vertexcachesize > MO_EVAL_VERTEX_CACHE_MAX )
    vertexcachesize = MO_EVAL_VERTEX_CACHE_MAX;
  for( cacheindex = 0 ; cacheindex < vertexcachesize ; cacheindex++ )
    vertexcache[cacheindex] = -1;

  cachemiss = 0;
  for( triindex = 0 ; triindex < tricount ; triindex++ )
  {
    indicesUserToNative( triv, indices );
    indices = ADDRESS( indices, indicesstride );
    cachemiss += moEvalCacheInsert( vertexcache, vertexcachesize, triv[0] );
    cachemiss += moEvalCacheInsert( vertexcache, vertexcachesize, triv[1] );
    cachemiss += moEvalCacheInsert( vertexcache, vertexcachesize, triv[2] );
  }

#if 0
  printf( "Cache Miss : %d / %d\n", cachemiss, (int)tricount*3 );
  printf( "ACMR : %f\n", (double)cachemiss / (double)( 3 * tricount ) );
#endif

  return (double)cachemiss / (double)( 3 * tricount );
}

