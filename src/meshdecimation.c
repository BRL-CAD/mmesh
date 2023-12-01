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

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include "cc.h"
#include "cchash.h"

#include "mmcore.h"
#include "mm.h"
#include "mmthread.h"
#include "mmatomic.h"
#include "mmhash.h"

#include "mmbinsort.h"
#include "meshdecimation.h"


////


#if __SSE__ || _M_X64 || _M_IX86_FP >= 1  || CPU_ENABLE_SSE
 #include <xmmintrin.h>
 #define CPU_SSE_SUPPORT (1)
#endif
#if __SSE2__ || _M_X64 || _M_IX86_FP >= 2  || CPU_ENABLE_SSE2
 #include <emmintrin.h>
 #define CPU_SSE2_SUPPORT (1)
#endif
#if __SSE3__ || __AVX__ || CPU_ENABLE_SSE3
 #include <pmmintrin.h>
 #define CPU_SSE3_SUPPORT (1)
#endif
#if __SSSE3__ || __AVX__  || CPU_ENABLE_SSSE3
 #include <tmmintrin.h>
 #define CPU_SSSE3_SUPPORT (1)
#endif
#if __SSE4_1__ || __AVX__  || CPU_ENABLE_SSE4_1
 #include <smmintrin.h>
 #define CPU_SSE4_1_SUPPORT (1)
#endif


#if defined(__GNUC__) || defined(__INTEL_COMPILER)
 #define CPU_ALIGN16 __attribute__((aligned(16)))
 #define CPU_ALIGN32 __attribute__((aligned(32)))
 #define CPU_ALIGN64 __attribute__((aligned(64)))
#elif defined(_MSC_VER)
 #define CPU_ALIGN16 __declspec(align(16))
 #define CPU_ALIGN64 __declspec(align(64))
#else
 #define CPU_ALIGN16
 #define CPU_ALIGN32
 #define CPU_ALIGN64
 #warning "SSE/AVX Disabled: Unsupported Compiler."
 #undef CPU_SSE_SUPPORT
 #undef CPU_SSE2_SUPPORT
 #undef CPU_SSE3_SUPPORT
 #undef CPU_SSSE3_SUPPORT
 #undef CPU_SSE4_1_SUPPORT
#endif


////


/* Define to use double floating point precision */
#define MD_CONF_DOUBLE_PRECISION (1)

/* Define to use double floating point precision */
#define MD_CONF_USE_SHEWCHUK_SUMMATION (1)

/* Define to use double precision quadric maths. Very strongly recommended. */
#define MD_CONF_QUADRICS_DOUBLE_PRECISION (1)

/* Enable progress report callback */
#define MD_CONF_ENABLE_PROGRESS (1)

/* Align all ops on 64 bytes to reduce cache line fetches */
#define MD_CONF_OP_ALIGNMENT (0x40)

/* Pack the mdEdge struct to save 4 bytes, only enabled on platforms with safe misaligned access ~ slower, but better than swap if constrained in memory! */
#define MD_CONF_PACKED_EDGE_STRUCT (1)

/* Track a per-vertex frame of reference for quadrics */
/* Greatly improves the numerical accuracy when the dataset is gigantic and highly accurate results are expected */
#define MD_CONF_LOCAL_VERTEX_ORIGINS (1)

/* Try to use some crazy __float128 precision to track the d^2 accumulated error, if available */
/* Not needed anymore thanks to local quadric origins */
#define MD_CONFIG_HIGH_QUADRICS (0)

/* Use approximations for some unimportant calculations, single precision only */
#define MD_CONFIG_APPROX_MATH (0)

/* Fix some pathological numerical robustness cases */
/* TODO: Check if that's still required with local quadric origins? */
#if 1
 #define MD_CONFIG_DISTANCE_BIAS (1)
#else
 #define MD_CONFIG_DISTANCE_BIAS (0)
#endif

/* How many bytes to use for vertex indices */
#define MD_SIZEOF_MDI (4)



#define MD_COLLAPSE_COST_COMPACTNESS_TARGET (0.25)
#define MD_COLLAPSE_COST_COMPACTNESS_FACTOR (1.0)

#define MD_BOUNDARY_WEIGHT (4.0)

#define MD_SYNC_STEP_COUNT (64)

#define MD_QUADRIC_DETERMINANT_MIN (0.0000000001)

#define MD_GLOBAL_LOCK_THRESHOLD (16)

#define MD_THREAD_COUNT_DEFAULT (16)

#define MD_THREAD_COUNT_MAX (256)

#define MD_TRIREF_AVAIL_MIN_COUNT (256*8)

#define MD_OP_FAIL_VALUE (0.25*FLT_MAX)

#define MD_COLINEAR_REJECTION (0.0000001)

/* Maximum factor for any edgeexpand or boundaryedgeexpand factor (also avoids any division by false zero) */
#define MD_EXPAND_FACTOR_CLAMP (65536.0)


////


#if 1
 #define DEBUG_VERBOSE_WORK (1)
 #define DEBUG_VERBOSE_QUADRIC (0)
 #define DEBUG_VERBOSE_BOUNDARY (0)
 #define DEBUG_VERBOSE_TOPOLOGY (0)
 #define DEBUG_VERBOSE_COLLISION (0)
 #define DEBUG_VERBOSE_COST (0)
 #define DEBUG_VERBOSE_COLLAPSE (0)
 #define DEBUG_VERBOSE_MEMORY (0+1)
 #define DEBUG_VERBOSE_OUTPUT (0)
 #define DEBUG_VERBOSE_CHECKS (0)
#elif 0
 #define DEBUG_VERBOSE_WORK (0)
 #define DEBUG_VERBOSE_QUADRIC (0+1)
 #define DEBUG_VERBOSE_BOUNDARY (0+1)
 #define DEBUG_VERBOSE_TOPOLOGY (1)
 #define DEBUG_VERBOSE_COLLISION (0)
 #define DEBUG_VERBOSE_COST (0+2)
 #define DEBUG_VERBOSE_COLLAPSE (0+1)
 #define DEBUG_VERBOSE_MEMORY (0)
 #define DEBUG_VERBOSE_OUTPUT (0)
 #define DEBUG_VERBOSE_CHECKS (0)
#elif 1
 #define DEBUG_VERBOSE_WORK (0)
 #define DEBUG_VERBOSE_QUADRIC (1)
 #define DEBUG_VERBOSE_BOUNDARY (1)
 #define DEBUG_VERBOSE_TOPOLOGY (1)
 #define DEBUG_VERBOSE_COLLISION (1)
 #define DEBUG_VERBOSE_COST (2)
 #define DEBUG_VERBOSE_COLLAPSE (1)
 #define DEBUG_VERBOSE_MEMORY (1)
 #define DEBUG_VERBOSE_OUTPUT (1)
 #define DEBUG_VERBOSE_CHECKS (1)
#else
 #define DEBUG_VERBOSE_WORK (0)
 #define DEBUG_VERBOSE_QUADRIC (1)
 #define DEBUG_VERBOSE_BOUNDARY (1)
 #define DEBUG_VERBOSE_TOPOLOGY (1)
 #define DEBUG_VERBOSE_COLLISION (1)
 #define DEBUG_VERBOSE_COST (1)
 #define DEBUG_VERBOSE_COLLAPSE (1)
 #define DEBUG_VERBOSE_MEMORY (0)
 #define DEBUG_VERBOSE_OUTPUT (0)
 #define DEBUG_VERBOSE_CHECKS (1)
#endif


/* Investigate weird sync issue? */
#define DEBUG_DEBUG_CHECK_SOMETHING (0)


#if DEBUG_VERBOSE || DEBUG_VERBOSE_COLLISION || DEBUG_VERBOSE_COLLAPSE
 #if 0
  #define MD_ERROR(s,f,...) ({fprintf(stderr,s,__VA_ARGS__);if(f) exit(1);})
 #else
  #define MD_ERROR(s,f,...) ({fprintf(stdout,s,__VA_ARGS__);fflush(stdout);})
 #endif
#else
 #define MD_ERROR(s,f,...) ({fprintf(stderr,s,__VA_ARGS__);})
#endif


////


#if MM_ATOMIC_SUPPORT
 #define MD_CONFIG_ATOMIC_SUPPORT (1)
#endif


////


#if MD_CONF_DOUBLE_PRECISION
typedef double mdf;
 #define mdfmin(x,y) fmin((x),(y))
 #define mdfmax(x,y) fmax((x),(y))
 #define mdffloor(x) floor(x)
 #define mdfceil(x) ceil(x)
 #define mdfround(x) round(x)
 #define mdfsqrt(x) sqrt(x)
 #define mdfcbrt(x) cbrt(x)
 #define mdfabs(x) fabs(x)
 #define mdflog2(x) log2(x)
 #define mdfacos(x) acos(x)
#else
typedef float mdf;
 #define mdfmin(x,y) fminf((x),(y))
 #define mdfmax(x,y) fmaxf((x),(y))
 #define mdffloor(x) floorf(x)
 #define mdfceil(x) ceilf(x)
 #define mdfround(x) roundf(x)
 #define mdfsqrt(x) sqrtf(x)
 #define mdfcbrt(x) cbrtf(x)
 #define mdfabs(x) fabsf(x)
 #define mdflog2(x) log2f(x)
 #define mdfacos(x) acosf(x)
#endif

#if MD_CONF_DOUBLE_PRECISION
 #if !MD_CONF_QUADRICS_DOUBLE_PRECISION
  #define MD_CONF_QUADRICS_DOUBLE_PRECISION (1)
 #endif
#endif

#if MD_CONF_QUADRICS_DOUBLE_PRECISION
typedef double mdqf;
#else
typedef float mdqf;
#endif

#if MD_CONFIG_HIGH_QUADRICS
 #if ( __GNUC__ > 4 || ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 ) ) && !defined(__INTEL_COMPILER) && ( defined(__i386__) || defined(__x86_64__) || defined(__ia64__) )
typedef __float128 mdqfhigh;
 #else
  #undef MD_CONFIG_HIGH_QUADRICS
typedef double mdqfhigh;
 #endif
#elif MD_CONF_QUADRICS_DOUBLE_PRECISION
typedef double mdqfhigh;
#else
typedef float mdqfhigh;
#endif

#if MD_SIZEOF_MDI == 8
typedef int64_t mdi;
#elif MD_SIZEOF_MDI == 4
typedef int32_t mdi;
#else
 #error MD_SIZEOF_MDI must be 4 or 8
#endif

#if !defined(__GNUC__) || defined(__clang__)
 /* "mathshewchuk.h" only supports GCC, clang doesn't support __attribute__((__optimize__(""))) syntax */
 #undef MD_CONF_USE_SHEWCHUK_SUMMATION
#endif

#if MD_CONF_USE_SHEWCHUK_SUMMATION
 #include "mathshewchuk.h"
#endif

#if MD_CONF_PACKED_EDGE_STRUCT
 #define MD_PACKED_EDGE_STRUCT MM_PACKED_SAFE
#else
 #define MD_PACKED_EDGE_STRUCT
#endif


////


#ifndef M_PI
 #define M_PI (3.14159265358979323846264338327)
#endif

#define MD_VectorMagnitude(x) (mdfsqrt((x)[0]*(x)[0]+(x)[1]*(x)[1]+(x)[2]*(x)[2]))
#define MD_VectorDotProduct(x,y) ((x)[0]*(y)[0]+(x)[1]*(y)[1]+(x)[2]*(y)[2])

#if !defined(_MSC_VER)
 #define MD_VectorSubStore(x,y,z) ({(x)[0]=(y)[0]-(z)[0];(x)[1]=(y)[1]-(z)[1];(x)[2]=(y)[2]-(z)[2];})
 #define MD_VectorMulScalar(x,y) ({(x)[0]*=(y);(x)[1]*=(y);(x)[2]*=(y);})
 #define MD_VectorAddMulScalar(x,y,z) ({(x)[0]+=(y)[0]*(z);(x)[1]+=(y)[1]*(z);(x)[2]+=(y)[2]*(z);})
 #define MD_VectorCrossProduct(x,y,z) ({(x)[0]=((y)[1]*(z)[2])-((y)[2]*(z)[1]);(x)[1]=((y)[2]*(z)[0])-((y)[0]*(z)[2]);(x)[2]=((y)[0]*(z)[1])-((y)[1]*(z)[0]);})
 #define MD_VectorCopy(x,y) ({(x)[0]=(y)[0];(x)[1]=(y)[1];(x)[2]=(y)[2];})
 #define MD_VectorNormalize(x) ({mdf _f;_f=1.0/sqrt((x)[0]*(x)[0]+(x)[1]*(x)[1]+(x)[2]*(x)[2]);(x)[0]*=_f;(x)[1]*=_f;(x)[2]*=_f;})
 #define MD_VectorZero(x) ({(x)[0]=0.0;(x)[1]=0.0;(x)[2]=0.0;})
#else
/* MSVC doesn't like the #define syntax */
static inline void MD_VectorSubStore(mdf *x, mdf *y, mdf *z) {(x)[0]=(y)[0]-(z)[0];(x)[1]=(y)[1]-(z)[1];(x)[2]=(y)[2]-(z)[2];}
static inline void MD_VectorMulScalar(mdf *x,mdf y) {(x)[0]*=(y);(x)[1]*=(y);(x)[2]*=(y);}
static inline void MD_VectorAddMulScalar(mdf *x,mdf *y,mdf z) {(x)[0]+=(y)[0]*(z);(x)[1]+=(y)[1]*(z);(x)[2]+=(y)[2]*(z);}
static inline void MD_VectorCrossProduct(mdf *x,mdf *y,mdf *z) {(x)[0]=((y)[1]*(z)[2])-((y)[2]*(z)[1]);(x)[1]=((y)[2]*(z)[0])-((y)[0]*(z)[2]);(x)[2]=((y)[0]*(z)[1])-((y)[1]*(z)[0]);}
static inline void MD_VectorCopy(mdf *x,mdf *y) {(x)[0]=(y)[0];(x)[1]=(y)[1];(x)[2]=(y)[2];}
static inline void MD_VectorNormalize(mdf *x) {mdf _f;_f=1.0/sqrt((x)[0]*(x)[0]+(x)[1]*(x)[1]+(x)[2]*(x)[2]);(x)[0]*=_f;(x)[1]*=_f;(x)[2]*=_f;}
static inline void MD_VectorZero(mdf *x) {(x)[0]=0.0;(x)[1]=0.0;(x)[2]=0.0;}
#endif


////


typedef struct
{
  mdqf area;
  mdqf a2, ab, ac, ad;
  mdqf b2, bc, bd;
  mdqf c2, cd;
  mdqfhigh d2;
} mathQuadric;

static inline void mathQuadricInit( mathQuadric *q, mdqf a, mdqf b, mdqf c, mdqf d, mdqf area )
{
  q->area = area;
  q->a2 = a * a;
  q->ab = a * b;
  q->ac = a * c;
  q->ad = a * d;
  q->b2 = b * b;
  q->bc = b * c;
  q->bd = b * d;
  q->c2 = c * c;
  q->cd = c * d;
  q->d2 = (mdqfhigh)d * (mdqfhigh)d;
#if DEBUG_VERBOSE_QUADRIC
  printf( "    Q Init %e ; %e %e %e %e %e %e %e %e %e %e\n", (double)q->area, (double)q->a2, (double)q->ab, (double)q->ac, (double)q->ad, (double)q->b2, (double)q->bc, (double)q->bd, (double)q->c2, (double)q->cd, (double)q->d2 );
#endif
  return;
}


////


static inline mdqf mathMatrix3x3Determinant( mdqf *m )
{
  mdqf det;
  det  = m[0*3+0] * ( ( m[2*3+2] * m[1*3+1] ) - ( m[2*3+1] * m[1*3+2] ) );
  det -= m[1*3+0] * ( ( m[2*3+2] * m[0*3+1] ) - ( m[2*3+1] * m[0*3+2] ) );
  det += m[2*3+0] * ( ( m[1*3+2] * m[0*3+1] ) - ( m[1*3+1] * m[0*3+2] ) );
  return det;
}

static inline void mathMatrix3x3MulVector( mdqf *vdst, mdqf *m, mdqf *v )
{
  vdst[0] = ( v[0] * m[0*3+0] ) + ( v[1] * m[1*3+0] ) + ( v[2] * m[2*3+0] );
  vdst[1] = ( v[0] * m[0*3+1] ) + ( v[1] * m[1*3+1] ) + ( v[2] * m[2*3+1] );
  vdst[2] = ( v[0] * m[0*3+2] ) + ( v[1] * m[1*3+2] ) + ( v[2] * m[2*3+2] );
  return;
}

static inline void mathQuadricToMatrix3x3( mdqf *m, mathQuadric *q )
{
  m[0*3+0] = q->a2;
  m[0*3+1] = q->ab;
  m[0*3+2] = q->ac;
  m[1*3+0] = q->ab;
  m[1*3+1] = q->b2;
  m[1*3+2] = q->bc;
  m[2*3+0] = q->ac;
  m[2*3+1] = q->bc;
  m[2*3+2] = q->c2;
  return;
}

static inline void mathMatrix3x3Invert( mdqf *mdst, mdqf *m, mdqf det )
{
  mdqf detinv;
  detinv = 1.0 / det;
  mdst[0*3+0] =  ( ( m[2*3+2] * m[1*3+1] ) - ( m[2*3+1] * m[1*3+2] ) ) * detinv;
  mdst[0*3+1] = -( ( m[2*3+2] * m[0*3+1] ) - ( m[2*3+1] * m[0*3+2] ) ) * detinv;
  mdst[0*3+2] =  ( ( m[1*3+2] * m[0*3+1] ) - ( m[1*3+1] * m[0*3+2] ) ) * detinv;
  mdst[1*3+0] = -( ( m[2*3+2] * m[1*3+0] ) - ( m[2*3+0] * m[1*3+2] ) ) * detinv;
  mdst[1*3+1] =  ( ( m[2*3+2] * m[0*3+0] ) - ( m[2*3+0] * m[0*3+2] ) ) * detinv;
  mdst[1*3+2] = -( ( m[1*3+2] * m[0*3+0] ) - ( m[1*3+0] * m[0*3+2] ) ) * detinv;
  mdst[2*3+0] =  ( ( m[2*3+1] * m[1*3+0] ) - ( m[2*3+0] * m[1*3+1] ) ) * detinv;
  mdst[2*3+1] = -( ( m[2*3+1] * m[0*3+0] ) - ( m[2*3+0] * m[0*3+1] ) ) * detinv;
  mdst[2*3+2] =  ( ( m[1*3+1] * m[0*3+0] ) - ( m[1*3+0] * m[0*3+1] ) ) * detinv;
  return;
}


////


static inline int mathQuadricSolve( mathQuadric *q, mdf *v )
{
  mdqf det, m[9], minv[9], vector[3], vres[3];
  mdqf areascale;
  mathQuadricToMatrix3x3( m, q );
  det = mathMatrix3x3Determinant( m );
/*
Det scale diagnotic
Scale 10x
  QAdd Sr0 1.120477 ; 0.022271 0.030416 -0.065258 -1.217629 0.042473 -0.090558 -1.704271 0.196298 3.621082 68.438599
  QAdd Sr0 112.048033 ; 222.705246 304.155192 -652.582478 -121762.889543 424.729952 -905.579057 -170427.095051 1962.994524 362109.207338 68438554.421304
  Det 0.000000440467
  Det 440472.329126436263
If scale is 10x, then:
  q0->area : 100x (10^2)
  q0->a2 : 10000x (10^4)
  q0->ab : 10000x (10^4)
  q0->ac : 10000x (10^4)
  q0->ad : 100000x (10^5)
  q0->b2 : 10000x (10^4)
  q0->bc : 10000x (10^4)
  q0->bd : 100000x (10^5)
  q0->c2 : 10000x (10^4)
  q0->cd : 100000x (10^5)
  q0->d2 : 1000000x (10^6)
  det : 1000000000000x (10^12)
*/
  areascale = q->area * q->area;
  areascale = areascale * areascale * areascale;
  if( mdfabs( det ) <= ( MD_QUADRIC_DETERMINANT_MIN * areascale ) )
  {
#if DEBUG_VERBOSE_QUADRIC
    printf( "        Solve Det : %.16f ; Fail (<= %.16f)\n", (double)det, MD_QUADRIC_DETERMINANT_MIN * areascale );
#endif
    return 0;
  }
#if DEBUG_VERBOSE_QUADRIC
  printf( "        Solve Det : %.16f ; Pass\n", (double)det );
#endif
  mathMatrix3x3Invert( minv, m, det );
  vector[0] = -q->ad;
  vector[1] = -q->bd;
  vector[2] = -q->cd;
  mathMatrix3x3MulVector( vres, minv, vector );
  v[0] = (mdf)vres[0];
  v[1] = (mdf)vres[1];
  v[2] = (mdf)vres[2];
#if DEBUG_VERBOSE_QUADRIC
  printf( "        Solve Vector ; %f %f %f ; %f %f %f ( Det %.16f )\n", (double)vector[0], (double)vector[1], (double)vector[2], (double)v[0], (double)v[1], (double)v[2], det );
#endif
  return 1;
}

static inline void mathQuadricZero( mathQuadric *q )
{
  q->area = 0.0;
  q->a2 = 0.0;
  q->ab = 0.0;
  q->ac = 0.0;
  q->ad = 0.0;
  q->b2 = 0.0;
  q->bc = 0.0;
  q->bd = 0.0;
  q->c2 = 0.0;
  q->cd = 0.0;
  q->d2 = 0.0;
  return;
}

static void mathQuadricTranslateStore( mathQuadric *dstq, mathQuadric *q, mdqf tx, mdqf ty, mdqf tz )
{
  mdqf d2accum;
  dstq->area = q->area;
  dstq->a2 = q->a2;
  dstq->ab = q->ab;
  dstq->ac = q->ac;
  dstq->ad = q->ad + ( tx * q->a2 ) + ( ty * q->ab ) + ( tz * q->ac );
  dstq->b2 = q->b2;
  dstq->bc = q->bc;
  dstq->bd = q->bd + ( tx * q->ab ) + ( ty * q->b2 ) + ( tz * q->bc );
  dstq->c2 = q->c2;
  dstq->cd = q->cd + ( tx * q->ac ) + ( ty * q->bc ) + ( tz * q->c2 );
  d2accum = ( tx*tx * q->a2 ) + ( ty*ty * q->b2 ) + ( tz*tz * q->c2 );
  d2accum += 2.0 * ( ( tx*ty * q->ab ) + ( tx*tz * q->ac ) + ( ty*tz * q->bc ) );
  d2accum -= 2.0 * ( ( tx * dstq->ad ) + ( ty * dstq->bd ) + ( tz * dstq->cd ) );
  dstq->d2 = q->d2 - d2accum;
  return;
}

static void mathQuadricTranslate( mathQuadric *q, mdqf tx, mdqf ty, mdqf tz )
{
  mdqf d2accum;
  q->ad = q->ad + ( tx * q->a2 ) + ( ty * q->ab ) + ( tz * q->ac );
  q->bd = q->bd + ( tx * q->ab ) + ( ty * q->b2 ) + ( tz * q->bc );
  q->cd = q->cd + ( tx * q->ac ) + ( ty * q->bc ) + ( tz * q->c2 );
  d2accum = ( tx*tx * q->a2 ) + ( ty*ty * q->b2 ) + ( tz*tz * q->c2 );
  d2accum += 2.0 * ( ( tx*ty * q->ab ) + ( tx*tz * q->ac ) + ( ty*tz * q->bc ) );
  d2accum -= 2.0 * ( ( tx * q->ad ) + ( ty * q->bd ) + ( tz * q->cd ) );
  q->d2 -= d2accum;
  return;
}

static inline void mathQuadricAddStoreQuadric( mathQuadric *qdst, mathQuadric *q0, mathQuadric *q1 )
{
#if DEBUG_VERBOSE_QUADRIC
  printf( "    QAdd Sr0 %e ; %e %e %e %e %e %e %e %e %e %e\n", (double)q0->area, (double)q0->a2, (double)q0->ab, (double)q0->ac, (double)q0->ad, (double)q0->b2, (double)q0->bc, (double)q0->bd, (double)q0->c2, (double)q0->cd, (double)q0->d2 );
  printf( "    QAdd Sr1 %e ; %e %e %e %e %e %e %e %e %e %e\n", (double)q1->area, (double)q1->a2, (double)q1->ab, (double)q1->ac, (double)q1->ad, (double)q1->b2, (double)q1->bc, (double)q1->bd, (double)q1->c2, (double)q1->cd, (double)q1->d2 );
#endif
  qdst->area = q0->area + q1->area;
  qdst->a2 = q0->a2 + q1->a2;
  qdst->ab = q0->ab + q1->ab;
  qdst->ac = q0->ac + q1->ac;
  qdst->ad = q0->ad + q1->ad;
  qdst->b2 = q0->b2 + q1->b2;
  qdst->bc = q0->bc + q1->bc;
  qdst->bd = q0->bd + q1->bd;
  qdst->c2 = q0->c2 + q1->c2;
  qdst->cd = q0->cd + q1->cd;
  qdst->d2 = q0->d2 + q1->d2;
#if DEBUG_VERBOSE_QUADRIC
  printf( "      QSum   %e ; %e %e %e %e %e %e %e %e %e %e\n", (double)qdst->area, (double)qdst->a2, (double)qdst->ab, (double)qdst->ac, (double)qdst->ad, (double)qdst->b2, (double)qdst->bc, (double)qdst->bd, (double)qdst->c2, (double)qdst->cd, (double)qdst->d2 );
#endif
  return;
}

static inline void mathQuadricAddQuadric( mathQuadric *qdst, mathQuadric *q )
{
#if DEBUG_VERBOSE_QUADRIC
  printf( "    QAdd Src %e ; %e %e %e %e %e %e %e %e %e %e\n", (double)q->area, (double)q->a2, (double)q->ab, (double)q->ac, (double)q->ad, (double)q->b2, (double)q->bc, (double)q->bd, (double)q->c2, (double)q->cd, (double)q->d2 );
  printf( "    QAdd Dst %e ; %e %e %e %e %e %e %e %e %e %e\n", (double)qdst->area, (double)qdst->a2, (double)qdst->ab, (double)qdst->ac, (double)qdst->ad, (double)qdst->b2, (double)qdst->bc, (double)qdst->bd, (double)qdst->c2, (double)qdst->cd, (double)qdst->d2 );
#endif
  qdst->area += q->area;
  qdst->a2 += q->a2;
  qdst->ab += q->ab;
  qdst->ac += q->ac;
  qdst->ad += q->ad;
  qdst->b2 += q->b2;
  qdst->bc += q->bc;
  qdst->bd += q->bd;
  qdst->c2 += q->c2;
  qdst->cd += q->cd;
  qdst->d2 += q->d2;
#if DEBUG_VERBOSE_QUADRIC
  printf( "      QSum   %e ; %e %e %e %e %e %e %e %e %e %e\n", (double)qdst->area, (double)qdst->a2, (double)qdst->ab, (double)qdst->ac, (double)qdst->ad, (double)qdst->b2, (double)qdst->bc, (double)qdst->bd, (double)qdst->c2, (double)qdst->cd, (double)qdst->d2 );
#endif
  return;
}



/* A volatile variable is used to force the compiler to do the math strictly in the order specified. */
static mdf mathQuadricEvaluate( mathQuadric *q, mdf *v )
{
  volatile mdqfhigh d;
  mdqfhigh vh[3];
  vh[0] = v[0];
  vh[1] = v[1];
  vh[2] = v[2];

#if MD_CONF_USE_SHEWCHUK_SUMMATION && !MD_CONFIG_HIGH_QUADRICS
  mathShewchukSum sum;
  mathShewchukInit( &sum );
  mathShewchukAdd( &sum, vh[0] * vh[1] * q->ab );
  mathShewchukAdd( &sum, vh[0] * vh[2] * q->ac );
  mathShewchukAdd( &sum, vh[1] * vh[2] * q->bc );
  mathShewchukAdd( &sum, vh[0] * q->ad );
  mathShewchukAdd( &sum, vh[1] * q->bd );
  mathShewchukAdd( &sum, vh[2] * q->cd );
  mathShewchukMultiply( &sum, 2.0 );
  mathShewchukAdd( &sum, vh[0] * vh[0] * q->a2 );
  mathShewchukAdd( &sum, vh[1] * vh[1] * q->b2 );
  mathShewchukAdd( &sum, vh[2] * vh[2] * q->c2 );
  mathShewchukAdd( &sum, q->d2 );
  d = mathShewchukTotal( &sum );
#else
  d = (mdqfhigh)( vh[0] * vh[0] * (mdqfhigh)q->a2 ) + (mdqfhigh)( vh[1] * vh[1] * (mdqfhigh)q->b2 ) + (mdqfhigh)( vh[2] * vh[2] * (mdqfhigh)q->c2 );
  d += (mdqfhigh)2.0 * ( (mdqfhigh)( vh[0] * vh[1] * (mdqfhigh)q->ab ) + (mdqfhigh)( vh[0] * vh[2] * (mdqfhigh)q->ac ) + (mdqfhigh)( vh[1] * vh[2] * (mdqfhigh)q->bc ) );
  d += (mdqfhigh)2.0 * ( (mdqfhigh)( vh[0] * (mdqfhigh)q->ad ) + (mdqfhigh)( vh[1] * (mdqfhigh)q->bd ) + (mdqfhigh)( vh[2] * (mdqfhigh)q->cd ) );
  d += (mdqfhigh)q->d2;
#endif

#if DEBUG_VERBOSE_QUADRIC >= 2
  printf( "        Q Eval %e ; %e %e %e %e %e %e %e %e %e %e : %e\n", (double)q->area, (double)q->a2, (double)q->ab, (double)q->ac, (double)q->ad, (double)q->b2, (double)q->bc, (double)q->bd, (double)q->c2, (double)q->cd, (double)q->d2, (double)d );
#endif
#if DEBUG_VERBOSE_QUADRIC >= 3
  volatile mdqfhigh debugd;
  printf( "          V  %e %e %e\n", v[0], v[1], v[2] );
  debugd = ( vh[0] * vh[0] * (mdqfhigh)q->a2 ) + ( vh[1] * vh[1] * (mdqfhigh)q->b2 ) + ( vh[2] * vh[2] * (mdqfhigh)q->c2 );
  printf( "          Qa %e + %e + %e = %e\n", vh[0] * vh[0] * (mdqfhigh)q->a2, vh[1] * vh[1] * (mdqfhigh)q->b2, vh[2] * vh[2] * (mdqfhigh)q->c2, (double)debugd );
  debugd = 2.0 * ( ( vh[0] * vh[1] * (mdqfhigh)q->ab ) + ( vh[0] * vh[2] * (mdqfhigh)q->ac ) + ( vh[1] * vh[2] * (mdqfhigh)q->bc ) );
  printf( "          Qb %e + %e + %e = %e\n", vh[0] * vh[1] * (mdqfhigh)q->ab, vh[0] * vh[2] * (mdqfhigh)q->ac, vh[1] * vh[2] * (mdqfhigh)q->bc, (double)debugd );
  debugd = 2.0 * ( ( vh[0] * (mdqfhigh)q->ad ) + ( vh[1] * (mdqfhigh)q->bd ) + ( vh[2] * (mdqfhigh)q->cd ) );
  printf( "          Qc %e + %e + %e = %e\n", vh[0] * (mdqfhigh)q->ad, vh[1] * (mdqfhigh)q->bd, vh[2] * (mdqfhigh)q->cd, (double)debugd );
  debugd = q->d2;
  printf( "          Qd %e = %e\n", q->d2, (double)debugd );
  printf( "           = %e\n", d );
#endif
  return (mdf)d;
}


//////


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
} mdBarrier;

#define MD_BARRIER_LOCK_READY(barrier) (((barrier)->count[(barrier)->index])-(((barrier)->lockcount))==1)

static void mdBarrierInit( mdBarrier *barrier, int count )
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

static void mdBarrierDestroy( mdBarrier *barrier )
{
  mtMutexDestroy( &barrier->mutex );
  mtSignalDestroy( &barrier->signal );
  mtSignalDestroy( &barrier->locksignal );
  mtSignalDestroy( &barrier->lockwakesignal );
  return;
}

static int mdBarrierSync( mdBarrier *barrier )
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
      if( MD_BARRIER_LOCK_READY(barrier) )
        mtSignalBroadcast( &barrier->locksignal );
    }
    for( ; barrier->count[index] ; )
      mtSignalWait( &barrier->signal, &barrier->mutex );
  }
  mtMutexUnlock( &barrier->mutex );
  return ret;
}

static int mdBarrierSyncTimeout( mdBarrier *barrier, long milliseconds )
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
      if( MD_BARRIER_LOCK_READY(barrier) )
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
static void mdBarrierCheckGlobal( mdBarrier *barrier )
{
  if( barrier->lockflag )
  {
    mtMutexLock( &barrier->mutex );
    if( barrier->lockflag )
    {
      barrier->lockcount++;
      if( MD_BARRIER_LOCK_READY(barrier) )
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
static void mdBarrierLockGlobal( mdBarrier *barrier )
{
  mtMutexLock( &barrier->mutex );
  while( barrier->lockflag )
  {
    barrier->lockcount++;
    if( MD_BARRIER_LOCK_READY(barrier) )
      mtSignalBroadcast( &barrier->locksignal );
    for( ; barrier->lockflag ; )
      mtSignalWait( &barrier->lockwakesignal, &barrier->mutex );
    barrier->lockcount--;
  }
  barrier->lockflag = 1;
  while( !MD_BARRIER_LOCK_READY(barrier) )
    mtSignalWait( &barrier->locksignal, &barrier->mutex );
  mtMutexUnlock( &barrier->mutex );
  return;
}

/* Release global lock */
static void mdBarrierUnlockGlobal( mdBarrier *barrier )
{
  mtMutexLock( &barrier->mutex );
  barrier->lockflag = 0;
  mtSignalBroadcast( &barrier->lockwakesignal );
  mtMutexUnlock( &barrier->mutex );
  return;
}


//////


/* 16 bytes ; OK ~ Custom tridata packed after it, kept aligned on 8 bytes */
typedef struct
{
  mdi v[3];
  union
  {
    int edgeflags;
    mdi redirectindex;
  } u;
} mdTriangle;

#define MD_EDGEFLAGS_BOUNDARY01 (0x1)
#define MD_EDGEFLAGS_BOUNDARY12 (0x2)
#define MD_EDGEFLAGS_BOUNDARY20 (0x4)
#define MD_EDGEFLAGS_DENYEDGE01 (0x10)
#define MD_EDGEFLAGS_DENYEDGE12 (0x20)
#define MD_EDGEFLAGS_DENYEDGE20 (0x40)

typedef struct MD_PACKED_EDGE_STRUCT
{
  mdi v[2];
  mdi triindex;
  void *op;
} mdEdge;

/* Double precision storage: 48 + 88 bytes (mathQuadric) = 136 bytes */
#if CPU_SSE_SUPPORT && !MD_CONF_DOUBLE_PRECISION
typedef struct CPU_ALIGN16
#else
typedef struct
#endif
{
#if CPU_SSE_SUPPORT && !MD_CONF_DOUBLE_PRECISION
  mdf CPU_ALIGN16 point[4];
#else
  mdf point[3];
#endif
  size_t trirefbase;
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomic32 atomicowner;
#else
  int owner;
  mtSpin ownerspinlock;
#endif
  mdi trirefcount;
  mdi redirectindex;
#if MD_CONFIG_DISTANCE_BIAS
  mdf sumbias;
#endif
  mathQuadric quadric;
} mdVertex;


typedef struct CPU_ALIGN64
{
  void **opbuffer;
  int opcount;
  int opalloc;
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomic32 atomlock;
#else
  mtSpin spinlock;
#endif
} mdUpdateBuffer;


typedef struct
{
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomic32 flags;
#else
  int flags;
  mtSpin spinlock;
#endif
  mdUpdateBuffer *updatebuffer;
  mdi v0, v1;
#if CPU_SSE_SUPPORT
  mdf CPU_ALIGN16 collapsepoint[4];
#else
  mdf collapsepoint[3];
#endif
  mdf collapsecost;
  mdf value;
  mdf penalty;
  mmListNode list;
} mdOp;

/* If detached, the op is not present in tdata->binsort */
#define MD_OP_FLAGS_DETACHED (0x1)
/* The parent edge was removed, the edge's op is scheduled to be deleted by the owner */
#define MD_OP_FLAGS_DELETION_PENDING (0x2)
/* The op is present in a updatebuffer->opbuffer */
#define MD_OP_FLAGS_UPDATE_QUEUED (0x4)
/* The op needs an mdUpdateOp() due to neighborhood changes */
#define MD_OP_FLAGS_UPDATE_NEEDED (0x8)
/* The op is dead, don't touch it */
#define MD_OP_FLAGS_DELETED (0x10)


typedef struct
{
  int threadcount;
  uint32_t operationflags;
  int updatestatusflag;

  /* User supplied raw data */
  void *point;
  size_t pointstride;
  void *indices;
  size_t indicesstride;
  void *tridata;
  size_t tridatasize;
  void (*indicesUserToNative)( mdi *dst, void *src );
  void (*indicesNativeToUser)( void *dst, mdi *src );
  void (*vertexUserToNative)( mdf *dst, void *src, mdf factor );
  void (*vertexNativeToUser)( void *dst, mdf *src, mdf factor );
  double (*edgeweight)( void *tridata0, void *tridata1 );
  double (*collapsemultiplier)( void *collapsecontext, void *tridata0, void *tridata1, double *point0, double *point1 );
  void *collapsecontext;
  void (*vertexmerge)( void *mergecontext, int dstindex, int srcindex, double dstfactor, double srcfactor );
  void *mergecontext;
  int (*adjustcollapse)( void *adjustcontext, mdf *collapsepoint, mdf *v0point, mdf *v1point );
  void *adjustcontext;
  void (*vertexcopy)( void *copycontext, int dstindex, int srcindex );
  void *copycontext;
  void (*writenormal)( void *dst, mdf *src );

  /* Per-vertex triangle references */
  mdi *trireflist;
  size_t trireflistcount;
  size_t trireflistalloc;
  char paddingA[64];
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomic32 trireflock;
#else
  mtSpin trirefspinlock;
#endif
  char paddingB[64];

  /* Synchronization stuff */
  mdBarrier workbarrier;
  int updatebuffercount;
  int updatebuffershift;

  /* List of triangles */
  void *trilist;
  long tricount;
  long tripackcount;
  size_t trisize;

  /* List of vertices */
  mdVertex *vertexlist;
  long vertexcount;
  long vertexalloc;
  long vertexpackcount;

  /* Hash table to locate edges from their vertex indices */
  void *edgehashtable;

  /* Collapse penalty function */
  mdf (*collapsepenalty)( mdf *newpoint, mdf *oldpoint, mdf *leftpoint, mdf *rightpoint, int *denyflag, mdf compactnesstarget, int meshflags );

  /* To compute vertex normals */
  void *normalbase;
  int normalformat;
  size_t normalstride;

  mdf invfeaturesizearea;
  /* Decimation strength, max cost */
  mdf maxcollapsecost;
  /* Maximum op accept cost, equal to maxcollapsecost, or FLT_MAX when targetvertexcountmax>0 */
  mdf maxcollapseacceptcost;
  /* Perpendicular expansion for boundaries */
  mdf boundaryareafactor;

  /* Vertex normalization factor, to avoid any overflow/underflow in the x^6 math when the featuresize is high/low */
  double normalizationfactor;
#if MD_CONFIG_DISTANCE_BIAS
  /* Feature size distance bias ~ penalty up to that distance for a pair of vertices */
  double biasclampdistance;
  /* Cost factor for distance bias, CONF_VALUE*maxcollapsecost/biasclampdistance */
  double biascostfactor;
#endif
  /* Target vertex count, stop when count<min, continue while count>max */
  long targetvertexcountmin;
  /* When targetvertexcountmax is enabled, _all_ ops are added to binsort queue */
  long targetvertexcountmax;

  char paddingC[64];
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomic32 globalvertexlock;
#else
  mtSpin globalvertexspinlock;
#endif
  char paddingD[64];

  /* Target vertex count tracking */
  char paddingE[64];
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicL trackvertexcount;
#else
  long trackvertexcount;
  mtSpin trackspinlock;
#endif
  char paddingF[64];

  /* Optional vertex locking map, can be null if not used */
  uint32_t *lockmap;

  /* Advanced configuration options */
  mdf compactnesstarget;
  mdf compactnesspenalty;
  mdf boundaryweight;
  mdf areaexpand;
  mdf boundaryedgeexpand;
  int syncstepcount;
  int syncstepabort;
  mdf normalsearchangle;

  /* Normal recomputation buffers */
  int clonesearchindex;
  void *vertexnormal;
  void *trinormal;

  /* Finish status tracking */
  int finishcount;
  mtMutex finishmutex;
  mtSignal finishsignal;

} mdMesh;


////


static void mdIndicesCharToNative( mdi *dst, void *src )
{
  unsigned char *s;
  s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void mdIndicesShortToNative( mdi *dst, void *src )
{
  unsigned short *s;
  s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void mdIndicesIntToNative( mdi *dst, void *src )
{
  unsigned int *s;
  s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void mdIndicesInt8ToNative( mdi *dst, void *src )
{
  uint8_t *s;
  s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void mdIndicesInt16ToNative( mdi *dst, void *src )
{
  uint16_t *s;
  s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void mdIndicesInt32ToNative( mdi *dst, void *src )
{
  uint32_t *s;
  s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}

static void mdIndicesInt64ToNative( mdi *dst, void *src )
{
  uint64_t *s;
  s = src;
  dst[0] = s[0];
  dst[1] = s[1];
  dst[2] = s[2];
  return;
}


static void mdIndicesNativeToChar( void *dst, mdi *src )
{
  unsigned char *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void mdIndicesNativeToShort( void *dst, mdi *src )
{
  unsigned short *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void mdIndicesNativeToInt( void *dst, mdi *src )
{
  unsigned int *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void mdIndicesNativeToInt8( void *dst, mdi *src )
{
  uint8_t *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void mdIndicesNativeToInt16( void *dst, mdi *src )
{
  uint16_t *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void mdIndicesNativeToInt32( void *dst, mdi *src )
{
  uint32_t *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void mdIndicesNativeToInt64( void *dst, mdi *src )
{
  uint64_t *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}


static void mdVertexFloatToNative( mdf *dst, void *src, mdf factor )
{
  float *s;
  s = src;
  dst[0] = s[0] * factor;
  dst[1] = s[1] * factor;
  dst[2] = s[2] * factor;
  return;
}

static void mdVertexDoubleToNative( mdf *dst, void *src, mdf factor )
{
  double *s;
  s = src;
  dst[0] = s[0] * factor;
  dst[1] = s[1] * factor;
  dst[2] = s[2] * factor;
  return;
}

static void mdVertexShortToNative( mdf *dst, void *src, mdf factor )
{
  short *s;
  s = src;
  dst[0] = s[0] * factor;
  dst[1] = s[1] * factor;
  dst[2] = s[2] * factor;
  return;
}

static void mdVertexIntToNative( mdf *dst, void *src, mdf factor )
{
  int *s;
  s = src;
  dst[0] = s[0] * factor;
  dst[1] = s[1] * factor;
  dst[2] = s[2] * factor;
  return;
}

static void mdVertexInt16ToNative( mdf *dst, void *src, mdf factor )
{
  int16_t *s;
  s = src;
  dst[0] = s[0] * factor;
  dst[1] = s[1] * factor;
  dst[2] = s[2] * factor;
  return;
}

static void mdVertexInt32ToNative( mdf *dst, void *src, mdf factor )
{
  int32_t *s;
  s = src;
  dst[0] = s[0] * factor;
  dst[1] = s[1] * factor;
  dst[2] = s[2] * factor;
  return;
}


static void mdVertexNativeToFloat( void *dst, mdf *src, mdf factor )
{
  float *d;
  d = dst;
  d[0] = src[0] * factor;
  d[1] = src[1] * factor;
  d[2] = src[2] * factor;
  return;
}

static void mdVertexNativeToDouble( void *dst, mdf *src, mdf factor )
{
  double *d;
  d = dst;
  d[0] = src[0] * factor;
  d[1] = src[1] * factor;
  d[2] = src[2] * factor;
  return;
}

static void mdVertexNativeToShort( void *dst, mdf *src, mdf factor )
{
  short *d;
  d = dst;
  d[0] = src[0] * factor;
  d[1] = src[1] * factor;
  d[2] = src[2] * factor;
  return;
}

static void mdVertexNativeToInt( void *dst, mdf *src, mdf factor )
{
  int *d;
  d = dst;
  d[0] = src[0] * factor;
  d[1] = src[1] * factor;
  d[2] = src[2] * factor;
  return;
}

static void mdVertexNativeToInt16( void *dst, mdf *src, mdf factor )
{
  int16_t *d;
  d = dst;
  d[0] = src[0] * factor;
  d[1] = src[1] * factor;
  d[2] = src[2] * factor;
  return;
}

static void mdVertexNativeToInt32( void *dst, mdf *src, mdf factor )
{
  int32_t *d;
  d = dst;
  d[0] = src[0] * factor;
  d[1] = src[1] * factor;
  d[2] = src[2] * factor;
  return;
}


static void mdNormalNativeToFloat( void *dst, mdf *src )
{
  float *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void mdNormalNativeToDouble( void *dst, mdf *src )
{
  double *d;
  d = dst;
  d[0] = src[0];
  d[1] = src[1];
  d[2] = src[2];
  return;
}

static void mdNormalNativeToChar( void *dst, mdf *src )
{
  mdf v;
  char *d;
  d = dst;
  v = mdfmin( 1.0, mdfmax( -1.0, src[0] ) );
  if( v > 0.0 )
    d[0] = (char)( ( v * (mdf)((1<<CHAR_BIT)-1) ) + 0.5 );
  else
    d[0] = (char)( ( v * (mdf)(1<<CHAR_BIT) ) - 0.5 );
  v = mdfmin( 1.0, mdfmax( -1.0, src[1] ) );
  if( v > 0.0 )
    d[1] = (char)( ( v * (mdf)((1<<CHAR_BIT)-1) ) + 0.5 );
  else
    d[1] = (char)( ( v * (mdf)(1<<CHAR_BIT) ) - 0.5 );
  v = mdfmin( 1.0, mdfmax( -1.0, src[2] ) );
  if( v > 0.0 )
    d[2] = (char)( ( v * (mdf)((1<<CHAR_BIT)-1) ) + 0.5 );
  else
    d[2] = (char)( ( v * (mdf)(1<<CHAR_BIT) ) - 0.5 );
  return;
}

static void mdNormalNativeToShort( void *dst, mdf *src )
{
  mdf v;
  short *d;
  d = dst;
  v = mdfmin( 1.0, mdfmax( -1.0, src[0] ) );
  if( v > 0.0 )
    d[0] = (short)( ( v * (mdf)(((1<<CHAR_BIT)*sizeof(short))-1) ) + 0.5 );
  else
    d[0] = (short)( ( v * (mdf)((1<<CHAR_BIT)*sizeof(short)) ) - 0.5 );
  v = mdfmin( 1.0, mdfmax( -1.0, src[1] ) );
  if( v > 0.0 )
    d[1] = (short)( ( v * (mdf)(((1<<CHAR_BIT)*sizeof(short))-1) ) + 0.5 );
  else
    d[1] = (short)( ( v * (mdf)((1<<CHAR_BIT)*sizeof(short)) ) - 0.5 );
  v = mdfmin( 1.0, mdfmax( -1.0, src[2] ) );
  if( v > 0.0 )
    d[2] = (short)( ( v * (mdf)(((1<<CHAR_BIT)*sizeof(short))-1) ) + 0.5 );
  else
    d[2] = (short)( ( v * (mdf)((1<<CHAR_BIT)*sizeof(short)) ) - 0.5 );
  return;
}

static void mdNormalNativeToInt8( void *dst, mdf *src )
{
  mdf v;
  int8_t *d;
  d = dst;
  v = mdfmin( 1.0, mdfmax( -1.0, src[0] ) );
  if( v > 0.0 )
    d[0] = (int8_t)( ( v * 127.0 ) + 0.5 );
  else
    d[0] = (int8_t)( ( v * 128.0 ) - 0.5 );
  v = mdfmin( 1.0, mdfmax( -1.0, src[1] ) );
  if( v > 0.0 )
    d[1] = (int8_t)( ( v * 127.0 ) + 0.5 );
  else
    d[1] = (int8_t)( ( v * 128.0 ) - 0.5 );
  v = mdfmin( 1.0, mdfmax( -1.0, src[2] ) );
  if( v > 0.0 )
    d[2] = (int8_t)( ( v * 127.0 ) + 0.5 );
  else
    d[2] = (int8_t)( ( v * 128.0 ) - 0.5 );
  return;
}

static void mdNormalNativeToInt16( void *dst, mdf *src )
{
  mdf v;
  int16_t *d;
  d = dst;
  v = mdfmin( 1.0, mdfmax( -1.0, src[0] ) );
  if( v > 0.0 )
    d[0] = (int8_t)( ( v * 32767.0 ) + 0.5 );
  else
    d[0] = (int8_t)( ( v * 32768.0 ) - 0.5 );
  v = mdfmin( 1.0, mdfmax( -1.0, src[1] ) );
  if( v > 0.0 )
    d[1] = (int8_t)( ( v * 32767.0 ) + 0.5 );
  else
    d[1] = (int8_t)( ( v * 32768.0 ) - 0.5 );
  v = mdfmin( 1.0, mdfmax( -1.0, src[2] ) );
  if( v > 0.0 )
    d[2] = (int8_t)( ( v * 32767.0 ) + 0.5 );
  else
    d[2] = (int8_t)( ( v * 32768.0 ) - 0.5 );
  return;
}

static void mdNormalNativeTo10_10_10_2( void *dst,  mdf *src )
{
  int part;
  uint32_t sum;
  mdf v;
  sum = 0;
  v = mdfmin( 1.0, mdfmax( -1.0, src[0] ) );
  if( v > 0.0 )
    part = (int)( ( v * 511.0 ) + 0.5 );
  else
    part = (int)( ( v * 512.0 ) - 0.5 );
  sum |= ( (uint32_t)part & 1023 ) << 0;
  v = mdfmin( 1.0, mdfmax( -1.0, src[1] ) );
  if( v > 0.0 )
    part = (int)( ( v * 511.0 ) + 0.5 );
  else
    part = (int)( ( v * 512.0 ) - 0.5 );
  sum |= ( (uint32_t)part & 1023 ) << 10;
  v = mdfmin( 1.0, mdfmax( -1.0, src[2] ) );
  if( v > 0.0 )
    part = (int)( ( v * 511.0 ) + 0.5 );
  else
    part = (int)( ( v * 512.0 ) - 0.5 );
  sum |= ( (uint32_t)part & 1023 ) << 20;
#if 0
  v = mdfmin( 1.0, mdfmax( -1.0, src[3] ) );
  if( v > 0.0 )
    part = (int)( ( v * 1.0 ) + 0.5 );
  else
    part = (int)( ( v * 2.0 ) - 0.5 );
  sum |= ( (uint32_t)part & 3 ) << 30;
#endif
  *(uint32_t *)dst = sum;
  return;
}


////


static void mdTriangleComputeQuadric( mdMesh *mesh, mdTriangle *tri, mathQuadric *q )
{
  mdf area, vecta[3], vectb[3], plane[4], expandfactor;
  mdVertex *vertex0, *vertex1, *vertex2;

  vertex0 = &mesh->vertexlist[ tri->v[0] ];
  vertex1 = &mesh->vertexlist[ tri->v[1] ];
  vertex2 = &mesh->vertexlist[ tri->v[2] ];
  MD_VectorSubStore( vecta, vertex1->point, vertex0->point );
  MD_VectorSubStore( vectb, vertex2->point, vertex0->point );
  MD_VectorCrossProduct( plane, vectb, vecta );
  area = mdfsqrt( MD_VectorDotProduct( plane, plane ) );
  if( area )
  {
    if( mesh->areaexpand < ( MD_EXPAND_FACTOR_CLAMP * area ) )
      expandfactor = 1.0 + ( mesh->areaexpand / area );
    else
      expandfactor = 1.0 + MD_EXPAND_FACTOR_CLAMP;
    expandfactor *= 0.5;
    area *= expandfactor;
    plane[0] *= expandfactor;
    plane[1] *= expandfactor;
    plane[2] *= expandfactor;
    plane[3] = -MD_VectorDotProduct( plane, vertex0->point );
  }
  else
  {
    area = 0.0;
    plane[0] = 0.0;
    plane[1] = 0.0;
    plane[2] = 0.0;
    plane[3] = 0.0;
  }
#if DEBUG_VERBOSE_QUADRIC
  printf( "  Plane %f %f %f %f ( %e %e %e ) : Area %f\n", plane[0], plane[1], plane[2], plane[3], plane[0], plane[1], plane[2], area );
#endif
  mathQuadricInit( q, plane[0], plane[1], plane[2], plane[3], area );
  return;
}

/* Compute a quadric with a localorigin that belongs on the triangle's plane ~ plane[3] is just zero, q->ad,bd,cd,dd are zeroes */
static void mdTriangleComputeLocalQuadric( mdMesh *mesh, mdTriangle *tri, mathQuadric *q )
{
  mdf area, vecta[3], vectb[3], plane[4], expandfactor;
  mdVertex *vertex0, *vertex1, *vertex2;

  vertex0 = &mesh->vertexlist[ tri->v[0] ];
  vertex1 = &mesh->vertexlist[ tri->v[1] ];
  vertex2 = &mesh->vertexlist[ tri->v[2] ];
  MD_VectorSubStore( vecta, vertex1->point, vertex0->point );
  MD_VectorSubStore( vectb, vertex2->point, vertex0->point );
  MD_VectorCrossProduct( plane, vectb, vecta );
  area = mdfsqrt( MD_VectorDotProduct( plane, plane ) );
  plane[3] = 0.0;
  if( area )
  {
    if( mesh->areaexpand < ( MD_EXPAND_FACTOR_CLAMP * area ) )
      expandfactor = 1.0 + ( mesh->areaexpand / area );
    else
      expandfactor = 1.0 + MD_EXPAND_FACTOR_CLAMP;
    expandfactor *= 0.5;
    area *= expandfactor;
    plane[0] *= expandfactor;
    plane[1] *= expandfactor;
    plane[2] *= expandfactor;
  }
  else
  {
    area = 0.0;
    plane[0] = 0.0;
    plane[1] = 0.0;
    plane[2] = 0.0;
  }
#if DEBUG_VERBOSE_QUADRIC
  printf( "  Plane %f %f %f %f ( %e %e %e ) : Area %f\n", plane[0], plane[1], plane[2], plane[3], plane[0], plane[1], plane[2], area );
#endif
  mathQuadricInit( q, plane[0], plane[1], plane[2], plane[3], area );
  return;
}


////


#define MD_POINT_SOLVE_FLAGS_V0 (0x1)
#define MD_POINT_SOLVE_FLAGS_V1 (0x2)
#define MD_POINT_SOLVE_FLAGS_MIDPOINT (0x4)
#define MD_POINT_SOLVE_FLAGS_QUADRIC (0x8)



#if MD_CONF_LOCAL_VERTEX_ORIGINS

static mdf mdEdgeSolvePoint( mdVertex *vertex0, mdVertex *vertex1, mdf *point, int solveflags )
{
  mdf cost, bestcost;
  mdf trypoint[3];
  mathQuadric q;

  /* Translate v1->q into v0's frame of reference */
  mathQuadricTranslateStore( &q, &vertex1->quadric, vertex0->point[0] - vertex1->point[0], vertex0->point[1] - vertex1->point[1], vertex0->point[2] - vertex1->point[2] );
  mathQuadricAddQuadric( &q, &vertex0->quadric );
  bestcost = MD_OP_FAIL_VALUE;

  if( solveflags & MD_POINT_SOLVE_FLAGS_QUADRIC )
  {
    if( mathQuadricSolve( &q, trypoint ) )
    {
      /* In _theory_, solving the quadric should always provide the optimal cost solution */
      /* In practice, rare floating point cases can screw things up, so we compare against the midpoint for safety */
      cost = mathQuadricEvaluate( &q, trypoint );
#if DEBUG_VERBOSE_QUADRIC
      printf( "        QuadricEvalCost %f %f %f ; cost %.16f\n", trypoint[0] + vertex0->point[0], trypoint[1] + vertex0->point[1], trypoint[2] + vertex0->point[2], cost );
#endif
      if( cost < bestcost )
      {
        bestcost = cost;
        point[0] = trypoint[0] + vertex0->point[0];
        point[1] = trypoint[1] + vertex0->point[1];
        point[2] = trypoint[2] + vertex0->point[2];
      }
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_MIDPOINT )
  {
    trypoint[0] = 0.5 * vertex1->point[0];
    trypoint[1] = 0.5 * vertex1->point[1];
    trypoint[2] = 0.5 * vertex1->point[2];
    cost = mathQuadricEvaluate( &q, trypoint );
    if( cost < bestcost )
    {
      bestcost = cost;
      point[0] = trypoint[0] + vertex0->point[0];
      point[1] = trypoint[1] + vertex0->point[0];
      point[2] = trypoint[2] + vertex0->point[0];
    }
#if DEBUG_VERBOSE_QUADRIC
    printf( "        MidCost %f %f %f : cost %.16f\n", trypoint[0] + vertex0->point[0], trypoint[1] + vertex0->point[1], trypoint[2] + vertex0->point[2], cost );
#endif
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_V0 )
  {
    trypoint[0] = 0.0;
    trypoint[1] = 0.0;
    trypoint[2] = 0.0;
    cost = mathQuadricEvaluate( &q, trypoint );
#if DEBUG_VERBOSE_QUADRIC
    printf( "        Vx0Cost %f %f %f : %.16f\n", vertex0->point[0], vertex0->point[1], vertex0->point[2], cost );
#endif
    if( cost < bestcost )
    {
      bestcost = cost;
      point[0] = vertex0->point[0];
      point[1] = vertex0->point[1];
      point[2] = vertex0->point[2];
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_V1 )
  {
    trypoint[0] = vertex1->point[0] - vertex0->point[0];
    trypoint[1] = vertex1->point[1] - vertex0->point[1];
    trypoint[2] = vertex1->point[2] - vertex0->point[2];
    cost = mathQuadricEvaluate( &q, trypoint );
#if DEBUG_VERBOSE_QUADRIC
    printf( "        Vx1Cost %f %f %f : %.16f\n", vertex1->point[0], vertex1->point[1], vertex1->point[2], cost );
#endif
    if( cost < bestcost )
    {
      bestcost = cost;
      point[0] = vertex1->point[0];
      point[1] = vertex1->point[1];
      point[2] = vertex1->point[2];
    }
  }

  return bestcost;
}

static mdf mdEdgeSolvePointAdjust( mdVertex *vertex0, mdVertex *vertex1, mdf *point, int solveflags, int (*adjustcollapse)( void *adjustcontext, mdf *collapsepoint, mdf *v0point, mdf *v1point ), void *adjustcontext )
{
  mdf cost, bestcost;
  mdf trypoint[3], localpoint[3];
  mathQuadric q;

  /* Translate v1->q into v0's frame of reference */
  mathQuadricTranslateStore( &q, &vertex1->quadric, vertex0->point[0] - vertex1->point[0], vertex0->point[1] - vertex1->point[1], vertex0->point[2] - vertex1->point[2] );
  mathQuadricAddQuadric( &q, &vertex0->quadric );
  bestcost = MD_OP_FAIL_VALUE;

  if( solveflags & MD_POINT_SOLVE_FLAGS_QUADRIC )
  {
    if( mathQuadricSolve( &q, localpoint ) )
    {
      trypoint[0] = localpoint[0] + vertex0->point[0];
      trypoint[1] = localpoint[1] + vertex0->point[1];
      trypoint[2] = localpoint[2] + vertex0->point[2];
      if( adjustcollapse( adjustcontext, trypoint, vertex0->point, vertex1->point ) )
      {
        localpoint[0] = trypoint[0] - vertex0->point[0];
        localpoint[1] = trypoint[1] - vertex0->point[1];
        localpoint[2] = trypoint[2] - vertex0->point[2];
        /* In _theory_, solving the quadric should always provide the optimal cost solution */
        /* In practice, rare floating point cases can screw things up, so we compare against the midpoint for safety */
        cost = mathQuadricEvaluate( &q, localpoint );
#if DEBUG_VERBOSE_QUADRIC
        printf( "        QuadricEvalCost %f %f %f ; cost %.16f\n", trypoint[0], trypoint[1], trypoint[2], cost );
#endif
        if( cost < bestcost )
        {
          bestcost = cost;
          point[0] = trypoint[0];
          point[1] = trypoint[1];
          point[2] = trypoint[2];
        }
      }
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_MIDPOINT )
  {
    trypoint[0] = 0.5 * ( vertex0->point[0] + vertex1->point[0] );
    trypoint[1] = 0.5 * ( vertex0->point[1] + vertex1->point[1] );
    trypoint[2] = 0.5 * ( vertex0->point[2] + vertex1->point[2] );
    if( adjustcollapse( adjustcontext, trypoint, vertex0->point, vertex1->point ) )
    {
      localpoint[0] = trypoint[0] - vertex0->point[0];
      localpoint[1] = trypoint[1] - vertex0->point[1];
      localpoint[2] = trypoint[2] - vertex0->point[2];
      cost = mathQuadricEvaluate( &q, localpoint );
      if( cost < bestcost )
      {
        bestcost = cost;
        point[0] = trypoint[0];
        point[1] = trypoint[1];
        point[2] = trypoint[2];
      }
#if DEBUG_VERBOSE_QUADRIC
      printf( "        MidCost %f %f %f : cost %.16f\n", trypoint[0], trypoint[1], trypoint[2], cost );
#endif
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_V0 )
  {
    trypoint[0] = vertex0->point[0];
    trypoint[1] = vertex0->point[1];
    trypoint[2] = vertex0->point[2];
    if( adjustcollapse( adjustcontext, trypoint, vertex0->point, vertex1->point ) )
    {
      localpoint[0] = trypoint[0] - vertex0->point[0];
      localpoint[1] = trypoint[1] - vertex0->point[1];
      localpoint[2] = trypoint[2] - vertex0->point[2];
      cost = mathQuadricEvaluate( &q, localpoint );
#if DEBUG_VERBOSE_QUADRIC
      printf( "        Vx0Cost %f %f %f : %.16f\n", vertex0->point[0], vertex0->point[1], vertex0->point[2], cost );
#endif
      if( cost < bestcost )
      {
        bestcost = cost;
        point[0] = trypoint[0];
        point[1] = trypoint[1];
        point[2] = trypoint[2];
      }
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_V1 )
  {
    trypoint[0] = vertex1->point[0];
    trypoint[1] = vertex1->point[1];
    trypoint[2] = vertex1->point[2];
    if( adjustcollapse( adjustcontext, trypoint, vertex0->point, vertex1->point ) )
    {
      localpoint[0] = trypoint[0] - vertex0->point[0];
      localpoint[1] = trypoint[1] - vertex0->point[1];
      localpoint[2] = trypoint[2] - vertex0->point[2];
      cost = mathQuadricEvaluate( &q, localpoint );
#if DEBUG_VERBOSE_QUADRIC
      printf( "        Vx1Cost %f %f %f : %.16f\n", vertex1->point[0], vertex1->point[1], vertex1->point[2], cost );
#endif
      if( cost < bestcost )
      {
        bestcost = cost;
        point[0] = trypoint[0];
        point[1] = trypoint[1];
        point[2] = trypoint[2];
      }
    }
  }

  return bestcost;
}

#else

static mdf mdEdgeSolvePoint( mdVertex *vertex0, mdVertex *vertex1, mdf *point, int solveflags )
{
  mdf cost, bestcost;
  mdf trypoint[3];
  mathQuadric q;

  mathQuadricAddStoreQuadric( &q, &vertex0->quadric, &vertex1->quadric );
  bestcost = MD_OP_FAIL_VALUE;

  if( solveflags & MD_POINT_SOLVE_FLAGS_QUADRIC )
  {
    if( mathQuadricSolve( &q, trypoint ) )
    {
      /* In _theory_, solving the quadric should always provide the optimal cost solution */
      /* In practice, rare floating point cases can screw things up, so we compare against the midpoint for safety */
      cost = mathQuadricEvaluate( &q, trypoint );
#if DEBUG_VERBOSE_QUADRIC
      printf( "        QuadricEvalCost %f %f %f ; cost %.16f\n", trypoint[0], trypoint[1], trypoint[2], cost );
#endif
      if( cost < bestcost )
      {
        bestcost = cost;
        point[0] = trypoint[0];
        point[1] = trypoint[1];
        point[2] = trypoint[2];
      }
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_MIDPOINT )
  {
    trypoint[0] = 0.5 * ( vertex0->point[0] + vertex1->point[0] );
    trypoint[1] = 0.5 * ( vertex0->point[1] + vertex1->point[1] );
    trypoint[2] = 0.5 * ( vertex0->point[2] + vertex1->point[2] );
    cost = mathQuadricEvaluate( &q, trypoint );
    if( cost < bestcost )
    {
      bestcost = cost;
      point[0] = trypoint[0];
      point[1] = trypoint[1];
      point[2] = trypoint[2];
    }
#if DEBUG_VERBOSE_QUADRIC
    printf( "        MidCost %f %f %f : cost %.16f\n", trypoint[0], trypoint[1], trypoint[2], cost );
#endif
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_V0 )
  {
    cost = mathQuadricEvaluate( &q, vertex0->point );
#if DEBUG_VERBOSE_QUADRIC
    printf( "        Vx0Cost %f %f %f : %.16f\n", vertex0->point[0], vertex0->point[1], vertex0->point[2], cost );
#endif
    if( cost < bestcost )
    {
      bestcost = cost;
      point[0] = vertex0->point[0];
      point[1] = vertex0->point[1];
      point[2] = vertex0->point[2];
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_V1 )
  {
    cost = mathQuadricEvaluate( &q, vertex1->point );
#if DEBUG_VERBOSE_QUADRIC
    printf( "        Vx1Cost %f %f %f : %.16f\n", vertex1->point[0], vertex1->point[1], vertex1->point[2], cost );
#endif
    if( cost < bestcost )
    {
      bestcost = cost;
      point[0] = vertex1->point[0];
      point[1] = vertex1->point[1];
      point[2] = vertex1->point[2];
    }
  }

  return bestcost;
}

static mdf mdEdgeSolvePointAdjust( mdVertex *vertex0, mdVertex *vertex1, mdf *point, int solveflags, int (*adjustcollapse)( void *adjustcontext, mdf *collapsepoint, mdf *v0point, mdf *v1point ), void *adjustcontext )
{
  mdf cost, bestcost;
  mdf trypoint[3];
  mathQuadric q;

  mathQuadricAddStoreQuadric( &q, &vertex0->quadric, &vertex1->quadric );
  bestcost = MD_OP_FAIL_VALUE;

  if( solveflags & MD_POINT_SOLVE_FLAGS_QUADRIC )
  {
    if( mathQuadricSolve( &q, trypoint ) )
    {
      if( adjustcollapse( adjustcontext, trypoint, vertex0->point, vertex1->point ) )
      {
        /* In _theory_, solving the quadric should always provide the optimal cost solution */
        /* In practice, rare floating point cases can screw things up, so we compare against the midpoint for safety */
        cost = mathQuadricEvaluate( &q, trypoint );
#if DEBUG_VERBOSE_QUADRIC
        printf( "        QuadricEvalCost %f %f %f ; cost %.16f\n", trypoint[0], trypoint[1], trypoint[2], cost );
#endif
        if( cost < bestcost )
        {
          bestcost = cost;
          point[0] = trypoint[0];
          point[1] = trypoint[1];
          point[2] = trypoint[2];
        }
      }
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_MIDPOINT )
  {
    trypoint[0] = 0.5 * ( vertex0->point[0] + vertex1->point[0] );
    trypoint[1] = 0.5 * ( vertex0->point[1] + vertex1->point[1] );
    trypoint[2] = 0.5 * ( vertex0->point[2] + vertex1->point[2] );
    if( adjustcollapse( adjustcontext, trypoint, vertex0->point, vertex1->point ) )
    {
      cost = mathQuadricEvaluate( &q, trypoint );
      if( cost < bestcost )
      {
        bestcost = cost;
        point[0] = trypoint[0];
        point[1] = trypoint[1];
        point[2] = trypoint[2];
      }
#if DEBUG_VERBOSE_QUADRIC
      printf( "        MidCost %f %f %f : cost %.16f\n", trypoint[0], trypoint[1], trypoint[2], cost );
#endif
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_V0 )
  {
    trypoint[0] = vertex0->point[0];
    trypoint[1] = vertex0->point[1];
    trypoint[2] = vertex0->point[2];
    if( adjustcollapse( adjustcontext, trypoint, vertex0->point, vertex1->point ) )
    {
      cost = mathQuadricEvaluate( &q, trypoint );
#if DEBUG_VERBOSE_QUADRIC
      printf( "        Vx0Cost %f %f %f : %.16f\n", vertex0->point[0], vertex0->point[1], vertex0->point[2], cost );
#endif
      if( cost < bestcost )
      {
        bestcost = cost;
        point[0] = trypoint[0];
        point[1] = trypoint[1];
        point[2] = trypoint[2];
      }
    }
  }
  if( solveflags & MD_POINT_SOLVE_FLAGS_V1 )
  {
    trypoint[0] = vertex1->point[0];
    trypoint[1] = vertex1->point[1];
    trypoint[2] = vertex1->point[2];
    if( adjustcollapse( adjustcontext, trypoint, vertex0->point, vertex1->point ) )
    {
      cost = mathQuadricEvaluate( &q, trypoint );
#if DEBUG_VERBOSE_QUADRIC
      printf( "        Vx1Cost %f %f %f : %.16f\n", vertex1->point[0], vertex1->point[1], vertex1->point[2], cost );
#endif
      if( cost < bestcost )
      {
        bestcost = cost;
        point[0] = trypoint[0];
        point[1] = trypoint[1];
        point[2] = trypoint[2];
      }
    }
  }

  return bestcost;
}

#endif



////


static void mdMeshAccumulateBoundary( mdVertex *vertex0, mdVertex *vertex1, mdVertex *vertex2, mdf boundaryareafactor, mdf boundaryedgeexpand )
{
  mdf normal[3], sideplane[4], vecta[3], vectb[3], length, expandfactor;
  mathQuadric q;

  MD_VectorSubStore( vecta, vertex1->point, vertex0->point );
  MD_VectorSubStore( vectb, vertex2->point, vertex0->point );
  MD_VectorCrossProduct( normal, vectb, vecta );
  length = MD_VectorMagnitude( vecta );

  if( ( length == 0.0 ) || ( MD_VectorMagnitude( normal ) == 0.0 ) )
    return;

  MD_VectorNormalize( normal );
#if DEBUG_VERBOSE_BOUNDARY
  printf( "  Normal %f %f %f\n", normal[0], normal[1], normal[2] );
#endif
  if( boundaryedgeexpand > 0.0 )
  {
    if( boundaryedgeexpand < ( MD_EXPAND_FACTOR_CLAMP * length ) )
      expandfactor = 1.0 + ( boundaryedgeexpand / length );
    else
      expandfactor = 1.0 + MD_EXPAND_FACTOR_CLAMP;
#if DEBUG_VERBOSE_BOUNDARY
    printf( "  Expandfactor %f\n", expandfactor );
#endif
    MD_VectorMulScalar( vecta, expandfactor );
    length += boundaryedgeexpand;
  }
  MD_VectorCrossProduct( sideplane, vecta, normal );
  MD_VectorMulScalar( sideplane, boundaryareafactor );
#if MD_CONF_LOCAL_VERTEX_ORIGINS
  /* If we are tracking a local origin, the boundary's plane for a vertex always passes by that origin ~ d is zero */
  sideplane[3] = 0.0;
#else
  sideplane[3] = -MD_VectorDotProduct( sideplane, vertex0->point );
#endif

#if DEBUG_VERBOSE_BOUNDARY
  printf( "  Boundary expand %f\n", boundaryareafactor );
  printf( "  Boundary plane %f %f %f %f : Length %f\n", sideplane[0], sideplane[1], sideplane[2], sideplane[3], length );
#endif
  mathQuadricInit( &q, sideplane[0], sideplane[1], sideplane[2], sideplane[3], length * boundaryareafactor );

#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicSpin32( &vertex0->atomicowner, -1, 0xffff );
  mathQuadricAddQuadric( &vertex0->quadric, &q );
  mmAtomicWrite32( &vertex0->atomicowner, -1 );
#else
  mtSpinLock( &vertex0->ownerspinlock );
  mathQuadricAddQuadric( &vertex0->quadric, &q );
  mtSpinUnlock( &vertex0->ownerspinlock );
#endif
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicSpin32( &vertex1->atomicowner, -1, 0xffff );
  mathQuadricAddQuadric( &vertex1->quadric, &q );
  mmAtomicWrite32( &vertex1->atomicowner, -1 );
#else
  mtSpinLock( &vertex1->ownerspinlock );
  mathQuadricAddQuadric( &vertex1->quadric, &q );
  mtSpinUnlock( &vertex1->ownerspinlock );
#endif

  return;
}


////


static void mdEdgeHashClearEntry( void *context, void *entry )
{
  mdEdge *edge;
  edge = entry;
  edge->v[0] = -1;
  return;
}

static int mdEdgeHashEntryValid( void *context, void *entry )
{
  mdEdge *edge;
  edge = entry;
  return ( edge->v[0] >= 0 ? 1 : 0 );
}

static mmHashIndex mdEdgeHashEntryKey( void *context, void *entry )
{
  mmHashIndex hashkey;
  mdEdge *edge;
  edge = entry;
#if MD_SIZEOF_MDI == 4
 #if MM_HASH_INDEX_64_BITS
  hashkey = ccHash64Int32x2( edge->v[0], edge->v[1] );
 #elif
  hashkey = ccHash32Int32x2( edge->v[0], edge->v[1] );
 #endif
#elif MD_SIZEOF_MDI == 8
 #if MM_HASH_INDEX_64_BITS
  hashkey = ccHash64Int64x2( edge->v[0], edge->v[1] );
 #elif
  hashkey = (mmHashIndex)ccHash64Int64x2( edge->v[0], edge->v[1] );
 #endif
#else
 #if MM_HASH_INDEX_64_BITS
  hashkey = ccHash64Data( edge->v, 2*sizeof(mdi) );
 #else
  hashkey = ccHash32Data( edge->v, 2*sizeof(mdi) );
 #endif
#endif
  return hashkey;
}

static int mdEdgeHashEntryCmp( void *context, void *entry, void *entryref )
{
  mdEdge *edge, *edgeref;
  edge = entry;
  edgeref = entryref;
  if( edge->v[0] == -1 )
    return MM_HASH_ENTRYCMP_INVALID;
  if( ( edge->v[0] == edgeref->v[0] ) && ( edge->v[1] == edgeref->v[1] ) )
    return MM_HASH_ENTRYCMP_FOUND;
  return MM_HASH_ENTRYCMP_SKIP;
}

static mmHashAccess mdEdgeHashAccess =
{
  .clearentry = mdEdgeHashClearEntry,
  .entryvalid = mdEdgeHashEntryValid,
  .entrykey = mdEdgeHashEntryKey,
  .entrycmp = mdEdgeHashEntryCmp
};

static int mdMeshHashInit( mdMesh *mesh, size_t trianglecount, mdf hashsizefactor, uint32_t lockpageshift, size_t maxmemoryusage )
{
  size_t edgecount, hashmemsize, meshmemsize, trirefmemsize, jobmemsize, basememsize, totalmemorysize;
  size_t hashsize;

  /* lockpageshift = 7; works great, 128 hash entries per lock page */
  if( lockpageshift < 3 )
    lockpageshift = 3;
  else if( lockpageshift > 16 )
    lockpageshift = 16;

  edgecount = trianglecount * 3;
  hashsizefactor = fmax( hashsizefactor, 1.1 );

  /* Memory usage for mesh vertices and indices */
  meshmemsize = ( mesh->tricount * mesh->trisize ) + ( mesh->vertexcount * sizeof(mdVertex) );
  /* Memory usage for trirefs */
  trirefmemsize = ( 2 * 6 * mesh->tricount ) * sizeof(mdi);
  /* Memory usage for job queue */
  jobmemsize = ( ( mesh->tricount * 3 ) >> 1 ) * sizeof(mdOp);
  /* Base fixed memory usage */
  basememsize = meshmemsize + trirefmemsize + jobmemsize;

  for( ; ; hashsizefactor -= 0.1 )
  {
    hashsize = (size_t)( edgecount * hashsizefactor );
    if( hashsize < 4096 )
      hashsize = 4096;

    /* Memory usage for edge hash table */
    hashmemsize = mmHashRequiredSize( sizeof(mdEdge), hashsize, lockpageshift );

    totalmemorysize = hashmemsize + basememsize;

    /* Increase estimate of memory consumption by 25% to account for extra stuff not counted here */
    totalmemorysize += totalmemorysize >> 2;

#if DEBUG_VERBOSE_MEMORY
    printf( "  Hash size : %lld (%lld)\n", (long long)hashsize, (long long)edgecount );
    printf( "    Estimated Memory Requirements : %lld bytes (%lld MB)\n", (long long)totalmemorysize, (long long)totalmemorysize >> 20 );
    printf( "    Memory Hard Limit : %lld bytes (%lld MB)\n", (long long)maxmemoryusage, (long long)maxmemoryusage >> 20 );
#endif

    if( ( maxmemoryusage ) && ( totalmemorysize > maxmemoryusage ) && ( hashsizefactor > 1.15 ) )
      continue;
    mesh->edgehashtable = malloc( hashmemsize );
    if( mesh->edgehashtable )
      break;
  }

  mmHashInit( mesh->edgehashtable, &mdEdgeHashAccess, sizeof(mdEdge), hashsize, lockpageshift, MM_HASH_FLAGS_NO_COUNT, 0 );

  return 1;
}

static void mdMeshHashEnd( mdMesh *mesh )
{
  free( mesh->edgehashtable );
  return;
}


////


/* If threadcount exceeds this number, updatebuffers will be shared by nearby cores */
#define MD_THREAD_UPDATE_BUFFER_COUNTMAX (8)

typedef struct CPU_ALIGN64
{
  int threadid;

  /* Memory block for ops */
  mmBlockHead opblock;

  /* Hierarchical bucket sort of ops */
  void *binsort;

  /* List of ops flagged by other threads in need of update */
  mdUpdateBuffer updatebuffer[MD_THREAD_UPDATE_BUFFER_COUNTMAX];

  /* Per-thread status trackers */
  volatile long statusbuildtricount;
  volatile long statusbuildrefcount;
  volatile long statuspopulatecount;
  volatile long statusdeletioncount;
  volatile long statuscollisioncount;

} mdThreadData;


static void mdUpdateBufferInit( mdUpdateBuffer *updatebuffer, int opalloc )
{
  updatebuffer->opbuffer = malloc( opalloc * sizeof(mdOp *) );
  updatebuffer->opcount = 0;
  updatebuffer->opalloc = opalloc;
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicWrite32( &updatebuffer->atomlock, 0x0 );
#else
  mtSpinInit( &updatebuffer->spinlock );
#endif
  return;
}

static void mdUpdateBufferEnd( mdUpdateBuffer *updatebuffer )
{
#ifndef MD_CONFIG_ATOMIC_SUPPORT
  mtSpinDestroy( &updatebuffer->spinlock );
#endif
  free( updatebuffer->opbuffer );
  return;
}

static void mdUpdateBufferAdd( mdUpdateBuffer *updatebuffer, mdOp *op, int orflags )
{
  int32_t flags;
#if MD_CONFIG_ATOMIC_SUPPORT
  for( ; ; )
  {
    flags = mmAtomicRead32( &op->flags );
    if( flags & MD_OP_FLAGS_UPDATE_QUEUED )
    {
      if( !( flags & MD_OP_FLAGS_UPDATE_NEEDED ) || ( orflags ) )
        mmAtomicOr32( &op->flags, orflags | MD_OP_FLAGS_UPDATE_NEEDED );
      return;
    }
    if( mmAtomicCmpReplace32( &op->flags, flags, flags | orflags | MD_OP_FLAGS_UPDATE_QUEUED | MD_OP_FLAGS_UPDATE_NEEDED ) )
      break;
  }
  /* TODO: Avoid spin lock, use atomic increment for write offset? Careful with this realloc, pointer could become invalid */
  mmAtomicSpin32( &updatebuffer->atomlock, 0x0, 0x1 );
  if( updatebuffer->opcount >= updatebuffer->opalloc )
  {
    updatebuffer->opalloc <<= 1;
    updatebuffer->opbuffer = realloc( updatebuffer->opbuffer, updatebuffer->opalloc * sizeof(mdOp *) );
  }
  updatebuffer->opbuffer[ updatebuffer->opcount++ ] = op;
  mmAtomicWrite32( &updatebuffer->atomlock, 0x0 );
#else
  mtSpinLock( &op->spinlock );
  op->flags |= orflags;
  flags = op->flags;
  if( flags & MD_OP_FLAGS_UPDATE_QUEUED )
  {
    if( !( flags & MD_OP_FLAGS_UPDATE_NEEDED ) )
      op->flags |= MD_OP_FLAGS_UPDATE_NEEDED;
    mtSpinUnlock( &op->spinlock );
    return;
  }
  op->flags |= MD_OP_FLAGS_UPDATE_QUEUED | MD_OP_FLAGS_UPDATE_NEEDED;
  mtSpinUnlock( &op->spinlock );
  mtSpinLock( &updatebuffer->spinlock );
  if( updatebuffer->opcount >= updatebuffer->opalloc )
  {
    updatebuffer->opalloc <<= 1;
    updatebuffer->opbuffer = realloc( updatebuffer->opbuffer, updatebuffer->opalloc * sizeof(mdOp *) );
  }
  updatebuffer->opbuffer[ updatebuffer->opcount++ ] = op;
  mtSpinUnlock( &updatebuffer->spinlock );
#endif
  return;
}



////



#define MD_COMPACTNESS_NORMALIZATION_FACTOR (0.5*4.0*1.732050808)

static mdf mdEdgeCollapsePenaltyTriangle( mdf *newpoint, mdf *oldpoint, mdf *leftpoint, mdf *rightpoint, int *denyflag, mdf compactnesstarget, int meshflags )
{
  mdf penalty, compactness, oldcompactness, newcompactness, vecta2, norm;
  mdf vecta[3], oldvectb[3], oldvectc[3], newvectb[3], newvectc[3], oldnormal[3], newnormal[3];
  mdf oldmagnitude, newmagnitude;

  /* Normal of old triangle */
  MD_VectorSubStore( vecta, rightpoint, leftpoint );
  MD_VectorSubStore( oldvectb, oldpoint, leftpoint );
  MD_VectorCrossProduct( oldnormal, vecta, oldvectb );
  /* Normal of new triangle */
  MD_VectorSubStore( newvectb, newpoint, leftpoint );
  MD_VectorCrossProduct( newnormal, vecta, newvectb );
  if( meshflags & MD_FLAGS_PLANAR_MODE )
  {
    /* Detect planar normal Z inversion */
    if( ( oldnormal[2] * newnormal[2] ) <= 0.0 )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal Z inversion denied in planar mode %f -> %f\n", oldnormal[2], newnormal[2] );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  else
  {
    /* Detect normal inversion */
    if( MD_VectorDotProduct( oldnormal, newnormal ) < 0.0 )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal inversion denied %f -> %f\n", oldnormal[2], newnormal[2] );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  /* Prevent near-zero area triangles */
  oldmagnitude = MD_VectorMagnitude( oldnormal );
  newmagnitude = MD_VectorMagnitude( newnormal );
#if DEBUG_VERBOSE_COST >= 2
  printf( "      Magnitude ; Old %e ; New %e\n", oldmagnitude, newmagnitude );
#endif
  if( !( newmagnitude > ( MD_COLINEAR_REJECTION * oldmagnitude ) ) )
  {
#if DEBUG_VERBOSE_COST >= 2
    printf( "      !! Colinear magnitude denied\n" );
#endif
    *denyflag = 1;
    return 0.0;
  }
  /* Penalize long thin triangles */
  penalty = 0.0;
  vecta2 = MD_VectorDotProduct( vecta, vecta );
  MD_VectorSubStore( newvectc, newpoint, rightpoint );
  newcompactness = MD_COMPACTNESS_NORMALIZATION_FACTOR * newmagnitude;
  norm = vecta2 + MD_VectorDotProduct( newvectb, newvectb ) + MD_VectorDotProduct( newvectc, newvectc );
  if( newcompactness < ( compactnesstarget * norm ) )
  {
    newcompactness /= norm;
    MD_VectorSubStore( oldvectc, oldpoint, rightpoint );
    oldcompactness = ( MD_COMPACTNESS_NORMALIZATION_FACTOR * oldmagnitude ) / ( vecta2 + MD_VectorDotProduct( oldvectb, oldvectb ) + MD_VectorDotProduct( oldvectc, oldvectc ) );
    compactness = fmin( compactnesstarget, oldcompactness ) - newcompactness;
#if DEBUG_VERBOSE_COST >= 2
    printf( "      Compactness ; Old %f ; New %f ; Diff %f\n", oldcompactness, newcompactness, compactness );
#endif
    penalty = fmaxf( penalty, compactness );
  }
  return penalty;
}

#if CPU_SSE4_1_SUPPORT

 #if !MD_CONF_DOUBLE_PRECISION

static float mdEdgeCollapsePenaltyTriangleSSE4p1f( float *newpoint, float *oldpoint, float *leftpoint, float *rightpoint, int *denyflag, float compactnesstarget, int meshflags )
{
  float penalty, compactness;
  __m128 left, vecta, oldvectb, oldvectc, newvectb, newvectc, oldnormal, newnormal;
  __m128 invcheck;
  __m128 norm, oldmagnitude, newmagnitude, oldcompactness, newcompactness;
  /* Normal of old triangle */
  left = _mm_load_ps( leftpoint );
  vecta = _mm_sub_ps( _mm_load_ps( rightpoint ), left );
  oldvectb = _mm_sub_ps( _mm_load_ps( oldpoint ), left );
  oldnormal = _mm_sub_ps(
    _mm_mul_ps( _mm_shuffle_ps( vecta, vecta, _MM_SHUFFLE(3,0,2,1) ), _mm_shuffle_ps( oldvectb, oldvectb, _MM_SHUFFLE(3,1,0,2) ) ),
    _mm_mul_ps( _mm_shuffle_ps( vecta, vecta, _MM_SHUFFLE(3,1,0,2) ), _mm_shuffle_ps( oldvectb, oldvectb, _MM_SHUFFLE(3,0,2,1) ) )
  );
  /* Normal of new triangle */
  newvectb = _mm_sub_ps( _mm_load_ps( newpoint ), left );
  newnormal = _mm_sub_ps(
    _mm_mul_ps( _mm_shuffle_ps( vecta, vecta, _MM_SHUFFLE(3,0,2,1) ), _mm_shuffle_ps( newvectb, newvectb, _MM_SHUFFLE(3,1,0,2) ) ),
    _mm_mul_ps( _mm_shuffle_ps( vecta, vecta, _MM_SHUFFLE(3,1,0,2) ), _mm_shuffle_ps( newvectb, newvectb, _MM_SHUFFLE(3,0,2,1) ) )
  );
  if( meshflags & MD_FLAGS_PLANAR_MODE )
  {
    /* Detect planar normal Z inversion */
    invcheck = _mm_mul_ps( oldnormal, newnormal );
    if( _mm_comile_ss( _mm_shuffle_ps( invcheck, invcheck, _MM_SHUFFLE(3,3,3,3) ), _mm_set_ss( 0.0f ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal Z inversion denied in planar mode\n" );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  else
  {
    /* Detect normal inversion */
    if( _mm_comilt_ss( _mm_dp_ps( oldnormal, newnormal, 0x1 | 0x70 ), _mm_set_ss( 0.0f ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal inversion denied\n" );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  /* Prevent near-zero area triangles */
  oldnormal = _mm_dp_ps( oldnormal, oldnormal, 0x1 | 0x70 );
  newnormal = _mm_dp_ps( newnormal, newnormal, 0x1 | 0x70 );
  #if MD_CONFIG_APPROX_MATH
  oldmagnitude = _mm_mul_ss( _mm_rsqrt_ss( oldnormal ), oldnormal );
  newmagnitude = _mm_mul_ss( _mm_rsqrt_ss( newnormal ), newnormal );
  #else
  oldmagnitude = _mm_sqrt_ss( oldnormal );
  newmagnitude = _mm_sqrt_ss( newnormal );
  #endif
  if( _mm_comile_ss( newmagnitude, _mm_mul_ss( oldmagnitude, _mm_set_ss( MD_COLINEAR_REJECTION ) ) ) )
  {
#if DEBUG_VERBOSE_COST >= 2
    printf( "      !! Colinear magnitude denied\n" );
#endif
    *denyflag = 1;
    return 0.0;
  }
  /* Penalize long thin triangles */
  penalty = 0.0;
  vecta = _mm_dp_ps( vecta, vecta, 0x1 | 0x70 );
  newvectc = _mm_sub_ps( _mm_load_ps( newpoint ), _mm_load_ps( rightpoint ) );
  newnormal = _mm_dp_ps( newnormal, newnormal, 0x1 | 0x70 );
  newvectb = _mm_dp_ps( newvectb, newvectb, 0x1 | 0x70 );
  newvectc = _mm_dp_ps( newvectc, newvectc, 0x1 | 0x70 );
  norm = _mm_add_ss( _mm_add_ss( vecta, newvectb ), newvectc );
  newcompactness = _mm_mul_ss( _mm_set_ss( MD_COMPACTNESS_NORMALIZATION_FACTOR ), newmagnitude );
  if( _mm_comile_ss( newcompactness, _mm_mul_ss( compactnesstarget, norm ) ) )
  {
  #if MD_CONFIG_APPROX_MATH
    newcompactness = _mm_mul_ss( newcompactness, _mm_rcp_ss( norm ) );
  #else
    newcompactness = _mm_div_ss( newcompactness, norm );
  #endif
    oldvectc = _mm_sub_ps( _mm_load_ps( oldpoint ), _mm_load_ps( rightpoint ) );
    oldnormal = _mm_dp_ps( oldnormal, oldnormal, 0x1 | 0x70 );
    oldvectb = _mm_dp_ps( oldvectb, oldvectb, 0x1 | 0x70 );
    oldvectc = _mm_dp_ps( oldvectc, oldvectc, 0x1 | 0x70 );
  #if MD_CONFIG_APPROX_MATH
    oldcompactness = _mm_mul_ss( _mm_mul_ss( _mm_set_ss( MD_COMPACTNESS_NORMALIZATION_FACTOR ), oldmagnitude ), _mm_rcp_ss( _mm_add_ss( _mm_add_ss( vecta, oldvectb ), oldvectc ) ) );
  #else
    oldcompactness = _mm_div_ss( _mm_mul_ss( _mm_set_ss( MD_COMPACTNESS_NORMALIZATION_FACTOR ), oldmagnitude ), _mm_add_ss( _mm_add_ss( vecta, oldvectb ), oldvectc ) );
  #endif
    compactness = fmin( compactnesstarget, _mm_cvtss_f32( oldcompactness ) ) - _mm_cvtss_f32( newcompactness );
    penalty = fmaxf( penalty, compactness );
  }
  return penalty;
}

 #else

static double mdEdgeCollapsePenaltyTriangleSSE4p1d( double *newpoint, double *oldpoint, double *leftpoint, double *rightpoint, int *denyflag, double compactnesstarget, int meshflags )
{
  __m128d vecta0, vecta1, oldvectb0, oldvectb1, oldvectc0, oldvectc1, newvectb0, newvectb1, newvectc0, newvectc1;
  __m128d oldnormal0, oldnormal1, newnormal0, newnormal1;
  __m128d invcheck;
  __m128d left0, left1;
  __m128d oldmagnitude, newmagnitude;
  double newcompactness, oldcompactness, compactness, penalty, norm;
  /* Normal of old triangle */
  left0 = _mm_loadu_pd( leftpoint+0 );
  left1 = _mm_load_sd( leftpoint+2 );
  vecta0 = _mm_sub_pd( _mm_loadu_pd( rightpoint+0 ), left0 );
  vecta1 = _mm_sub_pd( _mm_load_sd( rightpoint+2 ), left1 );
  oldvectb0 = _mm_sub_pd( _mm_loadu_pd( oldpoint+0 ), left0 );
  oldvectb1 = _mm_sub_pd( _mm_load_sd( oldpoint+2 ), left1 );
  oldnormal0 = _mm_sub_pd(
    _mm_mul_pd( _mm_shuffle_pd( vecta0, vecta1, _MM_SHUFFLE2(0,1) ), _mm_unpacklo_pd( oldvectb1, oldvectb0 ) ),
    _mm_mul_pd( _mm_unpacklo_pd( vecta1, vecta0 ), _mm_shuffle_pd( oldvectb0, oldvectb1, _MM_SHUFFLE2(0,1) ) )
  );
  oldnormal1 = _mm_sub_sd(
    _mm_mul_sd( vecta0, _mm_unpackhi_pd( oldvectb0, oldvectb0 ) ),
    _mm_mul_sd( _mm_unpackhi_pd( vecta0, vecta0 ), oldvectb0 )
  );
  /* Normal of new triangle */
  newvectb0 = _mm_sub_pd( _mm_loadu_pd( newpoint+0 ), left0 );
  newvectb1 = _mm_sub_sd( _mm_load_sd( newpoint+2 ), left1 );
  newnormal0 = _mm_sub_pd(
    _mm_mul_pd( _mm_shuffle_pd( vecta0, vecta1, _MM_SHUFFLE2(0,1) ), _mm_unpacklo_pd( newvectb1, newvectb0 ) ),
    _mm_mul_pd( _mm_unpacklo_pd( vecta1, vecta0 ), _mm_shuffle_pd( newvectb0, newvectb1, _MM_SHUFFLE2(0,1) ) )
  );
  newnormal1 = _mm_sub_sd(
    _mm_mul_sd( vecta0, _mm_unpackhi_pd( newvectb0, newvectb0 ) ),
    _mm_mul_sd( _mm_unpackhi_pd( vecta0, vecta0 ), newvectb0 )
  );
  if( meshflags & MD_FLAGS_PLANAR_MODE )
  {
    /* Detect planar normal Z inversion */
    invcheck = _mm_mul_sd( oldnormal1, newnormal1 );
    if( _mm_comile_sd( invcheck, _mm_set_sd( 0.0 ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal Z inversion denied in planar mode %f -> %f\n", _mm_cvtsd_f64( oldnormal1 ), _mm_cvtsd_f64( newnormal1 ) );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  else
  {
    /* Detect normal inversion */
    invcheck = _mm_add_sd( _mm_dp_pd( oldnormal0, oldnormal0, 0x1 | 0x30 ), _mm_mul_sd( oldnormal1, newnormal1 ) );
    if( _mm_comilt_sd( invcheck, _mm_set_sd( 0.0 ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal inversion denied\n" );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  /* Prevent near-zero area triangles */
  oldnormal0 = _mm_add_sd( _mm_dp_pd( oldnormal0, oldnormal0, 0x1 | 0x30 ), _mm_mul_sd( oldnormal1, oldnormal1 ) );
  newnormal0 = _mm_add_sd( _mm_dp_pd( newnormal0, newnormal0, 0x1 | 0x30 ), _mm_mul_sd( newnormal1, newnormal1 ) );
  oldmagnitude = _mm_sqrt_sd( oldnormal0, oldnormal0 );
  newmagnitude = _mm_sqrt_sd( newnormal0, newnormal0 );
  if( _mm_comile_sd( newmagnitude, _mm_mul_sd( oldmagnitude, _mm_set_sd( MD_COLINEAR_REJECTION ) ) ) )
  {
#if DEBUG_VERBOSE_COST >= 2
    printf( "      !! Colinear magnitude denied\n" );
#endif
    *denyflag = 1;
    return 0.0;
  }
  /* Penalize long thin triangles */
  penalty = 0.0;
  vecta0 = _mm_add_sd( _mm_dp_pd( vecta0, vecta0, 0x1 | 0x30 ), _mm_mul_sd( vecta1, vecta1 ) );
  newvectc0 = _mm_sub_pd( _mm_loadu_pd( newpoint+0 ), _mm_loadu_pd( rightpoint+0 ) );
  newvectc1 = _mm_sub_sd( _mm_load_sd( newpoint+2 ), _mm_load_sd( rightpoint+2 ) );
  newvectb0 = _mm_add_sd( _mm_dp_pd( newvectb0, newvectb0, 0x1 | 0x30 ), _mm_mul_sd( newvectb1, newvectb1 ) );
  newvectc0 = _mm_add_sd( _mm_dp_pd( newvectc0, newvectc0, 0x1 | 0x30 ), _mm_mul_sd( newvectc1, newvectc1 ) );
  norm = _mm_cvtsd_f64( _mm_add_sd( vecta0, _mm_add_sd( newvectb0, newvectc0 ) ) );
  newcompactness = _mm_cvtsd_f64( _mm_mul_sd( _mm_set_sd( MD_COMPACTNESS_NORMALIZATION_FACTOR ), newmagnitude ) );
  if( newcompactness < ( compactnesstarget * norm ) )
  {
    newcompactness /= norm;
    oldvectc0 = _mm_sub_pd( _mm_loadu_pd( oldpoint+0 ), _mm_loadu_pd( rightpoint+0 ) );
    oldvectc1 = _mm_sub_sd( _mm_load_sd( oldpoint+2 ), _mm_load_sd( rightpoint+2 ) );
    oldvectb0 = _mm_add_sd( _mm_dp_pd( oldvectb0, oldvectb0, 0x1 | 0x30 ), _mm_mul_sd( oldvectb1, oldvectb1 ) );
    oldvectc0 = _mm_add_sd( _mm_dp_pd( oldvectc0, oldvectc0, 0x1 | 0x30 ), _mm_mul_sd( oldvectc1, oldvectc1 ) );
    oldcompactness = _mm_cvtsd_f64( _mm_div_sd( _mm_mul_sd( _mm_set_sd( MD_COMPACTNESS_NORMALIZATION_FACTOR ), oldmagnitude ), _mm_add_sd( vecta0, _mm_add_sd( oldvectb0, oldvectc0 ) ) ) );
    compactness = fmin( compactnesstarget, oldcompactness ) - newcompactness;
    penalty = fmaxf( penalty, compactness );
  }
  return penalty;
}

 #endif

#elif CPU_SSE3_SUPPORT

 #if !MD_CONF_DOUBLE_PRECISION

static float mdEdgeCollapsePenaltyTriangleSSE3f( float *newpoint, float *oldpoint, float *leftpoint, float *rightpoint, int *denyflag, float compactnesstarget, int meshflags )
{
  float penalty, compactness;
  __m128 left, vecta, oldvectb, oldvectc, newvectb, newvectc, oldnormal, newnormal;
  __m128 norm, oldmagnitude, newmagnitude, oldcompactness, newcompactness;
  /* Normal of old triangle */
  left = _mm_load_ps( leftpoint );
  vecta = _mm_sub_ps( _mm_load_ps( rightpoint ), left );
  oldvectb = _mm_sub_ps( _mm_load_ps( oldpoint ), left );
  oldnormal = _mm_sub_ps(
    _mm_mul_ps( _mm_shuffle_ps( vecta, vecta, _MM_SHUFFLE(3,0,2,1) ), _mm_shuffle_ps( oldvectb, oldvectb, _MM_SHUFFLE(3,1,0,2) ) ),
    _mm_mul_ps( _mm_shuffle_ps( vecta, vecta, _MM_SHUFFLE(3,1,0,2) ), _mm_shuffle_ps( oldvectb, oldvectb, _MM_SHUFFLE(3,0,2,1) ) )
  );
  /* Normal of new triangle */
  newvectb = _mm_sub_ps( _mm_load_ps( newpoint ), left );
  newnormal = _mm_sub_ps(
    _mm_mul_ps( _mm_shuffle_ps( vecta, vecta, _MM_SHUFFLE(3,0,2,1) ), _mm_shuffle_ps( newvectb, newvectb, _MM_SHUFFLE(3,1,0,2) ) ),
    _mm_mul_ps( _mm_shuffle_ps( vecta, vecta, _MM_SHUFFLE(3,1,0,2) ), _mm_shuffle_ps( newvectb, newvectb, _MM_SHUFFLE(3,0,2,1) ) )
  );
  if( meshflags & MD_FLAGS_PLANAR_MODE )
  {
    /* Detect planar normal Z inversion */
    invcheck = _mm_mul_ps( oldnormal, newnormal );
    if( _mm_comile_ss( _mm_shuffle_ps( invcheck, invcheck, _MM_SHUFFLE(3,3,3,3) ), _mm_set_ss( 0.0f ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal Z inversion denied in planar mode\n" );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  else
  {
    /* Detect normal inversion */
    dotproduct = _mm_mul_ps( oldnormal, newnormal );
    dotproduct = _mm_hadd_ps( dotproduct, dotproduct );
    dotproduct = _mm_hadd_ps( dotproduct, dotproduct );
    if( _mm_comilt_ss( dotproduct, _mm_set_ss( 0.0f ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal inversion denied\n" );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  /* Prevent near-zero area triangles */
  newnormal = _mm_hadd_ps( _mm_mul_ps( newnormal, newnormal ), _mm_mul_ps( oldnormal, oldnormal ) );
  newnormal = _mm_hadd_ps( newnormal, newnormal );
  oldnormal = _mm_movehdup_ps( newnormal );
  #if MD_CONFIG_APPROX_MATH
  oldmagnitude = _mm_mul_ss( _mm_rsqrt_ss( oldnormal ), oldnormal );
  newmagnitude = _mm_mul_ss( _mm_rsqrt_ss( newnormal ), newnormal );
  #else
  oldmagnitude = _mm_sqrt_ss( oldnormal );
  newmagnitude = _mm_sqrt_ss( newnormal );
  #endif
  if( _mm_comile_ss( newmagnitude, _mm_mul_ss( oldmagnitude, _mm_set_ss( MD_COLINEAR_REJECTION ) ) ) )
  {
#if DEBUG_VERBOSE_COST >= 2
    printf( "      !! Colinear magnitude denied\n" );
#endif
    *denyflag = 1;
    return 0.0;
  }
  /* Penalize long thin triangles */
  penalty = 0.0;
  vecta = _mm_dp_ps( vecta, vecta, 0x1 | 0x70 );
  newvectc = _mm_sub_ps( _mm_load_ps( newpoint ), _mm_load_ps( rightpoint ) );
  newvectb = _mm_hadd_ps( _mm_mul_ps( newvectb, newvectb ), _mm_mul_ps( newvectc, newvectc ) );
  newvectb = _mm_hadd_ps( newvectb, newvectb );
  newvectc = _mm_movehdup_ps( newvectb );
  norm = _mm_add_ss( _mm_add_ss( vecta, newvectb ), newvectc );
  newcompactness = _mm_mul_ss( _mm_set_ss( MD_COMPACTNESS_NORMALIZATION_FACTOR ), newmagnitude );
  if( _mm_comile_ss( newcompactness, _mm_mul_ss( compactnesstarget, norm ) ) )
  {
  #if MD_CONFIG_APPROX_MATH
    newcompactness = _mm_mul_ss( newcompactness, _mm_rcp_ss( norm ) );
  #else
    newcompactness = _mm_div_ss( newcompactness, norm );
  #endif
    oldvectc = _mm_sub_ps( _mm_load_ps( oldpoint ), _mm_load_ps( rightpoint ) );
    oldvectb = _mm_hadd_ps( _mm_mul_ps( oldvectb, oldvectb ), _mm_mul_ps( oldvectc, oldvectc ) );
    oldvectb = _mm_hadd_ps( oldvectb, oldvectb );
    oldvectc = _mm_movehdup_ps( oldvectb );
  #if MD_CONFIG_APPROX_MATH
    oldcompactness = _mm_mul_ss( _mm_mul_ss( _mm_set_ss( MD_COMPACTNESS_NORMALIZATION_FACTOR ), oldmagnitude ), _mm_rcp_ss( _mm_add_ss( _mm_add_ss( vecta, oldvectb ), oldvectc ) ) );
  #else
    oldcompactness = _mm_div_ss( _mm_mul_ss( _mm_set_ss( MD_COMPACTNESS_NORMALIZATION_FACTOR ), oldmagnitude ), _mm_add_ss( _mm_add_ss( vecta, oldvectb ), oldvectc ) );
  #endif
    compactness = fmin( compactnesstarget, _mm_cvtss_f32( oldcompactness ) ) - _mm_cvtss_f32( newcompactness );
    penalty = fmaxf( penalty, compactness );
  }
  return penalty;
}

 #else

static double mdEdgeCollapsePenaltyTriangleSSE3d( double *newpoint, double *oldpoint, double *leftpoint, double *rightpoint, int *denyflag, double compactnesstarget, int meshflags )
{
  __m128d vecta0, vecta1, oldvectb0, oldvectb1, oldvectc0, oldvectc1, newvectb0, newvectb1, newvectc0, newvectc1;
  __m128d oldnormal0, oldnormal1, newnormal0, newnormal1;
  __m128d invcheck;
  __m128d left0, left1;
  __m128d oldmagnitude, newmagnitude;
  double newcompactness, oldcompactness, compactness, penalty, norm;
  /* Normal of old triangle */
  left0 = _mm_loadu_pd( leftpoint+0 );
  left1 = _mm_load_sd( leftpoint+2 );
  vecta0 = _mm_sub_pd( _mm_loadu_pd( rightpoint+0 ), left0 );
  vecta1 = _mm_sub_pd( _mm_load_ps( rightpoint+2 ), left1 );
  oldvectb0 = _mm_sub_pd( _mm_loadu_pd( oldpoint+0 ), left0 );
  oldvectb1 = _mm_sub_pd( _mm_load_sd( oldpoint+2 ), left1 );
  oldnormal0 = _mm_sub_pd(
    _mm_mul_pd( _mm_shuffle_pd( vecta0, vecta1, _MM_SHUFFLE2(0,1) ), _mm_unpacklo_pd( oldvectb1, oldvectb0 ) ),
    _mm_mul_pd( _mm_unpacklo_pd( vecta1, vecta0 ), _mm_shuffle_pd( oldvectb0, oldvectb1, _MM_SHUFFLE2(0,1) ) )
  );
  oldnormal1 = _mm_sub_sd(
    _mm_mul_sd( vecta0, _mm_unpackhi_pd( oldvectb0, oldvectb0 ) ),
    _mm_mul_sd( _mm_unpackhi_pd( vecta0, vecta0 ), oldvectb0 )
  );
  /* Normal of new triangle */
  newvectb0 = _mm_sub_pd( _mm_loadu_pd( newpoint+0 ), left0 );
  newvectb1 = _mm_sub_pd( _mm_load_sd( newpoint+2 ), left1 );
  newnormal0 = _mm_sub_pd(
    _mm_mul_pd( _mm_shuffle_pd( vecta0, vecta1, _MM_SHUFFLE2(0,1) ), _mm_unpacklo_pd( newvectb1, newvectb0 ) ),
    _mm_mul_pd( _mm_unpacklo_pd( vecta1, vecta0 ), _mm_shuffle_pd( newvectb0, newvectb1, _MM_SHUFFLE2(0,1) ) )
  );
  newnormal1 = _mm_sub_sd(
    _mm_mul_sd( vecta0, _mm_unpackhi_pd( newvectb0, newvectb0 ) ),
    _mm_mul_sd( _mm_unpackhi_pd( vecta0, vecta0 ), newvectb0 )
  );
  if( meshflags & MD_FLAGS_PLANAR_MODE )
  {
    /* Detect planar normal Z inversion */
    invcheck = _mm_mul_sd( oldnormal1, newnormal1 );
    if( _mm_comile_sd( invcheck, _mm_set_sd( 0.0 ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal Z inversion denied in planar mode %f -> %f\n", _mm_cvtsd_f64( oldnormal1 ), _mm_cvtsd_f64( newnormal1 ) );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  else
  {
    /* Detect normal inversion */
    invcheck = _mm_mul_pd( oldnormal0, newnormal0 );
    invcheck = _mm_add_sd( _mm_hadd_pd( invcheck, invcheck ), _mm_mul_sd( oldnormal1, newnormal1 ) );
    if( _mm_comilt_sd( invcheck, _mm_set_sd( 0.0 ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal inversion denied\n" );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  /* Prevent near-zero area triangles */
  oldnormal0 = _mm_mul_pd( oldnormal0, oldnormal0 );
  oldnormal0 = _mm_add_sd( _mm_hadd_pd( oldnormal0, oldnormal0 ), _mm_mul_sd( oldnormal1, oldnormal1 ) );
  newnormal0 = _mm_mul_pd( newnormal0, newnormal0 );
  newnormal0 = _mm_add_sd( _mm_hadd_pd( newnormal0, newnormal0 ), _mm_mul_sd( newnormal1, newnormal1 ) );
  oldmagnitude = _mm_sqrt_sd( oldnormal0, oldnormal0 );
  newmagnitude = _mm_sqrt_sd( newnormal0, newnormal0 );
  if( _mm_comile_sd( newmagnitude, _mm_mul_sd( oldmagnitude, _mm_set_sd( MD_COLINEAR_REJECTION ) ) ) )
  {
#if DEBUG_VERBOSE_COST >= 2
    printf( "      !! Colinear magnitude denied\n" );
#endif
    *denyflag = 1;
    return 0.0;
  }
  /* Penalize long thin triangles */
  penalty = 0.0;
  vecta0 = _mm_mul_pd( vecta0, vecta0 );
  vecta0 = _mm_add_sd( _mm_hadd_pd( vecta0, vecta0 ), _mm_mul_sd( vecta1, vecta1 ) );
  newvectc0 = _mm_sub_pd( _mm_loadu_pd( newpoint+0 ), _mm_loadu_pd( rightpoint+0 ) );
  newvectc1 = _mm_sub_sd( _mm_load_sd( newpoint+2 ), _mm_load_sd( rightpoint+2 ) );
  newnormal0 = _mm_mul_pd( newnormal0, newnormal0 );
  newnormal0 = _mm_add_sd( _mm_hadd_pd( newnormal0, newnormal0 ), _mm_mul_sd( newnormal1, newnormal1 ) );
  newvectb0 = _mm_mul_pd( newvectb0, newvectb0 );
  newvectb0 = _mm_add_sd( _mm_hadd_pd( newvectb0, newvectb0 ), _mm_mul_sd( newvectb1, newvectb1 ) );
  newvectc0 = _mm_mul_pd( newvectc0, newvectc0 );
  newvectc0 = _mm_add_sd( _mm_hadd_pd( newvectc0, newvectc0 ), _mm_mul_sd( newvectc1, newvectc1 ) );
  norm = _mm_cvtsd_f64( _mm_add_sd( vecta0, _mm_add_sd( newvectb0, newvectc0 ) ) );
  newcompactness = _mm_cvtsd_f64( _mm_mul_sd( _mm_set_sd( MD_COMPACTNESS_NORMALIZATION_FACTOR ), newmagnitude ) );
  if( newcompactness < ( compactnesstarget * norm ) )
  {
    newcompactness /= norm;
    oldvectc0 = _mm_sub_pd( _mm_loadu_pd( oldpoint+0 ), _mm_loadu_pd( rightpoint+0 ) );
    oldvectc1 = _mm_sub_sd( _mm_load_sd( oldpoint+2 ), _mm_load_sd( rightpoint+2 ) );
    oldnormal0 = _mm_mul_pd( oldnormal0, oldnormal0 );
    oldnormal0 = _mm_add_sd( _mm_hadd_pd( oldnormal0, oldnormal0 ), _mm_mul_sd( oldnormal1, oldnormal1 ) );
    oldvectb0 = _mm_mul_pd( oldvectb0, oldvectb0 );
    oldvectb0 = _mm_add_sd( _mm_hadd_pd( oldvectb0, oldvectb0 ), _mm_mul_sd( oldvectb1, oldvectb1 ) );
    oldvectc0 = _mm_mul_pd( oldvectc0, oldvectc0 );
    oldvectc0 = _mm_add_sd( _mm_hadd_pd( oldvectc0, oldvectc0 ), _mm_mul_sd( oldvectc1, oldvectc1 ) );
    oldcompactness = _mm_cvtsd_f64( _mm_div_sd( _mm_mul_sd( _mm_set_sd( MD_COMPACTNESS_NORMALIZATION_FACTOR ), oldmagnitude ), _mm_add_sd( vecta0, _mm_add_sd( oldvectb0, oldvectc0 ) ) ) );
    compactness = fmin( compactnesstarget, oldcompactness ) - newcompactness;
    penalty = fmaxf( penalty, compactness );
  }

  return penalty;
}

 #endif

#elif CPU_SSE2_SUPPORT

 #if !MD_CONF_DOUBLE_PRECISION

static float mdEdgeCollapsePenaltyTriangleSSE2f( float *newpoint, float *oldpoint, float *leftpoint, float *rightpoint, int *denyflag, float compactnesstarget, int meshflags )
{
  return mdEdgeCollapsePenaltyTriangle( newpoint, oldpoint, leftpoint, rightpoint, denyflag, compactnesstarget );
}

 #else

static double mdEdgeCollapsePenaltyTriangleSSE2d( double *newpoint, double *oldpoint, double *leftpoint, double *rightpoint, int *denyflag, double compactnesstarget, int meshflags )
{
  __m128d vecta0, vecta1, oldvectb0, oldvectb1, oldvectc0, oldvectc1, newvectb0, newvectb1, newvectc0, newvectc1;
  __m128d oldnormal0, oldnormal1, newnormal0, newnormal1;
  __m128d invcheck;
  __m128d left0, left1;
  __m128d oldmagnitude, newmagnitude;
  double newcompactness, oldcompactness, compactness, penalty, norm;
  /* Normal of old triangle */
  left0 = _mm_loadu_pd( leftpoint+0 );
  left1 = _mm_load_sd( leftpoint+2 );
  vecta0 = _mm_sub_pd( _mm_loadu_pd( rightpoint+0 ), left0 );
  vecta1 = _mm_sub_pd( _mm_load_sd( rightpoint+2 ), left1 );
  oldvectb0 = _mm_sub_pd( _mm_loadu_pd( oldpoint+0 ), left0 );
  oldvectb1 = _mm_sub_pd( _mm_load_sd( oldpoint+2 ), left1 );
  oldnormal0 = _mm_sub_pd(
    _mm_mul_pd( _mm_shuffle_pd( vecta0, vecta1, _MM_SHUFFLE2(0,1) ), _mm_unpacklo_pd( oldvectb1, oldvectb0 ) ),
    _mm_mul_pd( _mm_unpacklo_pd( vecta1, vecta0 ), _mm_shuffle_pd( oldvectb0, oldvectb1, _MM_SHUFFLE2(0,1) ) )
  );
  oldnormal1 = _mm_sub_sd(
    _mm_mul_sd( vecta0, _mm_unpackhi_pd( oldvectb0, oldvectb0 ) ),
    _mm_mul_sd( _mm_unpackhi_pd( vecta0, vecta0 ), oldvectb0 )
  );
  /* Normal of new triangle */
  newvectb0 = _mm_sub_pd( _mm_loadu_pd( newpoint+0 ), left0 );
  newvectb1 = _mm_sub_pd( _mm_load_sd( newpoint+2 ), left1 );
  newnormal0 = _mm_sub_pd(
    _mm_mul_pd( _mm_shuffle_pd( vecta0, vecta1, _MM_SHUFFLE2(0,1) ), _mm_unpacklo_pd( newvectb1, newvectb0 ) ),
    _mm_mul_pd( _mm_unpacklo_pd( vecta1, vecta0 ), _mm_shuffle_pd( newvectb0, newvectb1, _MM_SHUFFLE2(0,1) ) )
  );
  newnormal1 = _mm_sub_sd(
    _mm_mul_sd( vecta0, _mm_unpackhi_pd( newvectb0, newvectb0 ) ),
    _mm_mul_sd( _mm_unpackhi_pd( vecta0, vecta0 ), newvectb0 )
  );
  if( meshflags & MD_FLAGS_PLANAR_MODE )
  {
    /* Detect planar normal Z inversion */
    invcheck = _mm_mul_sd( oldnormal1, newnormal1 );
    if( _mm_comile_sd( invcheck, _mm_set_sd( 0.0 ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal Z inversion denied in planar mode %f -> %f\n", _mm_cvtsd_f64( oldnormal1 ), _mm_cvtsd_f64( newnormal1 ) );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  else
  {
    /* Detect normal inversion */
    invcheck = _mm_mul_pd( oldnormal0, newnormal0 );
    invcheck = _mm_add_sd( _mm_add_sd( invcheck, _mm_unpackhi_pd( invcheck, invcheck ) ), _mm_mul_sd( oldnormal1, newnormal1 ) );
    if( _mm_comilt_sd( invcheck, _mm_set_sd( 0.0 ) ) )
    {
#if DEBUG_VERBOSE_COST >= 2
      printf( "      !! Normal inversion denied\n" );
#endif
      *denyflag = 1;
      return 0.0;
    }
  }
  /* Prevent near-zero area triangles */
  oldnormal0 = _mm_mul_pd( oldnormal0, oldnormal0 );
  oldnormal0 = _mm_add_sd( _mm_add_sd( oldnormal0, _mm_unpackhi_pd( oldnormal0, oldnormal0 ) ), _mm_mul_sd( oldnormal1, oldnormal1 ) );
  newnormal0 = _mm_mul_pd( newnormal0, newnormal0 );
  newnormal0 = _mm_add_sd( _mm_add_sd( newnormal0, _mm_unpackhi_pd( newnormal0, newnormal0 ) ), _mm_mul_sd( newnormal1, newnormal1 ) );
  oldmagnitude = _mm_sqrt_sd( oldnormal0, oldnormal0 );
  newmagnitude = _mm_sqrt_sd( newnormal0, newnormal0 );
  if( _mm_comile_sd( newmagnitude, _mm_mul_sd( oldmagnitude, _mm_set_sd( MD_COLINEAR_REJECTION ) ) ) )
  {
#if DEBUG_VERBOSE_COST >= 2
    printf( "      !! Colinear magnitude denied\n" );
#endif
    *denyflag = 1;
    return 0.0;
  }
  /* Penalize long thin triangles */
  penalty = 0.0;
  vecta0 = _mm_mul_pd( vecta0, vecta0 );
  vecta0 = _mm_add_sd( _mm_add_sd( vecta0, _mm_unpackhi_pd( vecta0, vecta0 ) ), _mm_mul_sd( vecta1, vecta1 ) );
  newvectc0 = _mm_sub_pd( _mm_loadu_pd( newpoint+0 ), _mm_loadu_pd( rightpoint+0 ) );
  newvectc1 = _mm_sub_sd( _mm_load_sd( newpoint+2 ), _mm_load_sd( rightpoint+2 ) );
  newvectb0 = _mm_mul_pd( newvectb0, newvectb0 );
  newvectb0 = _mm_add_sd( _mm_add_sd( newvectb0, _mm_unpackhi_pd( newvectb0, newvectb0 ) ), _mm_mul_sd( newvectb1, newvectb1 ) );
  newvectc0 = _mm_mul_pd( newvectc0, newvectc0 );
  newvectc0 = _mm_add_sd( _mm_add_sd( newvectc0, _mm_unpackhi_pd( newvectc0, newvectc0 ) ), _mm_mul_sd( newvectc1, newvectc1 ) );
  norm = _mm_cvtsd_f64( _mm_add_sd( vecta0, _mm_add_sd( newvectb0, newvectc0 ) ) );
  newcompactness = _mm_cvtsd_f64( _mm_mul_sd( _mm_set_sd( MD_COMPACTNESS_NORMALIZATION_FACTOR ), newmagnitude ) );
  if( newcompactness < ( compactnesstarget * norm ) )
  {
    newcompactness /= norm;
    oldvectc0 = _mm_sub_pd( _mm_loadu_pd( oldpoint+0 ), _mm_loadu_pd( rightpoint+0 ) );
    oldvectc1 = _mm_sub_sd( _mm_load_sd( oldpoint+2 ), _mm_load_sd( rightpoint+2 ) );
    oldvectb0 = _mm_mul_pd( oldvectb0, oldvectb0 );
    oldvectb0 = _mm_add_sd( _mm_add_sd( oldvectb0, _mm_unpackhi_pd( oldvectb0, oldvectb0 ) ), _mm_mul_sd( oldvectb1, oldvectb1 ) );
    oldvectc0 = _mm_mul_pd( oldvectc0, oldvectc0 );
    oldvectc0 = _mm_add_sd( _mm_add_sd( oldvectc0, _mm_unpackhi_pd( oldvectc0, oldvectc0 ) ), _mm_mul_sd( oldvectc1, oldvectc1 ) );
    oldcompactness = _mm_cvtsd_f64( _mm_div_sd( _mm_mul_sd( _mm_set_sd( MD_COMPACTNESS_NORMALIZATION_FACTOR ), oldmagnitude ), _mm_add_sd( vecta0, _mm_add_sd( oldvectb0, oldvectc0 ) ) ) );
    compactness = fmin( compactnesstarget, oldcompactness ) - newcompactness;
    penalty = fmaxf( penalty, compactness );
  }
  return penalty;
}

 #endif

#endif


////


static mdf mdEdgeCollapsePenaltyTriRefs( mdMesh *mesh, mdThreadData *tdata, mdi *trireflist, mdi trirefcount, mdi pivotindex, mdi skipindex, mdf *collapsepoint, int *denyflag )
{
  int index;
  mdf penalty, tripenalty;
  mdi triindex;
  mdTriangle *tri;
  mdf (*collapsepenalty)( mdf *newpoint, mdf *oldpoint, mdf *leftpoint, mdf *rightpoint, int *denyflag, mdf compactnesstarget, int meshflags );

  collapsepenalty = mesh->collapsepenalty;
  penalty = 0.0;
  for( index = 0 ; index < trirefcount ; index++ )
  {
    if( *denyflag )
      break;
    triindex = trireflist[ index ];
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;

#if DEBUG_DEBUG_CHECK_SOMETHING
    mdi triv[3];
    triv[0] = tri->v[0];
    triv[1] = tri->v[1];
    triv[2] = tri->v[2];
#endif

#if DEBUG_VERBOSE_COST >= 2
    printf( "    Penalty Tri %d,%d,%d ( Pivot %d ; Skip %d )\n", tri->v[0], tri->v[1], tri->v[2], pivotindex, skipindex );
#endif
    if( tri->v[0] == pivotindex )
    {
      if( ( tri->v[1] == skipindex ) || ( tri->v[2] == skipindex ) )
        continue;
      tripenalty = collapsepenalty( collapsepoint, mesh->vertexlist[ tri->v[0] ].point, mesh->vertexlist[ tri->v[2] ].point, mesh->vertexlist[ tri->v[1] ].point, denyflag, mesh->compactnesstarget, mesh->operationflags );
#if DEBUG_VERBOSE_COST >= 2
      printf( "      Penalty %f\n", tripenalty );
#endif
      penalty += tripenalty;
    }
    else if( tri->v[1] == pivotindex )
    {
      if( ( tri->v[2] == skipindex ) || ( tri->v[0] == skipindex ) )
        continue;
      tripenalty = collapsepenalty( collapsepoint, mesh->vertexlist[ tri->v[1] ].point, mesh->vertexlist[ tri->v[0] ].point, mesh->vertexlist[ tri->v[2] ].point, denyflag, mesh->compactnesstarget, mesh->operationflags );
#if DEBUG_VERBOSE_COST >= 2
      printf( "      Penalty %f\n", tripenalty );
#endif
      penalty += tripenalty;
    }
    else if( tri->v[2] == pivotindex )
    {
      if( ( tri->v[0] == skipindex ) || ( tri->v[1] == skipindex ) )
        continue;
      tripenalty = collapsepenalty( collapsepoint, mesh->vertexlist[ tri->v[2] ].point, mesh->vertexlist[ tri->v[1] ].point, mesh->vertexlist[ tri->v[0] ].point, denyflag, mesh->compactnesstarget, mesh->operationflags );
#if DEBUG_VERBOSE_COST >= 2
      printf( "      Penalty %f\n", tripenalty );
#endif
      penalty += tripenalty;
    }
    else
    {
#if DEBUG_DEBUG_CHECK_SOMETHING
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
      {
        int32_t debugowner;
        mdVertex *debugvertex;

        printf( "Failure to locate pivot %d in triref %d / %d\n", pivotindex, index, trirefcount );

        debugvertex = &mesh->vertexlist[ pivotindex ];
        debugowner = mmAtomicRead32( &debugvertex->atomicowner );
        printf( "  Pivot owner %d (self %d), redirect -> %d\n", debugowner, tdata->threadid, debugvertex->redirectindex );

        debugvertex = &mesh->vertexlist[ tri->v[0] ];
        debugowner = mmAtomicRead32( &debugvertex->atomicowner );
        printf( "  tri->v[0] owner %d (self %d), redirect -> %d\n", debugowner, tdata->threadid, debugvertex->redirectindex );

        debugvertex = &mesh->vertexlist[ tri->v[1] ];
        debugowner = mmAtomicRead32( &debugvertex->atomicowner );
        printf( "  tri->v[1] owner %d (self %d), redirect -> %d\n", debugowner, tdata->threadid, debugvertex->redirectindex );

        debugvertex = &mesh->vertexlist[ tri->v[2] ];
        debugowner = mmAtomicRead32( &debugvertex->atomicowner );
        printf( "  tri->v[2] owner %d (self %d), redirect -> %d\n", debugowner, tdata->threadid, debugvertex->redirectindex );

      }
      printf( "Triref %d / %d\n", index, trirefcount );
      printf( "CopyV : %d %d %d (%d)\n", triv[0], triv[1], triv[2], pivotindex );
      printf( "TriV  : %d %d %d (%d)\n", tri->v[0], tri->v[1], tri->v[2], pivotindex );
      sleep( 1 );
      printf( "CopyV : %d %d %d (%d)\n", triv[0], triv[1], triv[2], pivotindex );
      printf( "TriV  : %d %d %d (%d)\n", tri->v[0], tri->v[1], tri->v[2], pivotindex );
      printf( "\n" );
#endif
    }
  }

  return penalty;
}


static mdf mdEdgeCollapsePenalty( mdMesh *mesh, mdThreadData *tdata, mdi v0, mdi v1, mdf *collapsepoint, int *denyflag )
{
  mdf penalty, penaltyfactor;
  mdVertex *vertex0, *vertex1;

  vertex0 = &mesh->vertexlist[ v0 ];
  vertex1 = &mesh->vertexlist[ v1 ];

#if DEBUG_VERBOSE_COST
  printf( "  Compute Penalty Edge %d,%d\n", v0, v1 );
#endif

  *denyflag = 0;
  penalty  = mdEdgeCollapsePenaltyTriRefs( mesh, tdata, &mesh->trireflist[ vertex0->trirefbase ], vertex0->trirefcount, v0, v1, collapsepoint, denyflag );
  penalty += mdEdgeCollapsePenaltyTriRefs( mesh, tdata, &mesh->trireflist[ vertex1->trirefbase ], vertex1->trirefcount, v1, v0, collapsepoint, denyflag );

  if( penalty > 0.0 )
  {
#if DEBUG_VERBOSE_COST >= 2
    printf( "    Penalty Sum : %f\n", penalty );
#endif
    /* Normalize sum by count of trirefs, now from 0.0 to 1.0 */
    penalty /= (mdf)( vertex0->trirefcount + vertex1->trirefcount );
#if DEBUG_VERBOSE_COST >= 2
    printf( "    Penalty Normalized : %f\n", penalty );
#endif
#if 1
    /* Amplify, bring penalty closer to 1.0 */
    penalty = sqrt( penalty );
#endif
#if DEBUG_VERBOSE_COST >= 2
    printf( "    Penalty Amplified : %f\n", penalty );
#endif
    /* Apply global compactness penalty factor */
    penalty *= mesh->compactnesspenalty;
    /* Apply factor proportional to area compared to feature size, amplify/dampen with sqrt() */
    penaltyfactor = sqrt( ( vertex0->quadric.area + vertex1->quadric.area ) * mesh->invfeaturesizearea );
    penalty *= penaltyfactor * mesh->maxcollapsecost;
#if DEBUG_VERBOSE_COST
    printf( "    Penalty Total : %e (factor %f)\n", penalty, penaltyfactor );
#endif
 }

  return penalty;
}


static inline int mdGetVertexLockFlag( mdMesh *mesh, mdi vertexindex )
{
  return ( mesh->lockmap[ vertexindex >> 5 ] & (((uint32_t)1)<<(vertexindex&(32-1))) ) != 0;
}


#if MD_CONFIG_DISTANCE_BIAS
static mdf mdVertexDistance( mdVertex *vertex0, mdVertex *vertex1 )
{
  mdf distx, disty, distz;
  distx = vertex0->point[0] - vertex1->point[0];
  disty = vertex0->point[1] - vertex1->point[1];
  distz = vertex0->point[2] - vertex1->point[2];
  return mdfsqrt( ( distx * distx ) + ( disty * disty ) + ( distz * distz ) );
}
#endif


/* Experimental: collapsemultiplier */
static mdf mdSolveEdgeCollapse( mdMesh *mesh, mdi v0, mdi v1, mdf *point )
{
  int solveflags;
  mdf cost, costmultiplier;
  mdEdge edge;
  mdVertex *vertex0, *vertex1;
  void *tridata0, *tridata1;
  vertex0 = &mesh->vertexlist[v0];
  vertex1 = &mesh->vertexlist[v1];

  solveflags = MD_POINT_SOLVE_FLAGS_V0 | MD_POINT_SOLVE_FLAGS_V1 | MD_POINT_SOLVE_FLAGS_MIDPOINT | MD_POINT_SOLVE_FLAGS_QUADRIC;
  if( mesh->lockmap )
  {
    if( mdGetVertexLockFlag( mesh, v0 ) )
      solveflags &= MD_POINT_SOLVE_FLAGS_V0;
    if( mdGetVertexLockFlag( mesh, v1 ) )
      solveflags &= MD_POINT_SOLVE_FLAGS_V1;
  }

  if( !solveflags )
    return MD_OP_FAIL_VALUE;

  if( mesh->adjustcollapse )
    cost = mdEdgeSolvePointAdjust( vertex0, vertex1, point, solveflags, mesh->adjustcollapse, mesh->adjustcontext );
  else
    cost = mdEdgeSolvePoint( vertex0, vertex1, point, solveflags );

  if( mesh->collapsemultiplier )
  {
    /* TODO: Redundant hash checks with code calling mdSolveEdgeCollapse(), optimize */
    tridata0 = 0;
    edge.v[0] = v0;
    edge.v[1] = v1;
    if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
      tridata0 = ADDRESS( mesh->trilist, ( edge.triindex * mesh->trisize ) + sizeof(mdTriangle) );
    tridata1 = 0;
    edge.v[0] = v1;
    edge.v[1] = v0;
    if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
      tridata1 = ADDRESS( mesh->trilist, ( edge.triindex * mesh->trisize ) + sizeof(mdTriangle) );
#if MD_CONF_DOUBLE_PRECISION
    costmultiplier = mesh->collapsemultiplier( mesh->collapsecontext, tridata0, tridata1, vertex0->point, vertex1->point );
#else
    double pt0[3], pt1[3];
    MD_VectorCopy( pt0, vertex0->point );
    MD_VectorCopy( pt1, vertex1->point );
    costmultiplier = mesh->collapsemultiplier( mesh->collapsecontext, tridata0, tridata1, pt0, pt1 );
#endif
#if DEBUG_VERBOSE_COLLAPSE || DEBUG_VERBOSE_COST
    printf( "    Costmultiplier for %d,%d : %f\n", (int)v0, (int)v1, costmultiplier );
#endif
    if( costmultiplier < 0.0 )
      return MD_OP_FAIL_VALUE;
    cost *= costmultiplier;
  }

/* WWW */
#if MD_CONFIG_DISTANCE_BIAS
  mdf sumbias;
  sumbias = vertex0->sumbias + vertex1->sumbias;
  if( sumbias < mesh->biasclampdistance )
    sumbias += mdVertexDistance( vertex0, vertex1 );
  sumbias = mdfmin( sumbias, mesh->biasclampdistance );
  sumbias *= mesh->biascostfactor;
 #if DEBUG_VERBOSE_COST >= 2
  printf( "      Cost sumbias += %f %e\n", sumbias, sumbias );
 #endif
  cost += sumbias;
#endif

  return cost;
}



////



static void mdMeshEdgeSetOpCallback( void *opaque, void *entry, int newflag )
{
  mdEdge *edge;
  edge = entry;
  edge->op = opaque;
  return;
}


static double mdMeshOpValueCallback( void *item )
{
  mdOp *op;
  op = item;
  return (double)op->collapsecost;
}

static void mdMeshAddOp( mdMesh *mesh, mdThreadData *tdata, mdi v0, mdi v1 )
{
  int denyflag, opflags;
  mdOp *op;
  mdEdge edge;

#if DEBUG_VERBOSE_COLLAPSE || DEBUG_VERBOSE_COST
  mdVertex *vertex0, *vertex1;
  vertex0 = &mesh->vertexlist[ v0 ];
  vertex1 = &mesh->vertexlist[ v1 ];
  printf( "  Add Edge Op %d,%d ; %f %f %f ~ %f %f %f\n", (int)v0, (int)v1, vertex0->point[0], vertex0->point[1], vertex0->point[2], vertex1->point[0], vertex1->point[1], vertex1->point[2] );
#endif

  op = mmBlockAlloc( &tdata->opblock );
  opflags = 0x0;
  op->updatebuffer = tdata->updatebuffer;
  op->v0 = v0;
  op->v1 = v1;
  op->value = mdSolveEdgeCollapse( mesh, v0, v1, op->collapsepoint );
#if CPU_SSE_SUPPORT
  op->collapsepoint[3] = 0.0;
#endif
  if( op->value < MD_OP_FAIL_VALUE )
    op->penalty = mdEdgeCollapsePenalty( mesh, tdata, v0, v1, op->collapsepoint, &denyflag );
  else
  {
    op->penalty = 0.0;
    denyflag = 1;
  }
  op->collapsecost = op->value + op->penalty;
  if( ( denyflag ) || ( op->collapsecost >= mesh->maxcollapseacceptcost ) )
    opflags |= MD_OP_FLAGS_DETACHED;
  else
    mmBinSortAdd( tdata->binsort, op, op->collapsecost );
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicWrite32( &op->flags, opflags );
#else
  op->flags = opflags;
  mtSpinInit( &op->spinlock );
#endif

  edge.v[0] = v0;
  edge.v[1] = v1;
  if( mmHashLockCallEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, mdMeshEdgeSetOpCallback, op, 0 ) != MM_HASH_SUCCESS )
  {
/*
    MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
*/
  }

#if DEBUG_VERBOSE_COLLAPSE || DEBUG_VERBOSE_COST
  printf( "  Added Edge Op %d,%d%s ~ flags 0x%x ; Point %f %f %f ; Value %e ; Penalty %e ; Cost %e (max %e, ratio %f)\n", (int)op->v0, (int)op->v1, denyflag ? " {denied}" : "", mmAtomicRead32( &op->flags ), op->collapsepoint[0], op->collapsepoint[1], op->collapsepoint[2], op->value, op->penalty, op->collapsecost, mesh->maxcollapsecost, op->collapsecost / mesh->maxcollapsecost );
#endif

  return;
}

static void mdMeshPopulateOpList( mdMesh *mesh, mdThreadData *tdata, mdi tribase, mdi tricount )
{
  mdTriangle *tri, *tristart, *triend;
  long populatecount;

  tristart = ADDRESS( mesh->trilist, tribase * mesh->trisize );
  triend = ADDRESS( tristart, tricount * mesh->trisize );

  populatecount = 0;
  for( tri = tristart ; tri < triend ; tri = ADDRESS( tri, mesh->trisize ) )
  {
#if DEBUG_VERBOSE_COLLAPSE
    printf( "Triangle %d, edges %d,%d,%d ~ edgeflags 0x%02x\n", (int)( tri - tristart ), tri->v[0], tri->v[1], tri->v[2], tri->u.edgeflags );
#endif
#if 0
    if( ( ( tri->v[0] < tri->v[1] ) || ( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY01 ) ) && !( tri->u.edgeflags & MD_EDGEFLAGS_DENYEDGE01 ) )
      mdMeshAddOp( mesh, tdata, tri->v[0], tri->v[1] );
    if( ( ( tri->v[1] < tri->v[2] ) || ( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY12 ) ) && !( tri->u.edgeflags & MD_EDGEFLAGS_DENYEDGE12 ) )
      mdMeshAddOp( mesh, tdata, tri->v[1], tri->v[2] );
    if( ( ( tri->v[2] < tri->v[0] ) || ( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY20 ) ) && !( tri->u.edgeflags & MD_EDGEFLAGS_DENYEDGE20 ) )
      mdMeshAddOp( mesh, tdata, tri->v[2], tri->v[0] );
#else
    /* Don't add ops for the whole triangle if any of the triangle's edge was denied? */
    if( !( tri->u.edgeflags & (MD_EDGEFLAGS_DENYEDGE01|MD_EDGEFLAGS_DENYEDGE12|MD_EDGEFLAGS_DENYEDGE20) ) )
    {
      if( ( tri->v[0] < tri->v[1] ) || ( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY01 ) )
        mdMeshAddOp( mesh, tdata, tri->v[0], tri->v[1] );
      if( ( tri->v[1] < tri->v[2] ) || ( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY12 ) )
        mdMeshAddOp( mesh, tdata, tri->v[1], tri->v[2] );
      if( ( tri->v[2] < tri->v[0] ) || ( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY20 ) )
        mdMeshAddOp( mesh, tdata, tri->v[2], tri->v[0] );
    }
#endif
    populatecount++;
    tdata->statuspopulatecount = populatecount;
  }

  return;
}



////



/* Merge vertex attributes of v0 and v1, write to v0 */
static inline void mdEdgeCollapseMergeVertexAttribs( mdMesh *mesh, mdThreadData *tdata, mdi v0, mdi v1, mdf *collapsepoint )
{
  mdVertex *vertex0, *vertex1;
  mdf dist[3], dist0, dist1, weightsum, weightsuminv;
  mdf weight0, weight1;

  vertex0 = &mesh->vertexlist[ v0 ];
  MD_VectorSubStore( dist, collapsepoint, vertex0->point );
  dist0 = MD_VectorMagnitude( dist );
  vertex1 = &mesh->vertexlist[ v1 ];
  MD_VectorSubStore( dist, collapsepoint, vertex1->point );
  dist1 = MD_VectorMagnitude( dist );
  weight0 = dist1 * vertex0->quadric.area;
  weight1 = dist0 * vertex1->quadric.area;
  weightsum = weight0 + weight1;
  if( weightsum )
  {
    weightsuminv = 1.0 / weightsum;
    weight0 *= weightsuminv;
    weight1 *= weightsuminv;
  }
  else
  {
    weight0 = 0.5;
    weight1 = 0.5;
  }
  mesh->vertexmerge( mesh->mergecontext, v0, v1, weight0, weight1 );
  return;
}


/* Delete triangle and return outer vertex */
static mdi mdEdgeCollapseDeleteTriangle( mdMesh *mesh, mdThreadData *tdata, mdi v0, mdi v1, int *retdelflags )
{
  int delflags;
  mdi outer;
  mdEdge edge;
  mdTriangle *tri;
  mdOp *op;

  *retdelflags = 0x0;

  edge.v[0] = v0;
  edge.v[1] = v1;
  if( mmHashLockDeleteEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) != MM_HASH_SUCCESS )
    return -1;
  op = edge.op;
  if( op )
    mdUpdateBufferAdd( &op->updatebuffer[ tdata->threadid >> mesh->updatebuffershift ], op, MD_OP_FLAGS_DELETION_PENDING );

  tri = ADDRESS( mesh->trilist, edge.triindex * mesh->trisize );

#if DEBUG_VERBOSE_COLLAPSE
  printf( "  Delete Triangle %d,%d,%d\n", tri->v[0], tri->v[1], tri->v[2] );
#endif

  if( tri->v[0] != v0 )
  {
    edge.v[0] = tri->v[0];
    edge.v[1] = tri->v[1];
    if( mmHashLockDeleteEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) == MM_HASH_SUCCESS )
    {
      op = edge.op;
      if( op )
        mdUpdateBufferAdd( &op->updatebuffer[ tdata->threadid >> mesh->updatebuffershift ], op, MD_OP_FLAGS_DELETION_PENDING );
    }
    else
    {
#if 0
      /* Shouldn't happen with a proper watertight mesh, but it can happen if edges are reused... */
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#endif
    }
  }
  if( tri->v[1] != v0 )
  {
    edge.v[0] = tri->v[1];
    edge.v[1] = tri->v[2];
    if( mmHashLockDeleteEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) == MM_HASH_SUCCESS )
    {
      op = edge.op;
      if( op )
        mdUpdateBufferAdd( &op->updatebuffer[ tdata->threadid >> mesh->updatebuffershift ], op, MD_OP_FLAGS_DELETION_PENDING );
    }
    else
    {
#if 0
      /* Shouldn't happen with a proper watertight mesh, but it can happen if edges are reused... */
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#endif
    }
  }
  if( tri->v[2] != v0 )
  {
    edge.v[0] = tri->v[2];
    edge.v[1] = tri->v[0];
    if( mmHashLockDeleteEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) == MM_HASH_SUCCESS )
    {
      op = edge.op;
      if( op )
        mdUpdateBufferAdd( &op->updatebuffer[ tdata->threadid >> mesh->updatebuffershift ], op, MD_OP_FLAGS_DELETION_PENDING );
    }
    else
    {
#if 0
      /* Shouldn't happen with a proper watertight mesh, but it can happen if edges are reused... */
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#endif
    }
  }

  /* Determine outer vertex */
  /* TODO: Replace branches with bitwise arithmetics */
  delflags = 0x0;
  if( tri->v[0] == v0 )
  {
    outer = tri->v[2];
    if( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY12 )
      delflags |= 0x1;
    if( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY20 )
      delflags |= 0x2;
  }
  else if( tri->v[1] == v0 )
  {
    outer = tri->v[0];
    if( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY20 )
      delflags |= 0x1;
    if( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY01 )
      delflags |= 0x2;
  }
  else if( tri->v[2] == v0 )
  {
    outer = tri->v[1];
    if( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY01 )
      delflags |= 0x1;
    if( tri->u.edgeflags & MD_EDGEFLAGS_BOUNDARY12 )
      delflags |= 0x2;
  }
  else
  {
    MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
    outer = tri->v[0];
  }

  *retdelflags = delflags;

  /* Invalidate triangle */
  tri->v[0] = -1;

  return outer;
}


static void mdEdgeCollapseUpdateTriangle( mdMesh *mesh, mdThreadData *tdata, mdTriangle *tri, mdi newv, int pivot, int left, int right )
{
  mdEdge edge;
  mdOp *op;

#if DEBUG_VERBOSE_COLLAPSE
  printf( "  Collapse Update %d : Tri %d,%d,%d (%d,%d,%d)\n", newv, tri->v[pivot], tri->v[right], tri->v[left], tri->v[0], tri->v[1], tri->v[2] );
#endif

  /* Update op on right side of pivot, update edge's vertex to new vertex */
  edge.v[0] = tri->v[pivot];
  edge.v[1] = tri->v[right];
  edge.op = 0;
  if( edge.v[0] == newv )
  {
    if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) != MM_HASH_SUCCESS )
    {
#if 0
      /* Shouldn't happen with a proper watertight mesh, but it can happen if edges are reused... */
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#endif
    }
  }
  else
  {
    if( mmHashLockDeleteEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) == MM_HASH_SUCCESS )
    {
      edge.v[0] = newv;
      if( mmHashLockAddEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) != MM_HASH_SUCCESS )
        MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
    }
    else
    {
#if 0
      /* Shouldn't happen with a proper watertight mesh, but it can happen if edges are reused... */
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#endif
    }
  }
  op = edge.op;
  if( op )
  {
#if DEBUG_VERBOSE_COLLAPSE
    printf( "    Update Edge %d,%d Before ; Point %f %f %f ; Cost %.16f\n", op->v0, op->v1, op->collapsepoint[0], op->collapsepoint[1], op->collapsepoint[2], op->collapsecost );
#endif
    op->value = mdSolveEdgeCollapse( mesh, edge.v[0], edge.v[1], op->collapsepoint );
#if CPU_SSE_SUPPORT
    op->collapsepoint[3] = 0.0;
#endif
    mdUpdateBufferAdd( &op->updatebuffer[ tdata->threadid >> mesh->updatebuffershift ], op, 0x0 );
#if DEBUG_VERBOSE_COLLAPSE
    printf( "    Update Edge %d,%d After  ; Point %f %f %f ; Cost %e\n", op->v0, op->v1, op->collapsepoint[0], op->collapsepoint[1], op->collapsepoint[2], op->value + op->penalty );
    printf( "    Edge %d,%d ; Value %e ; Penalty %e ; Cost %e\n", op->v0, op->v1, op->value, op->penalty, op->value + op->penalty );
#endif
  }

  /* Update op on left side of pivot, update edge's vertex to new vertex */
  edge.v[0] = tri->v[left];
  edge.v[1] = tri->v[pivot];
  edge.op = 0;
  if( edge.v[1] == newv )
  {
    if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) != MM_HASH_SUCCESS )
    {
#if 0
      /* Shouldn't happen with a proper watertight mesh, but it can happen if edges are reused... */
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#endif
    }
  }
  else
  {
    if( mmHashLockDeleteEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) == MM_HASH_SUCCESS )
    {
      edge.v[1] = newv;
      if( mmHashLockAddEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) != MM_HASH_SUCCESS )
        MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
    }
    else
    {
#if 0
      /* Shouldn't happen with a proper watertight mesh, but it can happen if edges are reused... */
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#endif
    }
  }
  op = edge.op;
  if( op )
  {
#if DEBUG_VERBOSE_COLLAPSE
    printf( "    Update Edge %d,%d Before ; Point %f %f %f ; Cost %f\n", op->v0, op->v1, op->collapsepoint[0], op->collapsepoint[1], op->collapsepoint[2], op->collapsecost );
#endif
    op->value = mdSolveEdgeCollapse( mesh, edge.v[0], edge.v[1], op->collapsepoint );
#if CPU_SSE_SUPPORT
    op->collapsepoint[3] = 0.0;
#endif
    mdUpdateBufferAdd( &op->updatebuffer[ tdata->threadid >> mesh->updatebuffershift ], op, 0x0 );
#if DEBUG_VERBOSE_COLLAPSE
    printf( "    Update Edge %d,%d After  ; Point %f %f %f ; Cost %e\n", op->v0, op->v1, op->collapsepoint[0], op->collapsepoint[1], op->collapsepoint[2], op->value + op->penalty );
    printf( "    Edge %d,%d ; Value %e ; Penalty %e ; Cost %e\n", op->v0, op->v1, op->value, op->penalty, op->value + op->penalty );
#endif
  }

  tri->v[pivot] = newv;

#if DEBUG_VERBOSE_COLLAPSE
  printf( "    Updated Triangle ; %d,%d,%d\n", tri->v[0], tri->v[1], tri->v[2] );
#endif
#if DEBUG_VERBOSE_CHECKS
  if( ( tri->v[0] == tri->v[1] ) || ( tri->v[0] == tri->v[2] ) || ( tri->v[1] == tri->v[2] ) )
    printf( "      ERROR: Updated triangle has repeated vertices ; %d,%d,%d\n", tri->v[0], tri->v[1], tri->v[2] );
#endif

#if DEBUG_VERBOSE_CHECKS
  edge.v[0] = tri->v[0];
  edge.v[1] = tri->v[1];
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) != MM_HASH_SUCCESS )
    printf( "      ERROR: Updated triangle %d,%d,%d has missing hash edge %d,%d\n", (int)tri->v[0], (int)tri->v[1], (int)tri->v[2], (int)edge.v[0], (int)edge.v[1] );
  edge.v[0] = tri->v[1];
  edge.v[1] = tri->v[2];
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) != MM_HASH_SUCCESS )
    printf( "      ERROR: Updated triangle %d,%d,%d has missing hash edge %d,%d\n", (int)tri->v[0], (int)tri->v[1], (int)tri->v[2], (int)edge.v[0], (int)edge.v[1] );
  edge.v[0] = tri->v[2];
  edge.v[1] = tri->v[0];
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) != MM_HASH_SUCCESS )
    printf( "      ERROR: Updated triangle %d,%d,%d has missing hash edge %d,%d\n", (int)tri->v[0], (int)tri->v[1], (int)tri->v[2], (int)edge.v[0], (int)edge.v[1] );
#endif

  return;
}



/*
Walk through the list of all triangles attached to a vertex
- Update cost of collapse for other edges of triangles
- Build up the updated list of triangle references for new vertex
*/
static mdi *mdEdgeCollapseUpdateAll( mdMesh *mesh, mdThreadData *tdata, mdi *trireflist, mdi trirefcount, mdi oldv, mdi newv, mdi *trirefstore )
{
  int index;
  mdi triindex;
  mdTriangle *tri;

#if DEBUG_VERBOSE_COLLAPSE
  printf( "  Triref collapse, %d refs ; oldv %d ; newv %d\n", (int)trirefcount, (int)oldv, (int)newv );
#endif

  for( index = 0 ; index < trirefcount ; index++ )
  {
    triindex = trireflist[ index ];
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;

#if DEBUG_VERBOSE_COLLAPSE
    printf( "  Tri %d : %d %d %d\n", index, tri->v[0], tri->v[1], tri->v[2] );
#endif

    *trirefstore = triindex;
    trirefstore++;
    if( tri->v[0] == oldv )
      mdEdgeCollapseUpdateTriangle( mesh, tdata, tri, newv, 0, 2, 1 );
    else if( tri->v[1] == oldv )
      mdEdgeCollapseUpdateTriangle( mesh, tdata, tri, newv, 1, 0, 2 );
    else if( tri->v[2] == oldv )
      mdEdgeCollapseUpdateTriangle( mesh, tdata, tri, newv, 2, 1, 0 );
    else
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
  }

  return trirefstore;
}


static void mdVertexInvalidateTri( mdMesh *mesh, mdThreadData *tdata, mdi v0, mdi v1 )
{
  mdEdge edge;
  mdOp *op;

  edge.v[0] = v0;
  edge.v[1] = v1;
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
  {
    op = edge.op;
    if( op )
      mdUpdateBufferAdd( &op->updatebuffer[ tdata->threadid >> mesh->updatebuffershift ], op, 0x0 );
  }
#if 0
  /* Shouldn't happen with a proper watertight mesh, but it can happen if edges are reused... */
  else
    MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#endif

  return;
}

static void mdVertexInvalidate( mdMesh *mesh, mdThreadData *tdata, mdi vertexindex )
{
  mdi index, triindex, trirefcount;
  mdi *trireflist;
  mdTriangle *tri;
  mdVertex *vertex;

  /* Vertices of the collapsed edge */
  vertex = &mesh->vertexlist[ vertexindex ];
  trireflist = &mesh->trireflist[ vertex->trirefbase ];
  trirefcount = vertex->trirefcount;

  for( index = 0 ; index < trirefcount ; index++ )
  {
    triindex = trireflist[ index ];
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;
    if( tri->v[0] == vertexindex )
    {
      mdVertexInvalidateTri( mesh, tdata, tri->v[0], tri->v[1] );
      mdVertexInvalidateTri( mesh, tdata, tri->v[2], tri->v[0] );
    }
    else if( tri->v[1] == vertexindex )
    {
      mdVertexInvalidateTri( mesh, tdata, tri->v[1], tri->v[2] );
      mdVertexInvalidateTri( mesh, tdata, tri->v[0], tri->v[1] );
    }
    else if( tri->v[2] == vertexindex )
    {
      mdVertexInvalidateTri( mesh, tdata, tri->v[2], tri->v[0] );
      mdVertexInvalidateTri( mesh, tdata, tri->v[1], tri->v[2] );
    }
    else
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
  }

  return;
}


static void mdVertexInvalidateAll( mdMesh *mesh, mdThreadData *tdata, mdi *trireflist, mdi trirefcount, mdi pivotindex )
{
  int index;
  mdi triindex;
  mdTriangle *tri;

  for( index = 0 ; index < trirefcount ; index++ )
  {
    triindex = trireflist[ index ];
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;
    if( tri->v[0] == pivotindex )
    {
      mdVertexInvalidate( mesh, tdata, tri->v[1] );
      mdVertexInvalidate( mesh, tdata, tri->v[2] );
    }
    else if( tri->v[1] == pivotindex )
    {
      mdVertexInvalidate( mesh, tdata, tri->v[2] );
      mdVertexInvalidate( mesh, tdata, tri->v[0] );
    }
    else if( tri->v[2] == pivotindex )
    {
      mdVertexInvalidate( mesh, tdata, tri->v[0] );
      mdVertexInvalidate( mesh, tdata, tri->v[1] );
    }
    else
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
  }

  return;
}


static void mdEdgeCollapseLinkOuter( mdMesh *mesh, mdThreadData *tdata, mdi newv, mdi outer )
{
  int sideflags;
  mdEdge edge;
  mdOp *op;

  if( outer == -1 )
    return;
  sideflags = 0x0;

  edge.v[0] = newv;
  edge.v[1] = outer;
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
  {
    sideflags |= 0x1;
    op = edge.op;
    if( op )
      return;
  }
  edge.v[0] = outer;
  edge.v[1] = newv;
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
  {
    sideflags |= 0x2;
    op = edge.op;
    if( op )
      return;
  }

  if( ( newv < outer ) && ( sideflags & 0x1 ) )
    mdMeshAddOp( mesh, tdata, newv, outer );
  else if( sideflags & 0x2 )
    mdMeshAddOp( mesh, tdata, outer, newv );

  return;
}



static void mdEdgeCollapsePropagateBoundary( mdMesh *mesh, mdi v0, mdi v1 )
{
  mdEdge edge;
  mdTriangle *tri;

  edge.v[0] = v1;
  edge.v[1] = v0;
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
    return;
  edge.v[0] = v0;
  edge.v[1] = v1;
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) != MM_HASH_SUCCESS )
    return;

  tri = ADDRESS( mesh->trilist, edge.triindex * mesh->trisize );
  if( tri->v[0] == v0 )
    tri->u.edgeflags |= MD_EDGEFLAGS_BOUNDARY01;
  else if( tri->v[1] == v0 )
    tri->u.edgeflags |= MD_EDGEFLAGS_BOUNDARY12;
  else if( tri->v[2] == v0 )
    tri->u.edgeflags |= MD_EDGEFLAGS_BOUNDARY20;
#if 0
  else
  {
    /* NOTE: Yes, this CAN happen ~ if you fully decimate a hole away, previous boundaries are just gone */
    MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
  }
#endif

  /* TODO: Grow the boundary for affected vertex quadrics */

  return;
}




#define MD_LOCK_BUFFER_STATIC (512)

typedef struct
{
  mdi vertexstatic[MD_LOCK_BUFFER_STATIC];
  mdi *vertexlist;
  int vertexcount;
  int vertexalloc;
} mdLockBuffer;

static inline void mdLockBufferInit( mdLockBuffer *buffer, int maxvertexcount )
{
  buffer->vertexlist = buffer->vertexstatic;
  buffer->vertexalloc = MD_LOCK_BUFFER_STATIC;
  if( maxvertexcount > MD_LOCK_BUFFER_STATIC )
  {
    buffer->vertexlist = malloc( maxvertexcount * sizeof(mdi) );
    buffer->vertexalloc = maxvertexcount;
  }
  buffer->vertexcount = 0;
  return;
}

static inline void mdLockBufferResize( mdLockBuffer *buffer, int maxvertexcount )
{
  int index, vertexalloc;
  mdi *vertexlist;
  if( CC_UNLIKELY( maxvertexcount >= buffer->vertexalloc ) )
  {
    vertexalloc = maxvertexcount;
    if( vertexalloc < ( 2 * MD_LOCK_BUFFER_STATIC ) )
      vertexalloc = 2 * MD_LOCK_BUFFER_STATIC;
    else
      vertexalloc <<= 1;
    vertexlist = malloc( vertexalloc * sizeof(mdi) );
    for( index = 0 ; index < buffer->vertexcount ; index++ )
      vertexlist[index] = buffer->vertexlist[index];
    if( buffer->vertexlist != buffer->vertexstatic )
      free( buffer->vertexlist );
    buffer->vertexlist = vertexlist;
    buffer->vertexalloc = vertexalloc;
  }
  return;
}

static inline void mdLockBufferEnd( mdLockBuffer *buffer )
{
  if( buffer->vertexlist != buffer->vertexstatic )
    free( buffer->vertexlist );
  buffer->vertexlist = buffer->vertexstatic;
  buffer->vertexalloc = MD_LOCK_BUFFER_STATIC;
  buffer->vertexcount = 0;
  return;
}

void mdLockBufferUnlockAll( mdMesh *mesh, mdThreadData *tdata, mdLockBuffer *buffer )
{
  int index;
  mdVertex *vertex;
  for( index = 0 ; index < buffer->vertexcount ; index++ )
  {
    vertex = &mesh->vertexlist[ buffer->vertexlist[index] ];
#if MD_CONFIG_ATOMIC_SUPPORT
    if( mmAtomicRead32( &vertex->atomicowner ) == tdata->threadid )
      mmAtomicCmpXchg32( &vertex->atomicowner, tdata->threadid, -1 );
#else
    mtSpinLock( &vertex->ownerspinlock );
    if( vertex->owner == tdata->threadid )
      vertex->owner = -1;
    mtSpinUnlock( &vertex->ownerspinlock );
#endif
  }
  buffer->vertexcount = 0;
  return;
}

/* If it fails, release all locks then return zero ~ return 1 when lock is already owned or acquired (and added to lockbuffer) */
int mdLockBufferTryLock( mdMesh *mesh, mdThreadData *tdata, mdLockBuffer *buffer, mdi vertexindex )
{
  int32_t owner;
  mdVertex *vertex;
  vertex = &mesh->vertexlist[ vertexindex ];

#if MD_CONFIG_ATOMIC_SUPPORT
  owner = mmAtomicRead32( &vertex->atomicowner );
  if( owner == tdata->threadid )
    return 1;
  if( ( owner != -1 ) || !( mmAtomicCmpReplace32( &vertex->atomicowner, -1, tdata->threadid ) ) )
  {
    mdLockBufferUnlockAll( mesh, tdata, buffer );
    return 0;
  }
#else
  mtSpinLock( &vertex->ownerspinlock );
  owner = vertex->owner;
  if( owner == tdata->threadid )
  {
    mtSpinUnlock( &vertex->ownerspinlock );
    return 1;
  }
  if( owner != -1 )
  {
    mtSpinUnlock( &vertex->ownerspinlock );
    mdLockBufferUnlockAll( mesh, tdata, buffer );
    return 0;
  }
  vertex->owner = tdata->threadid;
  mtSpinUnlock( &vertex->ownerspinlock );
#endif
  mdLockBufferResize( buffer, buffer->vertexcount );
  buffer->vertexlist[buffer->vertexcount++] = vertexindex;
  return 1;
}

/* If it fails, release all locks then wait for the desired lock to become available */
int mdLockBufferLock( mdMesh *mesh, mdThreadData *tdata, mdLockBuffer *buffer, mdi vertexindex )
{
  int32_t owner;
  mdVertex *vertex;
  vertex = &mesh->vertexlist[ vertexindex ];

#if MD_CONFIG_ATOMIC_SUPPORT
  owner = mmAtomicRead32( &vertex->atomicowner );
  if( owner == tdata->threadid )
    return 1;
  if( ( owner == -1 ) && mmAtomicCmpReplace32( &vertex->atomicowner, -1, tdata->threadid ) )
  {
    mdLockBufferResize( buffer, buffer->vertexcount );
    buffer->vertexlist[buffer->vertexcount++] = vertexindex;
    return 1;
  }
  /* Lock failed, release all locks and wait until we get the lock we got stuck on */
  mdLockBufferUnlockAll( mesh, tdata, buffer );
  mmAtomicSpinWaitEq32( &vertex->atomicowner, -1 );
#else
  mtSpinLock( &vertex->ownerspinlock );
  owner = vertex->owner;
  if( owner == tdata->threadid )
  {
    mtSpinUnlock( &vertex->ownerspinlock );
    return 1;
  }
  if( owner == -1 )
  {
    vertex->owner = tdata->threadid;
    mtSpinUnlock( &vertex->ownerspinlock );
    mdLockBufferResize( buffer, buffer->vertexcount );
    buffer->vertexlist[buffer->vertexcount++] = vertexindex;
    return 1;
  }
  mtSpinUnlock( &vertex->ownerspinlock );
  /* Lock failed, release all locks */
  mdLockBufferUnlockAll( mesh, tdata, buffer );
#endif
  return 0;
}


static int mdPivotLockRefs( mdMesh *mesh, mdThreadData *tdata, mdLockBuffer *buffer, mdi vertexindex )
{
  int index;
  mdi triindex, iav, ibv, trirefcount;
  mdi *trireflist;
  mdTriangle *tri;
  mdVertex *vertex;

  vertex = &mesh->vertexlist[ vertexindex ];
  trireflist = &mesh->trireflist[ vertex->trirefbase ];
  trirefcount = vertex->trirefcount;

  for( index = 0 ; index < trirefcount ; index++ )
  {
    triindex = trireflist[ index ];
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;
    if( tri->v[0] == vertexindex )
    {
      iav = tri->v[1];
      ibv = tri->v[2];
    }
    else if( tri->v[1] == vertexindex )
    {
      iav = tri->v[2];
      ibv = tri->v[0];
    }
    else if( tri->v[2] == vertexindex )
    {
      iav = tri->v[0];
      ibv = tri->v[1];
    }
    else
    {
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
      continue;
    }
    if( !( mdLockBufferLock( mesh, tdata, buffer, iav ) ) )
      return 0;
    if( !( mdLockBufferLock( mesh, tdata, buffer, ibv ) ) )
      return 0;
  }

  return 1;
}


static void mdOpResolveLockEdge( mdMesh *mesh, mdThreadData *tdata, mdLockBuffer *lockbuffer, mdOp *op )
{
  int failcount, globalflag;
  mdVertex *vertex0, *vertex1;

  failcount = 0;
  globalflag = 0;
  for( ; ; )
  {
    if( ( failcount > MD_GLOBAL_LOCK_THRESHOLD ) && !( globalflag ) )
    {
      mdLockBufferUnlockAll( mesh, tdata, lockbuffer );
#if MD_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &mesh->globalvertexlock, 0x0, 0x1 );
#else
      mtSpinLock( &mesh->globalvertexspinlock );
#endif
      globalflag = 1;
    }
    if( !( mdLockBufferLock( mesh, tdata, lockbuffer, op->v0 ) ) || !( mdLockBufferLock( mesh, tdata, lockbuffer, op->v1 ) ) )
    {
      failcount++;
      continue;
    }
    vertex0 = &mesh->vertexlist[ op->v0 ];
    vertex1 = &mesh->vertexlist[ op->v1 ];
    if( vertex0->redirectindex != -1 )
      op->v0 = vertex0->redirectindex;
    else if( vertex1->redirectindex != -1 )
      op->v1 = vertex1->redirectindex;
    else
      break;
  }

#if MD_CONFIG_ATOMIC_SUPPORT
  if( globalflag )
    mmAtomicWrite32( &mesh->globalvertexlock, 0x0 );
#else
  if( globalflag )
    mtSpinUnlock( &mesh->globalvertexspinlock );
#endif

  return;
}

/* Return 1 on succesful lock, return 0 on failed lock (which also releases any lock) */
static int mdOpResolveLockEdgeTry( mdMesh *mesh, mdThreadData *tdata, mdLockBuffer *lockbuffer, mdOp *op )
{
  mdVertex *vertex0, *vertex1;
  for( ; ; )
  {
    if( !( mdLockBufferTryLock( mesh, tdata, lockbuffer, op->v0 ) ) || !( mdLockBufferTryLock( mesh, tdata, lockbuffer, op->v1 ) ) )
      return 0;
    vertex0 = &mesh->vertexlist[ op->v0 ];
    vertex1 = &mesh->vertexlist[ op->v1 ];
    if( vertex0->redirectindex != -1 )
      op->v0 = vertex0->redirectindex;
    else if( vertex1->redirectindex != -1 )
      op->v1 = vertex1->redirectindex;
    else
      break;
  }
  return 1;
}

/* Acquire lock for op edge and all trirefs vertices */
static void mdOpResolveLockFull( mdMesh *mesh, mdThreadData *tdata, mdLockBuffer *lockbuffer, mdOp *op )
{
  int failcount, globalflag;
  mdVertex *vertex0, *vertex1;

  failcount = 0;
  globalflag = 0;
  for( ; ; )
  {
    if( ( failcount > MD_GLOBAL_LOCK_THRESHOLD ) && !( globalflag ) )
    {
      mdLockBufferUnlockAll( mesh, tdata, lockbuffer );
#if MD_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &mesh->globalvertexlock, 0x0, 0x1 );
#else
      mtSpinLock( &mesh->globalvertexspinlock );
#endif
      globalflag = 1;
    }
    if( !( mdLockBufferLock( mesh, tdata, lockbuffer, op->v0 ) ) || !( mdLockBufferLock( mesh, tdata, lockbuffer, op->v1 ) ) )
    {
      failcount++;
      continue;
    }
    vertex0 = &mesh->vertexlist[ op->v0 ];
    vertex1 = &mesh->vertexlist[ op->v1 ];
    /* If vertices have collapsed away, they have been redirected, update the top to follow the redirect and retry lock */
    if( vertex0->redirectindex != -1 )
      op->v0 = vertex0->redirectindex;
    else if( vertex1->redirectindex != -1 )
      op->v1 = vertex1->redirectindex;
    else
    {
      mdLockBufferResize( lockbuffer, 2 + ( ( vertex0->trirefcount + vertex1->trirefcount ) << 1 ) );
      if( !( mdPivotLockRefs( mesh, tdata, lockbuffer, op->v0 ) ) || !( mdPivotLockRefs( mesh, tdata, lockbuffer, op->v1 ) ) )
      {
        failcount++;
        continue;
      }
      break;
    }
  }

#if MD_CONFIG_ATOMIC_SUPPORT
  if( globalflag )
    mmAtomicWrite32( &mesh->globalvertexlock, 0x0 );
#else
  if( globalflag )
    mtSpinUnlock( &mesh->globalvertexspinlock );
#endif

  return;
}


#define MD_EDGE_COLLAPSE_TRIREF_STATIC (512)

static void mdEdgeCollapse( mdMesh *mesh, mdThreadData *tdata, mdi v0, mdi v1, mdf *collapsepoint, int *growtriref )
{
  int index, delflags0, delflags1;
  long deletioncount;
  mdi newv, trirefcount, trirefmax, outer0, outer1;
  mdi *trireflist, *trirefstore;
  mdVertex *vertex0, *vertex1;
  mdi trirefstatic[MD_EDGE_COLLAPSE_TRIREF_STATIC];

  /* If v1 vertex is locked, then v1 must be the vertex we overwrite while v0 is deleted, so swap them */
  if( ( mesh->lockmap ) && mdGetVertexLockFlag( mesh, v1 ) )
  {
    newv = v0;
    v0 = v1;
    v1 = newv;
  }

#if DEBUG_VERBOSE_COLLAPSE
  printf( "Collapse %d,%d ; Point %f %f %f ( Overwrite %d ; Delete %d )\n", (int)v0, (int)v1, collapsepoint[0], collapsepoint[1], collapsepoint[2], (int)v0, (int)v1 );
#endif
  /* New vertex overwriting v0 */
  newv = v0;

  /* Collapse other custom vertex attributes */
  if( mesh->vertexmerge )
    mdEdgeCollapseMergeVertexAttribs( mesh, tdata, v0, v1, collapsepoint );

  /* Vertices of the collapsed edge */
  vertex0 = &mesh->vertexlist[ v0 ];
  vertex1 = &mesh->vertexlist[ v1 ];

  /* Delete the triangles on both sides of the edge and all associated edges */
  outer0 = mdEdgeCollapseDeleteTriangle( mesh, tdata, v0, v1, &delflags0 );
  outer1 = mdEdgeCollapseDeleteTriangle( mesh, tdata, v1, v0, &delflags1 );

  /* Track count of deletions */
  deletioncount = tdata->statusdeletioncount;
  if( outer0 != -1 )
    deletioncount++;
  if( outer1 != -1 )
    deletioncount++;
  tdata->statusdeletioncount = deletioncount;

#if DEBUG_VERBOSE_COLLAPSE
  printf( "  Redirect %d -> %d ; %d -> %d\n", (int)v0, (int)vertex0->redirectindex, (int)v1, (int)vertex1->redirectindex );
#endif

#if DEBUG_VERBOSE_COLLAPSE
  printf( "    Move Point %f %f %f ( %f %f %f ) -> %f %f %f\n", vertex0->point[0], vertex0->point[1], vertex0->point[2], vertex1->point[0], vertex1->point[1], vertex1->point[2], collapsepoint[0], collapsepoint[1], collapsepoint[2] );
#endif

#if MD_CONFIG_DISTANCE_BIAS
  vertex0->sumbias += mdVertexDistance( vertex0, vertex1 );
#endif

#if MD_CONF_LOCAL_VERTEX_ORIGINS
  /* We must move both v0->q and v1->q to the frame of reference "collapsepoint" */
  mathQuadricTranslate( &vertex0->quadric, collapsepoint[0] - vertex0->point[0], collapsepoint[1] - vertex0->point[1], collapsepoint[2] - vertex0->point[2] );
  mathQuadricTranslate( &vertex1->quadric, collapsepoint[0] - vertex1->point[0], collapsepoint[1] - vertex1->point[1], collapsepoint[2] - vertex1->point[2] );
  /* Sum quadrics */
  mathQuadricAddQuadric( &vertex0->quadric, &vertex1->quadric );
#else
  /* Sum quadrics */
  mathQuadricAddQuadric( &vertex0->quadric, &vertex1->quadric );
#endif

  /* Set up new vertex over v0 */
  MD_VectorCopy( vertex0->point, collapsepoint );

  /* Propagate boundaries from deleted triangles */
  if( delflags0 )
  {
    if( delflags0 & 0x1 )
      mdEdgeCollapsePropagateBoundary( mesh, newv, outer0 );
    if( delflags0 & 0x2 )
      mdEdgeCollapsePropagateBoundary( mesh, outer0, newv );
  }
  if( delflags1 )
  {
    if( delflags1 & 0x1 )
      mdEdgeCollapsePropagateBoundary( mesh, newv, outer1 );
    if( delflags1 & 0x2 )
      mdEdgeCollapsePropagateBoundary( mesh, outer1, newv );
  }

  /* Redirect vertex1 to vertex0 */
  vertex1->redirectindex = newv;

  /* Maximum theoritical count of triangle references for our new vertex, we need a chunk of memory that big */
  trirefmax = vertex0->trirefcount + vertex1->trirefcount;

  /* Buffer to temporarily store our new trirefs */
  trireflist = trirefstatic;
  if( trirefmax > MD_EDGE_COLLAPSE_TRIREF_STATIC )
    trireflist = malloc( trirefmax * sizeof(mdi) );

  /* Update all triangles connected to vertex0 and vertex1 */
  trirefstore = trireflist;
  trirefstore = mdEdgeCollapseUpdateAll( mesh, tdata, &mesh->trireflist[ vertex0->trirefbase ], vertex0->trirefcount, v0, newv, trirefstore );
  trirefstore = mdEdgeCollapseUpdateAll( mesh, tdata, &mesh->trireflist[ vertex1->trirefbase ], vertex1->trirefcount, v1, newv, trirefstore );

  /* Find where to store the trirefs */
  trirefcount = (int)( trirefstore - trireflist );

#if DEBUG_VERBOSE_COLLAPSE
  printf( "  TriRefCount %d ; Alloc %d\n", trirefcount, trirefmax );
#endif

  if( trirefcount > vertex0->trirefcount )
  {
    if( trirefcount <= vertex1->trirefcount )
      vertex0->trirefbase = vertex1->trirefbase;
    else
    {
      /* Multithreading, acquire lock ~ we need to grow the trireflist buffer, we'll store our stuff at the end */
#if MD_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &mesh->trireflock, 0x0, 0x1 );
#else
      mtSpinLock( &mesh->trirefspinlock );
#endif
      vertex0->trirefbase = mesh->trireflistcount;
#if 0
      printf( "Add %d trirefs, total %ld, alloc %ld\n", (int)trirefcount, (long)mesh->trireflistcount, (long)mesh->trireflistalloc );
#endif
      mesh->trireflistcount += trirefcount;
      if( ( mesh->trireflistalloc - mesh->trireflistcount ) < ( mesh->threadcount * MD_TRIREF_AVAIL_MIN_COUNT ) )
        *growtriref = 1;
      if( mesh->trireflistcount >= mesh->trireflistalloc )
        MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
#if MD_CONFIG_ATOMIC_SUPPORT
      mmAtomicWrite32( &mesh->trireflock, 0x0 );
#else
      mtSpinUnlock( &mesh->trirefspinlock );
#endif
    }
  }

  /* Mark vertex1 as unused */
  vertex1->trirefcount = 0;

  /* Store trirefs */
  vertex0->trirefcount = trirefcount;
  trirefstore = &mesh->trireflist[ vertex0->trirefbase ];
  for( index = 0 ; index < trirefcount ; index++ )
  {
#if DEBUG_VERBOSE_COLLAPSE
    mdTriangle *tri;
    tri = ADDRESS( mesh->trilist, trireflist[index] * mesh->trisize );
    printf( "    Triref %d : %d,%d,%d\n", index, tri->v[0], tri->v[1], tri->v[2] );
#endif
    trirefstore[index] = trireflist[index];
  }

  /* Invalidate all cost calculations in neighborhood of pivot vertex */
  mdVertexInvalidateAll( mesh, tdata, trireflist, trirefcount, newv );

  /* If buffer wasn't static, free it */
  if( trireflist != trirefstatic )
    free( trireflist );

  /* Verify if we should create new ops between newv and outer vertices of deleted triangles */
  mdEdgeCollapseLinkOuter( mesh, tdata, newv, outer0 );
  mdEdgeCollapseLinkOuter( mesh, tdata, newv, outer1 );

#if DEBUG_VERBOSE_COLLAPSE
  printf( "Collapse End %d,%d\n", (int)v0, (int)v1 );
  fflush( stdout );
#endif

  return;
}



////



typedef struct
{
  int collisionflag;
  mdi trileft;
  mdi triright;
} mdEdgeCollisionData;

static void mdEdgeCollisionCallback( void *opaque, void *entry, int newflag )
{
  mdEdge *edge;
  mdEdgeCollisionData *ecd;
  edge = entry;
  ecd = opaque;
  if( ( edge->triindex != ecd->trileft ) && ( edge->triindex != ecd->triright ) )
    ecd->collisionflag = 1;
  return;
}


/*
Prevent 2D collapses

Check all triangles attached to v1 that would have to attach back to v0
If any of the edge is already present in the hash table, deny the collapse
*/
static int mdEdgeCollisionCheck( mdMesh *mesh, mdThreadData *tdata, mdi v0, mdi v1 )
{
  int index, trirefcount, left, right;
  mdi triindex, vsrc, vdst;
  mdi *trireflist;
  mdEdge edge;
  mdTriangle *tri;
  mdVertex *vertex0, *vertex1;
  mdEdgeCollisionData ecd;

  vertex0 = &mesh->vertexlist[ v0 ];
  vertex1 = &mesh->vertexlist[ v1 ];
  if( vertex0->trirefcount < vertex1->trirefcount )
  {
    vsrc = v0;
    vdst = v1;
    trireflist = &mesh->trireflist[ vertex0->trirefbase ];
    trirefcount = vertex0->trirefcount;
  }
  else
  {
    vsrc = v1;
    vdst = v0;
    trireflist = &mesh->trireflist[ vertex1->trirefbase ];
    trirefcount = vertex1->trirefcount;
  }

#if DEBUG_VERBOSE_COLLISION
  printf( "Collision Check %d,%d\n", (int)v0, (int)v1 );
  printf( "  Src %d ; Dst %d\n", vsrc, vdst );
#endif

  /* Find the triangles that would be deleted so that we don't detect false collisions with them */
  ecd.collisionflag = 0;
  ecd.trileft = -1;
  edge.v[0] = v0;
  edge.v[1] = v1;
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
    ecd.trileft = edge.triindex;
  ecd.triright = -1;
  edge.v[0] = v1;
  edge.v[1] = v0;
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
    ecd.triright = edge.triindex;

  /* Check all trirefs for collision */
  for( index = 0 ; index < trirefcount ; index++ )
  {
    triindex = trireflist[ index ];
    if( ( triindex == ecd.trileft ) || ( triindex == ecd.triright ) )
      continue;
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;

#if DEBUG_VERBOSE_COLLISION
    printf( "    Tri %d : %d,%d,%d\n", index, tri->v[0], tri->v[1], tri->v[2] );
#endif

    if( tri->v[0] == vsrc )
    {
      left = 2;
      right = 1;
    }
    else if( tri->v[1] == vsrc )
    {
      left = 0;
      right = 2;
    }
    else if( tri->v[2] == vsrc )
    {
      left = 1;
      right = 0;
    }
    else
    {
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
      continue;
    }

    edge.v[0] = vdst;
    edge.v[1] = tri->v[right];
    mmHashLockCallEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, mdEdgeCollisionCallback, &ecd, 0 );
    if( ecd.collisionflag )
      return 0;
    edge.v[0] = tri->v[left];
    edge.v[1] = vdst;
    mmHashLockCallEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, mdEdgeCollisionCallback, &ecd, 0 );
    if( ecd.collisionflag )
      return 0;
  }

  return 1;
}



////



/* Mesh init step 0, allocate, NOT threaded */
static int mdMeshInit( mdMesh *mesh, size_t maxmemoryusage )
{
  int retval;
  mdf hashsizefactor;

  /* Allocate vertices, no extra room for vertices, we overwrite existing ones as we decimate */
  mesh->vertexlist = mmAlignAlloc( mesh->vertexalloc * sizeof(mdVertex), 0x40 );

  /* Allocate space for per-vertex lists of face references, including future vertices */
  mesh->trireflistcount = 0;
  mesh->trireflistalloc = ( 2 * 6 * mesh->tricount ) + ( mesh->threadcount * MD_TRIREF_AVAIL_MIN_COUNT );
  mesh->trireflist = malloc( mesh->trireflistalloc * sizeof(mdi) );

  /* Allocate triangles */
  mesh->trisize = ( sizeof(mdTriangle) + mesh->tridatasize + 0x7 ) & ~0x7;
  mesh->trilist = malloc( mesh->tricount * mesh->trisize );

  /* Allocate edge hash table */
  retval = 1;
  hashsizefactor = 1.7;
  if( !( mesh->operationflags & MD_FLAGS_NO_DECIMATION ) )
    retval = mdMeshHashInit( mesh, mesh->tricount, hashsizefactor, 7, maxmemoryusage );

#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicWrite32( &mesh->trireflock, 0x0 );
  mmAtomicWrite32( &mesh->globalvertexlock, 0x0 );
#else
  mtSpinInit( &mesh->trirefspinlock );
  mtSpinInit( &mesh->globalvertexspinlock );
  mtSpinInit( &mesh->trackspinlock );
#endif

  return retval;
}


/* Mesh init step 1, initialize vertices, threaded */
static void mdMeshInitVertices( mdMesh *mesh, mdThreadData *tdata, int threadcount )
{
  int vertexindex, vertexindexmax, vertexperthread;
  mdf factor;
  mdVertex *vertex;
  void *point;

  vertexperthread = ( mesh->vertexcount / threadcount ) + 1;
  vertexindex = tdata->threadid * vertexperthread;
  vertexindexmax = vertexindex + vertexperthread;
  if( vertexindexmax > mesh->vertexcount )
    vertexindexmax = mesh->vertexcount;

  factor = mesh->normalizationfactor;
  vertex = &mesh->vertexlist[vertexindex];
  point = ADDRESS( mesh->point, vertexindex * mesh->pointstride );
  for( ; vertexindex < vertexindexmax ; vertexindex++, vertex++ )
  {
#if MD_CONFIG_ATOMIC_SUPPORT
    mmAtomicWrite32( &vertex->atomicowner, -1 );
#else
    vertex->owner = -1;
    mtSpinInit( &vertex->ownerspinlock );
#endif
    mesh->vertexUserToNative( vertex->point, point, factor );
#if CPU_SSE_SUPPORT && !MD_CONF_DOUBLE_PRECISION
    vertex->point[3] = 0.0;
#endif
    vertex->trirefcount = 0;
    vertex->redirectindex = -1;
#if MD_CONFIG_DISTANCE_BIAS
    vertex->sumbias = 0.0;
#endif
    mathQuadricZero( &vertex->quadric );
    point = ADDRESS( point, mesh->pointstride );
  }

  return;
}


static inline void mdMeshForbidEdge( mdMesh *mesh, int v0, int v1 )
{
  int edgeflags;
  mdEdge edge;
  mdTriangle *tri;
  edge.v[0] = v0;
  edge.v[1] = v1;
  if( mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge ) == MM_HASH_SUCCESS )
  {
    tri = ADDRESS( mesh->trilist, edge.triindex * mesh->trisize );
    edgeflags = 0;
    if( tri->v[0] == v0 )
      edgeflags = ( tri->v[1] == v1 ? MD_EDGEFLAGS_DENYEDGE01 : MD_EDGEFLAGS_DENYEDGE20 );
    else if( tri->v[1] == v0 )
      edgeflags = ( tri->v[2] == v1 ? MD_EDGEFLAGS_DENYEDGE12 : MD_EDGEFLAGS_DENYEDGE01 );
    else if( tri->v[2] == v0 )
      edgeflags = ( tri->v[0] == v1 ? MD_EDGEFLAGS_DENYEDGE20 : MD_EDGEFLAGS_DENYEDGE12 );
    tri->u.edgeflags |= edgeflags;
  }
  return;
}


/* Mesh init step 2, initialize triangles, threaded */
static void mdMeshInitTriangles( mdMesh *mesh, mdThreadData *tdata, int threadcount )
{
  int i, triperthread, triindex, triindexmax;
  long buildtricount;
  void *indices, *tridata;
  mdTriangle *tri;
  mdVertex *vertex;
  mdEdge edge;
  mathQuadric q;

  triperthread = ( mesh->tricount / threadcount ) + 1;
  triindex = tdata->threadid * triperthread;
  triindexmax = triindex + triperthread;
  if( triindexmax > mesh->tricount )
    triindexmax = mesh->tricount;

  /* Initialize triangles */
  buildtricount = 0;
  indices = ADDRESS( mesh->indices, triindex * mesh->indicesstride );
  tridata = ADDRESS( mesh->tridata, triindex * mesh->tridatasize );
  tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
  edge.op = 0;
  for( ; triindex < triindexmax ; triindex++, indices = ADDRESS( indices, mesh->indicesstride ), tri = ADDRESS( tri, mesh->trisize ), tridata = ADDRESS( tridata, mesh->tridatasize ) )
  {
    mesh->indicesUserToNative( tri->v, indices );
#if DEBUG_VERBOSE_QUADRIC
    printf( "Triangle %d ; %d,%d,%d\n", triindex, (int)tri->v[0], (int)tri->v[1], (int)tri->v[2] );
#endif
#if DEBUG_VERBOSE_QUADRIC || DEBUG_VERBOSE_CHECKS
    if( ( tri->v[0] == tri->v[1] ) || ( tri->v[1] == tri->v[2] ) ||( tri->v[0] == tri->v[2] ) )
      printf( "    ERROR: Repeated indices in triangle %d ; %d,%d,%d\n", triindex, (int)tri->v[0], (int)tri->v[1], (int)tri->v[2] );
#endif
    tri->u.edgeflags = 0;
#if MD_CONF_LOCAL_VERTEX_ORIGINS
    mdTriangleComputeLocalQuadric( mesh, tri, &q );
#else
    mdTriangleComputeQuadric( mesh, tri, &q );
#endif
    for( i = 0 ; i < 3 ; i++ )
    {
      vertex = &mesh->vertexlist[ tri->v[i] ];
#if DEBUG_VERBOSE_QUADRIC
      printf( "  Accum quadric to vertex %d\n", (int)tri->v[i] );
#endif
#if MD_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &vertex->atomicowner, -1, tdata->threadid );
      mathQuadricAddQuadric( &vertex->quadric, &q );
      vertex->trirefcount++;
      mmAtomicWrite32( &vertex->atomicowner, -1 );
#else
      mtSpinLock( &vertex->ownerspinlock );
      mathQuadricAddQuadric( &vertex->quadric, &q );
      vertex->trirefcount++;
      mtSpinUnlock( &vertex->ownerspinlock );
#endif
    }
    if( mesh->tridatasize )
      memcpy( ADDRESS(tri,sizeof(mdTriangle)), tridata, mesh->tridatasize );

    if( !( mesh->operationflags & MD_FLAGS_NO_DECIMATION ) )
    {
      edge.triindex = triindex;
      edge.v[0] = tri->v[0];
      edge.v[1] = tri->v[1];
      if( mmHashLockAddEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) != MM_HASH_SUCCESS )
      {
#if DEBUG_VERBOSE_TOPOLOGY
        printf( "  WARNING: bad topology, collision on edge %d,%d\n", (int)edge.v[0], (int)edge.v[1] );
#endif
        tri->u.edgeflags |= MD_EDGEFLAGS_DENYEDGE01;
        mdMeshForbidEdge( mesh, edge.v[0], edge.v[1] );
        mdMeshForbidEdge( mesh, edge.v[1], edge.v[0] );
        tdata->statuscollisioncount++;
      }
      edge.v[0] = tri->v[1];
      edge.v[1] = tri->v[2];
      if( mmHashLockAddEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) != MM_HASH_SUCCESS )
      {
#if DEBUG_VERBOSE_TOPOLOGY
        printf( "  WARNING: bad topology, collision on edge %d,%d\n", (int)edge.v[0], (int)edge.v[1] );
#endif
        tri->u.edgeflags |= MD_EDGEFLAGS_DENYEDGE12;
        mdMeshForbidEdge( mesh, edge.v[0], edge.v[1] );
        mdMeshForbidEdge( mesh, edge.v[1], edge.v[0] );
        tdata->statuscollisioncount++;
      }
      edge.v[0] = tri->v[2];
      edge.v[1] = tri->v[0];
      if( mmHashLockAddEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge, 1 ) != MM_HASH_SUCCESS )
      {
#if DEBUG_VERBOSE_TOPOLOGY
        printf( "  WARNING: bad topology, collision on edge %d,%d\n", (int)edge.v[0], (int)edge.v[1] );
#endif
        tri->u.edgeflags |= MD_EDGEFLAGS_DENYEDGE20;
        mdMeshForbidEdge( mesh, edge.v[0], edge.v[1] );
        mdMeshForbidEdge( mesh, edge.v[1], edge.v[0] );
        tdata->statuscollisioncount++;
      }
    }

    buildtricount++;
    tdata->statusbuildtricount = buildtricount;
  }

  return;
}


/* Mesh init step 3, initialize vertex trirefbase, NOT threaded */
static void mdMeshInitTrirefs( mdMesh *mesh )
{
  mdi vertexindex;
  size_t trirefcount;
  mdVertex *vertex;

  /* Compute base of vertex triangle references */
  trirefcount = 0;
  vertex = mesh->vertexlist;
  for( vertexindex = 0 ; vertexindex < mesh->vertexcount ; vertexindex++, vertex++ )
  {
    vertex->trirefbase = trirefcount;
    trirefcount += vertex->trirefcount;
    vertex->trirefcount = 0;
  }
  mesh->trireflistcount = trirefcount;

  return;
}


/* Accumulate quadrics from boundaries or weighted edges as returned by user callback */
static inline void mdMeshAccumBoundaryEdges( mdMesh *mesh, mdTriangle *tri, mdVertex **trivertex )
{
  int hashread;
  mdf edgeweight, boundaryedgeexpand;
  mdEdge edge;
  mdTriangle *trilink;

  boundaryedgeexpand = mesh->boundaryedgeexpand;

  edge.v[0] = tri->v[1];
  edge.v[1] = tri->v[0];
  hashread = mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge );
  edgeweight = mesh->boundaryareafactor;
  if( !( tri->u.edgeflags & MD_EDGEFLAGS_DENYEDGE01 ) && ( hashread != MM_HASH_SUCCESS ) )
  {
#if DEBUG_VERBOSE_BOUNDARY
    printf( "Boundary %d,%d (%d)\n", tri->v[1], tri->v[0], tri->v[2] );
#endif
    tri->u.edgeflags |= MD_EDGEFLAGS_BOUNDARY01;
  }
  else if( ( mesh->edgeweight ) && ( hashread == MM_HASH_SUCCESS ) )
  {
    trilink = ADDRESS( mesh->trilist, edge.triindex * mesh->trisize );
    edgeweight *= mesh->edgeweight( ADDRESS( tri, sizeof(mdTriangle) ), ADDRESS( trilink, sizeof(mdTriangle) ) );
    if( edgeweight <= 0.0 )
      goto skip01;
#if DEBUG_VERBOSE_BOUNDARY
    printf( "Edge weight %d,%d (%d) ; %f\n", tri->v[1], tri->v[0], tri->v[2], edgeweight );
#endif
  }
  else
    goto skip01;
  mdMeshAccumulateBoundary( trivertex[0], trivertex[1], trivertex[2], edgeweight, boundaryedgeexpand );
  skip01:

  edge.v[0] = tri->v[2];
  edge.v[1] = tri->v[1];
  hashread = mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge );
  edgeweight = mesh->boundaryareafactor;
  if( !( tri->u.edgeflags & MD_EDGEFLAGS_DENYEDGE12 ) && ( hashread != MM_HASH_SUCCESS ) )
  {
#if DEBUG_VERBOSE_BOUNDARY
    printf( "Boundary %d,%d (%d)\n", tri->v[2], tri->v[1], tri->v[0] );
#endif
    tri->u.edgeflags |= MD_EDGEFLAGS_BOUNDARY12;
  }
  else if( ( mesh->edgeweight ) && ( hashread == MM_HASH_SUCCESS ) )
  {
    trilink = ADDRESS( mesh->trilist, edge.triindex * mesh->trisize );
    edgeweight *= mesh->edgeweight( ADDRESS( tri, sizeof(mdTriangle) ), ADDRESS( trilink, sizeof(mdTriangle) ) );
    if( edgeweight <= 0.0 )
      goto skip12;
#if DEBUG_VERBOSE_BOUNDARY
    printf( "Edge weight %d,%d (%d) ; %f\n", tri->v[2], tri->v[1], tri->v[0], edgeweight );
#endif
  }
  else
    goto skip12;
  mdMeshAccumulateBoundary( trivertex[1], trivertex[2], trivertex[0], edgeweight, boundaryedgeexpand );
  skip12:

  edge.v[0] = tri->v[0];
  edge.v[1] = tri->v[2];
  hashread = mmHashLockReadEntry( mesh->edgehashtable, &mdEdgeHashAccess, &edge );
  edgeweight = mesh->boundaryareafactor;
  if( !( tri->u.edgeflags & MD_EDGEFLAGS_DENYEDGE20 ) && ( hashread != MM_HASH_SUCCESS ) )
  {
#if DEBUG_VERBOSE_BOUNDARY
    printf( "Boundary %d,%d (%d)\n", tri->v[0], tri->v[2], tri->v[1] );
#endif
    tri->u.edgeflags |= MD_EDGEFLAGS_BOUNDARY20;
  }
  else if( ( mesh->edgeweight ) && ( hashread == MM_HASH_SUCCESS ) )
  {
    trilink = ADDRESS( mesh->trilist, edge.triindex * mesh->trisize );
    edgeweight *= mesh->edgeweight( ADDRESS( tri, sizeof(mdTriangle) ), ADDRESS( trilink, sizeof(mdTriangle) ) );
    if( edgeweight <= 0.0 )
      goto skip20;
#if DEBUG_VERBOSE_BOUNDARY
    printf( "Edge weight %d,%d (%d) ; %f\n", tri->v[0], tri->v[2], tri->v[1], edgeweight );
#endif
  }
  else
    goto skip20;
  mdMeshAccumulateBoundary( trivertex[2], trivertex[0], trivertex[1], edgeweight, boundaryedgeexpand );
  skip20:

  return;
}


/* Mesh init step 4, store vertex trirefs and accumulate boundary quadrics, threaded */
static void mdMeshBuildTrirefs( mdMesh *mesh, mdThreadData *tdata, int threadcount )
{
  int i, triperthread, triindex, triindexmax;
  long buildrefcount;
  mdTriangle *tri;
  mdi *trireflist;
  mdVertex *vertex, *trivertex[3];

  triperthread = ( mesh->tricount / threadcount ) + 1;
  triindex = tdata->threadid * triperthread;
  triindexmax = triindex + triperthread;
  if( triindexmax > mesh->tricount )
    triindexmax = mesh->tricount;

  /* Store vertex triangle references and accumulate boundary quadrics */
  buildrefcount = 0;
  tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
  trireflist = mesh->trireflist;
  for( ; triindex < triindexmax ; triindex++, tri = ADDRESS( tri, mesh->trisize ) )
  {
    for( i = 0 ; i < 3 ; i++ )
    {
      vertex = &mesh->vertexlist[ tri->v[i] ];
#if MD_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &vertex->atomicowner, -1, tdata->threadid );
      trireflist[ vertex->trirefbase + vertex->trirefcount++ ] = triindex;
      mmAtomicWrite32( &vertex->atomicowner, -1 );
#else
      mtSpinLock( &vertex->ownerspinlock );
      trireflist[ vertex->trirefbase + vertex->trirefcount++ ] = triindex;
      mtSpinUnlock( &vertex->ownerspinlock );
#endif
      trivertex[i] = vertex;
    }

    if( !( mesh->operationflags & MD_FLAGS_NO_DECIMATION ) )
      mdMeshAccumBoundaryEdges( mesh, tri, trivertex );

    buildrefcount++;
    tdata->statusbuildrefcount = buildrefcount;
  }

  return;
}


/* Mesh clean up */
static void mdMeshEnd( mdMesh *mesh )
{
#ifndef MD_CONFIG_ATOMIC_SUPPORT
  mdi index;
  mdVertex *vertex;
  vertex = mesh->vertexlist;
  for( index = 0 ; index < mesh->vertexcount ; index++, vertex++ )
    mtSpinDestroy( &vertex->ownerspinlock );
  mtSpinDestroy( &mesh->trirefspinlock );
  mtSpinDestroy( &mesh->globalvertexspinlock );
  mtSpinDestroy( &mesh->trackspinlock );
#endif
  mmAlignFree( mesh->vertexlist );
  free( mesh->trireflist );
  free( mesh->trilist );
  return;
}



////



static void mdMeshGrowTriRefBuffer( mdMesh *mesh, size_t trirefavailneed )
{
  size_t trirefalloc;
  mdBarrierLockGlobal( &mesh->workbarrier );
  trirefalloc = mesh->trireflistcount + trirefavailneed;
  if( trirefalloc > mesh->trireflistalloc )
  {
    trirefalloc += 4096;
    mesh->trireflistalloc = trirefalloc;
    mesh->trireflist = realloc( mesh->trireflist, mesh->trireflistalloc * sizeof(mdi) );
  }
  mdBarrierUnlockGlobal( &mesh->workbarrier );
  return;
}

/* Maximum count of trirefs required by op */
static int mdMeshCountOpTriRefNeed( mdMesh *mesh, mdOp *op )
{
  mdi trirefmax;
  mdVertex *vertex0, *vertex1;
  vertex0 = &mesh->vertexlist[op->v0];
  vertex1 = &mesh->vertexlist[op->v1];
  trirefmax = vertex0->trirefcount + vertex1->trirefcount;
  return trirefmax;
}

/* Count of available trirefs */
static size_t mdMeshTriRefAvail( mdMesh *mesh )
{
  size_t trirefavail;
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicSpin32( &mesh->trireflock, 0x0, 0x1 );
#else
  mtSpinLock( &mesh->trirefspinlock );
#endif
  trirefavail = mesh->trireflistalloc - mesh->trireflistcount;
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicWrite32( &mesh->trireflock, 0x0 );
#else
  mtSpinUnlock( &mesh->trirefspinlock );
#endif
  return trirefavail;
}



////



static void mdSortOp( mdMesh *mesh, mdThreadData *tdata, mdOp *op, int denyflag )
{
  mdf collapsecost;
  collapsecost = op->value + op->penalty;
  if( ( denyflag ) || ( collapsecost >= mesh->maxcollapseacceptcost ) )
  {
#if MD_CONFIG_ATOMIC_SUPPORT
    if( !( mmAtomicRead32( &op->flags ) & MD_OP_FLAGS_DETACHED ) )
    {
      mmBinSortRemove( tdata->binsort, op, op->collapsecost );
      mmAtomicOr32( &op->flags, MD_OP_FLAGS_DETACHED );
    }
#else
    mtSpinLock( &op->spinlock );
    if( !( op->flags & MD_OP_FLAGS_DETACHED ) )
    {
      mmBinSortRemove( tdata->binsort, op, op->collapsecost );
      op->flags |= MD_OP_FLAGS_DETACHED;
    }
    mtSpinUnlock( &op->spinlock );
#endif
  }
  else
  {
#if MD_CONFIG_ATOMIC_SUPPORT
    if( mmAtomicRead32( &op->flags ) & MD_OP_FLAGS_DETACHED )
    {
      mmBinSortAdd( tdata->binsort, op, collapsecost );
      mmAtomicAnd32( &op->flags, ~MD_OP_FLAGS_DETACHED );
    }
    else if( op->collapsecost != collapsecost )
      mmBinSortUpdate( tdata->binsort, op, op->collapsecost, collapsecost );
#else
    mtSpinLock( &op->spinlock );
    if( op->flags & MD_OP_FLAGS_DETACHED )
    {
      mmBinSortAdd( tdata->binsort, op, collapsecost );
      op->flags &= ~MD_OP_FLAGS_DETACHED;
    }
    else if( op->collapsecost != collapsecost )
      mmBinSortUpdate( tdata->binsort, op, op->collapsecost, collapsecost );
    mtSpinUnlock( &op->spinlock );
#endif
  }
  op->collapsecost = collapsecost;
  return;
}


static void mdUpdateOp( mdMesh *mesh, mdThreadData *tdata, mdOp *op, int32_t opflagsmask )
{
  int denyflag, flags;

#if MD_CONFIG_ATOMIC_SUPPORT
  for( ; ; )
  {
    flags = mmAtomicRead32( &op->flags );
    if( mmAtomicCmpReplace32( &op->flags, flags, flags & opflagsmask ) )
      break;
  }
#else
  mtSpinLock( &op->spinlock );
  flags = op->flags;
  op->flags &= opflagsmask;
  mtSpinUnlock( &op->spinlock );
#endif
  if( !( flags & MD_OP_FLAGS_UPDATE_NEEDED ) )
    return;
  if( flags & MD_OP_FLAGS_DELETED )
    return;
  if( flags & MD_OP_FLAGS_DELETION_PENDING )
  {
    if( !( flags & MD_OP_FLAGS_DETACHED ) )
      mmBinSortRemove( tdata->binsort, op, op->collapsecost );
    /* Race condition, flag the op as deleted but don't free it ~ Free them all at the end with FreeAll(). */
    /*    mmBlockFree( &tdata->opblock, op );  */
#if MD_CONFIG_ATOMIC_SUPPORT
    mmAtomicOr32( &op->flags, MD_OP_FLAGS_DELETED );
#else
    mtSpinLock( &op->spinlock );
    op->flags |= MD_OP_FLAGS_DELETED;
    mtSpinUnlock( &op->spinlock );
#endif
  }
  else
  {
    if( op->value < MD_OP_FAIL_VALUE )
      op->penalty = mdEdgeCollapsePenalty( mesh, tdata, op->v0, op->v1, op->collapsepoint, &denyflag );
    else
    {
      op->penalty = 0.0;
      denyflag = 0;
    }
    mdSortOp( mesh, tdata, op, denyflag );
#if DEBUG_VERBOSE_COST
    printf( "  Updated Op %d,%d ~ flags 0x%x ; Point %f %f %f ; Value %e ; Penalty %e ; Cost %e (max %e, ratio %f)\n", (int)op->v0, (int)op->v1, mmAtomicRead32( &op->flags ), op->collapsepoint[0], op->collapsepoint[1], op->collapsepoint[2], op->value, op->penalty, op->collapsecost, mesh->maxcollapsecost, op->collapsecost / mesh->maxcollapsecost );
#endif
  }
  return;
}

static void mdUpdateBufferOps( mdMesh *mesh, mdThreadData *tdata, mdUpdateBuffer *updatebuffer, mdLockBuffer *lockbuffer )
{
  int index;
  mdOp *op;

#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicSpin32( &updatebuffer->atomlock, 0x0, 0x1 );
#else
  mtSpinLock( &updatebuffer->spinlock );
#endif
  for( index = 0 ; index < updatebuffer->opcount ; index++ )
  {
    op = updatebuffer->opbuffer[index];
    if( mdOpResolveLockEdgeTry( mesh, tdata, lockbuffer, op ) )
    {
      mdUpdateOp( mesh, tdata, op, ~( MD_OP_FLAGS_UPDATE_QUEUED | MD_OP_FLAGS_UPDATE_NEEDED ) );
      mdLockBufferUnlockAll( mesh, tdata, lockbuffer );
    }
    else
    {
#if MD_CONFIG_ATOMIC_SUPPORT
      mmAtomicWrite32( &updatebuffer->atomlock, 0x0 );
#else
      mtSpinUnlock( &updatebuffer->spinlock );
#endif
      mdOpResolveLockEdge( mesh, tdata, lockbuffer, op );
      mdUpdateOp( mesh, tdata, op, ~( MD_OP_FLAGS_UPDATE_QUEUED | MD_OP_FLAGS_UPDATE_NEEDED ) );
      mdLockBufferUnlockAll( mesh, tdata, lockbuffer );
#if MD_CONFIG_ATOMIC_SUPPORT
      mmAtomicSpin32( &updatebuffer->atomlock, 0x0, 0x1 );
#else
      mtSpinLock( &updatebuffer->spinlock );
#endif
    }
  }
  updatebuffer->opcount = 0;
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicWrite32( &updatebuffer->atomlock, 0x0 );
#else
  mtSpinUnlock( &updatebuffer->spinlock );
#endif

  return;
}


static inline mdf mdfMeshProcessGetStepMaxCost( mdMesh *mesh, int stepindex )
{
  mdf stepf, maxcost;
  stepf = (mdf)stepindex / (mdf)mesh->syncstepcount;
  /* Spread out the steps based on x^2, very tight steps at first */
  stepf *= stepf;
  maxcost = mesh->maxcollapsecost * stepf;
  return maxcost;
}


/* The actual mesh decimation loop, per thread */
static int mdMeshProcessQueue( mdMesh *mesh, mdThreadData *tdata )
{
  int index, decimationcount, stepindex, growtriref;
  size_t trirefneed, trirefavail;
  long targetvertexcountmin, targetvertexcountmax, trackvertexcount;
  int32_t opflags;
  mdf maxcost;
  mdOp *op;
  mdLockBuffer lockbuffer;

  mdLockBufferInit( &lockbuffer, 2 );

  stepindex = 0;
  maxcost = 0.0;

#if DEBUG_VERBOSE_WORK >= 2
  printf( "Thread %d work, begin decimation, maxcollapsecost %f\n", mesh->maxcollapsecost );
#elif DEBUG_VERBOSE_WORK
  if( tdata->threadid == 0 )
    printf( "Begin decimation, maxcollapsecost %f\n", mesh->maxcollapsecost );
#endif

  decimationcount = 0;
  targetvertexcountmin = mesh->targetvertexcountmin;
  targetvertexcountmax = mesh->targetvertexcountmax;
  for( ; ; )
  {
    /* Update all ops flagged as requiring update */
    if( mesh->operationflags & MD_FLAGS_CONTINUOUS_UPDATE )
    {
      for( index = 0 ; index < mesh->updatebuffercount ; index++ )
        mdUpdateBufferOps( mesh, tdata, &tdata->updatebuffer[index], &lockbuffer );
    }

    /* Acquire first op from thread's "queue" */
    op = mmBinSortGetFirst( tdata->binsort, maxcost );
    if( !op )
    {
      /* TODO: Consider stealing an op from another queue? Many threads become idle, waiting for the next step */
      if( targetvertexcountmax )
      {
        mdBarrierSync( &mesh->workbarrier );
#if MD_CONFIG_ATOMIC_SUPPORT
        trackvertexcount = mmAtomicReadL( &mesh->trackvertexcount );
#else
        trackvertexcount = mesh->trackvertexcount;
#endif
        stepindex++;
        if( ( stepindex > mesh->syncstepcount ) && ( trackvertexcount < targetvertexcountmax ) )
          break;
        if( stepindex >= mesh->syncstepabort )
          break;
#if DEBUG_VERBOSE_WORK >= 2
        printf( "Thread %d work, wait to begin step %d\n", tdata->threadid, stepindex );
#endif
        mdBarrierSync( &mesh->workbarrier );
      }
      else
      {
        if( ++stepindex > mesh->syncstepcount )
          break;
#if DEBUG_VERBOSE_WORK >= 2
        printf( "Thread %d work, wait to begin step %d\n", tdata->threadid, stepindex );
#endif
        mdBarrierSync( &mesh->workbarrier );
      }
      maxcost = mdfMeshProcessGetStepMaxCost( mesh, stepindex );
#if DEBUG_VERBOSE_WORK >= 2
      printf( "Thread %d work, begin step %d, maxcost %e\n", tdata->threadid, stepindex, maxcost );
#elif DEBUG_VERBOSE_WORK
      if( tdata->threadid == 0 )
        printf( "Decimation, begin step %d, maxcost %e\n", stepindex, maxcost );
#endif
      /* Update all ops flagged as requiring update */
      if( !( mesh->operationflags & MD_FLAGS_CONTINUOUS_UPDATE ) )
      {
        for( index = 0 ; index < mesh->updatebuffercount ; index++ )
          mdUpdateBufferOps( mesh, tdata, &tdata->updatebuffer[index], &lockbuffer );
      }
      continue;
    }

#if DEBUG_VERBOSE || DEBUG_VERBOSE_COST
 #if MD_CONFIG_ATOMIC_SUPPORT
    printf( "Op step %d ; Edge %d,%d (0x%x) ; Point %f %f %f ; Value %e ; Penalty %e ; Cost %e\n", stepindex, op->v0, op->v1, mmAtomicRead32( &op->flags ), op->collapsepoint[0], op->collapsepoint[1], op->collapsepoint[2], op->value, op->penalty, op->collapsecost );
 #else
    printf( "Op step %d ; Edge %d,%d (0x%x) ; Point %f %f %f ; Value %e ; Penalty %e ; Cost %e\n", stepindex, op->v0, op->v1, op->flags, op->collapsepoint[0], op->collapsepoint[1], op->collapsepoint[2], op->value, op->penalty, op->collapsecost );
 #endif
#endif

    /* Check if a thread requested a global lock */
    mdBarrierCheckGlobal( &mesh->workbarrier );

    for( ; ; )
    {
      /* Acquire lock for op edge and all trirefs vertices */
      mdOpResolveLockFull( mesh, tdata, &lockbuffer, op );
      /* If the op may require many trirefs, make sure we have enough buffer for them */
      trirefneed = mdMeshCountOpTriRefNeed( mesh, op );
      if( trirefneed < MD_TRIREF_AVAIL_MIN_COUNT )
        break;
      trirefavail = mdMeshTriRefAvail( mesh );
      if( trirefavail >= ( trirefneed * mesh->threadcount ) )
        break;
      /* Release all locks for op */
      mdLockBufferUnlockAll( mesh, tdata, &lockbuffer );
      /* Grow triref buffer and try again */
      mdMeshGrowTriRefBuffer( mesh, trirefneed * mesh->threadcount );
    }

    /* If our op was flagged for update between mdUpdateBufferOps() and before we acquired lock, no big deal, catch the update */
#if MD_CONFIG_ATOMIC_SUPPORT
    opflags = mmAtomicRead32( &op->flags );
#else
    mtSpinLock( &op->spinlock );
    opflags = op->flags;
    mtSpinUnlock( &op->spinlock );
#endif
    if( opflags & MD_OP_FLAGS_UPDATE_NEEDED )
    {
      mdUpdateOp( mesh, tdata, op, ~MD_OP_FLAGS_UPDATE_NEEDED );
      mdLockBufferUnlockAll( mesh, tdata, &lockbuffer );
      continue;
    }

    growtriref = 0;

    /* Prevent 2D collapses */
    if( !( mdEdgeCollisionCheck( mesh, tdata, op->v0, op->v1 ) ) )
    {
#if MD_CONFIG_ATOMIC_SUPPORT
      if( mmAtomicRead32( &op->flags ) & MD_OP_FLAGS_DETACHED )
        MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
      mmAtomicOr32( &op->flags, MD_OP_FLAGS_DETACHED );
      mmBinSortRemove( tdata->binsort, op, op->collapsecost );
#else
      mtSpinLock( &op->spinlock );
      if( op->flags & MD_OP_FLAGS_DETACHED )
        MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
      op->flags |= MD_OP_FLAGS_DETACHED;
      mtSpinUnlock( &op->spinlock );
      mmBinSortRemove( tdata->binsort, op, op->collapsecost );
#endif
      goto opdone;
    }

    if( ( targetvertexcountmin | targetvertexcountmax ) )
    {
      /* Only track vertex count if we need it */
#if MD_CONFIG_ATOMIC_SUPPORT
      trackvertexcount = mmAtomicAddReadL( &mesh->trackvertexcount, -1 );
#else
      mtSpinLock( &mesh->trackspinlock );
      mesh->trackvertexcount--;
      trackvertexcount = mesh->trackvertexcount;
      mtSpinUnlock( &mesh->trackspinlock );
#endif
      /* Exit if we have reached our minimum count of vertices */
      if( targetvertexcountmin )
      {
        if( trackvertexcount <= targetvertexcountmin )
        {
          mdLockBufferUnlockAll( mesh, tdata, &lockbuffer );
          break;
        }
      }
      /* When targetvertexcountmax is enabled, _all_ ops are added to binsort queue */
      /* We continue until <targetvertexcountmax, or until op->collapsecost > mesh->maxcollapsecost */
      if( targetvertexcountmax )
      {
        /* Continue while count>max */
        if( ( trackvertexcount < targetvertexcountmax ) && ( op->collapsecost > mesh->maxcollapsecost ) )
        {
          mdLockBufferUnlockAll( mesh, tdata, &lockbuffer );
          break;
        }
      }
    }

    /* Perform the edge collapse */
    mdEdgeCollapse( mesh, tdata, op->v0, op->v1, op->collapsepoint, &growtriref );
    decimationcount++;

    opdone:
    /* Release all locks for op */
    mdLockBufferUnlockAll( mesh, tdata, &lockbuffer );

    /* Grow triref buffer if flagged as too small, we need MD_TRIREF_AVAIL_MIN_COUNT per thread */
    if( growtriref )
    {
      /* Check if a thread requested a global lock */
      mdBarrierCheckGlobal( &mesh->workbarrier );
      /* Grow triref buffer if still required */
      if( mdMeshTriRefAvail( mesh ) < ( mesh->threadcount * MD_TRIREF_AVAIL_MIN_COUNT ) )
        mdMeshGrowTriRefBuffer( mesh, mesh->threadcount * MD_TRIREF_AVAIL_MIN_COUNT );
    }
  }

  mdLockBufferEnd( &lockbuffer );

#if DEBUG_VERBOSE_WORK >= 2
  printf( "Thread %d work, end decimation, %d collapses\n", tdata->threadid, decimationcount );
#endif

#if DEBUG_VERBOSE_OUTPUT
  printf( "Final Count of Collapses : %d\n", decimationcount );
#endif

  return decimationcount;
}



//////



typedef struct
{
  mdf normal[3];
  mdf factor[3];
} mdTriNormal;


static mdi mdMeshPackCountTriangles( mdMesh *mesh )
{
  mdi tricount;
  mdTriangle *tri, *triend;

  tricount = 0;
  tri = mesh->trilist;
  triend = ADDRESS( tri, mesh->tricount * mesh->trisize );
  for( ; tri < triend ; tri = ADDRESS( tri, mesh->trisize ) )
  {
    if( tri->v[0] == -1 )
      continue;
    tri->u.redirectindex = tricount;
    tricount++;
  }

  mesh->tripackcount = tricount;
  return tricount;
}

static mdf mdMeshAngleFactor( mdf dotangle )
{
  mdf factor;
  if( dotangle >= 1.0 )
    factor = 0.0;
  else if( dotangle <= -1.0 )
    factor = 0.5 * M_PI;
  else
  {
    factor = mdfacos( dotangle );
    if( isnan( factor ) )
      factor = 0.0;
  }
  return factor;
}

static void mdMeshBuildTriangleNormals( mdMesh *mesh )
{
  mdTriangle *tri, *triend;
  mdVertex *vertex0, *vertex1, *vertex2;
  mdf vecta[3], vectb[3], vectc[3], normalfactor, magna, magnb, magnc, norm, norminv;
  mdTriNormal *trinormal;

  trinormal = mesh->trinormal;

  normalfactor = 1.0;
  if( mesh->operationflags & MD_FLAGS_TRIANGLE_WINDING_CCW )
    normalfactor = -1.0;

  tri = mesh->trilist;
  triend = ADDRESS( tri, mesh->tricount * mesh->trisize );
  for( ; tri < triend ; tri = ADDRESS( tri, mesh->trisize ) )
  {
    if( tri->v[0] == -1 )
      continue;

    /* Compute triangle normal */
    vertex0 = &mesh->vertexlist[ tri->v[0] ];
    vertex1 = &mesh->vertexlist[ tri->v[1] ];
    vertex2 = &mesh->vertexlist[ tri->v[2] ];
    MD_VectorSubStore( vecta, vertex1->point, vertex0->point );
    MD_VectorSubStore( vectb, vertex2->point, vertex0->point );
    MD_VectorCrossProduct( trinormal->normal, vectb, vecta );

    norm = mdfsqrt( MD_VectorDotProduct( trinormal->normal, trinormal->normal ) );
    if( norm )
    {
      norminv = normalfactor / norm;
      trinormal->normal[0] *= norminv;
      trinormal->normal[1] *= norminv;
      trinormal->normal[2] *= norminv;
    }

    MD_VectorSubStore( vectc, vertex2->point, vertex1->point );
    magna = MD_VectorMagnitude( vecta );
    magnb = MD_VectorMagnitude( vectb );
    magnc = MD_VectorMagnitude( vectc );
    trinormal->factor[0] = norm * mdMeshAngleFactor(  MD_VectorDotProduct( vecta, vectb ) / ( magna * magnb ) );
    trinormal->factor[1] = norm * mdMeshAngleFactor( -MD_VectorDotProduct( vecta, vectc ) / ( magna * magnc ) );
    trinormal->factor[2] = norm * mdMeshAngleFactor(  MD_VectorDotProduct( vectb, vectc ) / ( magnb * magnc ) );

    trinormal++;
  }

  return;
}


static int mdMeshVertexComputeNormal( mdMesh *mesh, mdi vertexindex, mdi *trireflist, int trirefcount, mdf *normal )
{
  int index, pivot, validflag;
  mdi triindex;
  mdf norm, norminv;
  mdTriangle *tri;
  mdTriNormal *trinormal, *tn;

  trinormal = mesh->trinormal;

  /* Loop through all trirefs associated with the vertex */
  validflag = 0;
  MD_VectorZero( normal );
  for( index = 0 ; index < trirefcount ; index++ )
  {
    triindex = trireflist[ index ];
    if( triindex == -1 )
      continue;
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;

    if( tri->v[0] == vertexindex )
      pivot = 0;
    else if( tri->v[1] == vertexindex )
      pivot = 1;
    else if( tri->v[2] == vertexindex )
      pivot = 2;
    else
    {
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
      continue;
    }

    tn = &trinormal[ tri->u.redirectindex ];
    MD_VectorAddMulScalar( normal, tn->normal, tn->factor[pivot] );
    validflag = 1;
  }

  if( !( validflag ) )
    return 0;

  norm = mdfsqrt( MD_VectorDotProduct( normal, normal ) );
  if( norm )
  {
    norminv = 1.0 / norm;
    normal[0] *= norminv;
    normal[1] *= norminv;
    normal[2] *= norminv;
  }

  return 1;
}


static mdi mdMeshCloneVertex( mdMesh *mesh, mdi cloneindex, mdf *point )
{
  mdi vertexindex, retindex;
  mdVertex *vertex;

  retindex = -1;
  vertex = &mesh->vertexlist[ mesh->clonesearchindex ];
  for( vertexindex = mesh->clonesearchindex ; vertexindex < mesh->vertexalloc ; vertexindex++, vertex++ )
  {
    if( ( vertexindex < mesh->vertexcount ) && ( vertex->trirefcount ) )
      continue;
    vertex->trirefcount = -1;
    vertex->redirectindex = -1;
    /* Copy the point from the cloned vertex */
    MD_VectorCopy( vertex->point, point );
    /* Copy custom vertex attributes, if any */
    if( mesh->vertexcopy )
      mesh->vertexcopy( mesh->copycontext, vertexindex, cloneindex );
    retindex = vertexindex;
    if( vertexindex >= mesh->vertexcount )
      mesh->vertexcount = vertexindex+1;
    break;
  }
  mesh->clonesearchindex = vertexindex;
  return retindex;
}


static void mdMeshVertexRedirectTriRefs( mdMesh *mesh, mdi vertexindex, mdi newvertexindex, mdi *trireflist, int trirefcount )
{
  int index;
  mdi triindex;
  mdTriangle *tri;

  for( index = 0 ; index < trirefcount ; index++ )
  {
    triindex = trireflist[ index ];
    if( triindex == -1 )
      continue;
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;
    if( tri->v[0] == vertexindex )
      tri->v[0] = newvertexindex;
    else if( tri->v[1] == vertexindex )
      tri->v[1] = newvertexindex;
    else if( tri->v[2] == vertexindex )
      tri->v[2] = newvertexindex;
    else
      MD_ERROR( "SHOULD NOT HAPPEN %s:%d\n", 1, __FILE__, __LINE__ );
  }

  return;
}


/* Find a target normal */
static int mdMeshVertexFindTarget( mdMesh *mesh, mdi *trireflist, int trirefcount, mdf **targetnormal )
{
  int i0, i1;
  mdi triindex0, triindex1;
  mdf dotangle, bestdotangle;
  mdTriangle *tri0, *tri1;
  mdTriNormal *trinormal, *tn0, *tn1;

  /* Of all triangles, find the most diverging pair, pick one */
  targetnormal[0] = 0;
  targetnormal[1] = 0;
  trinormal = mesh->trinormal;
  bestdotangle = mesh->normalsearchangle;
  for( i0 = 0 ; i0 < trirefcount ; i0++ )
  {
    triindex0 = trireflist[ i0 ];
    if( triindex0 == -1 )
      continue;
    tri0 = ADDRESS( mesh->trilist, triindex0 * mesh->trisize );
    if( tri0->v[0] == -1 )
      continue;
    tn0 = &trinormal[ tri0->u.redirectindex ];
    for( i1 = i0+1 ; i1 < trirefcount ; i1++ )
    {
      triindex1 = trireflist[ i1 ];
      if( triindex1 == -1 )
        continue;
      tri1 = ADDRESS( mesh->trilist, triindex1 * mesh->trisize );
      if( tri1->v[0] == -1 )
        continue;
      tn1 = &trinormal[ tri1->u.redirectindex ];
      dotangle = MD_VectorDotProduct( tn0->normal, tn1->normal );
      if( dotangle < bestdotangle )
      {
        bestdotangle = dotangle;
        targetnormal[0] = tn0->normal;
        targetnormal[1] = tn1->normal;
      }
    }
  }

  return ( targetnormal[0] != 0 );
}



#define MD_MESH_TRIREF_MAX (256)

static int mdMeshVertexBuildNormal( mdMesh *mesh, mdi vertexindex, mdi *trireflist, int trirefcount, mdf *point, mdf *normal )
{
  int index, trirefbuffercount;
  mdi triindex, newvertexindex;
  mdi trirefbuffer[MD_MESH_TRIREF_MAX];
  mdf dotangle0, dotangle1;
  mdf *newnormal, *targetnormal[2];
  mdTriangle *tri;
  mdTriNormal *trinormal, *tn;

  if( trirefcount > MD_MESH_TRIREF_MAX )
    return 1;

  /* Loop to repeat as we retire trirefs from the list */
  trinormal = mesh->trinormal;
  for( ; ; )
  {
    /* Compute normal for vertex */
    if( !( mdMeshVertexComputeNormal( mesh, vertexindex, trireflist, trirefcount, normal ) ) )
      return 0;

    /* If user doesn't allow vertex splitting, take the normal as it is */
    if( !( mesh->operationflags & MD_FLAGS_NORMAL_VERTEX_SPLITTING ) )
      break;

    /* Find a pair of target normals */
    if( !( mdMeshVertexFindTarget( mesh, trireflist, trirefcount, targetnormal ) ) )
      break;

    /* Find all trirefs that agree with targetnormal[1] and store them independently */
    trirefbuffercount = 0;
    for( index = 0 ; index < trirefcount ; index++ )
    {
      triindex = trireflist[ index ];
      if( triindex == -1 )
        continue;
      tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
      if( tri->v[0] == -1 )
        continue;
      tn = &trinormal[ tri->u.redirectindex ];
      dotangle1 = MD_VectorDotProduct( targetnormal[1], tn->normal );
      if( dotangle1 < mesh->normalsearchangle )
        continue;
      dotangle0 = MD_VectorDotProduct( targetnormal[0], tn->normal );
      if( dotangle0 > dotangle1 )
        continue;
      trirefbuffer[trirefbuffercount++] = triindex;
      trireflist[ index ] = -1;
    }
    if( !( trirefbuffercount ) )
      break;

    /* Find an unused vertex, bail out if none can be found */
    newvertexindex = mdMeshCloneVertex( mesh, vertexindex, point );
    if( newvertexindex == -1 )
      break;

    /* Correct all trirefs to new vertex */
    mdMeshVertexRedirectTriRefs( mesh, vertexindex, newvertexindex, trirefbuffer, trirefbuffercount );

    /* Spawn a new vertex */
    newnormal = ADDRESS( mesh->vertexnormal, newvertexindex * 3 * sizeof(mdf) );
    mdMeshVertexBuildNormal( mesh, newvertexindex, trirefbuffer, trirefbuffercount, point, newnormal );
  }

  return 1;
}


/* In some rare circumstances, a vertex can be unused even with redirectindex == -1 and trirefs leading to deleted triangles */
static int mdMeshVertexCheckUse( mdMesh *mesh, mdi *trireflist, int trirefcount )
{
  int index;
  mdi triindex;
  mdTriangle *tri;
  for( index = 0 ; index < trirefcount ; index++ )
  {
    triindex = trireflist[ index ];
    if( triindex == -1 )
      continue;
    tri = ADDRESS( mesh->trilist, triindex * mesh->trisize );
    if( tri->v[0] == -1 )
      continue;
    return 1;
  }
  return 0;
}


static void mdMeshWriteVertices( mdMesh *mesh )
{
  mdi vertexindex, writeindex;
  mdf factor;
  mdf *point;
  mdi *trireflist;
  mdVertex *vertex;

  factor = 1.0 / mesh->normalizationfactor;
  point = mesh->point;
  writeindex = 0;
  vertex = mesh->vertexlist;
  trireflist = mesh->trireflist;
  for( vertexindex = 0 ; vertexindex < mesh->vertexcount ; vertexindex++, vertex++ )
  {
    if( !( mesh->operationflags & MD_FLAGS_NO_VERTEX_PACKING ) )
    {
      if( vertex->redirectindex != -1 )
        continue;
      if( !( vertex->trirefcount ) )
        continue;
      if( ( vertex->trirefcount != -1 ) && !( mdMeshVertexCheckUse( mesh, &trireflist[ vertex->trirefbase ], vertex->trirefcount ) ) )
        continue;
    }
    vertex->redirectindex = writeindex;
    mesh->vertexNativeToUser( point, vertex->point, factor );
    if( ( mesh->vertexcopy ) && ( writeindex != vertexindex  ) )
      mesh->vertexcopy( mesh->copycontext, writeindex, vertexindex );
    point = ADDRESS( point, mesh->pointstride );
    writeindex++;
  }
  mesh->vertexpackcount = writeindex;
  if( mesh->operationflags & MD_FLAGS_NO_VERTEX_PACKING )
    mesh->vertexpackcount = mesh->vertexcount;
#if DEBUG_VERBOSE_OUTPUT
  printf( "Final vertex count: %d\n", (int)mesh->vertexpackcount );
#endif
  return;
}


static void mdMeshWriteIndices( mdMesh *mesh )
{
  mdi finaltricount, v[3];
  mdTriangle *tri, *triend;
  mdVertex *vertex0, *vertex1, *vertex2;
  void *indices, *tridata;

  indices = mesh->indices;
  finaltricount = 0;
  tri = mesh->trilist;
  triend = ADDRESS( tri, mesh->tricount * mesh->trisize );
  tridata = mesh->tridata;
#if DEBUG_VERBOSE_OUTPUT
  printf( "Final triangle list\n" );
#endif
  for( ; tri < triend ; tri = ADDRESS( tri, mesh->trisize ) )
  {
    if( tri->v[0] == -1 )
      continue;
    vertex0 = &mesh->vertexlist[ tri->v[0] ];
    v[0] = vertex0->redirectindex;
    vertex1 = &mesh->vertexlist[ tri->v[1] ];
    v[1] = vertex1->redirectindex;
    vertex2 = &mesh->vertexlist[ tri->v[2] ];
    v[2] = vertex2->redirectindex;
#if DEBUG_VERBOSE_OUTPUT
    printf( "  Tri %d ; %d,%d,%d -> %d,%d,%d\n", (int)finaltricount, (int)tri->v[0], (int)tri->v[1], (int)tri->v[2], (int)v[0], (int)v[1], (int)v[2] );
#endif
#if DEBUG_VERBOSE_OUTPUT || DEBUG_VERBOSE_CHECKS
    if( ( v[0] == v[1] ) || ( v[1] == v[2] ) ||( v[0] == v[2] ) )
      printf( "    ERROR: Repeated indices in triangle %d ; %d,%d,%d\n", (int)finaltricount, (int)v[0], (int)v[1], (int)v[2] );
    if( ( v[0] >= mesh->vertexpackcount ) || ( v[1] >= mesh->vertexpackcount ) ||( v[2] >= mesh->vertexpackcount ) )
      printf( "    ERROR: Out of range vertex in triangle %d ; %d,%d,%d >= %d\n", (int)finaltricount, (int)v[0], (int)v[1], (int)v[2], (int)mesh->vertexpackcount );
#endif

    mesh->indicesNativeToUser( indices, v );
    if( mesh->tridatasize )
    {
      memcpy( tridata, ADDRESS(tri,sizeof(mdTriangle)), mesh->tridatasize );
      tridata = ADDRESS( tridata, mesh->tridatasize );
    }
    indices = ADDRESS( indices, mesh->indicesstride );
    finaltricount++;
  }
#if DEBUG_VERBOSE_OUTPUT
  printf( "Final triangle count: %d\n", (int)finaltricount );
#endif

  mesh->tripackcount = finaltricount;
  return;
}


/* Write vertices and indices, recompute normals, store them along with vertices and indices at once */
static void mdMeshWriteVerticesAndNormals( mdMesh *mesh )
{
  mdi vertexindex, writeindex;
  mdf factor;
  mdf *point, *normal;
  mdVertex *vertex;
  mdi *trireflist;
  void (*writenormal)( void *dst, mdf *src );
  void *normaldst;

  /* Start search for free vertices to clone at 0 */
  mesh->clonesearchindex = 0;
  mesh->trinormal = malloc( mesh->tripackcount * sizeof(mdTriNormal) );
  mesh->vertexnormal = malloc( mesh->vertexalloc * 3 * sizeof(normal) );

  /* Count triangles and assign redirectindex to each in sequence */
  mdMeshPackCountTriangles( mesh );

  /* Build up mesh->trinormal, store normals, area and vertex angles of each triangle */
  mdMeshBuildTriangleNormals( mesh );

  /* Build each vertex normal */
  vertex = mesh->vertexlist;
  trireflist = mesh->trireflist;
  for( vertexindex = 0 ; vertexindex < mesh->vertexcount ; vertexindex++, vertex++ )
  {
    if( !( vertex->trirefcount ) || ( vertex->trirefcount == -1 ) )
      continue;
    normal = ADDRESS( mesh->vertexnormal, vertexindex * 3 * sizeof(mdf) );
    if( !( mdMeshVertexBuildNormal( mesh, vertexindex, &trireflist[ vertex->trirefbase ], vertex->trirefcount, vertex->point, normal ) ) )
      vertex->trirefcount = 0;
  }

  /* Write vertices along with normals and other attributes */
  factor = 1.0 / mesh->normalizationfactor;
  writenormal = mesh->writenormal;
  point = mesh->point;
  writeindex = 0;
  vertex = mesh->vertexlist;
  for( vertexindex = 0 ; vertexindex < mesh->vertexcount ; vertexindex++, vertex++ )
  {
    if( !( mesh->operationflags & MD_FLAGS_NO_VERTEX_PACKING ) )
    {
      if( vertex->redirectindex != -1 )
        continue;
      if( !( vertex->trirefcount ) )
        continue;
    }
    vertex->redirectindex = writeindex;
    mesh->vertexNativeToUser( point, vertex->point, factor );
    normal = ADDRESS( mesh->vertexnormal, vertexindex * 3 * sizeof(mdf) );
    normaldst = ADDRESS( mesh->normalbase, writeindex * mesh->normalstride );
    writenormal( normaldst, normal );
    if( ( mesh->vertexcopy ) && ( writeindex != vertexindex  ) )
      mesh->vertexcopy( mesh->copycontext, writeindex, vertexindex );
    point = ADDRESS( point, mesh->pointstride );
    writeindex++;
  }

  mesh->vertexpackcount = writeindex;
  if( mesh->operationflags & MD_FLAGS_NO_VERTEX_PACKING )
    mesh->vertexpackcount = mesh->vertexcount;
#if DEBUG_VERBOSE_OUTPUT
  printf( "Final vertex count: %d\n", (int)mesh->vertexpackcount );
#endif
  free( mesh->vertexnormal );
  free( mesh->trinormal );

  return;
}



//////



typedef struct
{
  int threadid;
  mdMesh *mesh;
  long deletioncount;
  long collisioncount;
  long decimationcount;
  mdThreadData *tdata;
  int stage;
} mdThreadInit;

#ifndef MD_CONFIG_ATOMIC_SUPPORT
int mdFreeOpCallback( void *chunk, void *userpointer )
{
  mdOp *op;
  op = chunk;
  mtSpinDestroy( &op->spinlock );
  return 0;
}
#endif


static void *mdThreadMain( void *value )
{
  int index, tribase, trimax, triperthread, nodeindex;
  int groupthreshold;
  mdThreadInit *tinit;
  mdThreadData tdata;
  mdMesh *mesh;

  tinit = value;
  mesh = tinit->mesh;
  tinit->tdata = &tdata;

  /* Thread memory initialization */
  memset( &tdata, 0, sizeof(mdThreadData) );
  tdata.threadid = tinit->threadid;
  tdata.statusbuildtricount = 0;
  tdata.statusbuildrefcount = 0;
  tdata.statuspopulatecount = 0;
  tdata.statusdeletioncount = 0;
  tdata.statuscollisioncount = 0;
  groupthreshold = mesh->tricount >> 10;
  if( groupthreshold < 256 )
    groupthreshold = 256;
  else if( groupthreshold > 4096 )
    groupthreshold = 4096;

  nodeindex = -1;
  if( ( mmcore.numa.capable ) && !( mesh->operationflags & MD_FLAGS_DISABLE_NUMA ) )
  {
    mmBindThreadToCpu( tdata.threadid );
    nodeindex = mmGetNodeForCpu( tdata.threadid );
    mmBlockNumaInit( &tdata.opblock, nodeindex, sizeof(mdOp), 16384, 16384, MD_CONF_OP_ALIGNMENT );
  }
  else
    mmBlockInit( &tdata.opblock, sizeof(mdOp), 16384, 16384, MD_CONF_OP_ALIGNMENT );

  if( !mesh->targetvertexcountmax )
    tdata.binsort = mmBinSortInit( offsetof(mdOp,list), 64, 32, -0.2 * mesh->maxcollapsecost, 1.2 * mesh->maxcollapsecost, groupthreshold, mdMeshOpValueCallback, 6, nodeindex );
  else
  {
    int rootbucketcount;
    double maxcostrange;
    rootbucketcount = 4096;
    maxcostrange = 64.0 * mesh->maxcollapsecost;
    tdata.binsort = mmBinSortInit( offsetof(mdOp,list), rootbucketcount, 16, -0.2 * mesh->maxcollapsecost, maxcostrange, groupthreshold, mdMeshOpValueCallback, 6, nodeindex );
  }

  for( index = 0 ; index < mesh->updatebuffercount ; index++ )
    mdUpdateBufferInit( &tdata.updatebuffer[index], 4096 );

  /* Wait until all threads have properly initialized */
  if( mesh->updatestatusflag )
    mdBarrierSync( &mesh->workbarrier );

  /* Build mesh step 1 */
  if( !( tdata.threadid ) )
    tinit->stage = MD_STATUS_STAGE_BUILDVERTICES;
  mdMeshInitVertices( mesh, &tdata, mesh->threadcount );
  mdBarrierSync( &mesh->workbarrier );

  /* Build mesh step 2 */
  if( !( tdata.threadid ) )
    tinit->stage = MD_STATUS_STAGE_BUILDTRIANGLES;
  mdMeshInitTriangles( mesh, &tdata, mesh->threadcount );
  mdBarrierSync( &mesh->workbarrier );

  /* Build mesh step 3 is not parallel, have the thread zero run it */
  if( !( tdata.threadid ) )
  {
    tinit->stage = MD_STATUS_STAGE_BUILDTRIREFS;
    /* Build mesh step 3 is not parallel, have the thread zero run it */
    mdMeshInitTrirefs( mesh );
  }
  mdBarrierSync( &mesh->workbarrier );

  /* Build mesh step 4 */
  mdMeshBuildTrirefs( mesh, &tdata, mesh->threadcount );
  mdBarrierSync( &mesh->workbarrier );

  if( !( mesh->operationflags & MD_FLAGS_NO_DECIMATION ) )
  {
    /* Initialize the thread's op queue */
    if( !( tdata.threadid ) )
      tinit->stage = MD_STATUS_STAGE_BUILDQUEUE;

    triperthread = ( mesh->tricount / mesh->threadcount ) + 1;
    tribase = tdata.threadid * triperthread;
    trimax = tribase + triperthread;
    if( trimax > mesh->tricount )
      trimax = mesh->tricount;

    /* Initialize a list of ops for all edges */
    mdMeshPopulateOpList( mesh, &tdata, tribase, trimax - tribase );

    /* Wait for all threads to reach this point */
    mdBarrierSync( &mesh->workbarrier );

    /* Process the thread's op queue */
    if( !( tdata.threadid ) )
      tinit->stage = MD_STATUS_STAGE_DECIMATION;
    tinit->decimationcount = mdMeshProcessQueue( mesh, &tdata );
  }

  /* We need to synchronize the work barrier first, in case we had a request for a global lock on it */
  mdBarrierSync( &mesh->workbarrier );

  /* Wait for all threads to reach this point */
  tinit->deletioncount = tdata.statusdeletioncount;
  tinit->collisioncount = tdata.statuscollisioncount;

  /* If we didn't use atomic operations, we have spinlocks to destroy in each op */
#ifndef MD_CONFIG_ATOMIC_SUPPORT
  mmBlockProcessList( &tdata.opblock, 0, mdFreeOpCallback );
#endif

  /* Free thread memory allocations */
  mmBlockFreeAll( &tdata.opblock );
  for( index = 0 ; index < mesh->updatebuffercount ; index++ )
    mdUpdateBufferEnd( &tdata.updatebuffer[index] );
  mmBinSortFree( tdata.binsort );

  /* Send finish signal */
  mtMutexLock( &mesh->finishmutex );
  tinit->tdata = 0;
  mesh->finishcount--;
  if( mesh->finishcount == 0 )
    mtSignalBroadcast( &mesh->finishsignal );
  mtMutexUnlock( &mesh->finishmutex );

  return 0;
}



//////////



static const char *mdStatusStageName[] =
{
 [MD_STATUS_STAGE_INIT] = "Initializing",
 [MD_STATUS_STAGE_BUILDVERTICES] = "Building Vertices",
 [MD_STATUS_STAGE_BUILDTRIANGLES] = "Building Triangles",
 [MD_STATUS_STAGE_BUILDTRIREFS] = "Building Trirefs",
 [MD_STATUS_STAGE_BUILDQUEUE] = "Building Queues",
 [MD_STATUS_STAGE_DECIMATION] = "Decimating Mesh",
 [MD_STATUS_STAGE_STORE] = "Storing Geometry",
 [MD_STATUS_STAGE_DONE] = "Done"
};

static double mdStatusStageProgress[] =
{
 [MD_STATUS_STAGE_INIT] = 0.0,
 [MD_STATUS_STAGE_BUILDVERTICES] = 2.0,
 [MD_STATUS_STAGE_BUILDTRIANGLES] = 6.0,
 [MD_STATUS_STAGE_BUILDTRIREFS] = 6.0,
 [MD_STATUS_STAGE_BUILDQUEUE] = 8.0,
 [MD_STATUS_STAGE_DECIMATION] = 75.0,
 [MD_STATUS_STAGE_STORE] = 3.0,
 [MD_STATUS_STAGE_DONE] = 0.0
};

static void mdUpdateStatus( mdMesh *mesh, mdThreadInit *threadinit, mdStatus *status )
{
  int threadid, stageindex;
  long buildtricount, buildrefcount, populatecount, deletioncount;
  double progress, subprogress;
  mdThreadInit *tinit;
  mdThreadData *tdata;

  buildtricount = 0;
  buildrefcount = 0;
  populatecount = 0;
  deletioncount = 0;
  for( threadid = 0 ; threadid < mesh->threadcount ; threadid++ )
  {
    tinit = &threadinit[threadid];
    tdata = tinit->tdata;
    if( tdata )
    {
      buildtricount += tdata->statusbuildtricount;
      buildrefcount += tdata->statusbuildrefcount;
      populatecount += tdata->statuspopulatecount;
      deletioncount += tdata->statusdeletioncount;
    }
    else
      deletioncount += tinit->deletioncount;
  }
  status->trianglecount = mesh->tricount - deletioncount;

  subprogress = 0.0;
  tinit = &threadinit[0];
  status->stage = tinit->stage;
  if( status->stage == MD_STATUS_STAGE_DECIMATION )
    subprogress = 1.0 - ( (double)status->trianglecount / (double)mesh->tricount );
  else if( status->stage == MD_STATUS_STAGE_BUILDQUEUE )
    subprogress = (double)populatecount / (double)mesh->tricount;
  else if( status->stage == MD_STATUS_STAGE_BUILDTRIANGLES )
    subprogress = (double)buildtricount / (double)mesh->tricount;
  else if( status->stage == MD_STATUS_STAGE_BUILDTRIREFS )
    subprogress = (double)buildrefcount / (double)mesh->tricount;
  subprogress = fmax( 0.0, fmin( 1.0, subprogress ) );

  progress = 0.0;
  status->stagename = mdStatusStageName[status->stage];
  for( stageindex = 0 ; stageindex < status->stage ; stageindex++ )
    progress += mdStatusStageProgress[stageindex];

  progress += subprogress * mdStatusStageProgress[status->stage];
  status->progress = progress;

  return;
}



//////////



void mdOperationInit( mdOperation *op )
{
  /* Input */
  memset( op, 0, sizeof(mdOperation) );
  /* Advanced settings, default values */
  op->compactnesstarget = MD_COLLAPSE_COST_COMPACTNESS_TARGET;
  op->compactnesspenalty = MD_COLLAPSE_COST_COMPACTNESS_FACTOR;
  op->boundaryweight = MD_BOUNDARY_WEIGHT;
  op->edgeexpand = 0.0625;
  op->boundaryedgeexpand = 0.125;
  op->biaslengthfactor = 0.125;
  op->biascostfactor = 0.125;
  op->syncstepcount = MD_SYNC_STEP_COUNT;
  op->syncstepabort = 1048576;
  op->normalsearchangle = 45.0;
  mmInit();
  if( mmcore.sysmemory )
  {
    op->maxmemoryusage = ( mmcore.sysmemory >> 1 ) + ( mmcore.sysmemory >> 2 ); /* By default, allow to allocate up to 75% of system memory */
    if( op->maxmemoryusage < 1024*1024*1024 )
      op->maxmemoryusage = 1024*1024*1024;
  }
  return;
}

void mdOperationData( mdOperation *op, size_t vertexcount, void *vertex, int vertexformat, size_t vertexstride, size_t tricount, void *indices, int indicesformat, size_t indicesstride )
{
  op->vertexcount = vertexcount;
  op->vertex = vertex;
  op->vertexformat = vertexformat;
  op->vertexstride = vertexstride;
  op->indices = indices;
  op->indicesformat = indicesformat;
  op->indicesstride = indicesstride;
  op->tricount = tricount;
  return;
}

void mdOperationStrength( mdOperation *op, double featuresize )
{
  op->featuresize = featuresize;
  return;
}

void mdOperationBoundaryWeight( mdOperation *op, double boundaryweight )
{
  op->boundaryweight = boundaryweight;
  return;
}

void mdOperationTriData( mdOperation *op, void *tridata, size_t tridatasize, double (*edgeweight)( void *tridata0, void *tridata1 ), double (*collapsemultiplier)( void *collapsecontext, void *tridata0, void *tridata1, double *point0, double *point1 ), void *collapsecontext )
{
  op->tridata = tridata;
  op->tridatasize = tridatasize;
  op->edgeweight = edgeweight;
  op->collapsemultiplier = collapsemultiplier;
  op->collapsecontext = collapsecontext;
  return;
}

void mdOperationVertexCopy( mdOperation *op, void (*vertexcopy)( void *copycontext, int dstindex, int srcindex ), void *copycontext )
{
  op->vertexcopy = vertexcopy;
  op->copycontext = copycontext;
  return;
}

void mdOperationVertexMerge( mdOperation *op, void (*vertexmerge)( void *mergecontext, int dstindex, int srcindex, double dstfactor, double srcfactor ), void *mergecontext )
{
  op->vertexmerge = vertexmerge;
  op->mergecontext = mergecontext;
  return;
}

void mdOperationAdjustCollapse( mdOperation *op, int (*adjustcollapsef)( void *adjustcontext, float *collapsepoint, float *v0point, float *v1point ), int (*adjustcollapsed)( void *adjustcontext, double *collapsepoint, double *v0point, double *v1point ), void *adjustcontext )
{
  op->adjustcollapsef = adjustcollapsef;
  op->adjustcollapsed = adjustcollapsed;
  op->adjustcontext = adjustcontext;
  return;
}

void mdOperationComputeNormals( mdOperation *op, void *base, int format, size_t stride )
{
  op->normalbase = base;
  op->normalformat = format;
  op->normalstride = stride;
  return;
}

void mdOperationStatusCallback( mdOperation *op, void (*statuscallback)( void *statuscontext, const mdStatus *status ), void *statuscontext, long milliseconds )
{
  op->statusmilliseconds = milliseconds;
  op->statuscontext = statuscontext;
  op->statuscallback = statuscallback;
  return;
}

void mdOperationLockVertex( mdOperation *op, long vertexindex )
{
  size_t mapsize;
  if( CC_UNLIKELY( !op->lockmap ) )
  {
    mapsize = ( ( op->vertexcount + (32-1) ) >> 5 ) * sizeof(uint32_t);
    op->lockmap = malloc( mapsize );
    memset( op->lockmap, 0, mapsize );
  }
  op->lockmap[ vertexindex >> 5 ] |= ((uint32_t)1) << ( vertexindex & (32-1) );
  return;
}

void mdOperationFreeLocks( mdOperation *op )
{
  if( op->lockmap )
  {
    free( op->lockmap );
    op->lockmap = 0;
  }
  return;
}



//////


struct mdState
{
  mdOperation *operation;
  mdMesh mesh;
  mdThreadInit threadinit[MD_THREAD_COUNT_MAX];
  mdStatus status;
};

static void mdMeshDecimationFree( mdState *state )
{
  mdMesh *mesh;
  mesh = &state->mesh;
  if( !( mesh->operationflags & MD_FLAGS_NO_DECIMATION ) )
    mdMeshHashEnd( mesh );
  mdMeshEnd( mesh );
  mdBarrierDestroy( &mesh->workbarrier );
  mtMutexDestroy( &mesh->finishmutex );
  mtSignalDestroy( &mesh->finishsignal );
  free( state );
  return;
}

/* Initialize state to decimate the mesh specified by the mdOperation struct */
mdState *mdMeshDecimationInit( mdOperation *operation, int threadcount, int flags )
{
  int threadindex;
  double featuresize, normalizationfactor;
  mdState *state;
  mdMesh *mesh;
  mdThreadInit *tinit;
  mdStatus *status;

  if( threadcount <= 0 )
    return 0;
  if( threadcount > MD_THREAD_COUNT_MAX )
    threadcount = MD_THREAD_COUNT_MAX;

  state = malloc( sizeof(mdState) );
  memset( state, 0, sizeof(mdState) );
  state->operation = operation;
  mesh = &state->mesh;
  status = &state->status;

  operation->decimationcount = 0;
  operation->msecs = 0;

  /* Get operation general settings */
  mesh->point = operation->vertex;
  mesh->pointstride = operation->vertexstride;
  mesh->vertexcount = operation->vertexcount;
  mesh->indices = operation->indices;
  mesh->indicesstride = operation->indicesstride;
  mesh->tridata = operation->tridata;
  mesh->tridatasize = operation->tridatasize;
  switch( operation->indicesformat )
  {
    case MD_FORMAT_BYTE:
    case MD_FORMAT_UBYTE:
      mesh->indicesUserToNative = mdIndicesCharToNative;
      mesh->indicesNativeToUser = mdIndicesNativeToChar;
      break;
    case MD_FORMAT_SHORT:
    case MD_FORMAT_USHORT:
      mesh->indicesUserToNative = mdIndicesShortToNative;
      mesh->indicesNativeToUser = mdIndicesNativeToShort;
      break;
    case MD_FORMAT_INT:
    case MD_FORMAT_UINT:
      mesh->indicesUserToNative = mdIndicesIntToNative;
      mesh->indicesNativeToUser = mdIndicesNativeToInt;
      break;
    case MD_FORMAT_INT8:
    case MD_FORMAT_UINT8:
      mesh->indicesUserToNative = mdIndicesInt8ToNative;
      mesh->indicesNativeToUser = mdIndicesNativeToInt8;
      break;
    case MD_FORMAT_INT16:
    case MD_FORMAT_UINT16:
      mesh->indicesUserToNative = mdIndicesInt16ToNative;
      mesh->indicesNativeToUser = mdIndicesNativeToInt16;
      break;
    case MD_FORMAT_INT32:
    case MD_FORMAT_UINT32:
      mesh->indicesUserToNative = mdIndicesInt32ToNative;
      mesh->indicesNativeToUser = mdIndicesNativeToInt32;
      break;
    case MD_FORMAT_INT64:
    case MD_FORMAT_UINT64:
      mesh->indicesUserToNative = mdIndicesInt64ToNative;
      mesh->indicesNativeToUser = mdIndicesNativeToInt64;
      break;
    default:
      goto error;
  }
  switch( operation->vertexformat )
  {
    case MD_FORMAT_FLOAT:
      mesh->vertexUserToNative = mdVertexFloatToNative;
      mesh->vertexNativeToUser = mdVertexNativeToFloat;
      break;
    case MD_FORMAT_DOUBLE:
      mesh->vertexUserToNative = mdVertexDoubleToNative;
      mesh->vertexNativeToUser = mdVertexNativeToDouble;
      break;
    case MD_FORMAT_SHORT:
      mesh->vertexUserToNative = mdVertexShortToNative;
      mesh->vertexNativeToUser = mdVertexNativeToShort;
      break;
    case MD_FORMAT_INT:
      mesh->vertexUserToNative = mdVertexIntToNative;
      mesh->vertexNativeToUser = mdVertexNativeToInt;
      break;
    case MD_FORMAT_INT16:
      mesh->vertexUserToNative = mdVertexInt16ToNative;
      mesh->vertexNativeToUser = mdVertexNativeToInt16;
      break;
    case MD_FORMAT_INT32:
      mesh->vertexUserToNative = mdVertexInt32ToNative;
      mesh->vertexNativeToUser = mdVertexNativeToInt32;
      break;
    default:
      goto error;
  }
  mesh->edgeweight = operation->edgeweight;
  mesh->collapsemultiplier = operation->collapsemultiplier;
  mesh->collapsecontext = operation->collapsecontext;
  mesh->vertexmerge = operation->vertexmerge;
  mesh->mergecontext = operation->mergecontext;
#if MD_CONF_DOUBLE_PRECISION
  mesh->adjustcollapse = (void *)operation->adjustcollapsed;
#else
  mesh->adjustcollapse = (void *)operation->adjustcollapsef;
#endif
  mesh->adjustcontext = operation->adjustcontext;
  mesh->vertexcopy = operation->vertexcopy;
  mesh->copycontext = operation->copycontext;
  mesh->tricount = operation->tricount;
  if( mesh->tricount < 2 )
    goto error;

  /* Pick normalization factor, ensure math doesn't explode with overflow/underflow in the x^6 math */
  featuresize = operation->featuresize;
  normalizationfactor = 1.0;

  /* WWW YYY ZZZ */
#if 0
  while( featuresize < 2.0 )
  {
    normalizationfactor *= 2.0;
    featuresize *= 2.0;
  }
  while( featuresize > 4.0 )
  {
    normalizationfactor *= 0.5;
    featuresize *= 0.5;
  }
#endif


#if DEBUG_VERBOSE_QUADRIC || DEBUG_VERBOSE_WORK
  printf( "Decimation scaling ~ normalizationfactor %f ~ featuresize %f\n", normalizationfactor, featuresize );
#endif

  /* pow( featuresize/4.0, 6.0 ) */
  mesh->invfeaturesizearea = 1.0 / ( featuresize * featuresize );
  mesh->maxcollapsecost = pow( 0.25 * featuresize, 6.0 );
  mesh->maxcollapseacceptcost = ( operation->targetvertexcountmax == 0 ? mesh->maxcollapsecost : MD_OP_FAIL_VALUE );
  mesh->normalizationfactor = normalizationfactor;
#if MD_CONFIG_DISTANCE_BIAS
  mesh->biasclampdistance = operation->biaslengthfactor * featuresize;
  mesh->biascostfactor = operation->biascostfactor * mesh->maxcollapseacceptcost / mesh->biasclampdistance;
#endif
  mesh->targetvertexcountmin = operation->targetvertexcountmin;
  mesh->targetvertexcountmax = operation->targetvertexcountmax;
#if MD_CONFIG_ATOMIC_SUPPORT
  mmAtomicWriteL( &mesh->trackvertexcount, operation->vertexcount );
#else
  mesh->trackvertexcount = operation->vertexcount;
#endif

  /* Record start time */
  operation->msecs = mmGetMillisecondsTime();

  mesh->threadcount = threadcount;
  mesh->operationflags = flags;

  /* To compute vertex normals */
  mesh->normalbase = operation->normalbase;
  mesh->normalformat = operation->normalformat;
  mesh->normalstride = operation->normalstride;
  if( mesh->normalbase )
  {
    switch( mesh->normalformat )
    {
      case MD_FORMAT_FLOAT:
        mesh->writenormal = mdNormalNativeToFloat;
        break;
      case MD_FORMAT_DOUBLE:
        mesh->writenormal = mdNormalNativeToDouble;
        break;
      case MD_FORMAT_BYTE:
        mesh->writenormal = mdNormalNativeToChar;
        break;
      case MD_FORMAT_SHORT:
        mesh->writenormal = mdNormalNativeToShort;
        break;
      case MD_FORMAT_INT8:
        mesh->writenormal = mdNormalNativeToInt8;
        break;
      case MD_FORMAT_INT16:
        mesh->writenormal = mdNormalNativeToInt16;
        break;
      case MD_FORMAT_INT_2_10_10_10_REV:
        mesh->writenormal = mdNormalNativeTo10_10_10_2;
        break;
      default:
        goto error;
    }
  }

  /* Vertex lock map */
  mesh->lockmap = operation->lockmap;

  /* Advanced configuration options */
  mesh->compactnesstarget = operation->compactnesstarget;
  mesh->compactnesspenalty = operation->compactnesspenalty;
  mesh->boundaryareafactor = featuresize * operation->boundaryweight;
  mesh->areaexpand = featuresize * featuresize * operation->edgeexpand;
  mesh->boundaryedgeexpand = featuresize * operation->boundaryedgeexpand;
  mesh->syncstepcount = operation->syncstepcount;
  mesh->syncstepabort = operation->syncstepabort;
  if( mesh->syncstepcount < 1 )
    mesh->syncstepcount = 1;
  if( mesh->syncstepcount > 1024 )
    mesh->syncstepcount = 1024;
  mesh->normalsearchangle = cos( 1.0 * operation->normalsearchangle * (M_PI/180.0) );
  if( mesh->normalsearchangle > 0.9 )
    mesh->normalsearchangle = 0.9;

  /* Synchronization */
  mdBarrierInit( &mesh->workbarrier, threadcount );

  /* Determine update buffer shift required, find the count of updatebuffers */
  for( mesh->updatebuffershift = 0 ; ( threadcount >> mesh->updatebuffershift ) > MD_THREAD_UPDATE_BUFFER_COUNTMAX ; mesh->updatebuffershift++ );
  mesh->updatebuffercount = ( ( threadcount - 1 ) >> mesh->updatebuffershift ) + 1;

  /* Runtime picking of collapse penalty computation path */
  mesh->collapsepenalty = mdEdgeCollapsePenaltyTriangle;
#if CPU_SSE_SUPPORT && 1
 #if !MD_CONF_DOUBLE_PRECISION
  #if CPU_SSE4_1_SUPPORT
    mesh->collapsepenalty = mdEdgeCollapsePenaltyTriangleSSE4p1f;
  #elif CPU_SSE3_SUPPORT
    mesh->collapsepenalty = mdEdgeCollapsePenaltyTriangleSSE3f;
  #elif CPU_SSE2_SUPPORT
    mesh->collapsepenalty = mdEdgeCollapsePenaltyTriangleSSE2f;
  #endif
 #else
  #if CPU_SSE4_1_SUPPORT
    mesh->collapsepenalty = mdEdgeCollapsePenaltyTriangleSSE4p1d;
  #elif CPU_SSE3_SUPPORT
    mesh->collapsepenalty = mdEdgeCollapsePenaltyTriangleSSE3d;
  #elif CPU_SSE2_SUPPORT
    mesh->collapsepenalty = mdEdgeCollapsePenaltyTriangleSSE2d;
  #endif
 #endif
#endif

  /* Finish status tracking */
  mesh->finishcount = threadcount;
  mtMutexInit( &mesh->finishmutex );
  mtSignalInit( &mesh->finishsignal );

  /* Initialize entire mesh storage */
  mesh->vertexalloc = operation->vertexalloc;
  if( mesh->vertexalloc < mesh->vertexcount )
    mesh->vertexalloc = mesh->vertexcount;
  if( !( mdMeshInit( mesh, operation->maxmemoryusage ) ) )
  {
    mdMeshDecimationFree( state );
    return 0;
  }

  /* Initialize thread init data */
  for( threadindex = 0 ; threadindex < mesh->threadcount ; threadindex++ )
  {
    tinit = &state->threadinit[threadindex];
    tinit->threadid = threadindex;
    tinit->mesh = mesh;
    tinit->stage = MD_STATUS_STAGE_INIT;
    tinit->tdata = 0;
  }

  /* Status update */
  mesh->updatestatusflag = 0;
  status->progress = 0.0;
  status->stage = MD_STATUS_STAGE_INIT;
  status->trianglecount = mesh->tricount;
  if( operation->statuscallback )
  {
    mesh->updatestatusflag = 1;
    mdUpdateStatus( mesh, state->threadinit, status );
    operation->statuscallback( operation->statuscontext, status );
  }

  return state;

  /* Free all global data */
  error:
  free( state );
  return 0;
}

/* Perform the work for specified thread, must be called synchronously for all threadcount */
void mdMeshDecimationThread( mdState *state, int threadindex )
{
  mdMesh *mesh;
  mdThreadInit *tinit;
  mesh = &state->mesh;
  if( threadindex < mesh->threadcount )
  {
    tinit = &state->threadinit[threadindex];
    tinit->threadid = threadindex;
    tinit->mesh = mesh;
    tinit->stage = MD_STATUS_STAGE_INIT;
    tinit->tdata = 0;
    mdThreadMain( tinit );
  }
  return;
}

/* Wait until the work has completed */
void mdMeshDecimationEnd( mdState *state )
{
  int threadid, threadcount;
  long statuswait;
  mdOperation *operation;
  mdMesh *mesh;
  mdThreadInit *threadinit;
  mdStatus *status;
  mdThreadInit *tinit;

  operation = state->operation;
  mesh = &state->mesh;
  threadinit = state->threadinit;
  status = &state->status;

  threadcount = mesh->threadcount;
  statuswait = ( operation->statusmilliseconds > 2 ? operation->statusmilliseconds : 2 );

  /* Wait for all threads to complete */
#if MD_CONF_ENABLE_PROGRESS
  if( !( mesh->updatestatusflag ) )
  {
    mtMutexLock( &mesh->finishmutex );
    while( mesh->finishcount )
      mtSignalWait( &mesh->finishsignal, &mesh->finishmutex );
    mtMutexUnlock( &mesh->finishmutex );
  }
  else
  {
    mtMutexLock( &mesh->finishmutex );
    while( mesh->finishcount )
    {
      mdUpdateStatus( mesh, threadinit, status );
      operation->statuscallback( operation->statuscontext, status );
      mtSignalWaitTimeout( &mesh->finishsignal, &mesh->finishmutex, statuswait );
    }
    mtMutexUnlock( &mesh->finishmutex );
  }
#else
  mtMutexLock( &mesh->finishmutex );
  while( mesh->finishcount )
    mtSignalWait( &barrier->signal, &barrier->mutex );
  mtMutexUnlock( &mesh->finishmutex );
#endif

  /* Count sums of all threads */
  operation->decimationcount = 0;
  operation->collisioncount = 0;
  tinit = threadinit;
  for( threadid = 0 ; threadid < threadcount ; threadid++, tinit++ )
  {
    operation->decimationcount += tinit->decimationcount;
    operation->collisioncount += tinit->collisioncount;
  }

  if( mesh->updatestatusflag )
  {
    threadinit->stage = MD_STATUS_STAGE_STORE;
    mdUpdateStatus( mesh, threadinit, status );
    operation->statuscallback( operation->statuscontext, status );
  }

  /* Write out the final mesh */
  if( ( mesh->normalbase ) && ( mesh->writenormal ) )
    mdMeshWriteVerticesAndNormals( mesh );
  else
    mdMeshWriteVertices( mesh );
  mdMeshWriteIndices( mesh );
  operation->vertexcount = mesh->vertexpackcount;
  operation->tricount = mesh->tripackcount;

  if( mesh->updatestatusflag )
  {
    threadinit->stage = MD_STATUS_STAGE_DONE;
    mdUpdateStatus( mesh, state->threadinit, status );
    operation->statuscallback( operation->statuscontext, status );
  }

  /* Requires mmhash.c compiled with MM_HASH_DEBUG_STATISTICS */
#if MM_HASH_DEBUG_STATISTICS
  mmHashPrintStatistics( mesh->edgehashtable );
#endif

  mdMeshDecimationFree( state );
  /* Store total processing time */
  operation->msecs = mmGetMillisecondsTime() - operation->msecs;

  return;
}


////


typedef struct
{
  mdState *state;
  int threadindex;
} mdMeshDecimationThreadLaunch;

static void *mdMeshDecimationThreadMain( void *value )
{
  mdMeshDecimationThreadLaunch *threadlaunch;
  threadlaunch = (mdMeshDecimationThreadLaunch *)value;
  mdMeshDecimationThread( threadlaunch->state, threadlaunch->threadindex );
  return 0;
}

int mdMeshDecimation( mdOperation *operation, int threadcount, int flags )
{
  int threadindex, maxthreadcount;
  mdState *state;
  mtThread thread[MD_THREAD_COUNT_MAX];
  mdMeshDecimationThreadLaunch threadlaunch[MD_THREAD_COUNT_MAX];

  maxthreadcount = operation->tricount / 128;
  if( maxthreadcount > MD_THREAD_COUNT_MAX )
    maxthreadcount = MD_THREAD_COUNT_MAX;
  if( maxthreadcount == 0 )
    maxthreadcount = 1;
  if( threadcount <= 0 )
  {
    threadcount = mmcore.cpucount;
    if( threadcount <= 0 )
      threadcount = MD_THREAD_COUNT_DEFAULT;
  }
  if( threadcount > maxthreadcount )
    threadcount = maxthreadcount;

  state = mdMeshDecimationInit( operation, threadcount, flags );
  if( !state )
    return 0;
  for( threadindex = 0 ; threadindex < threadcount ; threadindex++ )
  {
    threadlaunch[threadindex].state = state;
    threadlaunch[threadindex].threadindex = threadindex;
    mtThreadCreate( &thread[threadindex], mdMeshDecimationThreadMain, &threadlaunch[threadindex], MT_THREAD_FLAGS_JOINABLE );
  }
  mdMeshDecimationEnd( state );
  for( threadindex = 0 ; threadindex < threadcount ; threadindex++ )
    mtThreadJoin( &thread[threadindex] );

  return 1;
}


