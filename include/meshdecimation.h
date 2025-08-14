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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef COMPILER_DLLEXPORT
# if defined(_WIN32)
#  define COMPILER_DLLEXPORT __declspec(dllexport)
#  define COMPILER_DLLIMPORT __declspec(dllimport)
# else
#  define COMPILER_DLLEXPORT __attribute__ ((visibility ("default")))
#  define COMPILER_DLLIMPORT __attribute__ ((visibility ("default")))
# endif
#endif

#ifndef MMESH_EXPORT
#  if defined(MMESH_DLL_EXPORTS) && defined(MMESH_DLL_IMPORTS)
#    error "Only MMESH_DLL_EXPORTS or MMESH_DLL_IMPORTS can be defined, not both."
#  elif defined(MMESH_DLL_EXPORTS)
#    define MMESH_EXPORT COMPILER_DLLEXPORT
#  elif defined(MMESH_DLL_IMPORTS)
#    define MMESH_EXPORT COMPILER_DLLIMPORT
#  else
#    define MMESH_EXPORT
#  endif
#endif

typedef struct
{
  double progress;
  int stage;
  const char *stagename;
  long trianglecount;
} mdStatus;

enum
{
  MD_STATUS_STAGE_INIT,
  MD_STATUS_STAGE_BUILDVERTICES,
  MD_STATUS_STAGE_BUILDTRIANGLES,
  MD_STATUS_STAGE_BUILDTRIREFS,
  MD_STATUS_STAGE_BUILDQUEUE,
  MD_STATUS_STAGE_DECIMATION,
  MD_STATUS_STAGE_STORE,
  MD_STATUS_STAGE_DONE,

  MD_STATUS_STAGE_COUNT
};


typedef struct
{
  /* Input vertex data */
  size_t vertexcount;
  void *vertex;
  /* Supported vertex formats: MD_FORMAT_FLOAT, MD_FORMAT_DOUBLE, MD_FORMAT_SHORT, MD_FORMAT_INT, MD_FORMAT_INT16, MD_FORMAT_INT32 */
  int vertexformat;
  size_t vertexstride;
  /* Can allocate more vertices than the initial count, in order to have spare vertices for cloning when computing normals */
  size_t vertexalloc;

  /* Input indices data */
  void *indices;
  /* Supported index formats: MD_FORMAT_UBYTE, MD_FORMAT_USHORT, MD_FORMAT_UINT, MD_FORMAT_UINT8, MD_FORMAT_UINT16, MD_FORMAT_UINT32, MD_FORMAT_UINT64 */
  int indicesformat;
  size_t indicesstride;
  size_t tricount;

  /* Optional per-triangle custom data */
  void *tridata;
  size_t tridatasize;

  /* Optional callback, if tridata is provided, can return a non-zero edge factor between two connected triangles */
  double (*edgeweight)( void *tridata0, void *tridata1 );
  /* Optional callback, if tridata is provided, can return a cost multiplier for collapsing this edge ~ default should be 1.0 ~ a value<0.0 denies the edge for collapse */
  double (*collapsemultiplier)( void *collapsecontext, void *tridata0, void *tridata1, double *point0, double *point1 );
  void *collapsecontext;

  /* Optional callback, adjust XYZ of potential collapse point ~ return zero to deny that collapse */
  int (*adjustcollapsef)( void *adjustcontext, float *collapsepoint, float *v0point, float *v1point );
  int (*adjustcollapsed)( void *adjustcontext, double *collapsepoint, double *v0point, double *v1point );
  void *adjustcontext;

  /* Optional callback, merge attributes for two vertices with the given blending factors */
  void (*vertexmerge)( void *mergecontext, int dstindex, int srcindex, double weight0, double weight1 );
  void *mergecontext;

  /* Optional callback, copy custom vertex attributes, only used when repacking (moving) vertices to different indices */
  void (*vertexcopy)( void *mergecontext, int dstindex, int srcindex );
  void *copycontext;



#if 0
  /* Optional callback, provide the allowed collapse flags for a given edge */
  int (*merge
#endif



  /* Target feature size for decimation, how far to decimate the mesh, also scales internally a bunch of math */
  double featuresize;

  /* Factor by which to multiply the stiffness of boundaries, edges without opposite edges ~ default is 4.0 */
  double boundaryweight;


  /* EXPERIMENTAL */
  /* EXPERIMENTAL */
  /* EXPERIMENTAL */
  /* Any regular edge is expanded by (edgeexpand*featuresize) for the purpose of quadric weighting ~ default is 0.0625 */
  /* This value acts as a bias for even tiny triangles to have some weight */
  double edgeexpand;
  /* Any boundary edge is expanded by (boundaryedgeexpand*featuresize) for the purpose of quadric weighting ~ default is 0.125 */
  /* This value acts as a bias for even tiny boundary edges to have some weight */
  double boundaryedgeexpand;
  /* EXPERIMENTAL */
  /* EXPERIMENTAL */
  /* EXPERIMENTAL */



  /* Increase bias cost proportionally to edge length up to this length, computed as lengthbiasfactor*featuresize ~ default is 0.25 */
  double biaslengthfactor;
  /* Increase bias cost by a fraction of max collapse cost ~ default is 0.125 */
  double biascostfactor;



  /* Stop decimating at/below this count of vertices.  Set zero to disable */
  size_t targetvertexcountmin;
  /* Continue decimating while vertexcount > targetvertexcountmax.  Set zero to disable */
  size_t targetvertexcountmax;

  /* To compute vertex normals */
  void *normalbase;
  /* Supported normal formats: MD_FORMAT_FLOAT, MD_FORMAT_DOUBLE, MD_FORMAT_BYTE, MD_FORMAT_SHORT, MD_FORMAT_INT8, MD_FORMAT_INT16, MD_FORMAT_INT_2_10_10_10_REV */
  int normalformat;
  size_t normalstride;

  /* Output: Count of edge reductions */
  long decimationcount;
  /* Output: Count of edges that were reused by triangles */
  /* Any non-zero count indicates mesh topology errors in the input data */
  long collisioncount;

  /* Output: Time spent performing the decimation */
  long msecs;

  /* Status callback */
  long statusmilliseconds;
  void *statuscontext;
  void (*statuscallback)( void *statuscontext, const mdStatus *status );

  /* Optional vertex locking map, can be null if not used */
  uint32_t *lockmap;

  /* Advanced configuration options */
  double compactnesstarget; /* default 0.25 */
  double compactnesspenalty; /* default 1.0 */
  /* Count of buckets of queued pending operations, raise to increase quality when multithreaded ~ default is 64 */
  int syncstepcount;
  /* Abort decimation when the syncstep reaches this ~ can only happen with targetvertexcountmax>0 and featuresize waaay too small ~ default is 1048576 */
  int syncstepabort;
  /* For optional normal smoothing, maximum angle in degrees for merged smoothing */
  double normalsearchangle;
  /* Maximum memory usage, if possible ~ mdMeshDecimation() may still allocate more than that if necessary */
  size_t maxmemoryusage;

} mdOperation;


enum
{
  /* Type format for vertexformat, indicesformat, normalformat */
  MD_FORMAT_FLOAT,
  MD_FORMAT_DOUBLE,
  MD_FORMAT_BYTE,
  MD_FORMAT_SHORT,
  MD_FORMAT_INT,
  MD_FORMAT_UBYTE,
  MD_FORMAT_USHORT,
  MD_FORMAT_UINT,
  MD_FORMAT_INT8,
  MD_FORMAT_INT16,
  MD_FORMAT_INT32,
  MD_FORMAT_INT64,
  MD_FORMAT_UINT8,
  MD_FORMAT_UINT16,
  MD_FORMAT_UINT32,
  MD_FORMAT_UINT64,
  MD_FORMAT_INT_2_10_10_10_REV
};


/* Initialize mdOperation with default values */
MMESH_EXPORT void mdOperationInit( mdOperation *op );

/* Set vertex and indices input data */
MMESH_EXPORT void mdOperationData( mdOperation *op, size_t vertexcount, void *vertex, int vertexformat, size_t vertexstride, size_t tricount, void *indices, int indicesformat, size_t indicesstride );

/* Set decimation strength, feature size proportional to scale of model */
MMESH_EXPORT void mdOperationStrength( mdOperation *op, double featuresize );

/* Set optional boundary weight, default is 4.0 */
MMESH_EXPORT void mdOperationBoundaryWeight( mdOperation *op, double boundaryweight );

/* Set optional per-triangle data and callback to return a edge weight between two triangles */
MMESH_EXPORT void mdOperationTriData( mdOperation *op, void *tridata, size_t tridatasize, double (*edgeweight)( void *tridata0, void *tridata1 ), double (*collapsemultiplier)( void *collapsecontext, void *tridata0, void *tridata1, double *point0, double *point1 ), void *collapsecontext );

/* Set optional callback to copy vertex attributes (excluding vertex position) */
MMESH_EXPORT void mdOperationVertexCopy( mdOperation *op, void (*vertexcopy)( void *copycontext, int dstindex, int srcindex ), void *copycontext );

/* Set optional callback to blend vertex attributes (exclude vertex position) */
MMESH_EXPORT void mdOperationVertexMerge( mdOperation *op, void (*vertexmerge)( void *mergecontext, int dstindex, int srcindex, double weight0, double weight1 ), void *mergecontext );

/* Set optional callbacks to adjust the XYZ of potential collapse point */
MMESH_EXPORT void mdOperationAdjustCollapse( mdOperation *op, int (*adjustcollapsef)( void *adjustcontext, float *collapsepoint, float *v0point, float *v1point ), int (*adjustcollapsed)( void *adjustcontext, double *collapsepoint, double *v0point, double *v1point ), void *adjustcontext );

/* Set optional computation and storage of normals */
MMESH_EXPORT void mdOperationComputeNormals( mdOperation *op, void *base, int format, size_t stride );

/* Set optional callback to receive progress updates */
MMESH_EXPORT void mdOperationStatusCallback( mdOperation *op, void (*statuscallback)( void *statuscontext, const mdStatus *status ), void *statuscontext, long milliseconds );

/* Optional, flag vertex for locking, op->vertexcount must already be set ; op->lockmap is allocated by malloc() if null */
MMESH_EXPORT void mdOperationLockVertex( mdOperation *op, long vertexindex );

/* Optional, free op->lockmap if allocated and set to zero */
MMESH_EXPORT void mdOperationFreeLocks( mdOperation *op );



/* Decimate the mesh specified by the mdOperation struct */
MMESH_EXPORT int mdMeshDecimation( mdOperation *operation, int threadcount, int flags );


/* Slightly increase the quality of aggressive mesh decimations, about 50% slower (or >100% slower without SSE) */
#define MD_FLAGS_CONTINUOUS_UPDATE (0x1)
/* Do not pack vertices, leave unused vertices in place */
#define MD_FLAGS_NO_VERTEX_PACKING (0x2)
/* When recomputing normals, allow cloning/splitting of vertices with diverging normals */
#define MD_FLAGS_NORMAL_VERTEX_SPLITTING (0x4)
/* Define orientation of triangles when rebuilding normals */
#define MD_FLAGS_TRIANGLE_WINDING_CW (0x8)
#define MD_FLAGS_TRIANGLE_WINDING_CCW (0x10)
/* Do not actually do any decimation, in case all you actually want to do is to recompute normals */
#define MD_FLAGS_NO_DECIMATION (0x20)
/* For a height field, XY are assumed mostly planar, deny inversion of Z normals */
#define MD_FLAGS_PLANAR_MODE (0x40)
/* Disable allocating memory strictly from local NUMA nodes on which threads are locked */
#define MD_FLAGS_DISABLE_NUMA (0x80)


/* Low-level mesh decimation interface, allows reuse of external threads */

typedef struct mdState mdState;

/* Initialize state to decimate the mesh specified by the mdOperation struct */
MMESH_EXPORT mdState *mdMeshDecimationInit( mdOperation *operation, int threadcount, int flags );

/* Perform the work for specified thread, must be called synchronously for all threadcount (they wait for each other) */
MMESH_EXPORT void mdMeshDecimationThread( mdState *state, int threadindex );

/* Wait until the work has completed */
MMESH_EXPORT void mdMeshDecimationEnd( mdState *state );


#ifdef __cplusplus
}
#endif

