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

#include <stddef.h> /* for size_t */

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



#define MO_FLAGS_FIXED_CACHE_SIZE (0x1)
#define MO_FLAGS_DISABLE_LOOK_AHEAD (0x2)
#define MO_FLAGS_ENABLE_LAZY_SEARCH (0x4)
#define MO_FLAGS_FAST_SEED_SELECT (0x8)


/* Optimize the mesh using the specified count of threads */
MMESH_EXPORT int moOptimizeMesh( size_t vertexcount, size_t tricount, void *indices, int indiceswidth, size_t indicesstride, void (*shufflecallback)( void *opaquepointer, long newvertexindex, long oldvertexindex ), void *shuffleopaquepointer, int vertexcachesize, int threadcount, int flags );


/* Low-level mesh optimization interface, allows reuse of external threads */

typedef struct moMesh moMesh;

/* Initialize mesh to optimize the mesh specified */
MMESH_EXPORT moMesh *moMeshOptimizationInit( size_t vertexcount, size_t tricount, void *indices, int indiceswidth, size_t indicesstride, void (*shufflecallback)( void *opaquepointer, long newvertexindex, long oldvertexindex ), void *shuffleopaquepointer, int vertexcachesize, int threadcount, int flags );

/* Perform the work for specified thread, must be called synchronously for all threadcount (they wait for each other) */
MMESH_EXPORT void moMeshOptimizationThread( moMesh *mesh, int threadindex );

/* Wait until the work has completed */
MMESH_EXPORT void moMeshOptimizationEnd( moMesh *mesh );


/*
Returns the ACMR (Average Cache Miss Rate) for the mesh.
ACMR is the sum of vertex cache miss divided by the number of triangles in the mesh. 

flags argument should be zero for now
*/
MMESH_EXPORT double moEvaluateMesh( size_t tricount, void *indices, int indiceswidth, size_t indicesstride, int vertexcachesize, int flags );


#ifdef __cplusplus
}
#endif

