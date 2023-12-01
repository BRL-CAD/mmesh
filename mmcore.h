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

#ifndef MMCORE_H
#define MMCORE_H


////


/* CPU information and portable NUMA interface (without linking directly against libnuma) */

#define MM_CPU_COUNT_MAXIMUM (256)
#define MM_NODE_COUNT_MAXIMUM (32)
#define MM_STAGE_COUNT_MAXIMUM (16) /* >=log2(cpu_max) */

typedef struct
{
  /* Non-zero if NUMA calls are usable and strict */
  int capable;
  /* Count of NUMA nodes */
  int nodecount;
  /* Table of NUMA node for each cpu */
  uint16_t cpunode[MM_CPU_COUNT_MAXIMUM];
  /* Memory size for each NUMA node */
  uint64_t nodesize[MM_NODE_COUNT_MAXIMUM];
  /* Count of CPU cores for each NUMA node */
  uint16_t nodecpucount[MM_NODE_COUNT_MAXIMUM];

  /* Internal implementation-dependent table to assign our CPU indices to native indices */
  uint32_t cputable[MM_CPU_COUNT_MAXIMUM];

} mmCoreNuma;

typedef struct
{
  int vendor;
  int socketcount;
  int socketphysicalcores;
  int socketlogicalcores;
  char vendorstring[12+1];
  char identifier[48+1];
} mmCoreCpuid;

typedef struct
{
  int32_t lineL1code, sizeL1code, associativityL1code, sharedL1code;
  int32_t lineL1data, sizeL1data, associativityL1data, sharedL1data;
  int32_t unifiedL1;
  int32_t lineL2, sizeL2, associativityL2, sharedL2;
  int32_t lineL3, sizeL3, associativityL3, sharedL3;
} mmCoreCache;

typedef struct
{
  int stagecount; /* Count of stages */
  int stageblockcount[MM_STAGE_COUNT_MAXIMUM]; /* Count of blocks for that stage */
  int stagecorecount[MM_STAGE_COUNT_MAXIMUM]; /* Count of cores per block in that stage */
  int cpugroupsize; /* Recommended thread CPU lock group size, logical CPUs sharing the L1 cache */
} mmCoreWork;

typedef struct
{
  /* Usually 4096 bytes */
  int pagesize;
  /* Total count of active and configured CPU cores */
  int cpucount;
  /* Total system memory */
  int64_t sysmemory;
  /* Count of CPU cores, even offline ones ~ don't use externally ~ can we add CPU hot-plug support? */
  int cpuallcount;

  mmCoreNuma numa;
  mmCoreCpuid cpuid;
  mmCoreCache cache;
  mmCoreWork workmap;
} mmCore;

extern mmCore mmcore;

enum
{
  MM_CPUID_VENDOR_AMD,
  MM_CPUID_VENDOR_INTEL,
  MM_CPUID_VENDOR_UNKNOWN
};



////


void mmInit();
void mmEnd();


int mmGetCpuCount();

int mmGetNodeCount();

uint64_t mmGetNodeSize( int nodeindex );

int mmGetNodeForCpu( int cpuindex );

int mmGetNodeCpuCount( int nodeindex );


void mmBindThreadToCpu( int cpuindex );
void mmBindThreadToCpuGroup( int cpuindex, int cpucount );
void mmBindThreadToNode( int nodeindex );


void *mmNumaAlloc( int nodeindex, size_t size );
void mmNumaFree( int nodeindex, void *v, size_t size );
int mmNumaMigrate( int nodeindex, void *start, size_t size );


void mmPrintSystemTopology();


////


/* Enable NUMA memory leak debugging */
#define MM_NUMA_DEBUG (0)

#define MM_NUMA_DEBUG_PRINT_INTERVAL (5000)

#if MM_NUMA_DEBUG
void *mmDebugNumaAlloc( int nodeindex, size_t size );
void mmDebugNumaFree( int nodeindex, void *v, size_t size );
 #define mmNumaAlloc mmDebugNumaAlloc
 #define mmNumaFree mmDebugNumaFree
#endif


////


#endif

