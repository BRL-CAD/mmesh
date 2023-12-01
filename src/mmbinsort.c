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


#include "cc.h"
#include "mm.h"

#include "mmbinsort.h"


////


/* Enable to use insertion sort items out of range for the very last bucket of the tree, to ensure proper order even when spread out */
#define MM_BINSORT_LASTBUCKET_INSERTION_SORT (1)


/* Should _never_ be used ~ available only for debugging */
#define DEBUG_INSERTION_SORT (0)

/* Should _never_ be used ~ available only for debugging */
#define DEBUG_NO_SUBGROUPS (0)


////


typedef float mmbsf;
#define mmbsffloor(x) floorf(x)
#define mmbsfceil(x) ceilf(x)



typedef struct
{
  int itemcount;
  int flags;
  void *p;
  void **last;
} mmBinSortBucket;

#define MM_BINSORT_BUCKET_FLAGS_SUBGROUP (0x1)


typedef struct
{
  mmbsf groupbase, groupmax, bucketrange;
  int bucketmax;
  mmBinSortBucket bucket[];
} mmBinSortGroup;


struct mmBinSort
{
  int numanodeindex;
  size_t memsize;
  size_t itemlistoffset;
  int rootbucketcount; /* Count of buckets for root */
  int groupbucketcount; /* Count of buckets per group */
  int bucketmaxsize; /* Max count of items in a bucket before we break it up in groups */
  int collapsethreshold;
  int maxdepth;

  /* Callback to user code to obtain the value of an item */
  double (*itemvalue)( void *item );

  /* Memory management */
  mmBlockHead bucketblock;
  mmBlockHead groupblock;

  /* Top-level group, *MUST* be at end of struct due to bucket[] zero-length array */
  mmBinSortGroup root;

};


////


mmBinSort *mmBinSortInit( size_t itemlistoffset, int rootbucketcount, int groupbucketcount, double rootmin, double rootmax, int bucketmaxsize, double (*itemvaluecallback)( void *item ), int maxdepth, int numanodeindex )
{
  int bucketindex;
  size_t memsize;
  mmBinSort *binsort;
  mmBinSortGroup *group;
  mmBinSortBucket *bucket;

  memsize = sizeof(mmBinSort) + ( rootbucketcount * sizeof(mmBinSortBucket) );
  if( numanodeindex >= 0 )
  {
    binsort = mmNumaAlloc( numanodeindex, memsize );
    mmBlockNumaInit( &binsort->bucketblock, numanodeindex, sizeof(mmBinSortBucket), 1024, 1024, 0x40 );
    mmBlockNumaInit( &binsort->groupblock, numanodeindex, sizeof(mmBinSortGroup) + ( groupbucketcount * sizeof(mmBinSortBucket) ), 16, 16, 0x40 );
  }
  else
  {
    binsort = malloc( memsize );
    mmBlockInit( &binsort->bucketblock, sizeof(mmBinSortBucket), 1024, 1024, 0x40 );
    mmBlockInit( &binsort->groupblock, sizeof(mmBinSortGroup) + ( groupbucketcount * sizeof(mmBinSortBucket) ), 16, 16, 0x40 );
  }
  binsort->numanodeindex = numanodeindex;
  binsort->memsize = memsize;

  binsort->itemlistoffset = itemlistoffset;
  binsort->rootbucketcount = rootbucketcount;
  binsort->groupbucketcount = groupbucketcount;
  binsort->bucketmaxsize = bucketmaxsize;
  binsort->collapsethreshold = bucketmaxsize >> 2;
  binsort->maxdepth = maxdepth;

  binsort->itemvalue = itemvaluecallback;

  group = &binsort->root;
  group->groupbase = rootmin;
  group->groupmax = rootmax;
  group->bucketrange = ( rootmax - rootmin ) / (double)rootbucketcount;
  group->bucketmax = rootbucketcount - 1;
  bucket = group->bucket;
  for( bucketindex = 0 ; bucketindex < rootbucketcount ; bucketindex++ )
  {
    bucket->flags = 0;
    bucket->itemcount = 0;
    bucket->p = 0;
    bucket++;
  }

  return binsort;
}


void mmBinSortFree( mmBinSort *binsort )
{
  mmBlockFreeAll( &binsort->bucketblock );
  mmBlockFreeAll( &binsort->groupblock );
  if( binsort->numanodeindex >= 0 )
    mmNumaFree( binsort->numanodeindex, binsort, binsort->memsize );
  else
    free( binsort );
  return;
}


static int MM_NOINLINE mmBinSortBucketIndex( mmBinSortGroup *group, mmbsf value )
{
  int bucketindex = (int)mmbsffloor( ( value - group->groupbase ) / group->bucketrange );
  if( bucketindex < 0 )
    bucketindex = 0;
  if( bucketindex > group->bucketmax )
    bucketindex = group->bucketmax;
  return bucketindex;
}


////


static mmBinSortGroup *mmBinSortSpawnGroup( mmBinSort *binsort, void *itembase, mmbsf base, mmbsf range, int lastflag )
{
  int bucketindex;
  mmbsf value, maxvalue;
  void *item, *itemnext;
  mmBinSortGroup *group;
  mmBinSortBucket *bucket;

  if( lastflag )
  {
    /* If this is the last bucket, we can raise the range based on the max value of contained items */
    maxvalue = -FLT_MAX;
    for( item = itembase ; item ; item = itemnext )
    {
      itemnext = ((mmListNode *)ADDRESS( item, binsort->itemlistoffset ))->next;
      value = binsort->itemvalue( item );
      maxvalue = fmaxf( maxvalue, value );
    }
    range = fmaxf( range, maxvalue - base );
  }

  group = mmBlockAlloc( &binsort->groupblock );
  group->groupbase = base;
  group->groupmax = base + range;
  group->bucketrange = range / (mmbsf)binsort->groupbucketcount;
  group->bucketmax = binsort->groupbucketcount - 1;

  bucket = group->bucket;
  for( bucketindex = 0 ; bucketindex < binsort->groupbucketcount ; bucketindex++ )
  {
    bucket->flags = 0;
    bucket->itemcount = 0;
    bucket->p = 0;
    bucket->last = &bucket->p;
    bucket++;
  }

  for( item = itembase ; item ; item = itemnext )
  {
    itemnext = ((mmListNode *)ADDRESS( item, binsort->itemlistoffset ))->next;
    value = binsort->itemvalue( item );
    bucketindex = mmBinSortBucketIndex( group, value );
    bucket = &group->bucket[ bucketindex ];
    bucket->itemcount++;
    mmListAdd( bucket->last, item, binsort->itemlistoffset );
    bucket->last = &((mmListNode *)ADDRESS( item, binsort->itemlistoffset ))->next;
  }

  return group;
}


static void mmBinSortCollapseGroup( mmBinSort *binsort, mmBinSortBucket *parentbucket )
{
  int bucketindex, itemcount;
  mmBinSortGroup *group;
  mmBinSortBucket *bucket;
  void *item, *itemnext;

  group = parentbucket->p;
  parentbucket->p = 0;
  parentbucket->last = &parentbucket->p;

  itemcount = 0;
  for( bucketindex = 0, bucket = group->bucket ; bucketindex < binsort->groupbucketcount ; bucketindex++, bucket++ )
  {
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
      mmBinSortCollapseGroup( binsort, bucket );
    for( item = bucket->p ; item ; item = itemnext )
    {
      itemnext = ((mmListNode *)ADDRESS( item, binsort->itemlistoffset ))->next;
      mmListAdd( parentbucket->last, item, binsort->itemlistoffset );
      parentbucket->last = &((mmListNode *)ADDRESS( item, binsort->itemlistoffset ))->next;
      itemcount++;
    }
  }
  parentbucket->flags &= ~MM_BINSORT_BUCKET_FLAGS_SUBGROUP;
  parentbucket->itemcount = itemcount;

  return;
}


////


void mmBinSortAdd( mmBinSort *binsort, void *item, double itemvalue )
{
  int bucketindex, depth, lastflag;
  mmbsf value;
  mmBinSortGroup *group, *subgroup;
  mmBinSortBucket *bucket;

  lastflag = 1;
  value = itemvalue;
  group = &binsort->root;
  for( depth = 0 ; ; group = bucket->p, depth++ )
  {
    bucketindex = mmBinSortBucketIndex( group, value );
    lastflag &= ( bucketindex == group->bucketmax );
    bucket = &group->bucket[ bucketindex ];
    bucket->itemcount++;
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
      continue;
#if !DEBUG_NO_SUBGROUPS
    if( ( bucket->itemcount >= binsort->bucketmaxsize ) && ( depth < binsort->maxdepth ) )
    {
      /* Build a new sub group, sort all entries into it */
      subgroup = mmBinSortSpawnGroup( binsort, bucket->p, group->groupbase + ( (mmbsf)bucketindex * group->bucketrange ), group->bucketrange, lastflag );
      bucket->flags |= MM_BINSORT_BUCKET_FLAGS_SUBGROUP;
      bucket->p = subgroup;
      continue;
    }
#endif

#if DEBUG_INSERTION_SORT
 #undef MM_BINSORT_LASTBUCKET_INSERTION_SORT
 #define MM_BINSORT_LASTBUCKET_INSERTION_SORT (1)
    lastflag = 1;
#endif

#if MM_BINSORT_LASTBUCKET_INSERTION_SORT
    /* When the item is out of root range, and in the very last bucket, add with insertion sort */
    if( !lastflag || ( value < binsort->root.groupmax ) )
      mmListAdd( &bucket->p, item, binsort->itemlistoffset );
    else
    {
      void *itemfind, *itemnext;
      void **insert;
      insert = &bucket->p;
      for( itemfind = bucket->p ; itemfind ; itemfind = itemnext )
      {
        itemnext = ((mmListNode *)ADDRESS( itemfind, binsort->itemlistoffset ))->next;
        if( itemvalue <= binsort->itemvalue( itemfind ) )
          break;
        insert = &((mmListNode *)ADDRESS( itemfind, binsort->itemlistoffset ))->next;
      }
      mmListAdd( insert, item, binsort->itemlistoffset );
    }
#else
    mmListAdd( &bucket->p, item, binsort->itemlistoffset );
#endif
    break;
  }

  return;
}

void mmBinSortRemove( mmBinSort *binsort, void *item, double itemvalue )
{
  int bucketindex;
  mmbsf value;
  mmBinSortGroup *group;
  mmBinSortBucket *bucket;

  value = itemvalue;
  group = &binsort->root;
  for( ; ; group = bucket->p )
  {
    bucketindex = mmBinSortBucketIndex( group, value );
    bucket = &group->bucket[ bucketindex ];
    bucket->itemcount--;
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
    {
      if( bucket->itemcount >= binsort->collapsethreshold )
        continue;
      mmBinSortCollapseGroup( binsort, bucket );
    }
    break;
  }
  mmListRemove( item, binsort->itemlistoffset );

  return;
}


void mmBinSortUpdate( mmBinSort *binsort, void *item, double olditemvalue, double newitemvalue )
{
  mmBinSortRemove( binsort, item, olditemvalue );
  mmBinSortAdd( binsort, item, newitemvalue );
  return;
}


////


static void *mmBinSortGroupGetFirst( mmBinSort *binsort, mmBinSortGroup *group, mmbsf failmax )
{
  int bucketindex, topbucket;
  void *item;
  mmBinSortBucket *bucket;
  mmBinSortGroup *subgroup;

  if( failmax > group->groupmax )
    topbucket = group->bucketmax;
  else
  {
    topbucket = (int)mmbsfceil( ( failmax - group->groupbase ) / group->bucketrange );
    if( ( topbucket < 0 ) || ( topbucket > group->bucketmax ) )
      topbucket = group->bucketmax;
  }
/*
printf( "  Failmax %f ; Base %f ; Range %f ; Index %f\n", failmax, group->groupbase, group->bucketrange, mmbsfceil( failmax - group->groupbase ) / group->bucketrange );
printf( "  Top bucket : %d\n", topbucket );
*/
  bucket = group->bucket;
  for( bucketindex = 0 ; bucketindex <= topbucket ; bucketindex++, bucket++ )
  {
/*
printf( "  Bucket %d ; Count %d ; Flags 0x%x ; %p\n", bucketindex, bucket->itemcount, bucket->flags, bucket->p );
*/
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
    {
      subgroup = bucket->p;
      item = mmBinSortGroupGetFirst( binsort, subgroup, failmax );
      if( item )
        return item;
    }
    else if( bucket->p )
      return bucket->p;
  }

  return 0;
}


void *mmBinSortGetFirst( mmBinSort *binsort, double failmax )
{
  int bucketindex, topbucket;
  mmBinSortGroup *group;
  mmBinSortBucket *bucket;
  void *item;

  if( failmax > binsort->root.groupmax )
    topbucket = binsort->root.bucketmax;
  else
  {
    topbucket = (int)mmbsfceil( ( (mmbsf)failmax - binsort->root.groupbase ) / binsort->root.bucketrange );
    if( ( topbucket < 0 ) || ( topbucket > binsort->root.bucketmax ) )
      topbucket = binsort->root.bucketmax;
  }

/*
  printf( "TopBucket : %d / %d\n", topbucket, binsort->root.bucketmax );
*/
  bucket = binsort->root.bucket;
  for( bucketindex = 0 ; bucketindex <= topbucket ; bucketindex++, bucket++ )
  {
/*
printf( "  Bucket %d ; Count %d ; Flags 0x%x ; %p\n", bucketindex, bucket->itemcount, bucket->flags, bucket->p );
*/
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
    {
      group = bucket->p;
      item = mmBinSortGroupGetFirst( binsort, group, (mmbsf)failmax );
      if( item )
        return item;
    }
    else if( bucket->p )
      return bucket->p;
  }

  return 0;
}


////


static void *mmBinSortGroupGetRemoveFirst( mmBinSort *binsort, mmBinSortGroup *group, mmbsf failmax )
{
  int bucketindex, topbucket;
  void *item;
  mmBinSortBucket *bucket;
  mmBinSortGroup *subgroup;

  if( failmax > group->groupmax )
    topbucket = group->bucketmax;
  else
  {
    topbucket = (int)mmbsfceil( ( failmax - group->groupbase ) / group->bucketrange );
    if( ( topbucket < 0 ) || ( topbucket > group->bucketmax ) )
      topbucket = group->bucketmax;
  }
/*
printf( "  Failmax %f ; Base %f ; Range %f ; Index %f\n", failmax, group->groupbase, group->bucketrange, mmbsfceil( failmax - group->groupbase ) / group->bucketrange );
printf( "  Top bucket : %d\n", topbucket );
*/
  bucket = group->bucket;
  for( bucketindex = 0 ; bucketindex <= topbucket ; bucketindex++, bucket++ )
  {
/*
printf( "  Bucket %d ; Count %d ; Flags 0x%x ; %p\n", bucketindex, bucket->itemcount, bucket->flags, bucket->p );
*/
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
    {
      subgroup = bucket->p;
      item = mmBinSortGroupGetRemoveFirst( binsort, subgroup, failmax );
      if( item )
      {
        bucket->itemcount--;
        if( bucket->itemcount < binsort->collapsethreshold )
          mmBinSortCollapseGroup( binsort, bucket );
        return item;
      }
    }
    else if( bucket->p )
    {
      item = bucket->p;
      bucket->itemcount--;
      return item;
    }
  }

  return 0;
}


void *mmBinSortGetRemoveFirst( mmBinSort *binsort, double failmax )
{
  int bucketindex, topbucket;
  mmBinSortGroup *group;
  mmBinSortBucket *bucket;
  void *item;

  if( failmax > binsort->root.groupmax )
    topbucket = binsort->root.bucketmax;
  else
  {
    topbucket = (int)mmbsfceil( ( (mmbsf)failmax - binsort->root.groupbase ) / binsort->root.bucketrange );
    if( ( topbucket < 0 ) || ( topbucket > binsort->root.bucketmax ) )
      topbucket = binsort->root.bucketmax;
  }

  bucket = binsort->root.bucket;
  for( bucketindex = 0 ; bucketindex <= topbucket ; bucketindex++, bucket++ )
  {
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
    {
      group = bucket->p;
      item = mmBinSortGroupGetRemoveFirst( binsort, group, (mmbsf)failmax );
      if( item )
      {
        bucket->itemcount--;
        if( bucket->itemcount < binsort->collapsethreshold )
          mmBinSortCollapseGroup( binsort, bucket );
        mmListRemove( item, binsort->itemlistoffset );
        return item;
      }
    }
    else if( bucket->p )
    {
      item = bucket->p;
      bucket->itemcount--;
      mmListRemove( item, binsort->itemlistoffset );
      return item;
    }
  }

  return 0;
}


////



/* Debugging */

static int mmBinSortDebugItemCount( mmBinSort *binsort, mmBinSortBucket *bucket )
{
  int itemcount;
  void  *item;
  itemcount = 0;
  for( item = bucket->p ; item ; item = ((mmListNode *)ADDRESS( item, binsort->itemlistoffset ))->next )
    itemcount++;
  return itemcount;
}


static int mmBinSortDebugGroupCount( mmBinSort *binsort, mmBinSortGroup *group, int depth )
{
  int itemcount, bucketindex;
  mmBinSortBucket *bucket;
  bucket = group->bucket;
  itemcount = 0;
  for( bucketindex = 0, bucket = group->bucket ; bucketindex < binsort->groupbucketcount ; bucketindex++, bucket++ )
  {
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
      itemcount += mmBinSortDebugGroupCount( binsort, bucket->p, depth + 1 );
    else
      itemcount += mmBinSortDebugItemCount( binsort, bucket );
  }
  return itemcount;
}


static int mmBinSortDebugBucket( mmBinSort *binsort, mmBinSortBucket *bucket )
{
  int itemcount;
  if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
    itemcount = mmBinSortDebugGroupCount( binsort, bucket->p, 0 );
  else
    itemcount = mmBinSortDebugItemCount( binsort, bucket );
  if( itemcount != bucket->itemcount )
  {
    printf( "mmBinSort : Item count mismatch %d != %d\n", itemcount, bucket->itemcount );
    return 0;
  }
  return 1;
}



static int mmBinSortDebugGroupPrint( mmBinSort *binsort, mmBinSortGroup *group, int depth )
{
  int itemcount, bucketindex, spacing;
  mmBinSortBucket *bucket;
  bucket = group->bucket;
  itemcount = 0;
  for( bucketindex = 0, bucket = group->bucket ; bucketindex < binsort->groupbucketcount ; bucketindex++, bucket++ )
  {
    for( spacing = 0 ; spacing < depth ; spacing++ )
      printf( "  " );
    printf( "Bucket %d/%d ; itemcount %d ; %s\n", bucketindex, binsort->groupbucketcount, bucket->itemcount, ( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP ? "Subgroups" : "LeafBucket" ) );
    if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
      mmBinSortDebugGroupPrint( binsort, bucket->p, depth + 1 );
  }
  return itemcount;
}


void mmBinSortBebug( mmBinSort *binsort, int verbose )
{
  int bucketindex;
  mmBinSortBucket *bucket;

  bucket = binsort->root.bucket;
  for( bucketindex = 0 ; bucketindex < binsort->root.bucketmax ; bucketindex++, bucket++ )
  {
    if( !mmBinSortDebugBucket( binsort, bucket ) )
      exit( 0 );
  }
  if( verbose )
  {
    printf( "mmBinSort %p\n", binsort );
    bucket = binsort->root.bucket;
    for( bucketindex = 0 ; bucketindex < binsort->root.bucketmax ; bucketindex++, bucket++ )
    {
      printf( "Bucket %d/%d ; itemcount %d ; %s\n", bucketindex, binsort->root.bucketmax, bucket->itemcount, ( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP ? "Subgroups" : "LeafBucket" ) );
      if( bucket->flags & MM_BINSORT_BUCKET_FLAGS_SUBGROUP )
        mmBinSortDebugGroupPrint( binsort, bucket->p, 0 );
    }
  }
  return;
}

