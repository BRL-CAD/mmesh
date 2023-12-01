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


typedef struct mmBinSort mmBinSort;


mmBinSort *mmBinSortInit( size_t itemlistoffset, int rootbucketcount, int groupbucketcount, double rootmin, double rootmax, int bucketmaxsize, double (*itemvaluecallback)( void *item ), int maxdepth, int numanodeindex );
void mmBinSortFree( mmBinSort *binsort );

void mmBinSortAdd( mmBinSort *binsort, void *item, double itemvalue );
void mmBinSortRemove( mmBinSort *binsort, void *item, double itemvalue );
void mmBinSortUpdate( mmBinSort *binsort, void *item, double olditemvalue, double newitemvalue );

void *mmBinSortGetFirst( mmBinSort *binsort, double failmax );

void *mmBinSortGetRemoveFirst( mmBinSort *binsort, double failmax );


void mmBinSortBebug( mmBinSort *binsort, int verbose );

