/* -----------------------------------------------------------------------------
 *
 * Copyright (c) 2019-2023 Alexis Naveros.
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
 * -----------------------------------------------------------------------------
 */


#ifndef CC_STRICT_MATH
 #if defined(__GNUC__)
  #define CC_STRICT_MATH __attribute__((__optimize__("no-associative-math", "no-unsafe-math-optimizations", "no-fast-math")))
 #else
  #define CC_STRICT_MATH
 #endif
#endif


#define MATH_SHEWCHUK_STACK 48
// max bits = 1023 - (-1074) + 1 = 2098
// MATH_SHEWCHUK_STACK > ceil(2098/53) = 40 doubles


typedef struct
{
  int last;
  double p[MATH_SHEWCHUK_STACK];
} mathShewchukSum;

static inline CC_STRICT_MATH void mathShewchukInit( mathShewchukSum *sum )
{
  sum->last = 0;
  sum->p[0] = 0.0;
  return;
}

static inline CC_STRICT_MATH void mathShewchukAdd( mathShewchukSum *sum, double x )
{
  int i, j;
  double y, hi, lo;
  i = 0;
  for( j = 0 ; j <= sum->last ; j++ )
  {
    y = sum->p[j];
    hi = x + y;
    lo = hi - x;
    lo = ( y - lo ) + ( x - ( hi - lo ) );
    x = hi;
    if( lo )
    {
      sum->p[i++] = x;
      x = lo;
      /* Can only happen with NaN input? */
      if( i >= MATH_SHEWCHUK_STACK )
      {
        i = MATH_SHEWCHUK_STACK - 1;
        break;
      }
    }
  }
  sum->last = i;
  sum->p[i] = x;
  if( i == MATH_SHEWCHUK_STACK-1 )
    mathShewchukAdd( sum, 0.0 );
  return;
}

static inline CC_STRICT_MATH void mathShewchukMultiply( mathShewchukSum *sum, double x )
{
  int i;
  for( i = 0 ; i <= sum->last ; i++ )
    sum->p[i] *= x;
  return;
}

static inline CC_STRICT_MATH double mathShewchukTotal( mathShewchukSum *sum )
{
  int n;
  double y, r;
  double *pv;
  for( ; ; mathShewchukAdd( sum, 0.0 ) )
  {
    n = sum->last;
    if( n == 0 )
      return sum->p[0];
    if( n == 1 )
      return sum->p[0] + sum->p[1];
    pv = &sum->p[n];
    do
    {
      y = 0.5 * *pv--;
    } while( ( --n ) && ( *pv == ( *pv + y ) ) );
    if( *pv == ( *pv + y ) )
    {
      y += y;
      r = *pv + y;
      y = 2.0 * ( y - ( r - *pv ) );
      if( y != ( ( y + r ) - r ) )
        break;
      r = ( ( ( y < 0.0 ) == ( pv[2] < 0.0 ) && pv[2] ) ? r + y : r );
      break;
    }
  }
  return r;
}


