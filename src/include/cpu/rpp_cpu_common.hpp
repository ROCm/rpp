#ifndef RPP_CPU_COMMON_H
#define RPP_CPU_COMMON_H

#include <math.h>
#include <algorithm>

#include <rppdefs.h>

#define PI 3.14159265
#define RAD(deg) (deg * PI / 180)
#define RPPABS(a)       ((a < 0) ? (-a) : (a))
#define RPPMIN2(a,b)    ((a < b) ? a : b)
#define RPPMIN3(a,b,c)  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define RPPMAX2(a,b)    ((a > b) ? a : b)
#define RPPMAX3(a,b,c)  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define RPPGAUSSIAN(x, sigma) (exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * PI * pow(sigma, 2)))
#define RPPDISTANCE(x, y, i, j) (sqrt(pow(x - i, 2) + pow(y - j, 2)))
#define RPPINRANGE(a, x, y) ((a >= x) && (a <= y) ? 1 : 0)

#endif //RPP_CPU_COMMON_H