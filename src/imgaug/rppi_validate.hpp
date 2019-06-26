#ifndef RPPI_VALIDATE_OPERATIONS_FUNCTIONS
#define RPPI_VALIDATE_OPERATIONS_FUNCTIONS
#include <iostream>
#include <stdlib.h>
#include <rppdefs.h>

inline void validate_image_size(RppiSize imgSize){
    if(!(imgSize.width >= 0) || !(imgSize.height >= 0)){
        std::cerr<<"\n Image width and height should be positive ";
        exit(0);
    }
}
inline void validate_float_range(Rpp32f min, Rpp32f max, Rpp32f value) {
    if( !(value <= max) || !(value >= min)){
        std::cerr<<"\nOut of bounds: "<<value;
        exit(0);
    }
}

inline void validate_int_range(Rpp32s min, Rpp32s max, Rpp32s value) {
    if( !(value <= max) || !(value >= min)){
        std::cerr<<"\nOut of bounds: "<<value;
        exit(0);
    }
}

inline void validate_int_max(Rpp32s max, Rpp32s value) {
    if( !(value <=max) ){
       std::cerr<<"\nOut of bounds: "<<value;
       exit(0);
    }
}

inline void validate_int_min(Rpp32s min, Rpp32s value) {
    if( !(value >= min) ){
       std::cerr<<"\nOut of bounds: "<<value;
       exit(0);
    }
}

inline void validate_float_max(Rpp32f max, Rpp32f value) {
    if( !(value <=max) ){
       std::cerr<<"\nOut of bounds: "<<value;
       exit(0);
    }
}

inline void validate_float_min(Rpp32f min, Rpp32f value) {
    if( !(value >= min) ){
       std::cerr<<"\nOut of bounds: "<<value;
       exit(0);
    }
}
#endif