
#ifndef RPP_EXPORT_H
#define RPP_EXPORT_H

#ifdef RPP_STATIC_DEFINE
#  define RPP_EXPORT
#  define RPP_NO_EXPORT
#else
#  ifndef RPP_EXPORT
#    ifdef MIOpen_EXPORTS
        /* We are building this library */
#      define RPP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define RPP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef RPP_NO_EXPORT
#    define RPP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef RPP_DEPRECATED
#  define RPP_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef RPP_DEPRECATED_EXPORT
#  define RPP_DEPRECATED_EXPORT RPP_EXPORT RPP_DEPRECATED
#endif

#ifndef RPP_DEPRECATED_NO_EXPORT
#  define RPP_DEPRECATED_NO_EXPORT RPP_NO_EXPORT RPP_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef RPP_NO_DEPRECATED
#    define RPP_NO_DEPRECATED
#  endif
#endif

#endif /* RPP_EXPORT_H */
