#ifndef BRASS_EXPORT_H
#define BRASS_EXPORT_H

/* Symbol export macros shared by brass's C headers (brass_c_api.h and
 * embedding/brass_c_api.h). BRASS_BUILD_SHARED is defined while building the
 * brass shared library; consumers of that library define BRASS_SHARED. */
#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(BRASS_BUILD_SHARED)
    #define BRASS_API __declspec(dllexport)
  #elif defined(BRASS_SHARED)
    #define BRASS_API __declspec(dllimport)
  #else
    #define BRASS_API
  #endif
  #define BRASS_CALL __cdecl
#else
  #if defined(BRASS_BUILD_SHARED)
    #define BRASS_API __attribute__((visibility("default")))
  #else
    #define BRASS_API
  #endif
  #define BRASS_CALL
#endif

#endif /* BRASS_EXPORT_H */
