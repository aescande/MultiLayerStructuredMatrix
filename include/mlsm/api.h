/* Copyright 2021 CNRS-AIST JRL*/

#pragma once

#if defined _WIN32 || defined __CYGWIN__
// On Microsoft Windows, use dllimport and dllexport to tag symbols.
#  define MLSM_DLLIMPORT __declspec(dllimport)
#  define MLSM_DLLEXPORT __declspec(dllexport)
#  define MLSM_DLLLOCAL
#else
// On Linux, for GCC >= 4, tag symbols using GCC extension.
#  if __GNUC__ >= 4
#    define MLSM_DLLIMPORT __attribute__((visibility("default")))
#    define MLSM_DLLEXPORT __attribute__((visibility("default")))
#    define MLSM_DLLLOCAL __attribute__((visibility("hidden")))
#  else
// Otherwise (GCC < 4 or another compiler is used), export everything.
#    define MLSM_DLLIMPORT
#    define MLSM_DLLEXPORT
#    define MLSM_DLLLOCAL
#  endif // __GNUC__ >= 4
#endif // defined _WIN32 || defined __CYGWIN__

#ifdef MLSM_STATIC
// If one is using the library statically, get rid of
// extra information.
#  define MLSM_DLLAPI
#  define MLSM_LOCAL
#else
// Depending on whether one is building or using the
// library define DLLAPI to import or export.
#  ifdef MLSM_EXPORTS
#    define MLSM_DLLAPI MLSM_DLLEXPORT
#  else
#    define MLSM_DLLAPI MLSM_DLLIMPORT
#  endif // MLSM_EXPORTS
#  define MLSM_LOCAL MLSM_DLLLOCAL
#endif // MLSM_STATIC
