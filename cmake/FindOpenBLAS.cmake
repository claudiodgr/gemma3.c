# ============================================================================
# FindOpenBLAS.cmake - Find OpenBLAS library
# ============================================================================
#
# This module finds the OpenBLAS library for optimized BLAS operations.
#
# Defines:
#   OpenBLAS_FOUND        - System has OpenBLAS
#   OpenBLAS_INCLUDE_DIRS - OpenBLAS include directories
#   OpenBLAS_LIBRARIES    - OpenBLAS libraries
#
# Imported targets:
#   OpenBLAS::OpenBLAS    - OpenBLAS library target
#
# ============================================================================

include(FindPackageHandleStandardArgs)

# Platform-specific search paths
set(_OpenBLAS_SEARCH_PATHS "")

if(WIN32)
    # Windows: Check common install locations
    list(APPEND _OpenBLAS_SEARCH_PATHS
        "C:/OpenBLAS"
        "C:/opt/OpenBLAS"
        "$ENV{OPENBLAS_HOME}"
        "$ENV{PROGRAMFILES}/OpenBLAS"
    )
    # vcpkg integration
    if(DEFINED CMAKE_TOOLCHAIN_FILE)
        get_filename_component(_vcpkg_root "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
        get_filename_component(_vcpkg_root "${_vcpkg_root}" DIRECTORY)
        list(APPEND _OpenBLAS_SEARCH_PATHS
            "${_vcpkg_root}/installed/x64-windows"
            "${_vcpkg_root}/installed/x86-windows"
        )
    endif()
elseif(APPLE)
    # macOS: Homebrew and MacPorts paths
    list(APPEND _OpenBLAS_SEARCH_PATHS
        "/opt/homebrew/opt/openblas"
        "/usr/local/opt/openblas"
        "/opt/local"
    )
else()
    # Linux: Standard paths
    list(APPEND _OpenBLAS_SEARCH_PATHS
        "/usr"
        "/usr/local"
        "/opt/OpenBLAS"
        "$ENV{OPENBLAS_HOME}"
    )
endif()

# Find include directory
find_path(OpenBLAS_INCLUDE_DIR
    NAMES cblas.h openblas/cblas.h
    PATHS ${_OpenBLAS_SEARCH_PATHS}
    PATH_SUFFIXES include include/openblas
)

# Find library
if(WIN32)
    set(_OpenBLAS_LIB_NAMES openblas libopenblas)
else()
    set(_OpenBLAS_LIB_NAMES openblas)
endif()

find_library(OpenBLAS_LIBRARY
    NAMES ${_OpenBLAS_LIB_NAMES}
    PATHS ${_OpenBLAS_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/openblas
)

# Handle standard args
find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR
)

if(OpenBLAS_FOUND)
    set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
    set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})

    # Create imported target
    if(NOT TARGET OpenBLAS::OpenBLAS)
        add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            IMPORTED_LOCATION "${OpenBLAS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
        )

        # On Windows, may need to link pthread
        if(NOT WIN32)
            find_package(Threads)
            if(Threads_FOUND)
                set_property(TARGET OpenBLAS::OpenBLAS APPEND PROPERTY
                    INTERFACE_LINK_LIBRARIES Threads::Threads
                )
            endif()
        endif()
    endif()
endif()

mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY)
