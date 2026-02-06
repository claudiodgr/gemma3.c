# ============================================================================
# FindWebGPU.cmake - Find wgpu-native library
# ============================================================================
#
# This module finds the wgpu-native WebGPU implementation.
#
# Defines:
#   WebGPU_FOUND        - System has wgpu-native
#   WebGPU_INCLUDE_DIRS - WebGPU include directories
#   WebGPU_LIBRARIES    - WebGPU libraries
#
# Imported targets:
#   WebGPU::WebGPU      - WebGPU library target
#
# User-configurable variables:
#   WEBGPU_DIR          - Path to wgpu-native installation
#   WEBGPU_ROOT         - Alternative to WEBGPU_DIR
#
# ============================================================================

include(FindPackageHandleStandardArgs)

# Get the source directory for lib/webgpu path
get_filename_component(_gemma3_source_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Platform and architecture detection
if(WIN32)
    set(_WebGPU_PLATFORM "windows")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_WebGPU_ARCH "x86_64")
    else()
        set(_WebGPU_ARCH "i686")
    endif()
elseif(APPLE)
    set(_WebGPU_PLATFORM "macos")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        set(_WebGPU_ARCH "arm64")
    else()
        set(_WebGPU_ARCH "x86_64")
    endif()
else()
    set(_WebGPU_PLATFORM "linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_WebGPU_ARCH "aarch64")
    else()
        set(_WebGPU_ARCH "x86_64")
    endif()
endif()

# Build search paths list
set(_WebGPU_SEARCH_PATHS "")

# User-specified paths (highest priority)
if(DEFINED WEBGPU_DIR)
    list(APPEND _WebGPU_SEARCH_PATHS "${WEBGPU_DIR}")
endif()
if(DEFINED WEBGPU_ROOT)
    list(APPEND _WebGPU_SEARCH_PATHS "${WEBGPU_ROOT}")
endif()
if(DEFINED ENV{WEBGPU_DIR})
    list(APPEND _WebGPU_SEARCH_PATHS "$ENV{WEBGPU_DIR}")
endif()

# Project-local lib directory (recommended for cross-platform builds)
list(APPEND _WebGPU_SEARCH_PATHS
    "${_gemma3_source_dir}/lib/webgpu/${_WebGPU_PLATFORM}-${_WebGPU_ARCH}"
    "${_gemma3_source_dir}/lib/webgpu/${_WebGPU_PLATFORM}"
    "${_gemma3_source_dir}/lib/webgpu"
)

# Platform-specific system paths
if(WIN32)
    list(APPEND _WebGPU_SEARCH_PATHS
        "C:/wgpu-native"
        "C:/opt/wgpu-native"
        "$ENV{PROGRAMFILES}/wgpu-native"
    )
    # vcpkg integration
    if(DEFINED CMAKE_TOOLCHAIN_FILE)
        get_filename_component(_vcpkg_root "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
        get_filename_component(_vcpkg_root "${_vcpkg_root}" DIRECTORY)
        list(APPEND _WebGPU_SEARCH_PATHS
            "${_vcpkg_root}/installed/x64-windows"
            "${_vcpkg_root}/installed/x86-windows"
        )
    endif()
elseif(APPLE)
    list(APPEND _WebGPU_SEARCH_PATHS
        "/opt/homebrew/opt/wgpu-native"
        "/usr/local/opt/wgpu-native"
        "/opt/wgpu-native"
        "/usr/local"
    )
else()
    list(APPEND _WebGPU_SEARCH_PATHS
        "/usr/local"
        "/usr"
        "/opt/wgpu-native"
    )
endif()

# Find include directory
find_path(WebGPU_INCLUDE_DIR
    NAMES webgpu/webgpu.h webgpu.h wgpu.h
    PATHS ${_WebGPU_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find library
if(WIN32)
    set(_WebGPU_LIB_NAMES wgpu_native wgpu)
    # Find both import lib and DLL
    find_library(WebGPU_LIBRARY
        NAMES ${_WebGPU_LIB_NAMES}
        PATHS ${_WebGPU_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64
    )
    # Find DLL for runtime
    find_file(WebGPU_DLL
        NAMES wgpu_native.dll wgpu.dll
        PATHS ${_WebGPU_SEARCH_PATHS}
        PATH_SUFFIXES bin lib
    )
elseif(APPLE)
    set(_WebGPU_LIB_NAMES wgpu_native wgpu)
    find_library(WebGPU_LIBRARY
        NAMES ${_WebGPU_LIB_NAMES}
        PATHS ${_WebGPU_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64
    )
else()
    set(_WebGPU_LIB_NAMES wgpu_native wgpu)
    find_library(WebGPU_LIBRARY
        NAMES ${_WebGPU_LIB_NAMES}
        PATHS ${_WebGPU_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64
    )
endif()

# Handle standard args
find_package_handle_standard_args(WebGPU
    REQUIRED_VARS WebGPU_LIBRARY WebGPU_INCLUDE_DIR
)

if(WebGPU_FOUND)
    set(WebGPU_LIBRARIES ${WebGPU_LIBRARY})
    set(WebGPU_INCLUDE_DIRS ${WebGPU_INCLUDE_DIR})

    # Create imported target
    if(NOT TARGET WebGPU::WebGPU)
        add_library(WebGPU::WebGPU UNKNOWN IMPORTED)
        set_target_properties(WebGPU::WebGPU PROPERTIES
            IMPORTED_LOCATION "${WebGPU_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${WebGPU_INCLUDE_DIR}"
        )

        # Platform-specific dependencies
        if(WIN32)
            # Windows: Link with required system libraries
            set_property(TARGET WebGPU::WebGPU APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES
                    user32
                    gdi32
                    ole32
                    shell32
                    d3dcompiler
                    ws2_32
                    userenv
                    bcrypt
                    ntdll
            )

            # Store DLL path for runtime copying
            if(WebGPU_DLL)
                set_property(TARGET WebGPU::WebGPU PROPERTY
                    IMPORTED_LOCATION_RUNTIME "${WebGPU_DLL}"
                )
            endif()

        elseif(APPLE)
            # macOS: Link with Metal and other frameworks
            find_library(METAL_FRAMEWORK Metal)
            find_library(QUARTZCORE_FRAMEWORK QuartzCore)
            find_library(FOUNDATION_FRAMEWORK Foundation)

            set_property(TARGET WebGPU::WebGPU APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES
                    ${METAL_FRAMEWORK}
                    ${QUARTZCORE_FRAMEWORK}
                    ${FOUNDATION_FRAMEWORK}
            )

        else()
            # Linux: Link with dl and pthread
            find_package(Threads)
            set_property(TARGET WebGPU::WebGPU APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES
                    ${CMAKE_DL_LIBS}
                    Threads::Threads
            )

            # Check for Vulkan (optional, but needed at runtime)
            find_package(Vulkan QUIET)
            if(Vulkan_FOUND)
                message(STATUS "Vulkan SDK found - runtime GPU support available")
            endif()
        endif()
    endif()

    # Provide helper function to copy DLLs on Windows
    if(WIN32 AND WebGPU_DLL)
        function(webgpu_copy_dll target)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${WebGPU_DLL}"
                    $<TARGET_FILE_DIR:${target}>
                COMMENT "Copying wgpu_native.dll to output directory"
            )
        endfunction()
    else()
        function(webgpu_copy_dll target)
            # No-op on non-Windows
        endfunction()
    endif()
endif()

mark_as_advanced(WebGPU_INCLUDE_DIR WebGPU_LIBRARY WebGPU_DLL)
