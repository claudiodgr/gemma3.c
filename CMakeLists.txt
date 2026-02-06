# ============================================================================
# gemma3.c - Modern CMake Build System
# ============================================================================
#
# Cross-platform build supporting Linux, Windows, and macOS
#
# Usage:
#   cmake -B build                           # Configure (Release by default)
#   cmake -B build -DCMAKE_BUILD_TYPE=Debug  # Configure Debug build
#   cmake --build build                       # Build
#   cmake --build build --config Release      # Build (Windows multi-config)
#
# Options:
#   -DGEMMA3_USE_BLAS=ON       Enable OpenBLAS acceleration
#   -DGEMMA3_USE_THREADS=ON    Enable multi-threading
#   -DGEMMA3_USE_WEBGPU=ON     Enable WebGPU GPU acceleration
#   -DGEMMA3_NATIVE=ON         Enable native CPU optimizations (-march=native)
#   -DGEMMA3_BUILD_SHARED=ON   Build as shared library (default: static)
#
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# Policy for consistent behavior
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)  # MSVC runtime library selection
endif()

project(gemma3
    VERSION 1.0.0
    DESCRIPTION "Pure C inference engine for Google Gemma 3 models"
    LANGUAGES C
)

# ============================================================================
# Build Options
# ============================================================================

option(GEMMA3_USE_BLAS "Enable OpenBLAS acceleration" OFF)
option(GEMMA3_USE_THREADS "Enable multi-threading support" OFF)
option(GEMMA3_USE_WEBGPU "Enable WebGPU GPU acceleration" OFF)
option(GEMMA3_NATIVE "Enable native CPU optimizations" OFF)
option(GEMMA3_BUILD_SHARED "Build shared library instead of static" OFF)
option(GEMMA3_BUILD_CLI "Build command-line interface" ON)

# ============================================================================
# CMake Module Path
# ============================================================================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ============================================================================
# Compiler Configuration
# ============================================================================

# Require C11
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ============================================================================
# Platform Detection
# ============================================================================

if(WIN32)
    set(GEMMA3_PLATFORM "windows")
    set(GEMMA3_PLATFORM_WINDOWS TRUE)
elseif(APPLE)
    set(GEMMA3_PLATFORM "macos")
    set(GEMMA3_PLATFORM_MACOS TRUE)
elseif(UNIX)
    set(GEMMA3_PLATFORM "linux")
    set(GEMMA3_PLATFORM_LINUX TRUE)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# Architecture detection
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(GEMMA3_ARCH "x86_64")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        set(GEMMA3_ARCH "arm64")
    endif()
else()
    set(GEMMA3_ARCH "x86")
endif()

message(STATUS "Platform: ${GEMMA3_PLATFORM} (${GEMMA3_ARCH})")

# ============================================================================
# Compiler Flags
# ============================================================================

# Common warning flags
if(MSVC)
    add_compile_options(/W4 /wd4996 /wd4244 /wd4267)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Use static runtime for easier distribution
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-Wno-unused-parameter)
    endif()
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(NOT MSVC)
        add_compile_options(-O3)
        if(GEMMA3_NATIVE)
            add_compile_options(-march=native -ffast-math)
            message(STATUS "Native optimizations enabled")
        endif()
    else()
        add_compile_options(/O2)
        if(GEMMA3_NATIVE)
            add_compile_options(/arch:AVX2)
            message(STATUS "AVX2 optimizations enabled")
        endif()
    endif()
endif()

# Debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
else()
    add_compile_definitions(NDEBUG)
endif()

# ============================================================================
# Source Files
# ============================================================================

set(GEMMA3_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_kernels.c
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_safetensors.c
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_tokenizer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_transformer.c
)

set(GEMMA3_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_kernels.h
)

# Platform abstraction
list(APPEND GEMMA3_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_platform.c
)
list(APPEND GEMMA3_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_platform.h
)

# Optional sources
set(GEMMA3_OPTIONAL_SOURCES "")

if(GEMMA3_USE_THREADS)
    list(APPEND GEMMA3_OPTIONAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_threads.c
    )
    list(APPEND GEMMA3_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_threads.h
    )
endif()

if(GEMMA3_USE_WEBGPU)
    list(APPEND GEMMA3_OPTIONAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_webgpu.c
    )
    list(APPEND GEMMA3_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/gemma3_webgpu.h
    )
endif()

# ============================================================================
# Library Target
# ============================================================================

if(GEMMA3_BUILD_SHARED)
    add_library(gemma3_lib SHARED
        ${GEMMA3_CORE_SOURCES}
        ${GEMMA3_OPTIONAL_SOURCES}
    )
    set_target_properties(gemma3_lib PROPERTIES
        OUTPUT_NAME gemma3
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
    )
    target_compile_definitions(gemma3_lib PRIVATE GEMMA3_EXPORTS)
else()
    add_library(gemma3_lib STATIC
        ${GEMMA3_CORE_SOURCES}
        ${GEMMA3_OPTIONAL_SOURCES}
    )
    set_target_properties(gemma3_lib PROPERTIES
        OUTPUT_NAME gemma3
    )
endif()

target_include_directories(gemma3_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# ============================================================================
# Dependencies
# ============================================================================

# Math library (required on POSIX, built-in on Windows)
if(NOT WIN32)
    target_link_libraries(gemma3_lib PRIVATE m)
endif()

# Windows-specific libraries
if(WIN32)
    # No additional libraries needed - we use Windows API for memory mapping
endif()

# Threading support
if(GEMMA3_USE_THREADS)
    target_compile_definitions(gemma3_lib PUBLIC USE_THREADS)

    if(WIN32)
        # Windows threads are built-in
    else()
        find_package(Threads REQUIRED)
        target_link_libraries(gemma3_lib PRIVATE Threads::Threads)
    endif()
    message(STATUS "Threading support: Enabled")
else()
    message(STATUS "Threading support: Disabled")
endif()

# OpenBLAS support
if(GEMMA3_USE_BLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        target_compile_definitions(gemma3_lib PUBLIC USE_BLAS)
        target_link_libraries(gemma3_lib PRIVATE OpenBLAS::OpenBLAS)
        message(STATUS "OpenBLAS: Found at ${OpenBLAS_LIBRARIES}")
    else()
        message(WARNING "OpenBLAS not found, building without BLAS support")
        set(GEMMA3_USE_BLAS OFF)
    endif()
else()
    message(STATUS "OpenBLAS: Disabled")
endif()

# WebGPU support
if(GEMMA3_USE_WEBGPU)
    find_package(WebGPU)
    if(WebGPU_FOUND)
        target_compile_definitions(gemma3_lib PUBLIC USE_WEBGPU)
        target_link_libraries(gemma3_lib PRIVATE WebGPU::WebGPU)

        # Copy shader file to build directory
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemma3_kernels.wgsl
            ${CMAKE_CURRENT_BINARY_DIR}/shaders/gemma3_kernels.wgsl
            COPYONLY
        )

        message(STATUS "WebGPU: Found at ${WebGPU_LIBRARIES}")
    else()
        message(WARNING "WebGPU (wgpu-native) not found, building without GPU support")
        set(GEMMA3_USE_WEBGPU OFF)
    endif()
else()
    message(STATUS "WebGPU: Disabled")
endif()

# ============================================================================
# CLI Executable
# ============================================================================

if(GEMMA3_BUILD_CLI)
    add_executable(gemma3 ${CMAKE_CURRENT_SOURCE_DIR}/main.c)
    target_link_libraries(gemma3 PRIVATE gemma3_lib)

    # Set RPATH for finding shared libraries
    if(GEMMA3_BUILD_SHARED AND NOT WIN32)
        set_target_properties(gemma3 PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Install library
install(TARGETS gemma3_lib
    EXPORT gemma3Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers
install(FILES ${GEMMA3_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemma3
)

# Install CLI
if(GEMMA3_BUILD_CLI)
    install(TARGETS gemma3
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Install shaders
if(GEMMA3_USE_WEBGPU)
    install(DIRECTORY shaders/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/gemma3/shaders
        FILES_MATCHING PATTERN "*.wgsl"
    )
endif()

# Export targets for find_package support
install(EXPORT gemma3Targets
    FILE gemma3Targets.cmake
    NAMESPACE gemma3::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gemma3
)

# Create package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gemma3Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gemma3Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gemma3
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gemma3ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gemma3Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gemma3ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gemma3
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== gemma3.c Build Configuration ===")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Platform:     ${GEMMA3_PLATFORM} (${GEMMA3_ARCH})")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler:   ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "  Options:")
message(STATUS "    Threading:  ${GEMMA3_USE_THREADS}")
message(STATUS "    OpenBLAS:   ${GEMMA3_USE_BLAS}")
message(STATUS "    WebGPU:     ${GEMMA3_USE_WEBGPU}")
message(STATUS "    Native Opt: ${GEMMA3_NATIVE}")
message(STATUS "    Shared Lib: ${GEMMA3_BUILD_SHARED}")
message(STATUS "    Build CLI:  ${GEMMA3_BUILD_CLI}")
message(STATUS "")
