cmake_minimum_required(VERSION 3.10)
project(example_1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- 1. FIND VTK ---
# Since we installed VTK to /usr/local in Docker, CMake can find it automatically.
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# --- 2. SETUP VITA PATHS ---
# In Docker, we installed headers to /usr/local/include/vita_source
include_directories("/usr/local/include/VItA-source")

# --- 3. FIND SYSTEM LIBRARIES ---
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)

add_executable(artery example_1.cpp)

# --- 4. LINK LIBRARIES ---
# libraries are in /usr/local/lib, which is a default search path.
# We no longer need the complex "brute force" linking or GLOB patterns.
target_link_libraries(artery 
    VItA
    ${VTK_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${X11_LIBRARIES}
    ${X11_Xt_LIB}
    -lpthread
    -ldl
)
