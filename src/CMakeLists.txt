set(MLSM_SOURCES
  #Matrix.cpp
  BlockMatrix.cpp
  MatrixBase.cpp
  SimpleMatrix.cpp
  StorageScheme.cpp
)

set(MLSM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/mlsm)
set(MLSM_HEADERS
  ${MLSM_INCLUDE_DIR}/api.h
  ${MLSM_INCLUDE_DIR}/defs.h
#  ${MLSM_INCLUDE_DIR}/enums.h
#  ${MLSM_INCLUDE_DIR}/Matrix.h
  ${MLSM_INCLUDE_DIR}/BlockMatrix.h
  ${MLSM_INCLUDE_DIR}/MatrixBase.h
  ${MLSM_INCLUDE_DIR}/SimpleMatrix.h
#  ${MLSM_INCLUDE_DIR}/ShapeDescriptor.h
  ${MLSM_INCLUDE_DIR}/internal/Shape.h
  ${MLSM_INCLUDE_DIR}/internal/Size.h
  ${MLSM_INCLUDE_DIR}/internal/SimpleStorage.h
  ${MLSM_INCLUDE_DIR}/internal/StorageScheme.h
)

#Helper functions to assign files to groups in IDE, mimicking the organization of files on disk
#Adapted from https://stackoverflow.com/a/31423421
function(assign_source_group folder)
  foreach(_source IN ITEMS ${ARGN})
    if (IS_ABSOLUTE "${_source}")
        file(RELATIVE_PATH _source_rel "${CMAKE_SOURCE_DIR}" "${_source}")
    else()
        set(_source_rel "${_source}")
    endif()
    #remove include/mlsm/ if present in the path
    string(REPLACE "include/mlsm/" "" _source_rel_modified "${_source_rel}")
    get_filename_component(_source_path "${_source_rel_modified}" PATH)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${folder}\\${_source_path_msvc}" FILES "${_source}")
  endforeach()
endfunction(assign_source_group)

assign_source_group("sources" ${MLSM_SOURCES})
assign_source_group("headers" ${MLSM_HEADERS})

add_library(mlsm SHARED ${MLSM_SOURCES} ${MLSM_HEADERS})
target_include_directories(mlsm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> $<INSTALL_INTERFACE:include>)
if(TARGET Eigen3::Eigen)
  target_link_libraries(mlsm PUBLIC Eigen3::Eigen)
else()
  target_include_directories(mlsm SYSTEM PUBLIC "${EIGEN3_INCLUDE_DIR}")
endif()
set_target_properties(mlsm PROPERTIES COMPILE_FLAGS "-DMLSM_EXPORTS -DEIGEN_RUNTIME_NO_MALLOC")
set_target_properties(mlsm PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR} VERSION ${PROJECT_VERSION})
set_target_properties(mlsm PROPERTIES CXX_STANDARD 17)
if(NOT WIN32)
  target_compile_options(mlsm PUBLIC "-Wall" "-Wextra" "-Wno-sign-conversion" "-pedantic")
else()
  target_compile_options(mlsm PUBLIC "-D_ENABLE_EXTENDED_ALIGNED_STORAGE" "-D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING")
endif()

if(GENERATE_COVERAGE)
  target_compile_options(mlsm PRIVATE "--coverage")
  set_target_properties(mlsm PROPERTIES LINK_FLAGS "--coverage")
endif()

install(TARGETS mlsm
  EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)
#install(DIRECTORY ${MLSM_INCLUDE_DIR}/ DESTINATION ${INCLUDE_INSTALL_DESTINATION})
#pkg_config_append_libs(mlsm)
