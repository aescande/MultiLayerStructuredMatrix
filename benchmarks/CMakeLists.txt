set(SOLVER_FLAGS "")
set(SOLVER_LIBS "")

add_custom_target(mlsm_benchmarks)

macro(addBenchmark name)
  set(benchName ${name}_Bench)
  add_executable(${benchName} ${name}.cpp ${ARGN})
  add_custom_command(TARGET mlsm_benchmarks
    COMMAND ${benchName} --benchmark_out="${CMAKE_CURRENT_SOURCE_DIR}/out/${name}.csv" --benchmark_out_format=csv
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ${name} benchmark"
    )
  target_link_libraries(${benchName} PUBLIC mlsm benchmark ${SOLVER_LIBS})
  if(NOT WIN32)
    target_link_libraries(${benchName} PUBLIC pthread)
  else()
    target_link_libraries(${benchName} PUBLIC shlwapi)
  endif()
  target_compile_definitions(${benchName} PUBLIC ${SOLVER_FLAGS})
  # Adding a project configuration file (for MSVC only)
  GENERATE_MSVC_DOT_USER_FILE(${benchName} "${CMAKE_INSTALL_PREFIX}/bin")
  set_target_properties(${benchName} PROPERTIES FOLDER "Benchmarks")
endmacro()

file(MAKE_DIRECTORY out)

#add_custom_command(TARGET mlsm_benchmarks
#    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/generatePlot.py"
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#	)
