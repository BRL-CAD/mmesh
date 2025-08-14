set(MMESH_SOURCES
  cc.c
  meshdecimation.c
  meshoptimizer.c
  mm.c
  mmbinsort.c
  mmcore.c
  mmhash.c
  mmthread.c
)

if (BUILD_SHARED_LIBS)
  add_library(mmesh SHARED ${MMESH_SOURCES})
  target_include_directories(
    mmesh PUBLIC
    $<INSTALL_INTERFACE:${INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )
  target_link_libraries(mmesh PRIVATE ${MM_LIBS})
  set_property(TARGET mmesh APPEND PROPERTY COMPILE_DEFINITIONS "MMESH_DLL_EXPORTS")
  set_property(TARGET mmesh APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "MMESH_DLL_IMPORTS")
  install(TARGETS mmesh
    EXPORT mmeshTargets
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}
    INCLUDES DESTINATION ${INCLUDE_DIR}
  )
endif (BUILD_SHARED_LIBS)

if (BUILD_STATIC_LIBS)

  set(mmesh_static_name mmesh)
  if (BUILD_SHARED_LIBS)
    set(mmesh_static_name mmesh-static)
  endif (BUILD_SHARED_LIBS)

  add_library(${mmesh_static_name} STATIC ${MMESH_SOURCES})
  if(MSVC)
    # msvc does not append 'lib' - do it here to have consistent name
    set_target_properties(${mmesh_static_name} PROPERTIES PREFIX "lib")
  else(MSVC)
    set_target_properties(${mmesh_static_name} PROPERTIES OUTPUT_NAME "mmesh")
  endif(MSVC)
  target_include_directories(
    ${mmesh_static_name} PUBLIC
    $<INSTALL_INTERFACE:${INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )
  target_link_libraries(${mmesh_static_name} PRIVATE ${MM_LIBS})
  set_property(TARGET ${mmesh_static_name} APPEND PROPERTY COMPILE_DEFINITIONS "MMESH_DLL_EXPORTS")
  set_property(TARGET ${mmesh_static_name} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "MMESH_DLL_IMPORTS")
  install(TARGETS ${mmesh_static_name}
    EXPORT mmeshTargets
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}
    INCLUDES DESTINATION ${INCLUDE_DIR}
  )
endif (BUILD_STATIC_LIBS)

# Export the targets to a file
install(EXPORT mmeshTargets
  FILE mmeshTargets.cmake
  NAMESPACE mmesh::
  DESTINATION lib/cmake/mmesh
)

# Generate the version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/mmeshConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Install the config and version files
install(FILES
  "${PROJECT_SOURCE_DIR}/cmake/mmeshConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/mmeshConfigVersion.cmake"
  DESTINATION lib/cmake/mmesh
)


# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8

