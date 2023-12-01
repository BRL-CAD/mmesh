include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

set(MMESH_SOURCES
  cc.c
  meshdecimation.c
  mmbinsort.c
  mm.c
  mmcore.c
  mmhash.c
  )

add_library(mmesh SHARED ${MMESH_SOURCES})
target_link_libraries(mmesh PRIVATE ${MM_LIBS})
#set_property(TARGET mmesh APPEND PROPERTY COMPILE_DEFINITIONS "MMESH_DLL_EXPORTS")
#set_property(TARGET mmesh APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "MMESH_DLL_IMPORTS")
install(TARGETS mmesh
  RUNTIME DESTINATION ${BIN_DIR}
  LIBRARY DESTINATION ${LIB_DIR}
  ARCHIVE DESTINATION ${LIB_DIR}
  )

# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8

