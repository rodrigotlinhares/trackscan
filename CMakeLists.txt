CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( Trackscan )

# Options -----------------------------------------------

SET( use_decklink TRUE CACHE BOOL "If compile with blackmagic support" )
SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -fopenmp" )

# cut here -------------------------------------------- >8

FIND_PACKAGE( OpenCV REQUIRED )
SET(LINK_LIBS ${OpenCV_LIBS}  SDL GLEW GL boost_thread boost_system z)

IF( ${use_decklink} )
	SET( decklink-sdk_DIR NOTFOUND CACHE PATH "Decklink sdk dir" )
	FIND_LIBRARY( decklink-sdk_LIBS NAMES DeckLinkAPI )
	add_definitions( -DUSE_DECKLINK )
	LIST(APPEND LINK_LIBS ${decklink-sdk_LIBS} )
ENDIF()


FILE (GLOB_RECURSE MY_SOURCES ${PROJECT_SOURCE_DIR}/source/*.cpp)
FILE (GLOB_RECURSE MY_INCLUDES ${PROJECT_SOURCE_DIR}/source/*.h)
FILE (GLOB_RECURSE MY_FERN_SOURCE ${PROJECT_SOURCE_DIR}/source/ferns/*.cc)
FILE (GLOB_RECURSE MY_FERN_INCLUDES ${PROJECT_SOURCE_DIR}/source/ferns/*.h)

ADD_EXECUTABLE (${PROJECT_NAME} ${PROJECT_SOURCE_DIR}  ${MY_INCLUDES}  ${MY_FERN_INCLUDES} ${MY_FERN_SOURCE} ${MY_SOURCES})

INCLUDE_DIRECTORIES( ${decklink-sdk_DIR} )

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${LINK_LIBS})
