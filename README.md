------------------
English version (See below for the one in portuguese)
-----------------

Hi there, 

So this is the file structure. When you checkout the repository, two folders are downloaded

/source -> which contains all source code for the example
/settings -> which contains the parameter files for the simulations

You will need to create two additional folders

/storage -> to save and load workspaces, as well as mosaic screenshots
/build (optional) -> for the binaries

To compile, all you need to do is 

cd build
ccmake ../source


(!) Make sure you use

CMAKE_BUILD_TYPE Release
CMAKE_CXX_FLAGS -lpthread -fopenmp

Dependencies:

BOOST (sudo apt-get install libboost-all-dev)
GLEW (sudo apt-get install libglew-dev)
SDL (sudo apt-get install libsdl1.2-dev)
DECKLINK (blackmagic driver)
ZLIB (sudo apt-get install zlib1g-dev)
OpenCV


questions/issues email me at rogerio.a.r@gmail.com
Enjoy

--------------------
Portuguese version
--------------------

Olá pessoal, 

A estrutura de arquivos é

/source -> para o código fonte deste repo
/build -> para os executaveis
/settings -> para o arquivo parameters.yml e keypoint_handler.yml (eles estao no repositorio também!)
/storage -> aqui é a pasta onde a gente salva screen shots do mosaico, salva o workspace e outros
arquivos que contem os índices de performance do sistema

para compilar é só fazer:

cd build
ccmake ../source

usem 
CMAKE_BUILD_TYPE Release
CMAKE_CXX_FLAGS -lpthread -fopenmp

Dependencies:

BOOST (sudo apt-get install libboost-all-dev)
GLEW (sudo apt-get install libglew-dev)
SDL (sudo apt-get install libsdl1.2-dev)
DECKLINK (blackmagic driver)
ZLIB (sudo apt-get install zlib1g-dev)
OpenCV

Qualquer dúvida, é só me achar.