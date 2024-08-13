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
