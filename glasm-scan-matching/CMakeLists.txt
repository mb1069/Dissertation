cmake_minimum_required(VERSION 3.6)
project(glasm_scan_matching)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS -O0)
#set(CMAKE_CXX_FLAGS_DEBUG -O0)


set(SOURCE_FILES
        experiments/doc/glasm.ini.explained
        experiments/doc/sm.ini.explained
        experiments/example1-rawseeds/bitmaps/empty_1000x1000.png
        experiments/example1-rawseeds/bitmaps/empty_2000x2000.png
        experiments/example1-rawseeds/bitmaps/empty_200x200.png
        experiments/example1-rawseeds/bitmaps/empty_400x400.png
        experiments/example1-rawseeds/scans_rawseeds/Bovisa_2008-09-01-SICK_FRONT_random_sample_1.csv
        experiments/example1-rawseeds/glasm_binary.ini
        experiments/example1-rawseeds/glasm_gradient.ini
        experiments/example1-rawseeds/hglasm.ini
        experiments/example1-rawseeds/icp.ini
        experiments/example1-rawseeds/mbicp_player.ini
        experiments/example1-rawseeds/pgaicp.ini
        experiments/example1-rawseeds/pgambicp.ini
        experiments/example1-rawseeds/polarga.ini
        experiments/example1-rawseeds/sm.ini
        experiments/example2-player-stage/bitmaps/cave.png
        experiments/example2-player-stage/bitmaps/cave_highres.png
        experiments/example2-player-stage/bitmaps/cave_mediumres.png
        experiments/example2-player-stage/bitmaps/cave_mediumres_static.png
        experiments/example2-player-stage/bitmaps/cave_mediumres_valid.png
        experiments/example2-player-stage/bitmaps/Drawings_02.png
        experiments/example2-player-stage/bitmaps/Drawings_03.png
        experiments/example2-player-stage/bitmaps/Drawings_03_valid.png
        experiments/example2-player-stage/bitmaps/hospital_room_lookup.png
        experiments/example2-player-stage/bitmaps/hospital_room_original.png
        experiments/example2-player-stage/bitmaps/hospital_room_valid.png
        experiments/example2-player-stage/other_scans/scans-autolab.7z
        experiments/example2-player-stage/other_scans/scans_cave.7z
        experiments/example2-player-stage/other_scans/scans_cave_png.7z
        experiments/example2-player-stage/other_scans/scans_hospital_room_valid_NOERROR.7z
        experiments/example2-player-stage/other_scans/scans_hospital_room_valid_WITHERROR.7z
        "experiments/example2-player-stage/player-stage configuration/klsimple.cfg"
        "experiments/example2-player-stage/player-stage configuration/map.inc"
        "experiments/example2-player-stage/player-stage configuration/pioneer.inc"
        "experiments/example2-player-stage/player-stage configuration/sick.inc"
        "experiments/example2-player-stage/player-stage configuration/simple.world"
        experiments/example2-player-stage/gcp.ini
        experiments/example2-player-stage/glasm.ini
        experiments/example2-player-stage/icp.ini
        experiments/example2-player-stage/mbicp.ini
        experiments/example2-player-stage/polarga.ini
        experiments/example2-player-stage/scans.7z
        experiments/example2-player-stage/sm.ini
        experiments/GA-comparison/glasm_binary.ini
        experiments/GA-comparison/glasm_gradient.ini
        experiments/GA-comparison/polarga.ini
        experiments/GA-comparison/sm.ini
        experiments/scripts/automate.sh
        experiments/scripts/launch_sm_cli.sh
        lib/libpngwriter0-dev_0.5.3-3_amd64.deb
        lib/libpngwriter0c2_0.5.3-3_amd64.deb
        src/mbicp/calcul.c
        src/mbicp/calcul.h
        src/mbicp/MbICP.c
        src/mbicp/MbICP.h
        src/mbicp/MbICP2.h
        src/mbicp/mbicp_aux.c
        src/mbicp/percolate.c
        src/mbicp/percolate.h
        src/mbicp/sp_matrix.c
        src/mbicp/sp_matrix.h
        src/mbicp/TData.h
        src/gcp.cpp
        src/gcp.h
        src/gcp_aux.cpp
        src/gcp_aux.h
        src/glasm.cpp
        src/glasm.h
        src/glasm_aux.cpp
        src/glasm_aux.h
        src/hglasm_aux.cpp
        src/hglasm_aux.h
        src/icp.cpp
        src/icp.h
        src/icp_aux.cpp
        src/icp_aux.h
        src/mbicp_aux.cpp
        src/mbicp_aux.h
        src/mydraw.cpp
        src/mydraw.h
        src/pgambicp_aux.cpp
        src/pgambicp_aux.h
        src/polarga.cpp
        src/polarga.h
        src/polarga_aux.cpp
        src/polarga_aux.h
        src/sga.cpp
        src/sga.h
        src/simulation.h
        src/sm_cli.cpp
        src/utils.cpp
        src/utils.h
        LICENSE
        README.md)

#link_directories(/usr/lib)
#link_directories(/usr/lib/x86_64-linux-gnu/)
#find_library(Freetype REQUIRED)
#find_library(png /usr/lib)
#find_library(pngwriter /usr/lib)
#find_library(boost_program_options /usr/lib)
#find_library(m /usr/lib)




find_package(Freetype REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
#find_package(Boost COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(/usr/include)

#find_library(pngwriter /usr/include)
#add_library(pngwriter)
include_directories(${PNGWRITER_INCLUDE_DIRS})





add_executable(glasm_scan_matching ${SOURCE_FILES})

target_link_libraries(glasm_scan_matching ${Boost_LIBRARIES})
#target_link_libraries(glasm_scan_matching /usr/include)
target_link_libraries(glasm_scan_matching  pngwriter)
target_link_libraries(glasm_scan_matching  png)
target_link_libraries(glasm_scan_matching  freetype)