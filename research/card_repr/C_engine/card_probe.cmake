# Path C probe tool (research/card_repr/C_engine). Added to the WSL build copy
# only, by appending to its CMakeLists.txt:
#   include(research/card_repr/C_engine/card_probe.cmake)
# Not part of `all`.
add_executable(card_probe EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/research/card_repr/C_engine/card_probe.cpp)
target_link_libraries(card_probe PRIVATE riftbound_core Threads::Threads)
target_include_directories(card_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
