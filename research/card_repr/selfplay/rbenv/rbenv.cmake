# rbenv shared library (research/card_repr/selfplay/rbenv). Added to the WSL
# build copy only, by appending to its CMakeLists.txt:
#   include(research/card_repr/selfplay/rbenv/rbenv.cmake)
# Built in a separate build dir configured with
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON (build_pic, see build.sh): the normal
# build's static engine libraries are not PIC. Not part of `all`.
add_library(rbenv SHARED EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/research/card_repr/selfplay/rbenv/rbenv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/openspiel/riftbound_game.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/openspiel/riftbound_state.cpp
    $<TARGET_OBJECTS:algorithms>
    $<TARGET_OBJECTS:game_transforms>
    $<TARGET_OBJECTS:utils>
)
target_link_libraries(rbenv PRIVATE
    riftbound_core open_spiel_core Threads::Threads
    absl::algorithm absl::flags absl::flags_parse absl::flat_hash_map
    absl::optional absl::random_random absl::str_format absl::strings absl::time
)
# An engine whose players can use their runes' own abilities (GameEngine::setManualRunes).
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/game_engine.h _rbenv_manual REGEX "void setManualRunes")
if(_rbenv_manual)
  target_compile_definitions(rbenv PRIVATE RB_MANUAL_RUNES)
endif()
target_include_directories(rbenv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${open_spiel_SOURCE_DIR})
