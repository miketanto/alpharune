# playweb shared library (tools/play_web): one human-vs-network game for the browser UI.
# Appended to the WSL build copy's CMakeLists.txt by tools/play_web/build.sh, next to the
# research rbenv target whose encoder it #includes; built in the PIC build dir (build_pic).
# Not part of `all`.
add_library(playweb SHARED EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/play_web/playweb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/openspiel/riftbound_game.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/openspiel/riftbound_state.cpp
    $<TARGET_OBJECTS:algorithms>
    $<TARGET_OBJECTS:game_transforms>
    $<TARGET_OBJECTS:utils>
)
target_link_libraries(playweb PRIVATE
    riftbound_core open_spiel_core Threads::Threads
    absl::algorithm absl::flags absl::flags_parse absl::flat_hash_map
    absl::optional absl::random_random absl::str_format absl::strings absl::time
)
# An engine whose players can use their runes' own abilities (GameEngine::setManualRunes).
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/game_engine.h _pw_manual REGEX "void setManualRunes")
if(_pw_manual)
  target_compile_definitions(playweb PRIVATE RB_MANUAL_RUNES)
endif()
target_include_directories(playweb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${open_spiel_SOURCE_DIR})
