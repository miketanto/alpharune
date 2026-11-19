# Decision dataset exporter (research/card_repr/smoke/data). Added to the WSL
# build copy only, by appending to its CMakeLists.txt:
#   include(research/card_repr/smoke/data/src/decision_export.cmake)
# Not part of `all`. mcts_agent.cpp is #included by the tool (it needs the
# TU-local IsMctsSearch), so it is NOT listed here.
add_executable(decision_export EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/research/card_repr/smoke/data/src/decision_export.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/openspiel/riftbound_game.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/openspiel/riftbound_state.cpp
    $<TARGET_OBJECTS:algorithms>
    $<TARGET_OBJECTS:game_transforms>
    $<TARGET_OBJECTS:utils>
)
target_link_libraries(decision_export PRIVATE
    riftbound_core open_spiel_core Threads::Threads
    absl::algorithm absl::flags absl::flags_parse absl::flat_hash_map
    absl::optional absl::random_random absl::str_format absl::strings absl::time
)
target_include_directories(decision_export PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${open_spiel_SOURCE_DIR})
