# Pool S sampler (research/card_repr/situations). Added to the WSL build copy
# only, by appending to its CMakeLists.txt:
#   include(research/card_repr/situations/situations.cmake)
# Not part of `all`. Links like search_strength (IsMctsAgent needs OpenSpiel).
add_executable(situation_sampler EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/research/card_repr/situations/sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/agents/mcts_agent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/openspiel/riftbound_game.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/openspiel/riftbound_state.cpp
    $<TARGET_OBJECTS:algorithms>
    $<TARGET_OBJECTS:game_transforms>
    $<TARGET_OBJECTS:utils>
)
target_link_libraries(situation_sampler PRIVATE
    riftbound_core open_spiel_core Threads::Threads
    absl::algorithm absl::flags absl::flags_parse absl::flat_hash_map
    absl::optional absl::random_random absl::str_format absl::strings absl::time
)
target_include_directories(situation_sampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${open_spiel_SOURCE_DIR})
