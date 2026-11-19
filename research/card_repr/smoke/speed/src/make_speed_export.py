"""Derives speed_export.cpp from ../../data/src/decision_export.cpp (unchanged copy plus
timing instrumentation and an optional auto-pay switch). Every patch must apply exactly once.
Added:
  * per decision: wall time of the observation export (Encoder) and of the agent's answer,
    by prompt kind, with searched / legal counts -> "timing" object in each games line;
  * env SPEED_AUTOPAY=1: "rune for energy" / "rune for power" prompts with >= 2 options are
    answered with legal[0] (the engine's default order, as autopay_check.py's act0) without
    calling the agent; the IsMctsAgent mirror re-syncs from history at its next decision.
  * env SPEED_NOREC=1: skip the observation export entirely (pure game+search time).
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "..", "..", "data", "src", "decision_export.cpp"), encoding="utf-8").read()

def sub(old, new):
    global src
    assert src.count(old) == 1, ("patch anchor not unique/absent", old[:80])
    src = src.replace(old, new)

sub('#include <random>\n', '#include <random>\n#include <array>\n#include <cstdlib>\n')
sub('    long n_multi = 0;\n',
    '    long n_multi = 0;\n'
    '    // speed: kind -> {n, n_searched, t_decide, t_encode, sum_legal}\n'
    '    std::map<std::string, std::array<double, 5>> tim;\n')
sub('        const PlayerId dec = deciderOf(s, legal, p);\n        std::string head;\n',
    '        const PlayerId dec = deciderOf(s, legal, p);\n        std::string head;\n'
    '        static const bool norec = std::getenv("SPEED_NOREC") != nullptr;\n'
    '        static const bool autopay = std::getenv("SPEED_AUTOPAY") != nullptr;\n'
    '        if (norec) ctx_.record = false;\n'
    '        const auto te0 = std::chrono::steady_clock::now();\n')
sub('        // swap-test candidates:',
    '        const double t_enc = std::chrono::duration<double>(std::chrono::steady_clock::now() - te0).count();\n'
    '        // swap-test candidates:')
sub('        const Intent pick = inner_->decide(s, legal, p);\n',
    '        const bool ap = autopay && legal.size() >= 2 &&\n'
    '            (p.kind == PromptKind::RuneForEnergy || p.kind == PromptKind::RuneForPower);\n'
    '        const auto td0 = std::chrono::steady_clock::now();\n'
    '        const Intent pick = ap ? legal.front() : inner_->decide(s, legal, p);\n'
    '        const double t_dec = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();\n'
    '        {\n'
    '            auto& a = ctx_.tim[std::string(toString(p.kind)) + (legal.size() >= 2 ? "" : "|single")];\n'
    '            a[0] += 1; a[1] += (ismcts_ && !ap && ismcts_->lastSearch().searched) ? 1 : 0;\n'
    '            a[2] += t_dec; a[3] += t_enc; a[4] += legal.size();\n'
    '        }\n')
sub('            if (ismcts_) {\n                const SearchSummary& ss',
    '            if (ismcts_ && !ap) {\n                const SearchSummary& ss')
sub('          << ",\\"obs_keys\\":{";\n',
    '          << ",\\"timing\\":{";\n'
    '        { bool ft = true; for (auto& [k, a] : ctx.tim) { o << (ft ? "" : ",") << jstr(k) << ":[" << a[0] << "," << a[1] << ","\n'
    '              << a[2] << "," << a[3] << "," << a[4] << "]"; ft = false; } }\n'
    '        o << "},\\"autopay\\":" << (std::getenv("SPEED_AUTOPAY") ? 1 : 0)\n'
    '          << ",\\"obs_keys\\":{";\n')
open(os.path.join(here, "speed_export.cpp"), "w", encoding="utf-8", newline="\n").write(src)
print("ok")
