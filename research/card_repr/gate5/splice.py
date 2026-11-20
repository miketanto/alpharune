# Splices gate5/rbenv_gate5_ext.cpp into selfplay/rbenv/rbenv.cpp (idempotent).
import re, os
H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, '..', 'selfplay', 'rbenv', 'rbenv.cpp')
src = open(R).read()
ext = open(os.path.join(H, 'rbenv_gate5_ext.cpp')).read()
A = ext[ext.index('// ---- PART A ----'):ext.index('// ---- END PART A ----') + len('// ---- END PART A ----')]
B = ext[ext.index('// ---- PART B ----'):ext.index('// ---- END PART B ----') + len('// ---- END PART B ----')]
for tag in ('PART A', 'PART B'):
    s = '// ---- %s ----' % tag; e = '// ---- END %s ----' % tag
    if s in src:
        src = src[:src.index(s)] + src[src.index(e) + len(e):]
src = src.replace('void applyEdit(GameState& s, const std::string& spec) {', A + '\nvoid applyEdit(GameState& s, const std::string& spec) {', 1)
op = '        } else if (op == "swap") {   // gate 5\n            GameObject& o = obj();\n            std::string key;\n            ss >> key;\n            swapCard(o, key);\n'
if 'op == "swap"' not in src:
    src = src.replace('        } else if (op == "might") {', op + '        } else if (op == "might") {', 1)
if "agents/determinizer.h" not in src:
    src = src.replace('#include "io/state_editor.h"', '#include "io/state_editor.h"\n#include "agents/determinizer.h"', 1)
src = src.rstrip() + '\n\n' + B + '\n'
open(R, 'w').write(src)
print('ok', src.count('PART A'), src.count('op == "swap"'))
