p='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/rbenv/rbenv.cpp'; s=open(p).read()
def rep(a,b,cnt=1):
    global s
    assert s.count(a)==cnt, (a, s.count(a)); s=s.replace(a,b)
rep("struct Result { int env; int64_t game_seed; int winner; int length; int n_ext; int turns; std::string error; };",
    "struct Result { int env; int64_t game_seed; int winner; int length; int n_ext; int turns; std::string error;\n                int var = -1; int64_t po_seed = 0; };   // gate 4 playouts")
rep("    PlayerId dec = PlayerId::None;\n};",
    "    PlayerId dec = PlayerId::None;\n    std::vector<int32_t> hist;   // gate 4: every applied choice index since game start (replay)\n    std::vector<int64_t> tok_obj; // gate 4: object id per token (-1 chain / pile)\n    int var = -1; int64_t po_seed = 0; bool idle = false;   // gate 4 playout job\n};")
rep("        noteChoice(e.mtrack[static_cast<int>(dec) % 3], sr.prompt, sr.legal[k]);\n    e.engine->applyChoice(k);",
    "        noteChoice(e.mtrack[static_cast<int>(dec) % 3], sr.prompt, sr.legal[k]);\n    e.hist.push_back(k);\n    e.engine->applyChoice(k);")
rep("struct Handle {\n", "struct Po;   // gate 4 playout state (end of file)\nstruct Handle;\nvoid poStart(Handle& h, Env& e);\nstruct Handle {\n")
rep("    std::string err;\n};", "    std::string err;\n    std::shared_ptr<Po> po;   // gate 4: playout handle\n};")
rep("    for (int t = 0; t < nt && t < T; ++t) {\n        const Tok& tk = all[t];",
    "    e.tok_obj.assign(T, -1);\n    for (int t = 0; t < nt && t < T; ++t) {\n        const Tok& tk = all[t];\n        e.tok_obj[t] = (tk.kind == 'c' || tk.kind == 'b') ? static_cast<int64_t>(tk.obj) : -1;")
rep("void startGame(Handle& h, Env& e) {\n", "void startGame(Handle& h, Env& e) {\n    if (h.po) { poStart(h, e); return; }\n    e.hist.clear();\n")
rep("    r.winner = 0; r.turns = 0; r.error = error;", "    r.winner = 0; r.turns = 0; r.error = error; r.var = e.var; r.po_seed = e.po_seed;")
rep("            if (!e.engine) startGame(h, e);\n", "            if (!e.engine) { startGame(h, e); if (e.idle) return; }\n")
open(p,'w').write(s)
print("ok")
