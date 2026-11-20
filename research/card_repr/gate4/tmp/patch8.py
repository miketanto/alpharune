p='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/REPORT.txt'; s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a; s=s.replace(a,b)
rep("""  - Normal rbenv behaviour is unchanged. Not re-run: test_determinism and the other rbenv tests (see "What is left").""",
    """  - Normal rbenv behaviour is unchanged. Re-run on the final build: test_determinism ok (410 steps, 24 games
    identical); test_fields OK (all v2 / v4 checks 0 mismatches, 300 games).""")
rep("""- rbenv tests (test_determinism, test_fields, test_agreement_v2) were not re-run after the extension. The changes are
  additive (history vector, token object ids, playout handles), but run tests/run_all.sh before relying on the
  build.
""", """- rbenv test_agreement_v2 / test_multivariant / test_random were not re-run after the extension (test_determinism and
  test_fields were, both OK). The changes are additive (history vector, token object ids, playout handles).
""")
open(p,'w').write(s); print('ok')
