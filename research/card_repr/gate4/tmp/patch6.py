p='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/analyze.py'; s=open(p).read()
def rep(a,b,c=1):
    global s
    assert s.count(a)==c, (a, s.count(a)); s=s.replace(a,b)
rep("""        hdr += ' | %-6s %-15s %5s %5s %6s %6s %-9s' % (nm + ' r', '[r CI]', 'sign', 'rho', '|dv|', 'TV', 'verdict')""",
    """        hdr += ' | %-6s %-15s %5s %5s %7s %6s %6s %-9s' % (nm + ' r', '[r CI]', 'sign', 'rho', 'mean dv', '|dv|', 'TV', 'verdict')""")
rep("""            line += ' | %+6.2f [%+.2f,%+.2f] %5.2f %+5.2f %6.3f %6.3f %-9s' % (s['r'], s['r_ci'][0], s['r_ci'][1], s['sign_agree'], s['rho'], s['model_abs'], s['tv'], st['verdict'][nm])""",
    """            line += ' | %+6.2f [%+.2f,%+.2f] %5.2f %+5.2f %+7.3f %6.3f %6.3f %-9s' % (s['r'], s['r_ci'][0], s['r_ci'][1], s['sign_agree'], s['rho'], s['model_mean'], s['model_abs'], s['tv'], st['verdict'][nm])""")
rep("""            line += ' | %+6.2f [%+.2f,%+.2f] %5.2f %+5.2f %6.3f %6.3f' % (s['r'], s['r_ci'][0], s['r_ci'][1], s['sign_agree'], s['rho'], s['model_abs'], s['tv'])""",
    """            line += ' | %+6.2f [%+.2f,%+.2f] %5.2f %+5.2f %+7.3f %6.3f %6.3f' % (s['r'], s['r_ci'][0], s['r_ci'][1], s['sign_agree'], s['rho'], s['model_mean'], s['model_abs'], s['tv'])""")
open(p,'w').write(s); print('ok')
