p='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/run.py'; s=open(p).read()
a="""    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    pos = json.load(open(a.positions))"""
b="""    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':   # coordinator 2026-11-19: cap each process's CUDA memory (WDDM oversubscription)
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('G4_CUDA_FRAC', '0.2')))
    pos = json.load(open(a.positions))"""
assert s.count(a)==1; s=s.replace(a,b); open(p,'w').write(s); print('ok')
