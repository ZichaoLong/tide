"""Compact topology witnesses: phase support, source identities and exchange."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from backbones import multiscale as ms
from backbones import settle_graphs as st
from hb_lh_checks import check_output_paths
from hb_settle_checks import check_alignment


def main():
    cases = 0
    for key,widths,extra in ms.COMPACT_PROFILES:
        for n in (8,16,32):
            ordinary = ms.construct(widths,n=n,placement='none',add_dd4=extra)
            plain_edges = {(ordinary[0][u],ordinary[0][v],kind)
                           for u,es in enumerate(ordinary[2]) for v,kind in es}
            for h in (0,4):
                graph=ms.construct(widths,n=n,placement='all' if h else 'none',add_dd4=extra)
                nodes,_,outs,_,_,_=graph
                check_alignment(widths,graph,len(widths)+1)
                check_output_paths(widths,graph)
                assert len(nodes)==n*sum(widths)+h*sum(widths[:-1])
                assert {(nodes[u],nodes[v],kind) for u,es in enumerate(outs)
                        for v,kind in es if kind.startswith('DD')} == plain_edges
                delays,_=ms.reachability(nodes,outs,widths,only_detail=True)
                expected = [1,2,3,4] if extra else list(range(len(widths)//2,len(widths)))
                assert all(ms.support(delays[u])==expected for u,x in enumerate(nodes)
                           if x[0]==len(widths)-1 and x[2]=='d')
                assert {d%4 for d in expected}==set(range(4))
                if len(widths)==15:
                    assert all(sum(d%4==p for d in expected)>=2 for p in range(4))
                total,full=ms.source_node_coverage(nodes,outs,widths)
                _,plain=ms.source_node_coverage(nodes,outs,widths,True)
                if key=='compact-15':
                    assert plain==[[total,total]]*4
                if key in ('compact-9','compact-9-two-inputs') and (n==8 or (n==32 and h==4)):
                    assert full==[[total,total]]*4
                cases+=1
    for m,R,n in ((1,3,8),(1,4,16),(1,5,32),(1,10,32),(2,5,32),(2,9,32)):
        widths=st.exchange_widths(m,R)
        assert len(widths)==2*m+R+1 and sum(widths)==(R+3)*2**m-2
        for h in (0,4):
            graph=st.construct(widths,n=n,h=h)
            check_alignment(widths,graph,len(widths)+1)
            check_output_paths(widths,graph)
        nodes,ids,outs,_,_,_=st.construct(widths,n=n)
        masks=[0]*len(nodes)
        for p in range(2**m):
            for i in range(n):
                masks[ids[m,p,'d',i]]=1<<(p*n+i)
        for l in range(m,m+R):
            for p in range(2**m):
                for i in range(n):
                    u=ids[l,p,'d',i]
                    for v,_ in outs[u]:
                        masks[v]|=masks[u]
            r=l-m+1
            expected=2**(min(r,m)+min(r,n.bit_length()-1))
            assert all(bin(masks[ids[l+1,p,'d',i]]).count('1')==expected
                       for p in range(2**m) for i in range(n))
        assert expected==2**m*n
    for depth in (25,33):
        widths=ms.extended_widths(depth)
        assert sum(widths)==152+(depth-17)*16
        graph=ms.construct(widths,n=8)
        check_alignment(widths,graph,depth+1)
        check_output_paths(widths,graph)
    print(f'PASS: {cases} compact Multiscale cases; 12 narrow Exchange cases; 25/33-layer extensions')
    print('PASS: exact phase support, first-layer node coverage, hub deletion and aligned Settle timing')


if __name__=='__main__':
    main()
