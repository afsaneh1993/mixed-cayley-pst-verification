"""Compare logs/tabnn_small.log and logs/tabnn16.log with Table tab:nn of the .tex file.
Usage: python3 compare_with_paper.py path/to/manuscript.tex LOG [LOG ...]   (LOG: output of t5_tabnn.py or t5_exact.py)"""
import re,sys,ast
tex=open(sys.argv[1],encoding="utf8").read()
rows=re.findall(r"^\$(.+?)\$ & (\d+) & (\d+) & (\d+) & (\d+) / (\d+) & (\d+) & (\d+) & (\d+) & (\d+)(?:\\\\)?$",tex,flags=re.M)
# map TeX group names -> script names (order of the table)
order=["D4","Q8","D6","Dic3","D8","Q16","SD16","M16","Z4sdZ4","Z2xD4","Z2xQ8","Z4oD4","Z4Z2sdZ2"]
paper={}
for name,r in zip(order,rows):
    sets,pst,nn,o2,o4,nc,oc,conn,cnn=map(int,r[1:])
    paper[name]=dict(sets=sets,pst=pst,nonnormal=nn,ord2=o2,ord4=o4,noncentral=nc,othercoset=oc,conn=conn,conn_nn=cnn)
ok=True
for log in sys.argv[2:]:
    for line in open(log):
        m=re.match(r"(\w+) (\{.*?\})",line)
        if not m: continue
        name,d=m.group(1),ast.literal_eval(m.group(2))
        if name in paper:
            for k,v in paper[name].items():
                if d[k]!=v: ok=False; print("MISMATCH",name,k,"paper",v,"computed",d[k])
        else:
            print("(no table row to compare)",name,{k:d[k] for k in("sets","pst","conn")})
print("tab:nn rows compared:",len([n for n in paper]),"-> ALL MATCH" if ok and len(rows)==13 else "-> CHECK OUTPUT ABOVE")
