import sys

inf = open(sys.argv[1])

of = open(sys.argv[2],'w')

entries = []

for l in inf:
	if not l.strip() or l[0] != ">" or "|" not in l:
		of.write(l)
		continue
	c = l.split("|")
	name = c[1]
	suf = ""
	if name in entries:
		suf = "_"+str(entries.count(name)+1)
	entries.append(name)
	c[1] = name+suf
	of.write("|".join(c))

inf.close()
of.close()
