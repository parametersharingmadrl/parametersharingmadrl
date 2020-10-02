from collections import defaultdict
import sys

fname = sys.argv[1]

with open(fname) as file:
    lines = file.readlines()

key_values = defaultdict(list)
for line in lines:
    if "| Episode:" in line:
        ep_num = line.split()[-1]
        key_values["episode"].append(ep_num)
    if line.startswith("q_taken_mean"):# or line.startswith("q_taken_mean") or line.startswith("td_error_abs"):
        data = line.split()
        value = data[3]
        key_values['return_mean'].append(float(value))

lines = []
headers = list(key_values)
lines.append(",".join(headers))
#print((key_values[headers[2d]]))
for i in range(len(key_values[headers[0]])-1):
    lines.append(",".join(str(key_values[head][i]) for head in headers))

with open(fname+".csv",'w') as out_file:
    out_file.write("\n".join(lines))
