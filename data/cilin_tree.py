#%%
import json
import yaml

all_keys = []
cilin = {}
with open('new_cilin.txt') as f:
    for l in f:
        l = l.strip().split()
        k = l[0]
        all_keys.append(k)
        v = {
                'tag': ' '.join(l[1:]),
                'sub': {}
            }
        if len(k) == 1:
            cilin[k] = v
        elif len(k) == 2:
            k1, k2 = k[0], k[1]
            cilin[k1]['sub'][k2] = v
        elif len(k) == 4:
            k1, k2, k3 = k[0], k[1], k[2:]
            cilin[k1]['sub'][k2]['sub'][k3] = v
        elif len(k) == 5:
            k1, k2, k3, k4 = k[0], k[1], k[2:4], k[4:]
            cilin[k1]['sub'][k2]['sub'][k3]['sub'][k4] = v
        elif len(k) == 8:
            k1, k2, k3, k4, k5 = k[0], k[1], k[2:4], k[4:5], k[5:]
            cilin[k1]['sub'][k2]['sub'][k3]['sub'][k4]['sub'][k5] = v['tag'].split()
        else:
            raise Exception('error')


#%%
with open('cilin_tree.yaml', "w") as f:
    yaml.dump(cilin, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

with open('cilin_tree.json', "w") as f:
    json.dump(cilin, f, ensure_ascii=False)

# with open('cilin_keys.txt', 'w') as f:
#     f.write('\n'.join(all_keys) + '\n')