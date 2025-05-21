import json
# 1D transverse-field Ising model with periodic-boundary condition.

jz = -1

def incar_file_generator(nsite, hx):
    labels = []

    data = {}

    h_list = [f'{hx}*'+"I"*i+"X"+"I"*(nsite-i-1) for i in range(nsite)]
    h_list.extend([f'{jz}*'+"I"*i+"ZZ"+"I"*(nsite-i-2) for i in range(nsite-1)])
    h_list.append(f'{jz}*'+"Z"+"I"*(nsite-2)+"Z")
    data['h'] = h_list

    pool = ["I"*i+"Y"+"I"*(nsite-i-1) for i in range(nsite)]
    pool.extend(['I'*i+op[0]+'I'*(j-i-1)+op[1]+'I'*(nsite-j-1)
            for op in ['YZ', 'ZY']
            for i in range(nsite-1)
            for j in range(i+1, nsite)
            ])
    data['pool'] = pool
    data['ref_state'] = "0"*nsite

    with open('incarN%sg%s' % (nsite,hx), 'w') as f:
        json.dump(data, f, indent=4)

if __name__=="__main__":
    for nsite in [8,10,12,14,16,18]:
        for hx in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
            incar_file_generator(nsite, hx)
