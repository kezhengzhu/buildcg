#!/usr/bin/env python
import re
import os
import copy
import math
import numpy as np

def checkerr(cond, message):
    if not (cond):
        raise Exception(message)
    return

def atype_calc(sig, epsi, lr, la):
    '''
    Takes in sigma (nm), epsilon (kJ/mol), lr, la
    returns C, A or V(c6), W(c12)
    '''
    c_mie = lr/(lr-la) * pow(lr/la, la/(lr-la))
    A = c_mie * epsi * pow(sig,lr) # repulsion coefficient
    C = c_mie * epsi * pow(sig,la)

    return (C,A)

def get_atomtype(name, atnum, mass, charge, ptype, C, A):
    '''
    Write a single line of atom type in ffnonbonded
    by default: ptype should be 'A'
    '''
    strcheck = [isinstance(s, str) for s in [name, atnum, ptype]]
    floatcheck = [isinstance(f, float) for f in [mass, charge, C, A]]
    checkerr(all(strcheck), "Name, at. num or ptype not of type str")
    checkerr(all(floatcheck), "Mass, charge, C or A not of type float")

    return "  {:<6}{:<11}{:<11.3f}{:<9.3f}{:<11}{:<13.5e}{:<11.5e}\n".format(name, atnum, mass, charge, ptype, C, A)

def get_atomnbp(name1, name2, C, A):
    '''
    Write the cross interaction between two different atom type in ffnonbonded
    '''
    strcheck = [isinstance(s, str) for s in [name1, name2]]
    floatcheck = [isinstance(f, float) for f in [C, A]]
    checkerr(all(strcheck), "Names not of type str")
    checkerr(all(floatcheck), "C or A not of type float")

    return "  {:<5}{:<6}{:<6}{:<15.5e}{:<15.5e}\n".format(name1,name2,"1",C,A)

def write_atomtype(template, atypes, nbp=None):
    '''
    Takes in template, search for $ATOM_TYPES and write in given str
    '''
    checkerr(isinstance(atypes, str), "Input atom_types info not of type str")
    f = open(template, 'r')
    content = f.read()
    f.close()

    ffnb = re.sub("[$]ATOM_TYPES", atypes, content)
    if nbp is None:
        ffnb = re.sub("[$]NBP_START[\s\S]+[$]NBP_STOP", "", ffnb)
    else:
        checkerr(isinstance(nbp, str), "Non-bonded params should be given in type str for input")
        ffnb = re.sub("[$]NBP_START", "", ffnb)
        ffnb = re.sub("[$]NBP_STOP", "", ffnb)
        ffnb = re.sub("[$]NONBOND_PARAMS", nbp, ffnb)

    return ffnb

def write_molecule(template, molecule_name, molecule_type, atoms, bonds=None):
    '''
    Takes in template, search for $MOLECULE_NAME, $MOLECULE_TYPE, $ATOMS and $BONDS
    And writes in the relevant info
    '''
    strcheck = [isinstance(s, str) for s in [molecule_name, molecule_type, atoms]]
    checkerr(all(strcheck), "Input information not all of type str")
    f = open(template, "r")
    content = f.read()
    f.close()

    molfile = re.sub("[$]MOLECULE_NAME", molecule_name, content)
    molfile = re.sub("[$]MOLECULE_TYPE", molecule_type, molfile)
    molfile = re.sub("[$]ATOMS", atoms, molfile)
    if bonds is None:
        molfile = re.sub("[$]BONDS_START[\s\S]+[$]BONDS_STOP", "", molfile)
    else:
        checkerr(isinstance(bonds, str), "Bonds should be of type str for input")
        molfile = re.sub("[$]BONDS_START", "", molfile)
        molfile = re.sub("[$]BONDS_STOP", "", molfile)
        molfile = re.sub("[$]BONDS", bonds, molfile)

    return molfile

def get_rings(edges, curr=None, found=None):
    '''
    Identifying the rings in the molecule. Make sure that the rings are created in order
    and triple connected rings doesn't work as of now.
    '''
    # initializing for zeroth layer
    if found is None: found = []
    if curr is None: curr = [0]
    # print("{:40s}\n{:10s} -- {:20s}".format(repr(edges),repr(curr),repr(found)))
    # Get current node
    current = curr[-1]
    # Check if ring is formed by tracking if this node has been visited
    if current in curr[:-1]:
        # remove traversed nodes and the ring, flip the ring, and return
        id = curr.index(current)
        f = curr[id:]
        addback = []
        for n in f[:-1]:
            if len(edges[n]) > 0:
                addback.append(n)

        curr[id:] = addback

        found.append(tuple(f))
        return found

    while len(edges[current])>0:
        # next in line is smallest in the current set that is left
        L = edges[current]
        next = min(L)

        # remove traversed
        L.remove(next)
        edges[next].remove(current)

        # successful ring would have removed current
        if len(curr) == 0 or curr[-1] != current: curr.append(current)
        curr.append(next)
        old = len(found)
        found = get_rings(edges, curr, found)
        if len(found) == old:
            curr.pop(-1)
    return found

def protate(x, n, a):
    # vector, normal of plane, angle to rotate by
    checkerr(isinstance(x, np.ndarray) and isinstance(n, np.ndarray), "Use numpy array for vectors")
    result = math.cos(a) * x + math.sin(a) * np.cross(n,x)
    return result

def norm_scale(x, m=1):
    # m is magnitude, reduce vector into unit vector then scale by magnitude
    checkerr(isinstance(x, np.ndarray), "Use numpy array for vectors")
    result = x / np.linalg.norm(x)
    if isinstance(m, (int,float)):
        result = result * m
    elif isinstance(m, np.ndarray):
        result = result * np.linalg.norm(m)
    else:
        raise Exception("Use numpy array or int/float for magnitude")
    return result

def make_zeroes(x):
    result = x * 1. # make duplicate
    result[np.abs(result) < 1e-5] = 0.
    return result

def get_bond_position(total_bonds, connected_pos):
    checkerr(isinstance(total_bonds, int), "No. of bonds specified not int")
    checkerr(total_bonds > 1 and total_bonds <= 4, "Total bonds should be between 2 to 4 to use this function")
    checkerr(isinstance(connected_pos, (list,tuple,np.ndarray)), "Connected positions should be an iterable")
    n_connected = len(connected_pos)
    checkerr(n_connected < total_bonds and n_connected > 0, "No. of connected bonds should be less than total bonds and more than 1")

    result = []

    if n_connected == 3:
        u1 = norm_scale(np.cross(connected_pos[0], connected_pos[1]))
        result = [u1]
    elif n_connected == 2:
        uimean_vec = norm_scale(-np.mean(connected_pos, axis=0))
        if total_bonds - n_connected == 1:
            result = [uimean_vec]
        else: # case total-connected == 2
            angle = math.acos(-1/3) # 109.47
            pn1 = norm_scale(np.cross(*connected_pos)) # normal vector to connected plane, which is on rotating plane
            pn2 = np.cross(uimean_vec, pn1) # rotating plane
            c1 = protate(uimean_vec, pn2, angle/2)
            c2 = protate(uimean_vec, pn2, -angle/2)
            result = [c1, c2]
    elif n_connected == 1:
        adding = total_bonds - n_connected
        if adding == 1:
            uimean_vec = norm_scale(-connected_pos[0])
            result = [uimean_vec]
        else:
            a = connected_pos[0]
            seed = np.array([1.,0,0])
            if np.linalg.norm(np.cross(a,seed)) < 1e-4:
                seed += np.array([0.,0.,1.])

            pn1 = norm_scale(np.cross(a, seed))
            if adding == 2:
                t = 2*math.pi/3
                uc1 = norm_scale(protate(a, pn1, t))
                uc2 = protate(uc1, pn1, t)
                result = [uc1, uc2]
            elif adding == 3:
                t = math.acos(-1/3)
                # rotate once by 109.47 on random plane to generate one bond
                ua = norm_scale(a)
                uc1 = protate(ua, pn1, t)
                # mean vec + norm vec = orthogonal plane against a,c1 plane
                uimean_vec = -np.mean([ua,uc1], axis=0)
                pn2 = norm_scale(np.cross(pn1, uimean_vec))
                # rotate mean vec up and down
                uc2 = protate(uimean_vec, pn2, t/2)
                uc3 = protate(uimean_vec, pn2, -t/2)
                result = [uc1, uc2, uc3]

    return result

def parse_mdp(mdpfile):
    f = open(mdpfile, "r")
    content = f.read()
    f.close()

    content = re.sub('[;][^\n]+', '', content)
    content = re.sub('[ ]{2,}', ' ', content)
    content = re.sub('[\n]{2,}', '\n', content)
    content = content.strip()
    listing = content.split("\n")

    mapping = dict()
    for item in listing:
        s = item.split("=")
        val = s[1].strip()
        if re.match('^[-+]?\d+$', val) != None:
            val = int(val)
        elif re.match('^[-+]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][-+]?((\d+\.\d*)|(\.\d+)|(\d+)))?$', val) != None:
            val = float(val)
        mapping[s[0].strip()] = val

    return mapping

def write_mdp(output_file, mapping, title=None):
    if title is None: title = "Auto generated .mdp file for Gromacs MD simulation with BuildCG"
    s = "; {:s}\n".format(title)
    for key, val in mapping.items():
        if val is None: continue
        if isinstance(val, float):
            val = "{:0.4f}".format(val) if val > 1e-3 else "{:0.4e}".format(val)
        elif not isinstance(val, str):
            val = str(val)
        if "_" in key:
            key = key.replace("_", "-")
        s += "{:<30s} = {:s}\n".format(key, val)

    fout = open(output_file, "w")
    fout.write(s)
    fout.close
    return output_file

def print_dict(dictionary):
    # Just prints dictionaries in the way that u can copy into python
    # Assuming that both sides are strings
    text = "{\n"
    for key, val in dictionary.items():
        if isinstance(val, str): val = '"' + val + '"'
        text += '    {:<30s}:   {},\n'.format('"'+key+'"', val)
    text += "}"
    print(text)
    return text

def write_topol(template, ff, sysname, molinfo):
    '''
    Takes in template, search for $FORCEFIELD, $SYSNAME and $N_MOLECULES
    And writes in the relevant info
    '''
    strcheck = [isinstance(s, str) for s in [ff, sysname, molinfo]]
    checkerr(all(strcheck), "Input information not all of type str")
    f = open(template, "r")
    content = f.read()
    f.close()

    topol = re.sub("[$]FORCEFIELD", ff, content)
    topol = re.sub("[$]SYSNAME", sysname, topol)
    topol = re.sub("[$]N_MOLECULES", molinfo, topol)

    return topol

def write_ff(template, ffnb, molecules):
    f = open(template, "r")
    content = f.read()
    f.close()

    ff = re.sub("[$]FF_NONBONDED", f'#include "{ffnb:s}"', content)

    if isinstance(molecules, (list, tuple)):
        mols = [f'#include "{molecules[i]:s}"' for i in range(len(molecules))]
        molinfo = "\n".join(mols)
    else:
        molinfo = f'#include "{molecules:s}"'
    ff = re.sub("[$]MOLECULES", molinfo, ff)

    return ff

def main():
    # print(parse_mdp("em.mdp"))
    d1 = parse_mdp("eql.mdp")
    d2 = parse_mdp("npt.mdp")
    missing = {"eql": [x for x in d1 if x not in d2], "npt": [x for x in d2 if x not in d1]}
    shared_diff  = {k: [d1[k], d2[k]] for k in d1 if k in d2 and d1[k] != d2[k]}
    shared_same = {k: d1[k] for k in d1 if k in d2 and d1[k] == d2[k]}

    print_dict(missing)
    print_dict(shared_diff)
    print(shared_same.keys())

    d1 = parse_mdp("nvt.mdp")
    d2 = parse_mdp("nvt_eq.mdp")
    missing = {"nvt": [x for x in d1 if x not in d2], "nvt_eq": [x for x in d2 if x not in d1]}
    shared_diff  = {k: [d1[k], d2[k]] for k in d1 if k in d2 and d1[k] != d2[k]}
    shared_same = {k: d1[k] for k in d1 if k in d2 and d1[k] == d2[k]}

    print_dict(missing)
    print_dict(shared_diff)
    print(shared_same.keys())

    print(write_ff("template/forcefield_template.itp", "ffnonbonded.itp", ["cgmolecule-eth.itp","cgmolecule-tmh.itp","cgmolecule-tm6.itp",]))

    #
    # print_dict(parse_mdp("npt.mdp"))
    # print_dict(parse_mdp("nvt.mdp"))
    # '^[-+]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][-+]?((\d+\.\d*)|(\.\d+)|(\d+)))?$'


    # print(parse_mdp("nvt_eq.mdp"))
    # print(parse_mdp("nvt.mdp"))

    # print(write_mdp("nvttest1.mdp", parse_mdp("nvt.mdp"), title="Testing a NVT mdp file"))
    # angle = math.acos(-1/3)
    # a = np.array([4,1,2])
    # seed = np.array([1,1,1])
    # n1 = norm_scale(np.cross(a, seed))
    # c1 = protate(a, n1, angle)
    # m_inv = norm_scale(-np.mean([a,c1], axis=0), a)
    # p2 = norm_scale(np.cross(a, c1))
    # n2 = norm_scale(np.cross(m_inv, p2))
    # c2 = protate(m_inv, n2, angle/2)
    # c3 = protate(m_inv, n2, -angle/2)

    # a1 = protate(a, n1, 2*math.pi/3)
    # a2 = protate(a1, n1, 2*math.pi/3)
    # c34 = get_bond_position(3,[a])
    # c1 = c34[0] * np.linalg.norm(a)
    # c2 = c34[1] * np.linalg.norm(a)
    # c3 = c34[2] * np.linalg.norm(a)
    # print(c1, c2)
    #
    # print(n1)
    # print(a,c1,c2)
    # rot = [a,c1,c2]
    # for i in rot:
    #     print(i, end=" ---- ")
    #     print(np.linalg.norm(i))
    #
    # for i in range(3):
    #     for j in range(i+1,3):
    #         print(i, j, math.acos(np.dot(rot[i],rot[j])/np.linalg.norm(rot[i])/np.linalg.norm(rot[j]))*180/math.pi)
    # f = open("ffnonbonded_template.itp", "r")
    # t = f.read()
    # f.close()

    # vars = re.sub("[$][A-Z0-9_]{1}[A-Z0-9_]+", write_atomtype("OCT",  "10", 10.0, 0.0, "A", 12.3e-3, 2.13e-4), t)
    # print(write_atomtype("ffnonbonded_template.itp", get_atomtype("OCT",  "10", 10.0, 0.0, "A", 12.3e-3, 2.13e-4)))
    # edges = [{1},{0,2,3},{1,3},{1,2,4,7},{3,5},{4,6},{5,7}, {3,6}]
    # print(get_rings(edges))
    #
    # edges = [{1},{0,2,3,5},{1,3},{1,2,4},{3,5},{1,4}]
    # print(get_rings(edges))
    #
    # edges = [{1,3},{0,2},{1,3},{0,2,4},{3,5},{4,6,7},{5,7},{5,6}]
    # print(get_rings(edges))
    #
    # edges = [{1,5},{0,2},{1,3},{2,4},{3,5},{0,4}]
    # print(get_rings(edges))
    #
    # edges = [{1,5},{0,2,8},{1,3},{2,4},{3,5},{0,4,6},{5,7},{6,8},{1,7}]
    # print(get_rings(edges))
    #
    # edges = [{1,10},{0,2,5},{1,3},{2,4},{3,5,6},{1,4,8},{4,7},{6,8},{5,7,9},{8,10},{0,9}]
    # print(get_rings(edges))
    #
    # edges = [{3},{3,9},{3,5},{0,1,2},{5,7,8},{2,4,9},{7},{4,6},{4},{1,5}]
    # print(get_rings(edges))
    #
    # edges = [{1},{0,2,5},{1,3},{2,4},{3,5,6},{1,4},{4}]
    # print(get_rings(edges))
    #
    # edges = [{1},{0,2,6},{1,3},{2,4},{3,5,6},{4},{1,4}]
    # print(get_rings(edges))
    return

if __name__ == '__main__':
    main()
