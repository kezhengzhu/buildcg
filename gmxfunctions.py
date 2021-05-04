#!/usr/bin/env python
import os
import re
import subprocess
# import gmxapi as gmx
# gmxapi doesn't work with 2019.6 version of gromacs

def make_box(boxsize, input_mol, nmolecules, output, del_temp=True):
    args = ['gmx', 'insert-molecules']
    # outputfile = {'-o': output}
    boxargs = ['-box']
    if isinstance(boxsize, (list,tuple)):
        assert(len(boxsize) == 3)
        boxargs += [str(boxsize[i]) for i in range(3)]
    else:
        assert(isinstance(boxsize, (int,float)))
        boxargs += [str(boxsize)]

    if isinstance(input_mol, (list, tuple)):
        assert(isinstance(nmolecules,(list,tuple)) and len(input_mol) == len(nmolecules))
        # insert many times!
        ntypes = len(nmolecules)
        temp_out = ['tempbox-{:d}.gro'.format(i+1) for i in range(ntypes-1)]
        allout = temp_out + [output]

        for i in range(ntypes):
            molargs = ['-nmol', str(nmolecules[i])]
            if i == 0:
                inputs = ['-ci', input_mol[0]]
                all_args = args + boxargs + molargs + inputs + ['-o', allout[i]]
                box = subprocess.run(all_args, capture_output=True, text=True)
                # box = gmx.commandline_operation('gmx',
                #                                 arguments=args + boxargs + molargs,
                #                                 input_files=inputs,
                #                                 output_files=allout[i])
                if box.returncode != 0:
                    print("="*20 + "GMX INSERT-MOLECULES FAILED, SEE ERROR MSG" + "="*20)
                    print(box.stderr)
                    return None
            else:
                inputs = ['-ci', input_mol[i], '-f', temp_out[i-1]]
                all_args = args + boxargs + molargs + inputs + ['-o', allout[i]]
                box = subprocess.run(all_args, capture_output=True, text=True)
                # box = gmx.commandline_operation('gmx',
                #                                 arguments=args + molargs,
                #                                 input_files=inputs,
                #                                 output_files=allout[i])
                if box.returncode != 0:
                    print("="*20 + "GMX INSERT-MOLECULES FAILED, SEE ERROR MSG" + "="*20)
                    print(box.stderr)
                    return None
        if del_temp:
            for d in temp_out:
                os.remove(d)

        return output

    else:
        assert(os.path.exists(input_mol) and isinstance(nmolecules, int))
        molargs = ['-nmol', str(nmolecules)]
        inputs = ['-ci', input_mol]
        all_args = args + boxargs + molargs + inputs + ['-o', output]

        box = subprocess.run(all_args, capture_output=True, text=True)
        # box = gmx.commandline_operation('gmx',
        #                                 arguments=all_args,
        #                                 input_files=inputs,
        #                                 output_files=outputfile)
        if box.returncode != 0:
            print("="*20 + "GMX INSERT-MOLECULES FAILED, SEE ERROR MSG" + "="*20)
            print(box.stderr)
            return None
        return output
    return

def make_ndx(input_file, output_file="index.ndx", perline=15):
    '''
    Assumes that this is a .gro file, and will automatically detect the molecules
    and atoms just like gmx make_ndx would do
    '''
    f = open(input_file, 'r')
    content = f.read()
    f.close()

    content = re.sub("\d+[.]\d+", "", content) # remove floats
    content = re.sub("[ ]{2,}", " ", content) # remove large spaces

    molecules = set(re.findall("(?<=\d)[A-Z]{1}[A-Z0-9]+", content))
    m_dict = dict.fromkeys(molecules, None)
    atoms = set(re.findall("(?<=[ ])[A-Z][A-Z0-9]{0,}(?=[ ]\d)", content))
    a_dict = dict.fromkeys(atoms, None)

    content = re.sub("[\d]+(?=[A-Z])", "", content)
    lines = content.split("\n")
    n_atoms = int(lines[1].strip())

    for i in range(2, 2+n_atoms):
        line = lines[i].strip()
        parse = line.split(" ")
        id = int(parse[2])
        m = parse[0]
        a = parse[1]

        assert(m in molecules and a in atoms)

        if m_dict[m] is None: m_dict[m] = []
        if a_dict[a] is None: a_dict[a] = []

        m_dict[m].append(id)
        a_dict[a].append(id)

    dform = "{:>4d}"
    # line_temp = "{:>4d}" + " {:>4d}" * 14 + "\n"
    fout = open(output_file, 'w')
    for s in ["System", "Other"]:
        fout.write(write_ndx_line(s,n_atoms,perline))

    for key, val in m_dict.items():
        fout.write(write_ndx_line(key,val,perline))

    for key, val in a_dict.items():
        fout.write(write_ndx_line(key,val,perline))

    fout.close()
    return output_file

def write_ndx_line(name, indexes, perline=15):
    result = "[ {:s} ]\n".format(name)
    dform = "{:>4d}"
    if isinstance(indexes, int):
        nlen = indexes
        lines = nlen // perline + (nlen%perline != 0)
        for i in range(lines):
            mult = perline-1 if (i<lines-1) or (nlen%perline == 0) else (nlen%perline)-1
            line = dform + (" " + dform) * mult + "\n"
            result += line.format(*[j+1 for j in range(i*perline, min((i+1)*perline, nlen))])
    else:
        nlen = len(indexes)
        lines = nlen // perline + (nlen%perline != 0)
        for i in range(lines):
            mult = perline-1 if (i<lines-1) or (nlen%perline == 0) else (nlen%perline)-1
            line = dform + (" " + dform) * mult + "\n"
            result += line.format(*[indexes[j] for j in range(i*perline, min((i+1)*perline, nlen))])
    return result

def prepare_tpr(mdpfile, topolfile, grofile, ndxfile, output=None, capture=True, stop_warn=True):
    if output is None:
        output = mdpfile[:-4] + ".tpr"
    args = [
        'gmx', 'grompp',
        '-f', mdpfile,
        '-p', topolfile,
        '-c', grofile,
        '-n', ndxfile,
        '-o', output
    ]
    if stop_warn:
        args += ['-maxwarn', '1']
    grompp = subprocess.run(args, capture_output=capture, text=True)
    # grompp = gmx.commandline_operation('gmx', 'grompp',
    #                                input_files={
    #                                    '-f': mdpfile,
    #                                    '-p': topolfile,
    #                                    '-c': grofile,
    #                                    '-n': ndxfile,
    #                                },
    #                                output_files={'-o': output})
    #
    if grompp.returncode != 0:
        print(grompp.stderr)
        return None
    return output

def mdrun(tprname, capture=True, **kwargs):
    args = ['gmx', 'mdrun']
    for key, val in kwargs.items():
        args.append('-'+key)
        if val is not None:
            args.append(val)
    args += ['-deffnm', tprname]
    md_run = subprocess.run(args,  capture_output=capture, text=True)

    if md_run.returncode != 0:
        print(md_run.stderr)
        return None
    if tprname.endswith('.tpr'): tprname = tprname[:-4]
    return tprname+".gro"

def get_energy(filename, idx, output='output.xvg', capture=True):
    if isinstance(idx, (tuple,list)):
        echos = ['echo'] + [f'{idx[i]:d}' for i in range(len(idx))]
    else:
        assert(isinstance(idx, int))
        echos = ['echo', f'{idx:d}']
    idin = subprocess.Popen(echos, stdout=subprocess.PIPE, text=True)
    args = ['gmx', 'energy', '-f', filename, '-o', output]
    ener = subprocess.run(args, stdin=idin.stdout, capture_output=capture, text=True)

    if ener.returncode != 0:
        print(ener.stderr)
        return False

    return True

def main():
    # b = make_box([8,8,15], ["cgmolecule-eth.gro","cgmolecule-tmh.gro"], [150,50], "boxtest.gro")
    # print(make_ndx("boxtest.gro"))
    # prepare_tpr('test/npt.mdp', 'test/topol.top', 'test/mdboxinit.gro', 'test/index.ndx', capture=False)
    # print(get_energy('LJ/tp6/md04nve.edr', [18,19,22], output='LJ/tp6/pressure_offdiag.xvg',capture=False))
    print(get_energy('LJ/tp7/md09nve.edr', [18,19,22], output='LJ/tp7/pressure_offdiag.xvg',capture=False))
    return

if __name__ == '__main__':
    main()
