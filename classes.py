#!/usr/bin/env python
import os
import copy
import numpy as np
import math
import const as cst
import pickle
import time
from functions import *
from gmxfunctions import *
from random import randint

class Project(object):
    def __init__(self, folder, cutoff=None):
        if os.path.exists(folder):
            print("Warning, project folder already exists, creating new project will risk overwriting it! Consider loading from .pickle if project is saved")
        else:
            os.mkdir(folder)
        self.path = folder
        self.procedure = []
        self.atoms = {}
        self.molecules = {}
        self.state_properties = {}
        self.cutoff = cutoff if isinstance(cutoff, (int,float)) else 5.0
        return

    # def add_cg(self, cg):
    #     checkerr(isinstance(cg, CG), "CG atom should be of type CG")
    #     self.atoms[cg.name] = cg
    #     return self
    #
    # def add_cgs(self, cgs):
    #     for cg in cgs:
    #         self.atoms[cg.name] = cg
    #     return self

    def add_molecule(self, mol, num):
        assert(isinstance(mol, Molecule) and isinstance(num, int))
        if mol in self.molecules:
            self.molecules[mol] += num
        else:
            self.molecules[mol] = num
        self.update_atoms()
        return self

    def quick_set(self, *args):
        molecules = []
        nmolecules = []
        for arg in args:
            checkerr(isinstance(arg, tuple) or isinstance(arg, list), "Use iterable tuple or list for quick_set")
            checkerr(isinstance(arg[0], Molecule), "First element of iterable must be of Molecule type")
            checkerr(isinstance(arg[1], int), "Second element of iterable must be of int type")
            molecules.append(arg[0])
            nmolecules.append(arg[1])

        for i in range(len(molecules)):
            mol = molecules[i]
            if mol in self.molecules:
                self.molecules[mol] += nmolecules[i]
            else:
                self.molecules[mol] = nmolecules[i]
        self.update_atoms()

        return self

    def update_atoms(self):
        new_dict = {}
        for mol, nmol in self.molecules.items():
            a_list = mol.atoms
            a_set = set(a_list)
            for a in a_set:
                count = a_list.count(a)
                if a in new_dict:
                    new_dict[a] += nmol * count
                else:
                    new_dict[a] = nmol * count
        self.atoms = new_dict
        return self

    def set_topol(self, output="topol.top", sysname="MDrun", table_name="table", template="template/topol_template.top"):
        '''
        Prepare topology file, including forcefields (ffnonbonded and molecular bonds)
        '''
        # ffnonbonded.itp setup
        ffnb_temp = "template/ffnonbonded_template.itp"
        atypes = ""
        nbp = ""
        table = table_name.strip().lower()
        alist = list(self.atoms)
        for i in range(len(alist)):
            cg = alist[i]
            atypes += cg.write_atype()

            fout = open(self.path + "/" + f"{table}_{cg.name}_{cg.name}.xvg", "w")
            t = cg.write_table()
            fout.write(t)
            fout.close()

            for j in range(i+1, len(alist)):
                other = alist[j]
                nbp += cg.write_atype(other)

                fout = open(self.path + "/" + f"{table}_{cg.name}_{other.name}.xvg", "w")
                t = cg.write_table()
                fout.write(t)
                fout.close()
        # Use last table as table.xvg holder for gromacs (will not actually be used)
        fout = open(self.path + "/" + "table.xvg", "w")
        fout.write(t)
        fout.close()

        if len(nbp) == 0: nbp = None
        text = write_atomtype(ffnb_temp, atypes, nbp)

        ffnb_file = "ffnonbonded.itp"
        fout = open(self.path + "/" +ffnb_file, "w")
        fout.write(text)
        fout.close()

        # molecules itp files setup
        mols = list(self.molecules)
        molfiles = []
        for m in mols:
            molfiles.append(m.write_ffentry(write_file=True, path=self.path))

        if len(molfiles) == 1: molfiles = molfiles[0]

        # forcefield.itp setup
        ff_temp = "template/forcefield_template.itp"

        text = write_ff(ff_temp, ffnb_file, molfiles)

        ff_file = "forcefield.itp"
        fout = open(self.path + "/" + ff_file, "w")
        fout.write(text)
        fout.close()

        # topol.top setup
        molinfo = ""
        for mol, nmol in self.molecules.items():
            molinfo += "  {:<10}{:<6d}\n".format(mol.name, nmol)
        text = write_topol(template, ff_file, sysname, molinfo)

        topol_file = "topol.top"

        fout = open(self.path + "/" + topol_file, "w")
        fout.write(text)
        fout.close()

        return

    def prepare_grofiles(self, boxsize, output="mdboxinit.gro"):
        '''
        Preparing individual gromacs files and adding into a box for MD simulation
        '''
        grofiles = []
        nmolecules = []
        for mol, nmol in self.molecules.items():
            b = mol.calc_positions()
            checkerr(b, "Calculate positions failed! Manually update positions of molecules if necessary")
            grofiles.append(self.path + "/" + mol.write_grofile(write_file=True, path=self.path))
            nmolecules.append(nmol)
        if len(nmolecules) == 1:
            grofiles = grofiles[0]
            nmolecules = nmolecules[0]

        gout = make_box(boxsize, grofiles, nmolecules, self.path+"/"+output)
        nout = make_ndx(gout, output_file=self.path+"/"+"index.ndx", perline=15)
        return (gout, nout)

    def add_mdrun(self, template, prev=None, add_path=None, **kwargs):
        '''
        Set up a single md run based on existing template
        '''
        id = len(self.procedure)
        if template is None:
            print("No template specified. Using default NVE template")
            template = "nve"
        name = "md{:02d}{:s}".format(id, template)

        if add_path is None:
            path = self.path
        else:
            path = self.path + "/" + add_path
        if isinstance(prev, str):
            grofile = prev
        elif isinstance(prev, MDProcedure):
            grofile = prev.path + "/" + prev.name + ".gro"
        else:
            grofile = self.path + "/" + "mdboxinit.gro"

        result = MDProcedure(template, name, path=path, grofile=grofile, **kwargs)
        self.procedure.append(result)

        return result

    def prepare_mdp(self):
        '''
        Set up all mdp files for the whole sequence of MD run
        '''
        T = self.state_properties['temperature']
        s = f"T = {T:0.2f}K"
        if 'pressure' in self.state_properties:
            P = self.state_properties['pressure']
            s += f", P = {P:0.2f}bar"
        for p in self.procedure:
            title = f"{p.name.upper():s} Simulation, auto-generated .mdp file at {s:s}"
            p.make_mdpfile(title=title)
        return self

    def tv_run(self, temperature, path=None,  prod_length=5000000, prod="nve"):
        '''
        Set volume constant run: (T: Kelvin)
        1. energy minimisation
        2. NVT equilibratiton run
        3. NVE production run
        '''
        checkerr(isinstance(temperature, (int,float)), "Temperature not of type int/float!")
        self.state_properties['temperature'] = temperature
        a = [cg.name for cg in self.atoms]
        g = ' '.join(a)
        glist = [f"{a[i]} {a[j]}" for i in range(len(a)) for j in range(i, len(a))]
        gtable = ' '.join(glist)
        grps = {
            "energygrps"        : g,
            "energygrp-table"   : gtable,
            "tc-grps"           : None
        }
        lensum = sum([a.sigma * cst.atonm for a in self.atoms])
        lenscale = lensum / len(self.atoms)
        cutoff_nm = lenscale * self.cutoff
        print(cutoff_nm)
        if path is not None:
            p = self.path + "/" + path
            if os.path.exists(p):
                print("Warning: Path exists! Risk carrying on at your own risk of overwriting data")
                time.sleep(3)
            else:
                os.mkdir(p)
        em = self.add_mdrun('em', prev=None, add_path=path,
                            rlist=cutoff_nm, rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)
        grps["tc-grps"] = g
        nvt = self.add_mdrun('nvt', prev=em, add_path=path,
                            dt=0.01, nsteps=1000000, tinit=0, nstcalcenergy=1000,
                            nstenergy=1000, ref_t=temperature, gen_vel="yes",
                            gen_temp=temperature, gen_seed=-1, rlist=cutoff_nm,
                            rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)
        if prod.lower() == "nve":
            # for gas phase use NVE, for liquid phase please use NVT due to energy drift
            grps["tc-grps"] = None
            nve = self.add_mdrun('nve', prev=nvt, add_path=path,
                                dt=0.002, nsteps=prod_length, tinit=0, nstcalcenergy=1,
                                nstenergy=1, tcoupl="no", pcoupl="no", rlist=cutoff_nm,
                                rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)
        else:
            nvt2 = self.add_mdrun('nvt', prev=nvt, add_path=path,
                                dt=0.002, nsteps=prod_length, tinit=0, nstcalcenergy=1,
                                nstenergy=1, ref_t=temperature, pcoupl="no", rlist=cutoff_nm,
                                rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)

        return self

    def std_run(self, temperature, pressure, path=None, prod_length=5000000, prod="nve"):
        '''
        Standard viscosity run: (T: Kelvin, P: bar)
        1. energy minimisation
        2. equilibrate with Berendsen barostat
        3. NPT equilibration run
        4. NVT equilibration run
        5. NVE production run
        '''
        checkerr(isinstance(temperature, (int,float)) and isinstance(pressure, (int,float)), "Temperature and pressure not of type int/float!")
        self.state_properties['temperature'] = temperature
        self.state_properties['pressure'] = pressure
        a = [cg.name for cg in self.atoms]
        g = ' '.join(a)
        glist = [f"{a[i]} {a[j]}" for i in range(len(a)) for j in range(i, len(a))]
        gtable = ' '.join(glist)
        grps = {
            "energygrps"        : g,
            "energygrp-table"   : gtable,
            "tc-grps"           : None
        }
        lensum = sum([a.sigma * cst.atonm for a in self.atoms])
        lenscale = lensum / len(self.atoms)
        cutoff_nm = lenscale * self.cutoff
        print(cutoff_nm)
        if path is not None:
            p = self.path + "/" + path
            if os.path.exists(p):
                print("Warning: Path exists! Risk carrying on at your own risk of overwriting data")
                time.sleep(3)
            else:
                os.mkdir(p)
        em = self.add_mdrun('em', prev=None, add_path=path,
                            rlist=cutoff_nm, rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)
        grps["tc-grps"] = g
        npt1 = self.add_mdrun('npt', prev=em, add_path=path,
                            pcoupl="Berendsen", tinit=0, nsteps=1000000, ref_t=temperature,
                            ref_p=pressure, gen_vel="yes", gen_temp=temperature,
                            gen_seed=-1, rlist=cutoff_nm, rcoulomb=cutoff_nm,
                            rvdw=cutoff_nm, **grps)
        npt2 = self.add_mdrun('npt', prev=npt1, add_path=path,
                            pcoupl="Parrinello-Rahman", tinit=1000000, nsteps=3000000,
                            ref_t=temperature, ref_p=pressure, gen_vel="no",
                            rlist=cutoff_nm, rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)
        nvt = self.add_mdrun('nvt', prev=npt2, add_path=path,
                            dt=0.01, nsteps=1000000, tinit=0, nstcalcenergy=1000,
                            nstenergy=1000, ref_t=temperature, gen_vel="no",
                            rlist=cutoff_nm, rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)

        if prod.lower() == "nve":
            # for gas phase use NVE, for liquid phase please use NVT due to energy drift
            grps["tc-grps"] = None
            nve = self.add_mdrun('nve', prev=nvt, add_path=path,
                                dt=0.002, nsteps=prod_length, tinit=0, nstcalcenergy=1,
                                nstenergy=1, tcoupl="no", pcoupl="no", rlist=cutoff_nm,
                                rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)
        else:
            nvt2 = self.add_mdrun('nvt', prev=nvt, add_path=path,
                                dt=0.002, nsteps=prod_length, tinit=0, nstcalcenergy=1,
                                nstenergy=1, ref_t=temperature, pcoupl="no", rlist=cutoff_nm,
                                rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)
        # nve = self.add_mdrun('nve', prev=nvt, add_path=path,
        #                     dt=0.002, nsteps=prod_length, tinit=0, nstcalcenergy=1,
        #                     nstenergy=1, tcoupl="no", pcoupl="no", rlist=cutoff_nm,
        #                     rcoulomb=cutoff_nm, rvdw=cutoff_nm, **grps)

        return self

    def run(self, capture=False):
        '''
        After completing prepare_mdp, run will recursively progress through the procedure
        until self.procedure is empty
        '''
        path = self.path + "/"
        try:
            while len(self.procedure) > 0:
                next = self.procedure[0]
                next.make_tprfile(path+"topol.top", path+"index.ndx", capture=False)
                next.run(table=path+"table.xvg", v=None, capture=capture)
                self.procedure.pop(0)

                print("="*20 + "{:40s}".format(next.name.upper()+" procedure done!") + "="*20)
        except:
            fout = self.save()
            raise Exception("Error occurred! Project saved at {:s}.".format(fout))

        print("="*20 + "{:40s}".format("All procedures complete!") + "="*20)
        return

    def save(self, file="project.pickle"):
        fw = open(self.path + "/" + file, "wb")
        pickle.dump(self, fw)
        fw.close()

        return file

    @classmethod
    def load(cls, file):
        fb = open(file, 'rb')
        p = pickle.load(fb)
        fb.close()
        if not isinstance(p, Project):
            print("Warning: project loaded is not of correct class \"Project\"")
        return p

class MDProcedure(object):
    def __init__(self, template, name, path="", grofile="mdboxinit.gro", **kwargs):
        if (not isinstance(template, str)) or template.lower() not in ["em","nve","nvt","npt"]:
            print("Warning, no template found, no pre-loaded mdp options in MDProcedure object. Define your own options!")
            self.template = "none"
            self.options = dict()
        else:
            self.template = template.lower()
            self.options = copy.deepcopy(cst.templates[template])
        self.name = name
        self.path = path
        self.set_options(**kwargs)
        self.mdpfile = None
        self.tprfile = None
        self.grofile = grofile

    def __repr__(self):
        return "<MDP_{:s}_from:{:s}>".format(self.name, self.template.upper())

    def set_options(self, w_ignore=False, **kwargs):
        for key, val in kwargs.items():
            if "_" in key:
                key = key.replace("_", "-")
            if key in cst.MDP_OPTIONS:
                self.options[key] = val
            else:
                print(f"Warning: key '{key:s}' not found in MDP options, carry on at your own risk!")
                time.sleep(3)
                if w_ignore:
                    self.options[key] = val
        return self

    def make_mdpfile(self, title=None):
        file = "{:s}.mdp".format(self.name)
        fout = write_mdp(self.path+'/'+file, self.options, title)
        self.mdpfile = file

        return fout

    def make_tprfile(self, topolfile, ndxfile, capture=True):
        file = "{:s}.tpr".format(self.name)
        path = self.path + "/"
        fout = prepare_tpr(path+self.mdpfile, topolfile, self.grofile, ndxfile, output=path+file, capture=capture)
        if fout is None: raise Exception("Error occured in grompp function, see error message")
        self.tprfile = file

        return file

    def run(self, capture=True, **kwargs):
        path = self.path + "/"
        out = mdrun(path+self.name, capture=capture, **kwargs)
        if out is None: raise Exception("Error occured in mdrun function, see error message")
        return out

class CG(object):
    '''
    CG class contains info of a coarse-grained bead, i.e. sigma, epsilon, lam_r and lam_a
    '''
    _total = 0
    _ecross = np.array([[0.]])
    _rcross = np.array([[0.]])
    _list = []
    def __init__(self, name, mass, charge, sigma, epsilon, lam_r, lam_a):
        checkerr(isinstance(name, str), "CG init failed, name given is not str")
        fcheck = [isinstance(f, (float,int)) for f in [mass,charge,sigma,epsilon,lam_r,lam_a]]
        checkerr(all(fcheck), "CG init failed, one or more properties is not given in float or int")
        name = "".join(name.strip().split(" "))
        name = name.upper()[:3]
        if name in CG._list:
            raise Exception("Name already used, please use a new name")
        self.name = name
        CG._list.append(self.name)
        self.mass = mass
        self.charge = charge
        self.sigma = sigma
        self.epsilon = epsilon
        self.lam_r = lam_r
        self.lam_a = lam_a

        self.index = CG._total
        CG._total += 1
        CG.add_elem()

    @classmethod
    def add_elem(cls):
        t1 = cls._ecross
        t2 = cls._rcross
        if cls._total > 1:
            t12 = np.append(t1, np.zeros((1, t1.shape[1])), axis=0)
            cls._ecross = np.append(t12, np.zeros((t12.shape[0], 1)), axis=1)
            t22 = np.append(t2, np.zeros((1, t2.shape[1])), axis=0)
            cls._rcross = np.append(t22, np.zeros((t22.shape[0], 1)), axis=1)

    @classmethod
    def print_table(cls):
        print("{:40s}".format("Total defined group types: "), cls._total)
        print("{:40s}".format("Combination k values for Epsilon: "), cls._etable)
        print("{:40s}".format("Combination k values for Lambda_r: "), cls._lrtable)

    @classmethod
    def combining_eval(cls, g1, g2, val):
        try:
            i1 = g1.index
            i2 = g2.index
        except:
            raise Exception("Index not available. Do not set combination rules for group combinations")
        cls._ecross[i1,i2] = val
        cls._ecross[i2,i1] = val

    @classmethod
    def combining_ecoeff(cls, g1, g2, val):
        try:
            i1 = g1.index
            i2 = g2.index
        except:
            raise Exception("Index not available. Do not set combination rules for group combinations")
        s1 = g1.sigma
        s2 = g2.sigma
        e1 = g1.epsilon
        e2 = g2.epsilon

        actl = sqrt(pow(s1,3) * pow(s2,3)) / pow((s1+s2)/2, 3) * sqrt(e1 * e2)
        ratio = val / actl
        kij = 1 - ratio
        cls._ecross[i1,i2] = kij
        cls._ecross[i2,i1] = kij

    @classmethod
    def combining_rval(cls, g1, g2, val):
        try:
            i1 = g1.index
            i2 = g2.index
        except:
            raise Exception("Index not available. Do not set combination rules for group combinations")
        cls._rcross[i1,i2] = val
        cls._rcross[i2,i1] = val

    @classmethod
    def combining_rcoeff(cls, g1, g2, val):
        try:
            i1 = g1.index
            i2 = g2.index
        except:
            raise Exception("Index not available. Do not set combination rules for group combinations")
        lr1 = g1.lam_r
        lr2 = g2.lam_r

        intm = val - 3
        ratio = intm / sqrt((lr1 - 3) * (lr2 - 3))
        gij = 1 - ratio
        cls._rcross[i1,i2] = gij
        cls._rcross[i2,i1] = gij

    def __repr__(self):
        return f"<CG {self.name:s}>"

    def write_atype(self, other=None):
        #get_atomtype(name, atnum, mass, charge, ptype, C, A)
        if other is None:
            lr = self.lam_r
            la = self.lam_a
            sig = self.sigma * cst.atonm
            epsi = self.epsilon * cst.K_to_kJmol
            (C,A) = atype_calc(sig, epsi, lr, la)
            s = get_atomtype(self.name, "6", self.mass, 0.0, "A", C, A)
        else:
            checkerr(isinstance(other, CG), 'Other has to be CG type as well')
            lr = CG._rcross[self.index, other.index]
            if lr < 8:
                lr = 3 + math.sqrt((self.lam_r - 3)*(self.lam_r - 3))

            la = 3 + math.sqrt( (self.lam_a - 3) * (other.lam_a - 3) )
            sig = (self.sigma + other.sigma) / 2 * cst.atonm
            epsi = CG._ecross[self.index, other.index]
            if epsi < 100.:
                sratio = math.sqrt(pow(self.sigma*cst.atonm, 3) * pow(other.sigma*cst.atonm, 3)) / pow(sig,3)
                epsi = sratio * math.sqrt(self.epsilon * other.epsilon)
            epsi = epsi * cst.K_to_kJmol
            (C,A) = atype_calc(sig, epsi, lr, la)
            s = get_atomnbp(self.name, other.name, C, A)

        return s

    def write_table(self, cutoff=5.0, delr=0.002, other=None):
        if other is None:
            lr = self.lam_r
            la = self.lam_a
        else:
            lr = CG._rcross[self.index, other.index]
            if lr < 8:
                lr = 3 + math.sqrt((self.lam_r - 3)*(self.lam_r - 3))

            la = 3 + math.sqrt( (self.lam_a - 3) * (other.lam_a - 3) )

        c_mie = lr/(lr-la) * pow(lr/la, la/(lr-la))
        nbins=int((cutoff+1)/delr)+1
        namecard = self.name if other is None else "cross between {:s}{:s}".format(self.name, other.name)
        table = ""
        table += "# Mie potential of {:s} lambda_r = {:<7.5f} and lambda_a = {:<7.5f}\n".format(namecard,lr,la)
        table += "# Calculate C and A (or V(6) W(12)) parameters in ffnonbonded.itp using:\n"
        table += "# C = {:<7.5f} x epsilon x sigma ^ {:<7.5f}, A = {:<7.5f} x epsilon x sigma ^ {:<7.5f}\n".format(c_mie,lr,c_mie,la)
        for j in range(nbins):
            r = delr * j
            if r == 0:
                table += "{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,0.0,0.0,0.0,0.0,0.0,0.0)
            elif lr/(pow(r,lr+1)) > 1e27:
                table += "{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,0.0,0.0,0.0,0.0,0.0,0.0)
            else:
                # Format provided as: f(r), -f'(r), g(r), -g'(r), h(r), -h'(r)
                f = 1 / r
                fp = 1 / (pow(r,2))
                g = -1 / (pow(r,la))
                gp = -la / (pow(r,la+1))
                h = 1 / (pow(r,lr))
                hp = lr / (pow(r,lr+1))

                table += "{:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}   {:12.10e} {:12.10e}\n".format(r,f,fp,g,gp,h,hp)
        return table

class Molecule(object):
    '''
    Molecule file contains a graph which arranges the atom, writes itp file and the base gro file
    __init__ assumes any passed CG atoms are arranged in a chain
    '''
    def __init__(self, *args, name=""):
        self.atoms = []
        self.edges = [] # or bonds
        self.angles = {}
        self.position = {}
        if len(name) == 0:
            c = [chr(randint(65,65+26)) for i in range(3)]
            self.name = "".join(c)
        else:
            checkerr(isinstance(name, str), "Use str for names")
            self.name = name.upper()
        for arg in args:
            self.add_atom(arg)
        return

    def __repr__(self):
        return "<Molecule {:s}>".format(self.name)

    def connect(self, i, j):
        L = len(self.atoms)
        checkerr(i < L and j < L and i != j, "Connect failed, out of range or trying to connect atom to itself")
        self.edges[i].add(j)
        self.edges[j].add(i)
        return

    def disconnect(self, i, j):
        L = len(self.atoms)
        checkerr(i < L and j < L and i != j, "Connect failed, out of range or trying to connect atom to itself")
        self.edges[i].remove(j)
        self.edges[j].remove(i)
        return

    def add_atom(self, cg, connectedto=None):
        i = len(self.atoms)
        if connectedto is None:
            self.atoms.append(cg)
            self.edges.append(set())
            if i > 0:
                self.connect(i,i-1)
        elif isinstance(connectedto, int):
            checkerr(connectedto < i, "Connected-to atom is not found!")
            self.atoms.append(cg)
            self.edges.append(set())
            self.connect(connectedto, i)
        elif isinstance(connectedto, (list,tuple)):
            icheck = [isinstance(j, int) and j < i for j in connectedto]
            checkerr(all(icheck), "Connected-to input error, not int or out of range")
            self.atoms.append(cg)
            self.edges.append(set())
            for j in connectedto: self.connect(i,j)
        else:
            print("Atom failed to add! No connections found, likely wrong connection specification")
        return

    def calc_positions(self):
        '''
        Calculates the vector positions of a molecule starting from 0,0,0
        Resolves for 1 ring system max only
        Returns True if all atoms are placed correctly. If function returns false or
        an error occur, manual input might be needed.
        '''
        ring = self.place_rings()
        placed = [False] * len(self.atoms)
        connections = [len(node) for node in self.edges]
        resolve_next = []
        if ring is not None:
            for i in ring[:-1]:
                placed[i] = True
                if connections[i] > 2:
                    resolve_next.append(i)

        if len(resolve_next) == 0:
            resolve_next.append(0)
            self.position[0] = np.array([0,0,0])
            placed[0] = True
        if all(placed): return True

        print(f"Start resolving atom from list: {resolve_next}")
        while len(resolve_next) > 0:
        # Resolve angles for a single atom
            atom = resolve_next.pop(0)
            print(f"Currently resolving atom no. {atom:d}")
            connected = self.edges[atom]
            pos = np.array(self.position[atom])
            total = len(connected)
            done = []
            placing = []

            for i in connected:
                if not placed[i]:
                    placing.append(i)
                else:
                    done.append(np.array(self.position[i])-pos)
            print(f"    Connected to this atom: ", connected)
            print(f"    Done: ", done)
            print(f"    Placing: ", placing)
            if len(placing) == 0:
                continue
            elif total == 1: # so total 1 placing 1, choose x direction and put it there
                a = placing[0]
                blen = (self.atoms[a].sigma+self.atoms[atom].sigma)/2
                self.position[a] = blen * np.array([1.,0.,0.]) + pos
                resolve_next.append(a)
                placed[a] = True
            else:
                u_npos = get_bond_position(total, done)
                for i in range(len(placing)):
                    a = placing[i]
                    blen = (self.atoms[a].sigma+self.atoms[atom].sigma)/2
                    self.position[a] = make_zeroes(blen * u_npos[i]) + pos
                    resolve_next.append(a)
                    placed[a] = True
            print(f"Next to resolve: {resolve_next}")
        return all(placed)

    def place_rings(self):
        '''
        Placing rings only works for one ring so far, and it will take the average
        sigma as the equal bond length for the whole ring system
        '''
        track = copy.deepcopy(self.edges)
        rings = get_rings(track)
        if len(rings) == 0: return None
        track = copy.deepcopy(self.edges)
        placed = [False] * len(self.atoms)
        for ring in rings:
            ringsize = len(ring)-1
            d = sum([self.atoms[a].sigma for a in ring[:-1]])/ringsize
            # inner_angle = (math.pi*(ringsize-2))/ringsize
            if len(self.position) == 0:
                self.position[ring[0]] = np.array([0.,0.,0.])
                x, y = 0, 0
                outer = 2*math.pi/ringsize
                for i in range(ringsize-1):
                    x, y = x + d*math.cos(i*outer), y + d*math.sin(i*outer)
                    self.position[ring[i+1]] = np.array([x,y,0.])
            else:
                raise Exception("Placing rings only works with one ring, do generate your own .gro file")

        return rings[0]

    def add_angles(self, loc, degree):
        # in degrees, loc defined as a 3 digit int or list of ints
        checkerr(isinstance(loc, (list,int,tuple)), "Angle location not in type int or list or tuple")
        if isinstance(loc, int):
            checkerr(loc < 999 and loc > 11, "Angle location should be a 2/3 digit int. If molecule is too big, use an iterable instead")
            A = loc // 100
            B = (loc // 10) % 10
            C = loc % 10
        else:
            icheck = [isinstance(g, int) for g in loc]
            checkerr(all(icheck), "Items in iterable is not of type int")
            A = loc[0]
            B = loc[1]
            C = loc[2]

        checkerr(max(A,B,C) < len(self.atoms), "Loc out of range of atoms in molecule")

        bina = B in self.edges[A]
        cinb = C in self.edges[B]
        checkerr(all([bina,cinb]), "Atoms {:d},{:d},{:d} doesn't form an angle!".format(A,B,C))
        if A > C: A, C = C, A
        checkerr((A,B,C) not in self.angles, "Angle already specified!")
        checkerr(degree > 0 and degree < 360, "Angle out of range!")
        self.angles[(A,B,C)] = degree

        return

    def write_grofile(self, title="", write_file=False, path=""):
        '''
        Writes grochain after positions have been filled
        molecule name required: such as MET, ETH, DEC, WATER, etc etc
        bead name: such as CH4, mustn't be the same as molecule
        '''
        mname = self.name
        natoms = len(self.atoms)
        if len(title) == 0:
            title = "CG molecule {} with {} beads".format(mname,natoms)
            longname = "cgmolecule-{}".format(mname.lower())
        else:
            longname = title.replace(" ", "").lower()

        checkerr(len(self.position) > 0, "Positions of molecules not resolved yet. Use .calc_positions method first.")
        content = ""

        content += "{}\n".format(title)
        content += "{:>5d}\n".format(natoms)

        for atom in range(natoms):
            p = self.position[atom] # This is in Armstrong
            aname = self.atoms[atom].name
            atonm = 0.1
            # Writing has the following format {:>5d}{:<5}{:>5}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}{:>8.4f}{:>8.4f}{:>8.4f}
            content += "{:>5d}{:<5}{:>5}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}\n".format(1,mname,aname,atom+1,p[0]*atonm,p[1]*atonm,p[2]*atonm)
        content += "{:>10.5f}{:>10.5f}{:>10.5f}\n".format(0.0,0.0,0.0)

        # Writing .gro file
        if write_file:
            if len(path) > 0: path = path + "/"
            filename = "{}.gro".format(longname)
            fout = open(path+filename, "w")
            fout.write(content)
            fout.close()
            return filename
        return content

    def write_ffentry(self, template="template/molecule_template.itp", title="", kb=6666, write_file=False, path=""):
        '''
        Writes grochain after positions have been filled
        molecule name required: such as MET, ETH, DEC, WATER, etc etc
        bead name: such as CH4, mustn't be the same as molecule
        '''
        mname = self.name
        natoms = len(self.atoms)
        if len(title) == 0:
            title = "CG molecule {} with {} beads".format(mname,natoms)
            longname = "cgmolecule{}".format(mname.lower())
        else:
            longname = title.replace(" ", "").lower()

        # Write .itp file
        moltype = "  {:<7}{:<3d}\n\n".format(mname,1)
        atoms_entry = ""
        for i in range(natoms):
            cg = self.atoms[i]
            atoms_entry += "  {:<6d}{:<6}{:<7d}{:<8}{:<6}{:<6d}\n".format(i+1,cg.name,1,mname,cg.name,i+1)

        if natoms == 1:
            bonds_entry = None
        else:
            bonds_entry = ""
            track = copy.deepcopy(self.edges)
            for i in range(natoms):
                bonded_to = track[i]
                for j in bonded_to:
                    atonm = 0.1
                    sigma = (self.atoms[i].sigma + self.atoms[j].sigma)/2 * atonm
                    # Add entry
                    bonds_entry += "  {:<5d}{:<5d}{:<6d}{:<10.4f}{:<10.1f}\n".format(i+1,j+1,1,sigma,kb)
                    # Remove duplicate from future entries
                    track[j].remove(i)

        text_out = write_molecule(template, longname, moltype, atoms_entry, bonds_entry)
        if write_file:
            if len(path) > 0: path = path + "/"
            filename = "{}.itp".format(longname)
            fout = open(path+filename, "w")
            fout.write(text_out)
            fout.close()
            return filename
        return text_out

def main():
    T = CG("T", 20.3, 0.0, 3.4, 300, 14.3, 6.0)
    M = CG("M", 18.2, 0.0, 3.2, 313, 15.2, 6.0)
    G = CG("G", 42.7510, 0.0, 4.41697863, 300.134, 14.40821934, 6.0)

    LJ = CG("LJ", 16.0, 0.0, 3.0, 300, 12.0, 6.0)
    ljmet = Molecule(LJ, name="LJM")
    CG.combining_eval(T,M,310)
    CG.combining_rval(T,M,15.0)
    eth = Molecule(T,M,M,T, name="eth")#M,M,T,M,
    tmh = Molecule(G,G,G, name="tmh")
    tmh.connect(0,2)
    # eth.connect(1,6)
    # eth.connect(4,6)
    # eth.disconnect(5,6)
    # print(eth.atoms)
    # print(eth.edges)
    # print(CG._total)
    # print(CG._ecross)
    # print(CG._rcross)
    # print(tmh.calc_positions())
    # print(tmh.write_grofile(write_file=True
    # ))
    # print(tmh.write_ffentry())
    #
    # print("="*30)
    # print(T.write_atype())
    # print(M.write_atype())
    # print(H.write_atype())
    # print(T.write_atype(other=H))

    sys1 = Project("LJ", cutoff=4.0)
    sys1.add_molecule(ljmet, 1000)
    print(sys1.molecules)
    print(sys1.atoms)

    sys1.set_topol()
    rhos = 0.85
    vol = 1000 * pow(0.3,3) / rhos
    boxlen = pow(vol, 1/3)
    sys1.prepare_grofiles(boxlen)
    # sys1.tv_run(300,path="t1", prod_length=10000000, prod='nvt')
    # sys1.std_run(300, 2, path="tp2")
    # sys1.std_run(250, 1, path="tp3")
    # sys1.std_run(285, 2, path="tp4")
    # sys1.std_run(293, 1, path="tp5")
    # sys1.std_run(314, 3, path="tp6", prod_length=20000000)
    sys1.std_run(332, 2, path="tp8", prod_length=20000000)
    sys1.prepare_mdp()
    #
    sys1.run()


    # sys1.procedure[0].make_tprfile(sys1.path+"/"+'topol.top', sys1.path+"/"+'index.ndx', capture=False)
    # sys1.procedure[0].run(capture=False, table=sys1.path+"/"+"table.xvg", v=None)

    # for p in sys1.procedure:
    #     print(p.grofile)
    # print(sys1.procedure[1].path)
    # print(MDProcedure("nve","nve1", test=100).options)
    # fout = open("table_test.xvg", "w")
    # fout.write((H.write_table()))
    # fout.close()
    return
# TODO: generate topol.top, table and mdp files
if __name__ == '__main__':
    main()
