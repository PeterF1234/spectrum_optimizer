#!/usr/bin/env python3

import os

# These strings will be searched in the filenames
# molecule = ["NCE_1","NCE_2","NCE_3","NCE_4","PTZ_1","PTZ_2","PTZ_3",
#             "PA_1","PA_2","PA_3","PDI_1","PDI_2","PDI_3","POZ_1","POZ_2","POZ_3","POZ_4","POZ_5",
#             "BOH-Acr_m","BOH-Acr_o","BOH-Acr_p","BF3-Acr_m","BF3-Acr_o","BF3-Acr_p",
#             "Ph-Acr_1","Ph-Acr_2","Me2-Acr_1","Me2-Acr_2","Me2-Acr_3","Mes-Acr_1",
#             "CA_1","CA_2","CA_3","Eos_1","Eos_2","Eos_3","Eos_Y","Rh_6G","Rh_B"]

# molecule = ["Au-alkyne_1", "Au-alkyne_2", "Au-alkyne_3", "Cu50_batho2", "Cu50_BINAP_batho", "Cu50_BINAP_dmbp",
#             "Cu50_BINAP_dmp", "Cu50_BINAP_dq", "Cu50_BINAP_dtbbp", "Cu50_BINAP_iquintri", "Cu50_BINAP_phen",
#             "Cu50_BINAP_pytri", "Cu50_BINAP_quintri", "Cu50_BINAP_tmp", "Cu50_dmp2", "Cu50_DPEPhos-batho",
#             "Cu50_DPEPhos-bmbp", "Cu50_DPEPhos-btbbp", "Cu50_DPEPhos-dmp", "Cu50_DPEPhos-dq",
#             "Cu50_DPEPhos-iquintri", "Cu50_DPEPhos-phen", "Cu50_DPEPhos-pytri", "Cu50_DPEPhos-quintri_b",
#             "Cu50_DPEPhos-quintri", "Cu50_DPEPhos-tmp", "Cu50_dppf-batho", "Cu50_dppf-dmbp", "Cu50_dppf-dmp",
#             "Cu50_dppf-dq", "Cu50_dppf-dtbbp", "Cu50_dppf-iquintri", "Cu50_dppf-phen", "Cu50_dppf-pytri",
#             "Cu50_dppf-tmp", "Cu50_dq2", "Cu50_phen2", "Cu50_tmp2", "Cu50_XantPhos-batho", "Cu50_XantPhos-dmbp",
#             "Cu50_XantPhos-dmp", "Cu50_XantPhos-dq", "Cu50_XantPhos-dtbbp", "Cu50_XantPhos-iquintri",
#             "Cu50_XantPhos-phen", "Cu50_XantPhos-pytri", "Cu50_XantPhos-quintri_b", "Cu50_XantPhos-quintri",
#             "Cu50_XantPhos-tmp", "Cu-phen_1", "Cu-phen_2", "Cu-xant-dmp", "Cu-xant-dppz", "Cu-xant-dpq",
#             "FeTP_highspin", "FeTP", "Ir-ppy-bpy_1", "Ir-ppy-bpy_2", "Ir-ppy", "Mo-CN_1", "Ni_co-cat_1",
#             "Ni_co-cat_2", "Rh_dimer_A", "Rh_dimer_B", "Rh_dimer_B_UHF-guess", "RhInd_1", "Ru-arylbpy_1",
#             "Ru-arylbpy_2", "Ru-arylbpy_3", "Ru-bpm", "Ru-bpy-dppz", "Ru-bpy-dpq", "Ru-bpy", "Ru-bpy-phen",
#             "Ru-bpyrz-bpy_2", "Ru-bpyrz-bpy_3", "Ru-bpz", "Ru-CN_1", "Ru-CN_2", "Ru-CN_3", "Ru-CN_4", "Ru-NHC_1",
#             "Ru-NHC_2", "Ru-NHC_3", "Ru-NHC_4", "Ru-NHC_5", "Ru-phen", "Ru-qd_1", "Ru-qd_2", "Ru-qd_3",
#             "Ru-qd_4", "W-CN_1", "W-CN_2", "Cu50_DPEPhos-dtbbp", "Cu50_DPEPhos-dmbp"]

molecule = ['FeC_1', 'FeC_2', 'FeC_3', 'FeC_4', 'FeC_5', 'FeC_6', 'FeC_7', 'FeTP', 
            'FeII_carbene_C1', 'FeII_carbene_C2', 'FeII_carbene_C3', 'FeII_carbene_C4', 
            'FeII_CNC_1', 'FeII_CNC_2', 'Fe_bpy', 'Fe_dcpp', 'Fe_tpy', 'FeII_HMTI', 
            'Mo0_carbonyl_1', 'Mo0_carbonyl_2', 'Mo-CN_1', 'Mo-CN_2', 'Au-alkyne_1', 'Au-alkyne_2', 'Au-alkyne_3', 
            'AuBCz', 'CuBCz', 'CuPhCz', 'Cu50_phen2', 'Cu50_dq2', 'Cu50_tmp2', 'Cu50_batho2', 
            'Cu50_dmp2', 'Cu50_BINAP_quintri', 'Cu50_BINAP_iquintri', 'Cu50_BINAP_tmp', 'Cu50_BINAP_dq', 
            'Cu50_BINAP_dmp', 'Cu50_BINAP_dtbbp', 'Cu50_BINAP_batho', 'Cu50_BINAP_phen', 'Cu50_BINAP_dmbp', 
            'Cu50_BINAP_pytri', 'Cu50_DPEPhos-quintri', 'Cu50_DPEPhos-iquintri', 'Cu50_DPEPhos-tmp', 
            'Cu50_DPEPhos-dq', 'Cu50_DPEPhos-dmp', 'Cu50_DPEPhos-dtbbp', 'Cu50_DPEPhos-batho', 
            'Cu50_DPEPhos-phen', 'Cu50_DPEPhos-dmbp', 'Cu50_DPEPhos-pytri', 'Cu50_XantPhos-quintri', 
            'Cu50_XantPhos-iquintri', 'Cu50_XantPhos-tmp', 'Cu50_XantPhos-dq', 'Cu50_XantPhos-dmp', 
            'Cu50_XantPhos-dtbbp', 'Cu50_XantPhos-batho', 'Cu50_XantPhos-phen', 'Cu50_XantPhos-dmbp', 
            'Cu50_XantPhos-pytri', 'Cu50_dppf-quintri', 'Cu50_dppf-iquintri', 'Cu50_dppf-tmp', 'Cu50_dppf-dq', 
            'Cu50_dppf-dmp', 'Cu50_dppf-dtbbp', 'Cu50_dppf-batho', 'Cu50_dppf-phen', 'Cu50_dppf-dmbp', 
            'Cu50_dppf-pytri', 'Cu-phen_1', 'Cu-phen_2', 'Cu-phen_3', 'Cu-xant-dppz', 'Cu-xant-dpq', 
            'Cu-xant-dmp', 'Ru-bpy', 'Ru-phen', 'Ru-bpz', 'Ru-bpm', 'Ru-bpy-dppz', 'Ru-bpy-dpq', 'Ru-bpy-phen', 
            'Ru-CN_1', 'Ru-CN_2', 'Ru-CN_3', 'Ru-CN_4', 'Ru-bpz', 'Ru-bpz-bpy_21', 'Ru-bpz-bpy_12', 
            'Ru-arylbpy_1', 'Ru-arylbpy_2', 'Ru-arylbpy_3', 'Ru-qd_1', 'Ru-qd_2', 'Ru-qd_3', 'Ru-qd_4', 
            'Ru-NHC_1', 'Ru-NHC_2', 'Ru-NHC_3', 'Ru-NHC_4', 'Ru-NHC_5', 'Ru-bpy-NN_2', 'Ru-bpy-NN_3', 
            'Ru-bpy-NN_4', 'Ru-bpy-NN_5', 'Ru-bpy-NN_6', 'Ru-PRC_1', 'Ru-PRC_2', 'Ru-PRC_3',
            'Ir-ppy-bpy_1', 'Ir-ppy-bpy_2', 
            'Ni_co-cat_1', 'Ir-ppy', 'Ni_co-cat_2', 'Ir-ho_1', 'Ir-ho_2', 'Ir-ho_3', 'Ir-ho_4', 'Ir-ho_5', 
            'Ir-ho_6', 'Ir-he_1', 'Ir-he_2', 'Ir-he_3', 'Ir-he_4', 'Ir-he_5', 'Ir-he_6', 'IrIII_encaps_5', 
            'IrIII_encaps_6', 'Ir_CNC_3', 'Ir_CNC_4', 'Ir_CNC_6', 'Ir_isoCN_1', 'Ir_isoCN_2', 'Ir_isoCN_3', 
            'Ir_isoCN_4', 'Ir_isoCN_5', 'Ir_isoCN_6', 'Ir_isoCN_7', 'Ir_isoCN-acac_1', 'Ir_isoCN-acac_2', 
            'Ir_isoCN-acac_3', 'W-CN_1', 'W-CN_2', 'W-CN_3']

def build_filedata(files,outfile,dumpfile,header,workdir='.'):

    functionals = ["DSD-BLYP","_B2PLYP","wB2PLYP","STEOM-DLPNO-CCSD",
                   "_B3LYP-D3","CAM-B3LYP-D3","_M06_","M062X","_M06L",
                   "_PBE-D3","PBE0","_wB97XD","TPSS","B97D3","TDHF",
                   "B2GPPLYP","SCS-PBE-QIDH","_wPBEPP86","SOS-wPBEPP86",
                   "SOC-B3LYP","SOC-PBE","SOC-M06L","SOC-M06_","SOC-wB97XD"]
    basis = ["_def2TZVP","_TZVP","_def2SVP","_DEF2-TZVP"]
    # Build the database
    print("Building file data...")
    G = {}
    for i in files:
        # extract info from the filenames
        fname_matches = []
        for m in molecule:
            if m in i:
                fname_matches.append(m)
        fname_matches.sort()
        print(f"fname_matches = {fname_matches}")
        G[i] = [fname_matches[-1]]
        try:
            len(G[i]) == 1 # can also test proper conditions with an if statement
        except KeyError:
            print("Warning: molecule identification was not found in " + i)
        for f in functionals:
            if f in i: # change to proper functional name
                if f == "_wB97XD":
                    G[i].append("ωB97X-D")
                    # G[i].append("$\\omega$B97X-D")
                elif f == "SOC-wB97XD":
                    G[i].append("SOC-ωB97X-D")
                    # G[i].append("SOC-$\\omega$B97X-D")
                elif f == "wB2PLYP":
                    G[i].append("ωB2PLYP")
                    # G[i].append("$\\omega$B2PLYP")
                elif f == "_B2PLYP": # avoid conflict with wB2PLYP
                    G[i].append("B2PLYP")
                elif f == "_B3LYP-D3": # avoid conflict with CAM-B3LYP
                    G[i].append("B3LYP-D3")
                elif f == "_M06_": # avoid conflict with M062X and M06L
                    G[i].append("M06")
                elif f == "SOC-M06_": # avoid conflict with SOC-M06L
                    G[i].append("SOC-M06")
                elif f == "_M06L": # avoid conflict with SOC-M06L
                    G[i].append("M06-L")
                elif f == "M062X":
                    G[i].append("M06-2X")
                elif f == "_PBE-D3": # avoid conflict with SOC-PBE
                    G[i].append("PBE-D3")
                elif f == "B97D3":
                    G[i].append("B97-D3")
                elif f == "B2GPPLYP":
                    G[i].append("B2GP-PLYP")
                elif f == "_wPBEPP86":
                    G[i].append("ωPBEPP86")
                    # G[i].append("$\\omega$PBEPP86")
                elif f == "SOS-wPBEPP86":
                    G[i].append("SOS-ωPBEPP86")
                    # G[i].append("SOS-$\\omega$PBEPP86")
                else:
                    G[i].append(f)
        if len(G[i]) != 2:
            print("Warning: functional was not found in " + i)
        for b in basis:
            if b in i:
                if b == "_def2TZVP" or b == "_DEF2-TZVP":
                    G[i].append("def2-TZVP")
                if b == "_TZVP":
                    G[i].append("TZVP")
                if b == "_def2SVP":
                    G[i].append("def2-SVP")
        if len(G[i]) != 3:
            print("Warning: basis was not found in " + i)
            print("Placing n/a in the database...")
            G[i].append("n/a")
        # extract the optimized parameters (error, bandwidth, shift factor, etc.)
        opt_params = os.path.join(workdir,dumpfile)
        with open(opt_params) as inp:
            for line in inp:
                line = line.strip()
                dat = line.split(',')
                if dat[0] == i:
                    G[i].append(dat[1:])
    print("Building file data done!")
    # G[i] will have this ordering: [molecule, functional, basis, error_function, error_value, bandwidth, shift_factor]
    # default header looks like this: "reference,molecule,functional,basis,error_function,error_value,bandwidth,shift_factor,Wavelength [nm],Intensity [a.u.]\n"
    k = open(outfile, "w")
    k.write(header)
    print("Merging the following files:")
    for filename, data in G.items(): # i = 0,1,2,3
        filename = os.path.join(workdir,filename)
        with open(filename) as inp:
            print(filename)
            for line in inp:
                newline = "no" + "," + data[0] + "," + data[1] + "," + data[2] + "," + data[3][0] + "," + data[3][1] + "," + data[3][2] + "," + data[3][3] + "," + line
                k.write(newline)
    print("Merge complete!\n")
    k.close()

def merge_db(databases,merged_name="database_merged.csv"):
    k = open(merged_name, "w")
    header = ""
    for db in databases:
        firstline = True
        with open(db) as inp:
            for line in inp:
                if firstline == True and header == "":
                    header = line
                    firstline = False
                    k.write(header)
                    continue
                elif firstline == True and line == header:
                    firstline = False
                    continue
                if firstline == True and line != header:
                    raise ValueError("The first lines (headers) in the input databases do not seem to match!")
                elif firstline == False:
                    k.write(line)
    k.close()

def add_reference(ref_file,dbfile):
    sortedmols = sorted(molecule,reverse=True,key=len) # inefficient to sort every time
    molname = "Molecule name was not found in " + ref_file
    for m in sortedmols:
        if m in ref_file:
            molname = m
            break
    db = open(dbfile, "a")
    with open(ref_file, "r") as ref:
        for line in ref:
            newline = "yes" + "," + molname + "," + "expt." + "," + "expt." + "," + "-" + "," + "-" + "," + "-" + "," + "-" + "," + line
            db.write(newline)
    db.close()


