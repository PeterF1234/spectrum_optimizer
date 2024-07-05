#!/usr/bin/env python3

import os
import time
import add_labels_optimized_defonly as labels

def merge_fulldb(db_names="database_merged.csv",pth='.'):
    molecules = labels.molecule
    current_time_seconds = int(time.time())
    fulldb_name = "TMPC_database_" + str(current_time_seconds) + ".csv"
    folders = next(os.walk(pth))[1]
    k = open(fulldb_name, "w")
    header = ""
    for mol in molecules:
        for i in folders:
            if i == mol:
                db_fullpath = os.path.join(i,db_names) # all databases should have the same name
                firstline = True
                with open(db_fullpath) as inp:
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
                break
    k.close()
    print("Database has been built!")

def main():
    merge_fulldb(db_names="database_merged.csv",pth='.')

if __name__ == "__main__":
    main()
