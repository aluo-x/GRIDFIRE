def label_merge(aneigh):
    # credit goes to jimifiki from stackoverflow
    def findroot(anode, aroot):
        while anode != aroot[anode][0]:
            anode = aroot[anode][0]
        return anode, aroot[anode][1]

    myroot = {}
    for myNode in aneigh.keys():
        myroot[myNode] = (myNode, 0)
    for myI in aneigh:
        for myJ in aneigh[myI]:
            (myroot_myi, mydepthmyi) = findroot(myI, myroot)
            (myroot_myj, mydepthmyj) = findroot(myJ, myroot)
            if myroot_myi != myroot_myj:
                mymin = myroot_myi
                mymax = myroot_myj
                if mydepthmyi > mydepthmyj:
                    mymin = myroot_myj
                    mymax = myroot_myi
                myroot[mymax] = (mymax, max(myroot[mymin][1] + 1, myroot[mymax][1]))
                myroot[mymin] = (myroot[mymax][0], -1)
    mytoret = {}
    for myI in aneigh:
        if myroot[myI][0] == myI:
            mytoret[myI] = []
    for myI in aneigh:
        mytoret[findroot(myI, myroot)[0]].append(myI)
    return mytoret


def label_overlap(cluster_1, cluster_2):
    return not set(cluster_1).isdisjoint(cluster_2)
