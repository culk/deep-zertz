def loc2coords(loc):
    '''
    Translates locations like "A3", "D5" to a coordinate tuple like (0, 1) and (3, 2)
    :param loc:
    :return:
    '''
    assert len(loc) == 2
    map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
    cap = min (map[loc[0]] + 4, 7)
    return (map[loc[0]], cap - int(loc[1]))

def test_loc2coords():
    assert loc2coords("A4") == (0, 0)
    assert loc2coords("C1") == (2, 5)
    assert loc2coords("G4") == (6, 3)
    print "loc2coords passes tests!"


if __name__ == '__main__':
    # test_loc2coords()
    loc1 = "D5"
    loc2 = "D3"
    loc3 = "C2"
    # action = (('PUT', 'w', loc2coords(loc1)), ('REM', loc2coords(loc2)))
    action = (('CAP', 'g', loc2coords(loc1)), ('w', loc2coords(loc2)))
    print action