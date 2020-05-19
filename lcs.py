from collections import namedtuple

'''
F = namedtuple('F', ('score', 'pointer'))

def NeedlemanWunsch(seq1, seq2):
    # init lcs array
    row, col = len(seq2) , len(seq1)
    array = [[0] * (col + 1) for _ in range(row + 1)]
    array[0][0] = F(0, None)
    for j in range(1, col + 1):
        array[0][j] = F((-5)*j, [0, j-1])
    for i in range(1, row + 1):
        array[i][0] = F((-5)*i, [i-1, 0])
    # compute lcs array
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if seq1[j-1] == seq2[i-1]:
                s = 10
            else:
                s = 5
            lu = [array[i-1][j-1].score+s, [i-1, j-1]]
            left = [array[i-1][j].score-5, [i-1, j]]
            up = [array[i][j-1].score-5, [i, j-1]]
            max_choice = max([lu, left, up], key=lambda x: x[0])
            score= max_choice[0]
            pointer = max_choice[1]
            array[i][j] = F(score, pointer)
    # backtrack longest subseq
    subseq1 = []
    subseq2 = []
    while array[row][col].score != 0:
        i, j = array[row][col].pointer
        if i+1 == row and j+1 == col:
            subseq1.append(seq1[col-1])
            subseq2.append(seq2[row-1])
            row, col = i, j
        elif row == i+1 and col == j:
            subseq1.append("-")
            subseq2.append(seq2[i])
            row, col = i, j
        elif row == i and col == j+1:
            subseq1.append(seq1[j])
            subseq2.append("-")
            row, col = i, j
    return subseq1[::-1], subseq2[::-1], array[row][col].score, array
'''

def LCS(seq1, seq2):
    len1 = len(seq1) + 1
    len2 = len(seq2) + 1
    lcs = [[["", 0] for j in list(range(len2))] for i in list(range(len1))]
    for i in list(range(1, len1)):
        lcs[i][0][0] = seq1[i - 1]
    for j in list(range(1, len2)):
        lcs[0][j][0] = seq2[j - 1]
    for i in list(range(1, len1)):
        for j in list(range(1, len2)):
            if seq1[i - 1] == seq2[j - 1]:
                lcs[i][j] = ['↖', lcs[i - 1][j - 1][1] + 1]
            elif lcs[i][j - 1][1] > lcs[i - 1][j][1]:
                lcs[i][j] = ['←', lcs[i][j - 1][1]]
            else:
                lcs[i][j] = ['↑', lcs[i - 1][j][1]]
    i = len1 - 1
    j = len2 - 1
    subseq = []
    while i > 0 and j > 0:
        if lcs[i][j][0] == '↖':
            subseq.append(lcs[i][0][0])
            i -= 1
            j -= 1
        if lcs[i][j][0] == '←':
            j -= 1
        if lcs[i][j][0] == '↑':
            i -= 1
    subseq.reverse()
    return subseq, lcs

def main(seq1, seq2):
    #subseq1, subseq2, max_score, _ = NeedlemanWunsch(seq1, seq2)

    #print("max score：", max_score)
    #print("subseq1:", subseq1)
    #print("subseq2:", subseq2)
    subseq, lcs = LCS(seq1, seq2)
    for i in range(len(lcs)):
        print(lcs[i])
    print(subseq)

if __name__ == '__main__':
    seq1 = [(7, 1), (11, 1), (32, 0), (33, 1), (37, 4), (52, 1), (56, 1), (63, 3)]
    seq2 = [(7, 0), (11, 3), (32, 0), (33, 5), (37, 2), (50, 4), (52, 1), (62, 3), (63, 2)]
    #seq1 = "GGGATCGA"
    #seq2 = "GAATTCAGTTA"

    main(seq1, seq2)