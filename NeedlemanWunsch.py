from collections import namedtuple

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
    subseq = []
    #s2 = []
    while array[row][col].score != 0:
        i, j = array[row][col].pointer
        if i+1 == row and j+1 == col:
            subseq.append(seq1[col-1])
            #s2.append(seq2[row-1])
            row, col = i, j
        elif row == i+1 and col == j:
            #s1.append("-")
            #s2.append(seq2[i])
            row, col = i, j
        elif row == i and col == j+1:
            #s1.append(seq1[j])
            #s2.append("-")
            row, col = i, j
    return subseq[::-1], array[row][col].score, array

def main(seq1, seq2):
    subseq, max_score, _ = NeedlemanWunsch(seq1, seq2)

    print("max score：", max_score)
    print("subsequence:", subseq)

if __name__ == '__main__':
    seq1 = "ATCGCGCAACTGCGCGC"
    seq2 = "ACGCGCACTGCGGC"
    main(seq1, seq2)