import sys
import os

input_file = sys.argv[1]
cand_file = sys.argv[2]
cnt_file = sys.argv[3]
output_file = sys.argv[4]

fi = open(input_file, 'r')
ficand = open(cand_file, 'r')
fic = open(cnt_file, 'r')
fo = open(output_file, 'w')

cand2score = {}

for str_cnt in fic:
	cnt = int(str_cnt)
	cand = ficand.readline().strip()
	sum_score = 0
	for i in range(cnt):
		line = fi.readline()
		lst = line.strip().split()
		score = float(lst[1])
		sum_score += score
	if cnt == 0:
		cand2score[cand] = 0
	else:
		cand2score[cand] = sum_score / cnt

cand2score = sorted(cand2score.items(), key = lambda x:x[1], reverse = True)

for tp in cand2score:
	fo.write(tp[0] + '\t' + str(tp[1]) + '\n')

fi.close()
ficand.close()
fic.close()
fo.close()
		
