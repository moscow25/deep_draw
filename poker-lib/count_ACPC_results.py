import sys
import csv
import re

"""
Author: Nikolai Yakovenko

Adds up SCORE: xxx lines from merged ACPC log files
"""

if len(sys.argv) >= 2:
	input_file = sys.argv[1]
else:
	print('Usage: python %s input_file' % 'count_ACPC_results.py')
        sys.exit(0)

filename = input_file
reader = open(filename, 'rU')
agent_sums = {}

# Look for SCORE summaries of form:
# SCORE:-99605|99605:tartanian6|slumbot
#acpc_line_regex = re.compile(r'(?:STATE)?:?(\d+):([^:]*):([^|]*)\|([^|/]*)([^:]*):(-?\d+)[:\|](-?\d+):([^|]*)\|([^|]*)')
# (score_1, score_2, name_1, name_2)
acpc_result_regexp = re.compile(r'SCORE:(-?\d+)[:\|](-?\d+):([^|]*)\|([^|]*)')
for line in reader:
	score_match = acpc_result_regexp.match(line.strip())
	if not score_match:
		continue
	else:
		print(line.strip())
		(score_1, score_2, name_1, name_2) = score_match.groups()
		if not(name_1 in agent_sums.keys()):
			agent_sums[name_1] = 0.0
		if not(name_2 in agent_sums.keys()):
			agent_sums[name_2] = 0.0
		agent_sums[name_1] += int(score_1)
		agent_sums[name_2] += int(score_2)
print(agent_sums)

