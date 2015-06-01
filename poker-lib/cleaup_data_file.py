import sys
#import csv


"""
Author: Nikolai Yakovenko

From here, simple utility to clean up messy CSV file, or other input file. Just removes the corrupted data.
"""

if len(sys.argv) >= 3:
        input_file = sys.argv[1]
	output_file = sys.argv[2]
else:
	print('Usage: python %s input_file output_file' % 'cleanup_data_file')
        sys.exit(0)

fi = open(input_file, 'rb')
data = fi.read()
fi.close()
fo = open(output_file, 'wb')
fo.write(data.replace('\x00', ''))
fo.close()
