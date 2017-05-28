INPUT_DATA = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/database.csv"
INPUT_LABEL = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/database_new_label.csv"
CLUSTER_OUT_PATH = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/cluster_result_raw"
CLUSTERS = 7

files = []

with open(INPUT_LABEL, 'r') as f:
    line = f.readline()
    labelset = [int(x) for x in line.split(', ')]
for n in range(0, CLUSTERS):
	print("Generate file handler according to cluster size.")
	fname = CLUSTER_OUT_PATH + "/cluster_" + str(n) + ".csv"
	file_handler = open(fname, 'w+')
	files.append(file_handler)
cluster_out_file = CLUSTER_OUT_PATH + "all_raw.csv"
with open(cluster_out_file, 'w+') as out_f:
	with open(INPUT_DATA, 'r') as in_f:
		for i,line in enumerate(in_f):
			new_line = (str(line).rstrip() + ";" + str(labelset[i]) + "\n")
			out_f.write(new_line)
			files[labelset[i]].write(new_line)
			if (i % 10000) == 0:
				print("Procesed %d lines." % (i+1))
for n in range(0, CLUSTERS):
	files[n].close()