
def diff_csv(prev, after, actual, write):
    with open(prev, "r") as p_file, open(after, "r") as a_file, open(actual, "r") as t_file, open(write+"_OFF.csv", "w+") as w_off, open(write+"_NOT.csv", "w+") as w_not:
        for p_line, a_line, t_line in zip(p_file, a_file, t_file):
            p = p_line.split(",")
            a = a_line.split(",")
            t = t_line.split("\t")
            if p[0] != a[0] or p[0] != t[0]:
                raise Exception("mismatch in id")

            if p[1].strip('\n') != a[1].strip('\n') and a[1].strip('\n') == t[2].strip('\n'):
                if a[1].strip('\n') == "OFF":
                    w_off.write(t_line)
                else:
                    w_not.write(t_line)


def diff_csv_gold(pred, actual, write):
    with open(pred, "r") as p_file, open(actual, "r") as t_file, open(write+"_OFF.csv", "w+") as w_off, open(write+"_NOT.csv", "w+") as w_not:
        for p_line, t_line in zip(p_file, t_file):
            p = p_line.split(",")
            t = t_line.split("\t")
            if p[0] != t[0]:
                raise Exception("mismatch in id")

            if p[1].strip('\n') != t[1].strip('\n'):
                if p[1].strip('\n') == "OFF":
                    w_off.write(t_line)
                else:
                    w_not.write(t_line)


if __name__ == "__main__":

    base = "analysis/dev/"
    prev = "bert"
    after = "ensemble"
    actual = "data/dev.txt"
    write = base + "diff/"+prev + "_"+after +"_write"
    diff_csv(base + prev + ".csv", base + after + ".csv", actual, write)

    prev = "ensemble"
    write = base + "diff/"+prev + "_gold_data_" +"_write"
    diff_csv_gold(base + prev + ".csv", actual, write)
