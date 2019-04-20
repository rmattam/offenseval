
def diff_csv(prev, after, actual, write):
    with open(prev, "r") as p_file, open(after, "r") as a_file, open(actual, "r") as t_file, open(write, "w+") as w:
        for p_line, a_line, t_line in zip(p_file, a_file, t_file):
            p = p_line.split(",")
            a = a_line.split(",")
            t = t_line.split("\t")
            if p[0] != a[0] and p[0] != t[0]:
                raise Exception("mismatch in id")

            if p[1] != a[1] and a[1] == t[2]:
                w.write(t_line)


if __name__ == "__main__":

    base = "analysis/test/"
    prev = "bert"
    after = "ensemble"
    actual = "data/test.txt"
    write = base + "diff/"+prev + "_"+after +"_write.csv"
    diff_csv(base + prev + ".csv", base + after + ".csv", actual, write)
