def dummy_results(read: str, write: str):
    with open(read, "r") as r, open(write, "w+") as w:
        for line in r:
            tokens = line.split('\t')
            output = tokens[0] + ',' "OFF\n"
            w.write(output)

    print("done")


if __name__ == "__main__":
    dummy_results('data/test.txt', 'submissions/test/task-a-submission.csv')
