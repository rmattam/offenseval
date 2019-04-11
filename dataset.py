# helper scripts to clean up the dataset provided to me by the organizers of offenseval


def create_test():
    with open("data/testset-levela.txt", "r") as test, open("data/labels-levela.txt", "r") as label, open("data/test.txt", "w+") as file:
        for a , b in zip(test, label):
            tokens = b.split(",")
            test_tokens = a.split("\t")
            assert test_tokens[0] == tokens[0]
            file.write(a[:-1] + "\t" + tokens[1])


def create_dev():
    with open("data/dev_orig.txt", "r") as orig, open("data/dev.txt", "w+") as file:
        for i, line in enumerate(orig):
            file.write(str(i) + "\t" + line)


if __name__ == "__main__":
    create_dev()
