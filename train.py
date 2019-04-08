import reader
import bert

if __name__ == "__main__":
    train_data = "data/train.txt"
    for dp in reader.read_data(train_data):
        print(dp.id)
        print('\n')
        print(dp.text)
        print('\n')
        print(dp.category)
        print('\n\n')

    bert.Run()
