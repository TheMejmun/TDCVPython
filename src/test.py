import os
import data

if __name__ == '__main__':  # Only execute if called
    dirname = os.path.dirname
    ROOT_DIR = dirname(dirname(__file__))
    folder = ROOT_DIR + '/dataset/fine/cat/'

    data.load_folder(folder)

    # s = 'acbdjhbfuineifnoinfioqejfiojeqoirjnhoqiwueh216.png'
    # print(int(re.findall('[0-9]+', s)[0]))
