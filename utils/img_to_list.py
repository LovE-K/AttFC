import os


def generate(dir):
    basename = os.path.basename(dir)
    files = os.listdir(dir)
    files.sort()
    listText = open('all_list.txt', 'a')
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = "data" + basename + '\\' + file + '\n'
        listText.write(name)
    listText.close()


outer_path = 'the path of your datasets'  # path of datasets

if __name__ == '__main__':
    i = 0
    folderlist = os.listdir(outer_path)
    folderlist.sort(key=lambda x: int(x.split('.')[0]))
    for folder in folderlist:
        generate(os.path.join(outer_path, folder))
        i += 1
