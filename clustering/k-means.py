######################################################
#                CS 434 Assignment 4 
#                   Spike Madden
#                   Jacob Fenger
######################################################

def get_data(filename):
    file = open(filename, 'r')
    
    data = [] 

    for line in file:
        data.append(line)

    return data

def main():
    train_data = get_data('data.txt')

if __name__ == '__main__':
    main()
