f = open("C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\averageVector.txt")
g= open("C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\averageVector1.txt",'w')

for line in f:
    line = line.split(',')
    x=float(line)
    print(x)
    input()

  
