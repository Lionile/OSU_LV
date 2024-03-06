try:
    grade = float(input("Grade: "))
except ValueError:
    print("Not a number")
    exit()

if(grade > 1 or grade < 0):
    print("Enter grade between 0 and 1")
    exit()

if(grade >= 0.9):
    print("A")
elif(grade >= 0.8):
    print("B")
elif(grade >= 0.7):
    print("C")
elif(grade >= 0.6):
    print("D")
else:
    print("F")