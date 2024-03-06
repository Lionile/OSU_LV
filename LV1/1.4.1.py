hours = float(input("Radni sati: "))
wage = float(input("eura/h: "))

def total_euro(hours, wage):
    return wage*hours

print("Ukupno: " + str(total_euro(hours, wage)))
