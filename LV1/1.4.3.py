nums = []

while(1):
    user_input = input()
    if(user_input == "Done"):
        break
    try:
        num = float(user_input)
        nums.append(num)
    except ValueError:
        print("Wrong input")
        continue

total = 0
for num in nums:
    total += num

print("Num count: " + str(len(nums)) + " average: " + str(total/len(nums)) + " min: " + str(min(nums)) + " max: " + str(max(nums)))

for num in sorted(nums):
    print(num)