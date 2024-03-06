file = open("song.txt")
words = {}

for line in file:
    line = line.rstrip()
    for word in line.split():
        if word not in words:
            words[word] = 0
        else:
            words[word] += 1


count = 0
for word in words:
    if words[word] == 1:
        count += 1

print(count)

file.close()