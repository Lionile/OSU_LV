file = open("SMSSpamCollection.txt")

spamc = 0
spam_wordc = 0
spam_exclamation = 0
hamc = 0
ham_wordc = 0

for line in file:
    line = line.rstrip()
    if line.startswith("spam"):
        spamc += 1
        if line.endswith("!"):
            spam_exclamation += 1
        line = line[5:]
        for word in line.split():
            spam_wordc += 1
    elif line.startswith("ham"):
        hamc += 1
        line = line[4:]
        for word in line.split():
            ham_wordc += 1

print("Spam messages: " + str(spamc) + ", average words: " + str(spam_wordc/spamc))
print("Spam messages ending with exclamation: " + str(spam_exclamation))
print("Ham messages: " + str(hamc) + ", average words: " + str(ham_wordc/hamc))

file.close()