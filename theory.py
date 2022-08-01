import random

notes = ['C','c','D','d','E','F','f','G','g','A','a','B']
chords = {
  "M": [0,4,7],
  "M7": [0,4,7,10],
  "Maj7": [0,4,7,11],
  "M6": [0,4,7,9],
  "m": [0,3,7],
  "m7": [0,3,7,10],
  "mM7": [0,3,7,10],
  "mMaj7": [0,3,7,11],
  "dim": [0,3,6],
  "dim7": [0,3,6,9],
  "dimH7": [0,3,6,10],
  "aug": [0,4,8],
  "sus4": [0,5,7],
  "sus2": [0,2,7],
}

def generateChordProgression():
    key = random.choice(notes)
    prog = ''
    progression = []
    for i in range(4):
        #prog += str(random.randint(1,7))
        progression.append(getChord(key,(random.randint(1,7))))

    return progression, key

def getNextNote(note,steps):
    curNote = notes.index(note)
    if (curNote + steps >= 12):
        return notes[steps+curNote-12]
    else:
        return notes[curNote+steps]

def getChord(key, num):
    if (num == 1):
        return key+'M'
    if (num == 2):
        return getNextNote(key,2)+'m'
    if (num == 3):
        return getNextNote(key,4)+'m'
    if (num == 4):
        return getNextNote(key,5)+'M'
    if (num == 5):
        return getNextNote(key,7)+'M'
    if (num == 6):
        return getNextNote(key,9)+'m'
    if (num == 7):
        return getNextNote(key,11)+'dim'
