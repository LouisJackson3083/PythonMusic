import numpy as np
from scipy.io import wavfile
import utils

# leadvoice_notes = ['C4', 'C4', 'G4', 'G4',
#                    'A4', 'A4', 'G4',
#                    'F4', 'F4', 'E4', 'E4',
#                    'D4', 'D4', 'C4',
#                    'G4', 'G4', 'F4', 'F4',
#                    'E4', 'E4', 'D4',
#                    'G4', 'G4', 'F4', 'F4',
#                    'E4', 'E4', 'D4',
#                    'C4', 'C4', 'G4', 'G4',
#                    'A4', 'A4', 'G4',
#                    'F4', 'F4', 'E4', 'E4',
#                    'D4', 'D4', 'C4',]
# leadvoice_duration = [0.5, 0.5, 0.5, 0.5,
#                        0.5, 0.5, 1]*6

voice1_notes = ['C3']
voice1_duration = [2]

voice2_notes = ['G3']
voice2_duration = [2]

voice3_notes = ['C4']
voice3_duration = [2]

voice4_notes = ['E4']
voice4_duration = [2]

# # Lead voice
# factor = [0.68, 0.26, 0.03, 0.  , 0.03]
# length = [0.01, 0.6, 0.29, 0.1]
# decay = [0.05,0.02,0.005,0.1]
# sustain_level = 0.1
# leadvoice = utils.get_song_data(leadvoice_notes, leadvoice_duration, 2,
#                                  factor, length, decay, sustain_level)

# Chord voice
factor = [0.73, 0.16, 0.06, 0.01, 0.02, 0.01  , 0.01]
length = [0.01, 0.29, 0.6, 0.1]
decay = [0.05,0.02,0.005,0.1]
sustain_level = 0.1
voice1 = utils.get_song_data(voice1_notes, voice1_duration, 2,
                                 factor, length, decay, sustain_level)
voice2 = utils.get_song_data(voice2_notes, voice2_duration, 2,
                                 factor, length, decay, sustain_level)
data = voice1+voice2
data = data * (4096/np.max(data))
wavfile.write('data/song_1.wav', 44100, data.astype(np.int16))