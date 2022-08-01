import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
import utils

# Get frequency of notes
note_freqs = utils.get_piano_notes()
C4 = utils.get_sine_wave(note_freqs['C4'], 2, amplitude=2048)  # Middle C
E4 = utils.get_sine_wave(note_freqs['E4'], 2, amplitude=2048)  # C one octave above
B4 = utils.get_sine_wave(note_freqs['B4'], 2, amplitude=2048)  # C one octave above

# Plot
plt.figure(figsize=(12,4))
plt.plot(C4[:2500], label='C4')
plt.plot(E4[:2500], label='E4')
plt.plot(B4[:2500], label='B4')
plt.plot((C4+E4+B4)[:2500], label='Octave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Perfect Consonance (Octave)')
plt.grid()
plt.legend()
plt.show()