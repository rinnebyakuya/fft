from mingus.core import chords, notes
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import plotly.graph_objects as go
import numpy as np
import tqdm

# Configuration
FPS = 30
FFT_WINDOW_SECONDS = 0.25 # how many seconds of audio make up an FFT window

# Note range to display
FREQ_MIN = 10
FREQ_MAX = 1000

# Notes to take in
TOP_NOTES = 4
NOTE_MIN_STRENGTH = 0.25
NOTE_MIN_FREQUENCY = 20

# Names of the notes
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Output size. Generally use SCALE for higher res, unless we need a non-standard aspect ratio.
RESOLUTION = (1920, 1080)
# 0.5=QHD(960x540), 1=HD(1920x1080), 2=4K(3840x2160)
SCALE = 2                                                                               

# load the data
fs, data = wavfile.read('./audio/C_min.wav')                                            

audio = data.T[0]
try: 
  # if this isn't a stereo audio
  AUDIO_LENGTH = len(audio)/fs                                                          
except TypeError:
  audio = data
  AUDIO_LENGTH = len(audio)/fs
FRAME_STEP = (fs / FPS) # audio samples per video frame
FFT_WINDOW_SIZE = int(fs * FFT_WINDOW_SECONDS)


# Utility functions
def plot_fft(p, xf, fs, notes, dimensions=(960,540)):
  layout = go.Layout(
      title="frequency spectrum",
      autosize=False,
      width=dimensions[0],
      height=dimensions[1],
      xaxis_title="Frequency (note)",
      yaxis_title="Magnitude",
      font={'size' : 24}
  )

  fig = go.Figure(layout=layout,
                  layout_xaxis_range=[FREQ_MIN,FREQ_MAX],
                  layout_yaxis_range=[0,1]
                  )

  fig.add_trace(go.Scatter(
      x = xf,
      y = p))

  for note in notes:
    fig.add_annotation(x=note[0]+10, y=note[2],
            text=note[1],
            font = {'size' : 48},
            showarrow=False)
  return fig

def extract_sample(audio, frame_number):
  end = frame_number * FRAME_OFFSET
  begin = int(end - FFT_WINDOW_SIZE)

  if end == 0:
    # We have no audio yet, return all zeros (very beginning)
    return np.zeros((np.abs(begin)),dtype=float)
  elif begin<0:
    # We have some audio, padd with zeros
    return np.concatenate([np.zeros((np.abs(begin)),dtype=float),audio[0:end]])
  else:
    # Usually this happens, return the next sample
    return audio[begin:end]

def find_top_notes(fft,num):
  if np.max(fft.real)<0.001:
    return []

  lst = [x for x in enumerate(fft.real)]
  lst = sorted(lst, key=lambda x: x[1],reverse=True)

  idx = 0
  found = []
  found_note = set()
  while( (idx<len(lst)) and (len(found)<num) ):
    f = xf[lst[idx][0]]
    y = lst[idx][1]
    n = abs(freq_to_number(f))
    try:
      n0 = int(round(n))
    except OverflowError:
      n0 = 0

    name = note_name(n0)
    if (f == 0):
      scaled_y = 0
    else:
      scaled_y = y * 300/f
    if name not in found_note:
      found_note.add(name)
      s = [f,note_name(n0),scaled_y]
      found.append(s)
    idx += 1

  return found

def convert_to_invariant(array):

  note_names = [note[1].split(' ')[0] for note in array if note[2] > NOTE_MIN_STRENGTH]
  note_string = ' '.join(note_names)
  no_digits = []
  for i in note_string:
    if not i.isdigit():
        no_digits.append(i)
  result = ''.join(no_digits)

  letters = result.split()
  mapped_indices = [NOTE_NAMES.index(letter) for letter in letters]
  sorted_letters = [letter for _, letter in sorted(zip(mapped_indices, letters))]
  sorted_string = ' '.join(sorted_letters)

  return sorted_string.strip()

def convert_to_flats(notes):
    # Create a dictionary with sharp notes as keys and flat equivalents as values
    sharp_to_flat = {
        'C#': 'Db',
        'D#': 'Eb',
        'F#': 'Gb',
        'G#': 'Ab',
        'A#': 'Bb'
    }

    # Split the input string into an array of notes
    note_list = notes.split()

    # Iterate through the array and replace any sharp notes with flat equivalents
    for i in range(len(note_list)):
        if note_list[i] in sharp_to_flat:
            note_list[i] = sharp_to_flat[note_list[i]]

    # Join the array back into a string and return
    return ' '.join(note_list)

# See https://newt.phys.unsw.edu.au/jw/notes.html
def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
def note_name(n): return NOTE_NAMES[n % 12] + str(int(n/12))

# Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, FFT_WINDOW_SIZE, False)))

xf = np.fft.rfftfreq(FFT_WINDOW_SIZE, 1/fs)
FRAME_COUNT = int(AUDIO_LENGTH*FPS)
FRAME_OFFSET = int(len(audio)/FRAME_COUNT)

# Pass 1, find out the maximum amplitude so we can scale.
mx = 0
for frame_number in range(FRAME_COUNT):
  sample = extract_sample(audio, frame_number)

  fft = np.fft.rfft(sample * window)
  fft = np.abs(fft).real
  mx = max(np.max(fft),mx)

print(f"Max amplitude: {mx}")

# Pass 2, produce the result
for frame_number in tqdm.tqdm(range(FRAME_COUNT)):
  sample = extract_sample(audio, frame_number)

  fft = np.fft.rfft(sample * window)
  fft = np.abs(fft) / mx

  s = find_top_notes(fft,TOP_NOTES)
  x = convert_to_invariant(s)
  
  print(x)
  chord_name = chords.determine(x.split(),  shorthand=True)

  # Chech for enharmonic notes as flats
  if (chord_name == []):
    flat_chord = convert_to_flats(x)
    chord_name = chords.determine(flat_chord.split(),  shorthand=True)
  
  if(chord_name == []):
    print('Couldn\t identify chord')
  else:
    print(chord_name)