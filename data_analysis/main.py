import mne
import seaborn as sns
import matplotlib.pyplot as plt

pth = f'raw_data/Pilot_2/S00_Vol.vhdr'
raw = mne.io.read_raw_brainvision(pth, preload=True)
raw.drop_channels(['VEOG', 'Resp'])
print(raw.ch_names)
# raw.set_eeg_reference('average', )
raw.filter(0, 1)
ch_name = 'Fp1'
ch_idx = raw.ch_names.index(ch_name)
# for ch_name in raw.ch_names:
#     ch_idx = raw.ch_names.index(ch_name)
plt.figure()
plt.plot(raw.times/60, raw._data[ch_idx, :]*1e6 - (raw._data[ch_idx, :]*1e6).mean())
plt.xlabel('Time [Minutes]')
plt.ylabel('Amplitude [Microvolts]')
plt.title(ch_name)
plt.show()


pth = f'raw_data/Pilot_2/S00_Exp.vhdr'
raw = mne.io.read_raw_brainvision(pth, preload=True)
raw.drop_channels(['VEOG', 'Resp'])
print(raw.ch_names)
raw.set_eeg_reference('average', )
raw.filter(0, 1)

ch_name = 'Cz'
ch_idx = raw.ch_names.index(ch_name)
plt.figure()
plt.plot(raw.times/60, raw._data[ch_idx, :]*1e6 - (raw._data[ch_idx, :]*1e6).mean())
plt.xlabel('Time [Minutes]')
plt.ylabel('Amplitude [Microvolts]')
plt.show()



pth = f'raw_data/Pilot_2/S00_DRMT.vhdr'
raw = mne.io.read_raw_brainvision(pth, preload=True)
raw.drop_channels(['VEOG', 'Resp'])
print(raw.ch_names)
raw.set_eeg_reference('average', )
raw.filter(0, 1)

ch_name = 'Cz'
ch_idx = raw.ch_names.index(ch_name)
plt.figure()
plt.plot(raw.times/60, raw._data[ch_idx, :]*1e6 - (raw._data[ch_idx, :]*1e6).mean())
plt.xlabel('Time [Minutes]')
plt.ylabel('Amplitude [Microvolts]')
plt.show()
