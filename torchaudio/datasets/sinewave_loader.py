from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import shutil
import errno
import torch
import ipdb
import numpy as np
from tools import checkexists_mkdir, mkdir_in_path
import librosa

try:
    import torchaudio
    def read_audio(fp, sample_rate, downsample=True):
        if sample_rate != 44100:
        # if downsample:
            E = torchaudio.sox_effects.SoxEffectsChain()
            E.set_input_file(fp)
            E.append_effect_to_chain("gain", ["-h"])
            E.append_effect_to_chain("channels", [1])
            E.append_effect_to_chain("rate", [sample_rate])
            E.append_effect_to_chain("gain", ["-rh"])
            E.append_effect_to_chain("dither", ["-s"])
            sig, sr = E.sox_build_flow_effects()
            # ipdb.set_trace()
        else:
            sig, sr = torchaudio.load(fp)
        sig = sig.contiguous()

        return sig, sr

except Exception as e:
    print(e)
    print("Error loading torchaudio. Loading librosa instead")
    from librosa.core import load as read_audio





class SinewaveLoader(data.Dataset):
    """`YesNo Hebrew <http://www.openslr.org/1/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Scale``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        dev_mode(bool, optional): if true, clean up is not performed on downloaded
            files.  Useful to keep raw audio and transcriptions.
    """

    def __init__(
            self, root, sample_rate=16000, processed_file='sine-waves',
            transform=None, chunk_size=1, freqs= [100], n_examples=100, rand_ph=True, overwrite=False
        ):
        self.processed_file = "{0}_{1}.pt".format(processed_file, sample_rate)
        self.root = os.path.join(
            os.path.expanduser(root), processed_file + f'_sr{str(sample_rate)}_randph_{rand_ph}'
        )
        self.freqs = freqs
        self.rand_ph = rand_ph
        self.n_examples = n_examples
        self.transform = transform
        self.data = []
        self.num_samples = 0
        self.max_len = 0
        self.sample_rate=sample_rate
        self.chunk_size = chunk_size
        self.processed_folder = 'processed'
        self.raw_folder = 'raw'
        self.overwrite = overwrite
        checkexists_mkdir(self.root)
        mkdir_in_path(self.root, self.processed_folder)
        mkdir_in_path(self.root, self.raw_folder)
        self.dump_torch_file()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self.data = torch.load(os.path.join(
            self.root, self.processed_folder, self.processed_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        audio = self.data[index]

        if self.transform is not None:
            audio = self.transform(audio)

        return audio[:, :int(self.chunk_size * self.sample_rate)], 0


    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.processed_file))

    def _gen_sinewave(self, ampl=1, f=100, ph=0):
        return (ampl *np.sin(2*np.pi * np.arange(self.chunk_size*self.sample_rate)*f/self.sample_rate + ph)).astype(np.float32)


    def dump_torch_file(self):
        """Download the yesno data if it doesn't exist in processed_folder already."""

        if self._check_exists() and not self.overwrite:
            return

        raw_abs_dir = os.path.join(self.root, self.raw_folder)
        processed_abs_dir = os.path.join(self.root, self.processed_folder)

        sine_tensors = []
        print('Processing...')
        for i in range(self.n_examples):

            f = self.freqs[np.random.randint(len(self.freqs))]
            if self.rand_ph:
                ph = np.random.uniform(-np.pi, np.pi)
            else:
                ph=0

            filename = f'sinusoid_{str(i)}_f{str(f)}_sr{str(self.sample_rate)}.wav'
            sinewave = torch.FloatTensor(self._gen_sinewave(f=f, ph=ph))
            torchaudio.save(
                os.path.join(raw_abs_dir, filename),
                sinewave,
                self.sample_rate
            )
            sine_tensors.append(sinewave)

        output_processed_path = os.path.join(self.root, self.processed_folder)
        if not os.path.exists(output_processed_path): os.mkdir(output_processed_path)

        torch.save(
            sine_tensors,
            os.path.join(
                output_processed_path,
                self.processed_file
            )
        )
        print('Done!')

if __name__=='__main__':
    output_path = "~/Developer/sandbox"
    pianods = SinewaveLoader(output_path, sample_rate=2000, overwrite=True, rand_ph=True, freqs=[100, 200, 50])
