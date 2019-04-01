from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import shutil
import errno
import torch
import ipdb

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





class YOUTUBE_PIANOS(data.Dataset):
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
    raw_folder = 'youtube_pianos/raw'
    processed_folder = 'youtube_pianos/processed'
    # url = 'http://www.openslr.org/resources/1/waves_yesno.tar.gz'
    # dset_path = 'waves_youtube_pianos'

    def __init__(
            self, root, sample_rate=16000, processed_file='youtube_pianos',
            transform=None, target_transform=None, dev_mode=False, chunk_size=16000
        ):
        self.processed_file = "{0}_{1}.pt".format(processed_file, sample_rate)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.data = []
        self.labels = []
        self.num_samples = 0
        self.max_len = 0
        self.sample_rate=sample_rate
        self.chunk_size = chunk_size

        self.save_as_torch_file()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self.data, self.labels = torch.load(os.path.join(
            self.root, self.processed_folder, self.processed_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        audio, target = self.data[index], self.labels[index]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio[:, :self.chunk_size], target


    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.processed_file))

    def save_as_torch_file(self):
        """Download the yesno data if it doesn't exist in processed_folder already."""
        import tarfile

        if self._check_exists():
            return

        raw_abs_dir = os.path.join(self.root, self.raw_folder)
        processed_abs_dir = os.path.join(self.root, self.processed_folder)
        dset_abs_path = os.path.join(
            self.root, self.raw_folder)


        # process and save as torch files
        print('Processing...')
        # shutil.copyfile(
        #     os.path.join(dset_abs_path, "README"),
        #     os.path.join(processed_abs_dir, "YESNO_README")
        # )
        audios = [x for x in os.listdir(dset_abs_path) if ".wav" in x]
        print("Found {} audio files".format(len(audios)))
        tensors = []
        labels = []
        lengths = []
        for i, f in enumerate(audios):
            print("Reading: {0}".format(f))
            full_path = os.path.join(dset_abs_path, f)

            sig, sr = read_audio(full_path, self.sample_rate)
            sig = sig.reshape((1, -1))

            sig = torch.FloatTensor(sig)

            tensors.append(sig)
            lengths.append(sig.size(1))

            labels.append(os.path.basename(f).split(".", 1)[0].split("_"))
        # sort sigs/labels: longest -> shortest
        tensors, labels = zip(*[(b, c) for (a, b, c) in sorted(
            zip(lengths, tensors, labels), key=lambda x: x[0], reverse=True)])
        self.max_len = tensors[0].size(1)

        output_processed_path = os.path.join(self.root, self.processed_folder)
        if not os.path.exists(output_processed_path): os.mkdir(output_processed_path)

        torch.save(
            (tensors, labels),
            os.path.join(
                output_processed_path,
                self.processed_file
            )
        )
        print('Done!')

if __name__=='__main__':
    youtube_pianos_path = "~/Developer/datasets/"
    pianods = YOUTUBE_PIANOS(youtube_pianos_path, sample_rate=16000)
