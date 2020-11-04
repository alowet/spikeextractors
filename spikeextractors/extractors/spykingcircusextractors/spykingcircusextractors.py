from spikeextractors import SortingExtractor
from spikeextractors.extractors.numpyextractors import NumpyRecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id

try:
    import h5py
    import scipy
    HAVE_SCSX = True
except ImportError:
    HAVE_SCSX = False


class SpykingCircusRecordingExtractor(NumpyRecordingExtractor):
    extractor_name = 'SpykingCircusRecordingExtractor'
    has_default_locations = False
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path):
        spykingcircus_folder = Path(folder_path)
        listfiles = spykingcircus_folder.iterdir()
        sample_rate = None
        recording_file = None

        parent_folder = None
        result_folder = None
        for f in listfiles:
            if f.is_dir():
                if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                    parent_folder = spykingcircus_folder
                    result_folder = f

        if parent_folder is None:
            parent_folder = spykingcircus_folder.parent
            for f in parent_folder.iterdir():
                if f.is_dir():
                    if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                        result_folder = spykingcircus_folder

        assert isinstance(parent_folder, Path) and isinstance(result_folder, Path), "Not a valid spyking circus folder"

        for f in parent_folder.iterdir():
            if f.suffix == '.params':
                sample_rate = _load_sample_rate(f)
            if f.suffix == '.npy':
                recording_file = str(f)
        NumpyRecordingExtractor.__init__(self, recording_file, sample_rate)
        self._kwargs = {'folder_path': str(Path(folder_path).absolute())}


class SpykingCircusSortingExtractor(SortingExtractor):
    extractor_name = 'SpykingCircusSortingExtractor'
    installed = HAVE_SCSX  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = "To use the SpykingCircusSortingExtractor install h5py and scipy: " \
                        "\n\n pip install h5py scipy\n\n"

    def __init__(self, file_or_folder_path, load_templates=True):
        assert HAVE_SCSX, self.installation_mesg
        SortingExtractor.__init__(self)
        file_or_folder_path = Path(file_or_folder_path)

        results = None
        sample_rate = None

        parent_folder = None
        result_folder = None

        if file_or_folder_path.is_dir():
            spykingcircus_folder = file_or_folder_path
            listfiles = spykingcircus_folder.iterdir()

            for f in listfiles:
                if f.is_dir():
                    if any([f_.suffix in ['.hdf5', '.h5'] for f_ in f.iterdir()]):
                        parent_folder = spykingcircus_folder
                        result_folder = f

            if parent_folder is None:
                parent_folder = spykingcircus_folder.parent
                for f in parent_folder.iterdir():
                    if f.is_dir():
                        if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                            result_folder = spykingcircus_folder

            assert isinstance(parent_folder, Path) and isinstance(result_folder, Path), \
                "Not a valid spyking circus folder"

            # load files
            for f in result_folder.iterdir():
                if 'result.hdf5' in str(f):
                    results = f
                if 'result-merged.hdf5' in str(f):
                    results = f
                    break
        else:
            results = file_or_folder_path
            assert results.suffix in ['.hdf5', '.h5'], "file path should be an hdf5 file"

            result_folder = results.parent
            parent_folder = result_folder.parent

        # load params
        for f in parent_folder.iterdir():
            if f.suffix == '.params':
                sample_rate = _load_sample_rate(f)

        if sample_rate is not None:
            self._sampling_frequency = sample_rate

        if results is None:
            raise Exception(f"{file_or_folder_path} is not a spyking circus file or folder")

        f_results = h5py.File(results, 'r')
        self._spiketrains = []
        self._unit_ids = []
        for temp in f_results['spiketimes'].keys():
            self._spiketrains.append(np.array(f_results['spiketimes'][temp]).astype('int64'))
            self._unit_ids.append(int(temp.split('_')[-1]))

        if load_templates:
            data_name = str(results.name).split('.')[0]
            results_name = str(results.name).split('.')[-2]
            results_start_idx = results_name.find('result')
            results_name = results_name[results_start_idx + len('result'):]
            template_file = result_folder / f"{data_name}.templates{results_name}.hdf5"

            if template_file.is_file():
                with h5py.File(template_file, 'r', libver='earliest') as myfile:
                    temp_x = myfile.get('temp_x')[:].ravel()
                    temp_y = myfile.get('temp_y')[:].ravel()
                    temp_data = myfile.get('temp_data')[:].ravel()
                    N_e, N_t, nb_templates = myfile.get('temp_shape')[:].ravel().astype(np.int32)
                templates = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e * N_t, nb_templates))
                templates_reshaped = np.array([templates[:, i].toarray().reshape(N_e, N_t)
                                               for i in np.arange(templates.shape[-1])])
                assert len(templates_reshaped) // 2 == len(self._unit_ids), "Number of templates does not correspond to " \
                                                                            "number of units and could not be loaded."
                for i_u, u in enumerate(self.get_unit_ids()):
                    self.set_unit_property(u, 'template', templates_reshaped[i_u])
            else:
                print("Couldn't find templates file")

        self._kwargs = {'file_or_folder_path': str(Path(file_or_folder_path).absolute()),
                        'load_templates': load_templates}

    def get_unit_ids(self):
        return list(self._unit_ids)

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spiketrains[self.get_unit_ids().index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    @staticmethod
    def write_sorting(sorting, save_path):
        assert HAVE_SCSX, SpykingCircusSortingExtractor.installation_mesg
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / 'data.result.hdf5'
        elif save_path.suffix == '.hdf5':
            if not str(save_path).endswith('result.hdf5') or not str(save_path).endswith('result-merged.hdf5'):
                raise AttributeError("'save_path' is either a folder or an hdf5 file "
                                     "ending with 'result.hdf5' or 'result-merged.hdf5")
        else:
            save_path.mkdir()
            save_path = save_path / 'data.result.hdf5'
        F = h5py.File(save_path, 'w')
        spiketimes = F.create_group('spiketimes')

        for id in sorting.get_unit_ids():
            spiketimes.create_dataset('tmp_' + str(id), data=sorting.get_unit_spike_train(id))


def _load_sample_rate(params_file):
    sample_rate = None
    with params_file.open('r') as f:
        for r in f.readlines():
            if 'sampling_rate' in r:
                sample_rate = r.split('=')[-1]
                if '#' in sample_rate:
                    sample_rate = sample_rate[:sample_rate.find('#')]
                sample_rate = float(sample_rate)
    return sample_rate
