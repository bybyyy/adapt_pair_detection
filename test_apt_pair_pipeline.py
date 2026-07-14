import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_apt_manifest import build_manifest
from build_apt_pair_dataset import convert, output_paths

try:
    import torch

    from EventDataset import AptPairDataset
    from train_2d_cnn import PairEventAPT2DCNN
    from train_apt_pair_models import campaign_split, leave_one_energy_out_split, stratified_split
    from train_hybrid_cnn import PairEventAPTHybridCNN, make_apt_wls_engineered_features
except ModuleNotFoundError:
    torch = None


class AptPairDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self, directory, include_second_truth=True):
        directory.mkdir(parents=True)
        digitizer = directory / "digitizer_final.txt"
        digitizer.write_text(
            "eventid layer type axis channelid x/y z energy pe time\n"
            "0 0 WLS_Fast x 0 0 0 0 1.5 0\n"
            "0 1 WLS_Slow y 2 0 0 0 2.5 0\n"
            "0 0 TKR x 0 0 0 0 999 0\n"
            "0 0 Edge_Detector x 0 0 0 0 999 0\n"
            "0 0 Calorimeter x 0 0 0 0 999 0\n"
            "1 0 WLS_Fast y 1 0 0 0 3.5 0\n"
            "1 1 WLS_Slow x 0 0 0 0 4.5 0\n"
            "1 0 GUNout n 0 0 0 0 999 0\n"
        )
        gun = directory / "GUNout_tmp.dat"
        gun.write_text("0 gamma 0 0 0 0 0 1 0\n1 gamma 0 0 0 0 0 1 0\n")
        csi = directory / "CsIout_tmp.dat"
        second = "1 1 1 e- 1 0 0 0 0 0 0 0 0 0 0 0 compt\n" if include_second_truth else ""
        csi.write_text("0 1 1 e- 1 0 0 0 0 0 0 0 0 0 0 0 conv\n" + second)
        return digitizer, gun, csi

    def convert_shard(self, name="run_a", energy=5.0):
        directory = self.root / name
        digitizer, gun, csi = self.write_inputs(directory)
        prefix = directory / "apt"
        args = argparse.Namespace(
            digitizer=digitizer,
            csi_truth=csi,
            gun_truth=gun,
            output_prefix=prefix,
            energy_mev=energy,
            seed=1,
            run_id=name,
            pipeline_config=None,
            effective_config_log=None,
            pipeline_repo=None,
            pipeline_commit="test-commit",
            layers=2,
            channels=3,
            min_class_count=1,
            allow_single_class=False,
            exclude_missing_csi_truth=False,
        )
        convert(args)
        return output_paths(prefix)["metadata"]

    def test_converter_uses_sig_and_wls_only(self):
        metadata_path = self.convert_shard()
        with metadata_path.open() as source:
            metadata = json.load(source)
        arrays = metadata["arrays"]
        features = np.load(metadata_path.parent / arrays["features"])
        labels = np.load(metadata_path.parent / arrays["labels"])
        self.assertEqual(features.shape, (2, 4, 2, 3))
        self.assertEqual(features[0, 0, 0, 0], 1.5)
        self.assertEqual(features[0, 3, 1, 2], 2.5)
        self.assertEqual(features[1, 1, 0, 1], 3.5)
        self.assertEqual(features[1, 2, 1, 0], 4.5)
        self.assertEqual(float(features.sum()), 12.0)
        np.testing.assert_array_equal(labels, [1, 0])

    def test_converter_fails_if_digitized_event_lacks_csi_truth(self):
        directory = self.root / "missing_truth"
        digitizer, gun, csi = self.write_inputs(directory, include_second_truth=False)
        args = argparse.Namespace(
            digitizer=digitizer,
            csi_truth=csi,
            gun_truth=gun,
            output_prefix=directory / "apt",
            energy_mev=5.0,
            seed=1,
            run_id="missing_truth",
            pipeline_config=None,
            effective_config_log=None,
            pipeline_repo=None,
            pipeline_commit=None,
            layers=2,
            channels=3,
            min_class_count=1,
            allow_single_class=False,
            exclude_missing_csi_truth=False,
        )
        with self.assertRaisesRegex(ValueError, "missing from CsI truth"):
            convert(args)

    def test_manifest_loader_is_lazy_and_geometry_aware(self):
        if torch is None:
            self.skipTest("PyTorch is not installed in this interpreter")
        first = self.convert_shard("run_a")
        second = self.convert_shard("run_b")
        manifest = self.root / "manifest.json"
        build_manifest([first, second], manifest)
        dataset = AptPairDataset(manifest)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.geometry["feature_shape"], [4, 2, 3])
        features, label, energy = dataset[0]
        self.assertEqual(tuple(features.shape), (4, 2, 3))
        self.assertEqual(label.item(), 1.0)
        self.assertEqual(energy.item(), 5.0)


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class AptPairModelTests(unittest.TestCase):
    def test_metrics_include_specificity_and_confidence_interval(self):
        from train_apt_pair_models import metrics_with_confidence

        metrics = metrics_with_confidence(8, 2, 2, 8)
        self.assertAlmostEqual(metrics["specificity"], 0.8)
        self.assertIn("specificity", metrics["confidence_intervals_95"])

    def test_models_accept_geometry_independent_wls_maps(self):
        wls = torch.rand(2, 4, 20, 32)
        self.assertEqual(PairEventAPT2DCNN()(wls).shape, (2,))
        self.assertEqual(PairEventAPTHybridCNN()(wls).shape, (2,))
        self.assertEqual(make_apt_wls_engineered_features(wls).shape, (2, 11))

    def test_splits_keep_seed_runs_disjoint(self):
        energies = []
        labels = []
        run_ids = []
        for energy in [5.0, 10.0, 50.0]:
            for seed in range(4):
                run_id = f"{energy:g}_seed_{seed}"
                for label in [0, 1, 0, 1]:
                    energies.append(energy)
                    labels.append(label)
                    run_ids.append(run_id)
        energies = np.asarray(energies)
        labels = np.asarray(labels)
        run_ids = np.asarray(run_ids, dtype=object)

        partitions = stratified_split(labels, energies, run_ids)
        partition_runs = [set(run_ids[index.numpy()].tolist()) for index in partitions]
        self.assertTrue(partition_runs[0].isdisjoint(partition_runs[1]))
        self.assertTrue(partition_runs[0].isdisjoint(partition_runs[2]))
        self.assertTrue(partition_runs[1].isdisjoint(partition_runs[2]))
        for indices in partitions:
            self.assertEqual(set(energies[indices.numpy()].tolist()), {5.0, 10.0, 50.0})

        loeo = leave_one_energy_out_split(labels, energies, run_ids, 50.0)
        loeo_runs = [set(run_ids[index.numpy()].tolist()) for index in loeo]
        self.assertTrue(loeo_runs[0].isdisjoint(loeo_runs[1]))
        self.assertTrue(loeo_runs[0].isdisjoint(loeo_runs[2]))
        self.assertTrue(loeo_runs[1].isdisjoint(loeo_runs[2]))
        self.assertEqual(set(energies[loeo[2].numpy()].tolist()), {50.0})

    def test_campaign_split_uses_seed_endings(self):
        energies = []
        labels = []
        run_ids = []
        random_seeds = []
        for energy, base in [(10.0, 21000), (15.0, 21500)]:
            for ending in range(10):
                for label in [0, 1, 0, 1]:
                    energies.append(energy)
                    labels.append(label)
                    run_ids.append(f"{energy:g}_seed_{base + ending}")
                    random_seeds.append(base + ending)
        energies = np.asarray(energies)
        labels = np.asarray(labels)
        run_ids = np.asarray(run_ids, dtype=object)
        random_seeds = np.asarray(random_seeds)

        partitions = campaign_split(labels, energies, run_ids, random_seeds)
        expected_endings = [set(range(8)), {8}, {9}]
        for indices, endings in zip(partitions, expected_endings):
            selected = random_seeds[indices.numpy()]
            self.assertEqual(set((selected % 10).tolist()), endings)



if __name__ == "__main__":
    unittest.main()
