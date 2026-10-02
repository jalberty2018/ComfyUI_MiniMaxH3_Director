"""Adapter contract tests; run with unittest, without ComfyUI/GPU dependencies."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'director_motion_under_test',
    Path(__file__).resolve().parents[1] / 'director/h3_motion_context.py',
)
motion = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'torch': types.ModuleType('torch')}):
    spec.loader.exec_module(motion)


class Tensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        indices = list(indices)
        if Ellipsis in indices:
            i = indices.index(Ellipsis)
            indices[i:i + 1] = [slice(None)] * (self.ndim - len(indices) + 1)
        indices += [slice(None)] * (self.ndim - len(indices))
        return Tensor(tuple(len(range(*s.indices(n))) for s, n in zip(indices, self.shape)))

    def contiguous(self):
        return self


class ExternalNode:
    FUNCTION = 'apply'
    legacy = False

    def apply(self, **kwargs):
        self.called = kwargs
        result = []
        for emb, meta in kwargs['conditioning']:
            meta = dict(meta)
            kfs = list(meta['minimax_keyframes'])
            kfs.append({'resolved_frame_index': 0, 'latent': object()})
            if self.legacy:
                meta['minimax_refs'] = list(meta.get('minimax_refs', [])) + [{'kind': 'audio'}]
            else:
                kfs.append({'resolved_frame_index': -2.4, 'audio_latent': object()})
            meta['minimax_keyframes'] = kfs
            result.append([emb, meta])
        return result, 22


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.node = ExternalNode()
        self.registry = types.SimpleNamespace(NODE_CLASS_MAPPINGS={
            'MiniMaxH3MotionContext': lambda: self.node,
        })
        self.context = {'samples': [Tensor((1, 4, 71, 2, 2)), Tensor((1, 4, 2, 398))]}
        self.positive = [[object(), {'minimax_refs': [{'kind': 'image'}],
            'minimax_keyframes': [{'resolved_frame_index': 238, 'latent': object()}]}]]
        self.patches = [
            patch.dict(sys.modules, {'nodes': self.registry}),
            patch.object(motion, '_streams_from_latent', side_effect=lambda x: x['samples']),
            patch.object(motion, '_repack_av_streams', side_effect=lambda x, _: x),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def apply(self, **kwargs):
        return motion.apply_motion_context(self.positive, self.context, vae=object(),
            context_length=22, context_latent=self.context, **kwargs)

    def test_registered_node_and_reference_and_endpoint_preserved(self):
        out, trim, gap = self.apply()
        self.assertEqual((trim, gap), (22, 0))
        self.assertIs(self.node.called['context_latent'], self.context)
        self.assertEqual(self.node.called['context_length'], '22')
        self.assertEqual(out[0][1]['minimax_refs'], self.positive[0][1]['minimax_refs'])
        kfs = out[0][1]['minimax_keyframes']
        self.assertNotIn(motion.CTX_FRAME_KEY, kfs[0])
        self.assertIn(motion.CTX_FRAME_KEY, kfs[1])
        self.assertEqual(len(self.positive[0][1]['minimax_keyframes']), 1)

    def test_missing_pack_gives_installation_error(self):
        self.registry.NODE_CLASS_MAPPINGS.clear()
        with self.assertRaisesRegex(RuntimeError, 'Install or enable.*github.com'):
            self.apply()

    def test_audio_disabled_removes_only_added_audio(self):
        out, _, _ = self.apply(continue_audio=False)
        self.assertFalse(any('audio_latent' in k for k in out[0][1]['minimax_keyframes']))
        self.assertEqual(out[0][1]['minimax_refs'], self.positive[0][1]['minimax_refs'])

    def test_legacy_external_audio_reference_removed_when_muted(self):
        self.node.legacy = True
        out, _, _ = self.apply(continue_audio=False)
        self.assertEqual(out[0][1]['minimax_refs'], self.positive[0][1]['minimax_refs'])

    def test_drop_old_pins_keep_endpoint(self):
        self.positive[0][1]['minimax_keyframes'].append({motion.CTX_FRAME_KEY: True})
        self.apply()
        self.assertEqual(len(self.node.called['conditioning'][0][1]['minimax_keyframes']), 1)

    def test_drop_endpoints_on_request(self):
        self.apply(keep_existing_keyframes=False)
        self.assertEqual(self.node.called['conditioning'][0][1]['minimax_keyframes'], [])

    def test_export_boundary_aligns_both_streams(self):
        _, _, gap = self.apply(context_end_frame=230)
        streams = self.node.called['context_latent']['samples']
        self.assertEqual(gap, 4)
        self.assertEqual(streams[0].shape[2], 67)
        self.assertEqual(streams[1].shape[-1], 377)
        self.assertEqual(self.context['samples'][0].shape[2], 71)

    def test_canvas_mismatch_uses_decoded_frames(self):
        target = {'samples': [Tensor((1, 4, 71, 4, 4)), Tensor((1, 4, 2, 398))]}
        frames = Tensor((100, 32, 32, 3))
        motion.apply_motion_context(self.positive, target, vae=object(), context_length=22,
            context_latent=self.context, context_frames=frames, context_end_frame=230)
        self.assertIsNone(self.node.called['context_latent'])
        self.assertIs(self.node.called['context_frames'], frames)


if __name__ == '__main__':
    unittest.main()
