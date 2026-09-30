"""Run with: python -m unittest discover -s r1/tests -p test_c_score.py -v"""
import sys
from pathlib import Path
import unittest
import importlib.util
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_r1 import engine, spec
from mica_r1.batch import ModelStack, BatchSession, ingest_batch, probe_batch, evaluate


class ExactScoringTests(unittest.TestCase):
    def test_original_conformance(self):
        path = Path(__file__).with_name('test_conformance.py')
        module_spec = importlib.util.spec_from_file_location('conformance', path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if name.startswith('test_'):
                with self.subTest(fixture=name):
                    getattr(module, name)()

    def test_ties_extreme_biases_and_duplicate_terms(self):
        m = engine.random_model(12)
        # All candidates read the same pair repeatedly; cancellation and +/-6.
        m.sc_nb[:] = 6
        m.sc_ch[:] = 0
        m.sc_co[:, 0] = [1,-1,1,-1,1,-1]
        m.sc_co[:, 1] = 1
        m.sc_co[:, 2] = -1
        m.sc_bias[:, 0] = 32767
        m.sc_bias[:, 1] = 32767
        m.sc_bias[:, 2] = -32768
        ms = ModelStack([m], score_backend='c')
        rng = np.random.default_rng(41)
        F = rng.choice(np.array([-127,0,127], np.int32), (3,spec.N_CELLS,spec.N_CHANNELS))
        b = rng.integers(0,3,300); i = rng.integers(0,spec.N_CELLS,300)
        p = rng.integers(0,64,300); k = np.zeros(300,dtype=int)
        vals=F[b[:,None,None],(i[:,None,None]+np.array(spec.OFFSETS)[ms.sc_nb[k,p]])%spec.N_CELLS,ms.sc_ch[k,p]]
        expected=(ms.sc_bias[k,p].astype(np.int32)+(ms.sc_co[k,p]*vals).sum(2)).argmax(1)
        np.testing.assert_array_equal(ms.scorer.winners(F,b,i,k,p), expected)
        F[:] = 0
        np.testing.assert_array_equal(ms.scorer.winners(F,b,i,k,p),0)

    def test_state_and_scalar_for_all_routing_modes(self):
        rng = np.random.default_rng(73)
        old_mode = engine.ROUTING_MODE
        try:
            for routing in ('r1','positive','pairdiff','parity'):
                engine.ROUTING_MODE = routing
                models=[engine.random_model(seed) for seed in (1,9)]
                stacks=[ModelStack(models,score_backend=x) for x in ('numpy','c')]
                states=[BatchSession(2,np.arange(2)) for _ in stacks]
                scalars=[engine.Session(),engine.Session()]
                for step in range(65):
                    symbols=np.full(2,spec.BOS) if step == 0 else rng.integers(0,256,2)
                    for ms,st in zip(stacks,states): ingest_batch(ms,st,symbols)
                    for j,(m,s) in enumerate(zip(models,scalars)):
                        engine.ingest(m,s,int(symbols[j]))
                        np.testing.assert_array_equal(states[1].F[j],s.F)
                        np.testing.assert_array_equal(states[1].phase[j],s.phase)
                        self.assertEqual(states[1].updates[j],s.updates)
                        self.assertEqual(states[1].position[j],s.position)
                    for attr in ('F','phase','position','updates'):
                        np.testing.assert_array_equal(getattr(states[0],attr),getattr(states[1],attr))
                    np.testing.assert_array_equal(probe_batch(stacks[0],states[0]),probe_batch(stacks[1],states[1]))
        finally:
            engine.ROUTING_MODE=old_mode

    def test_full_record_objective_and_child_selection(self):
        from mica_r1.search import XorShift32, make_child
        from mica_r1.batch import objective
        m=engine.random_model(7)
        models=[m,make_child(m,XorShift32(11)),make_child(m,XorShift32(31))]
        records=[bytes(np.random.default_rng(5).integers(0,256,256,dtype=np.uint8))]
        results=[evaluate(ModelStack(models,score_backend=b),records) for b in ('numpy','c')]
        for a,b in zip(*results): np.testing.assert_array_equal(a,b)
        self.assertEqual(objective(*results[0][:2]).argmin(),objective(*results[1][:2]).argmin())

if __name__ == '__main__': unittest.main()
