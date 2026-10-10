"""Bounded integer CA memory implementation of the frozen belief transition.

This is a separate radius-one cellular memory plane, not the native B-740 VM.
The raw-text adapter is still outside this engine. The 0.3.4 experimental package exposes this extension; it is not compiled
into B-740's native instruction set.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

@dataclass(frozen=True)
class Parcel:
    key: tuple[int, int]
    choice: int
    features: np.ndarray

def identity(agent, obj):
    raw = hashlib.blake2s((agent+'\0'+obj).encode(), digest_size=8).digest()
    return tuple(int.from_bytes(raw[i:i+4], 'little') for i in (0, 4))

def local_rule(occupied, key, mask, incoming, weight, bias, threshold, unknown_bit):
    """Identical finite integer transition at every site.

    Input is the parcel from the left neighbour. A full nonmatching site
    forwards it one position; a vacant/matching site consumes it. There is no
    global attention, dictionary lookup, classifier model call, or float math.
    """
    if incoming is None:
        return occupied, key, mask, None
    if occupied and key != incoming.key:
        return occupied, key, mask, incoming
    if not occupied:
        occupied, key, mask = True, incoming.key, unknown_bit
    x = incoming.features
    margin = int(x.astype(np.int32) @ weight + bias)
    if not x[-8] or not x[-7]:
        margin = -32768
    if margin >= threshold:
        mask = 1 << incoming.choice
    elif margin > -threshold:
        mask |= 1 << incoming.choice
    return occupied, key, mask, None

class BeliefCA:
    """32 rings of 256 sites, deterministic occupied-key preservation.

    Sparse simulation skips quiescent sites; it executes exactly the same local
    rule as synchronous tick(). No eviction is hidden: a full mismatched ring
    reports overflow, so a caller must abstain/fall back.
    """
    def __init__(self, weight, bias, threshold, choices=4, topics=32, sites=256):
        assert 1 <= choices <= 30 and topics > 0 and sites > 0
        w = np.asarray(weight)
        assert w.ndim == 1 and np.all(w >= -128) and np.all(w <= 127)
        self.weight = w.astype(np.int32)
        self.bias, self.threshold = int(bias), int(threshold)
        self.choices, self.unknown_bit = choices, 1 << choices
        self.topics, self.sites = topics, sites
        self.occupied = np.zeros((topics, sites), bool)
        self.keys = np.zeros((topics, sites, 2), np.uint32)
        self.masks = np.full((topics, sites), self.unknown_bit, np.uint32)
        self.parcels = [[None]*sites for _ in range(topics)]
        self.overflow = 0
        self.ticks = 0

    def _validate(self, parcel):
        x = np.asarray(parcel.features)
        assert x.shape == self.weight.shape and np.issubdtype(x.dtype, np.integer)
        assert np.all(x >= -32768) and np.all(x <= 32767)
        assert 0 <= parcel.choice < self.choices
        # int32 dot cannot overflow under the actual input bound.
        assert int(np.abs(x.astype(np.int64)) @ np.abs(self.weight.astype(np.int64))) + abs(self.bias) < 2**31

    def _site(self, topic, site, incoming):
        state = local_rule(bool(self.occupied[topic,site]),
            tuple(int(v) for v in self.keys[topic,site]), int(self.masks[topic,site]),
            incoming, self.weight, self.bias, self.threshold, self.unknown_bit)
        occupied, key, mask, outgoing = state
        self.occupied[topic,site] = occupied
        self.keys[topic,site] = key
        self.masks[topic,site] = mask
        return outgoing

    def send(self, topic, parcel):
        """Exact sparse simulation of a single injected left-neighbour parcel."""
        assert not any(p is not None for ring in self.parcels for p in ring)
        self._validate(parcel)
        assert 0 <= topic < self.topics
        incoming = parcel
        for site in range(self.sites):
            incoming = self._site(topic, site, incoming)
            self.ticks += 1
            if incoming is None:
                return True
        self.overflow += 1
        return False

    def inject(self, topic, parcel):
        """Place an external input on the left edge of a ring before tick()."""
        self._validate(parcel)
        assert self.parcels[topic][-1] is None
        self.parcels[topic][-1] = parcel

    def tick(self):
        """Dense reference: synchronous radius-one transitions, no skipped cells."""
        incoming = [ring[:] for ring in self.parcels]
        outgoing = [[None]*self.sites for _ in range(self.topics)]
        for topic in range(self.topics):
            for site in range(self.sites):
                outgoing[topic][site] = self._site(topic,site,incoming[topic][site-1])
        self.parcels = outgoing
        self.ticks += 1

    def read_mask(self, topic, key):
        # External readout scans tags; it does not alter cellular state.
        for site in range(self.sites):
            if self.occupied[topic,site] and tuple(int(v) for v in self.keys[topic,site]) == key:
                return int(self.masks[topic,site])
        return self.unknown_bit

    def answer(self, topic, key):
        mask = self.read_mask(topic,key)
        return mask.bit_length()-1 if mask and not(mask&(mask-1)) and not(mask&self.unknown_bit) else None
