class HParams(object):
    def __init__(self):
        self.num_mels=80
        self.num_freq=1025
        self.sample_rate=16000
        self.frame_length_ms=50
        self.frame_shift_ms=12.5
        self.preemphasis=0.97
        self.min_level_db=-100
        self.ref_level_db=20
        self.max_abs_value=4.0
        self.symmetric_specs=True
        self.griffin_lim_iters=60
        self.power=1.5


hparams = HParams()
