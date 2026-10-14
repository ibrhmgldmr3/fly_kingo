"""Brian2 referans modeli.

philshiu/Drosophila_brain_model `model.py` (MIT, Shiu et al. 2024) temel alınarak yazıldı.
Denklemler, parametreler ve Poisson uyarımı aynıdır. Farklar:
- Ağ bir kez kurulur ve denemeler arasında `store/restore` ile sıfırlanır (her deneme için yeniden
  kurmak yerine).
- Her uyarılan nöron için ayrı frekans verilebilir (Shiu'da en fazla iki frekans grubu var).

Bu modül doğruluk referansıdır. Günlük kullanımda `lif_torch.TorchLIF` çalışır.
"""

from textwrap import dedent

import numpy as np

from flyhire.brain.params import DEFAULT_PARAMS, LIFParams

EQS = dedent(
    """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
)


def run_brian2(
    n_neurons: int,
    pre: np.ndarray,
    post: np.ndarray,
    signed_count: np.ndarray,
    rates_hz: dict[int, float],
    duration_ms: float = 1000.0,
    n_trials: int = 30,
    seed: int = 0,
    params: LIFParams = DEFAULT_PARAMS,
    codegen_target: str = "cython",
) -> np.ndarray:
    """Her deneme için spike sayılarını döndürür: (n_trials, n_neurons)."""
    import brian2 as b2
    from brian2 import Hz, ms, mV

    b2.prefs.codegen.target = codegen_target
    b2.defaultclock.dt = params.dt * ms
    b2.seed(seed)

    namespace = {
        "v_0": params.v_0 * mV,
        "t_mbr": params.t_mbr * ms,
        "tau": params.tau * ms,
        "v_th": params.v_th * mV,
        "v_rst": params.v_rst * mV,
    }
    neu = b2.NeuronGroup(
        N=n_neurons,
        model=EQS,
        method="linear",
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace=namespace,
    )
    neu.v = params.v_0 * mV
    neu.g = 0 * mV
    neu.rfc = params.t_rfc * ms

    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=params.t_dly * ms)
    syn.connect(i=pre, j=post)
    syn.w = signed_count * params.w_syn * mV

    poisson = []
    for i, rate in rates_hz.items():
        if rate <= 0:
            continue
        poisson.append(
            b2.PoissonInput(
                target=neu[i : i + 1],
                target_var="v",
                N=1,
                rate=rate * Hz,
                weight=params.w_poi * mV,
            )
        )
        neu[i : i + 1].rfc = 0 * ms

    mon = b2.SpikeMonitor(neu)
    net = b2.Network(neu, syn, mon, *poisson)
    net.store()

    counts = np.zeros((n_trials, n_neurons), dtype=np.int32)
    for trial in range(n_trials):
        net.restore()
        b2.seed(seed + trial)
        net.run(duration_ms * ms)
        counts[trial] = np.asarray(mon.count)
    return counts
