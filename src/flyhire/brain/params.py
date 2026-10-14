"""Shiu LIF modelinin parametreleri.

Değerler philshiu/Drosophila_brain_model `model.py` içindeki `default_params` ile birebir aynıdır
(MIT lisansı, Shiu et al., Nature 634, 2024). Birimler: ms ve mV.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LIFParams:
    dt: float = 0.1  # ms, Brian2 varsayılan zaman adımı
    v_0: float = -52.0  # mV, dinlenme potansiyeli
    v_rst: float = -52.0  # mV, spike sonrası reset
    v_th: float = -45.0  # mV, eşik
    t_mbr: float = 20.0  # ms, membran zaman sabiti
    tau: float = 5.0  # ms, sinaptik iletkenlik zaman sabiti
    t_rfc: float = 2.2  # ms, refrakter süre
    t_dly: float = 1.8  # ms, sinaptik gecikme
    w_syn: float = 0.275  # mV, sinaps başına ağırlık (serbest parametre)
    f_poi: float = 250.0  # Poisson girdisi ölçeği: w_syn * f_poi = 68.75 mV, tek olay spike üretir

    @property
    def w_poi(self) -> float:
        return self.w_syn * self.f_poi

    @property
    def refractory_steps(self) -> int:
        return round(self.t_rfc / self.dt)

    @property
    def delay_steps(self) -> int:
        return round(self.t_dly / self.dt)


DEFAULT_PARAMS = LIFParams()
