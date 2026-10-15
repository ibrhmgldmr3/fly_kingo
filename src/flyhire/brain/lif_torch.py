"""Shiu LIF modelinin PyTorch karşılığı (CPU veya CUDA).

Brian2 modelindeki her zaman adımı sırası birebir izlenir:

1. durum güncellemesi (refrakter olmayan nöronlar, kesin doğrusal çözüm)
2. eşik kontrolü (refrakter olmayanlar için v > v_th)
3. sinapslar: `t_dly` önce atılan spike'lar hedeflerin g'sine, Poisson girdileri v'ye eklenir.
   Brian2'de `(unless refractory)` değişkenlere "koşullu yazma" uygulanır: refrakter (veya bu adımda
   spike atmış) bir hedefe gelen sinaptik girdi yok sayılır. Burada da aynısı yapılır.
4. reset: spike atan nöronlarda v = v_rst, g = 0

Denklemler: dv/dt = (v_0 - v + g) / t_mbr,  dg/dt = -g / tau.
Batch boyutu deneme (trial) sayısıdır; denemeler birbirinden bağımsızdır.

Ağırlıklar yükleme sırasında bir kez hesaplanır ve hiçbir yerde değiştirilmez. Bunu doğrulamak için
`weights_digest()` kullanılır.
"""

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from flyhire.brain.params import DEFAULT_PARAMS, LIFParams


@dataclass
class RunResult:
    counts: np.ndarray  # (n_trials, N) spike sayıları
    duration_ms: float
    # kayıt istenen nöronlar için spike'lar: (deneme, kayıt sırası, zaman ms) satırları
    raster: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    def rates_hz(self) -> np.ndarray:
        """(n_trials, N) ateşleme oranları."""
        return self.counts / (self.duration_ms / 1000.0)


class TorchLIF:
    def __init__(
        self,
        n_neurons: int,
        pre: np.ndarray,
        post: np.ndarray,
        signed_count: np.ndarray,
        params: LIFParams = DEFAULT_PARAMS,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.n = int(n_neurons)
        self.params = params
        self.device = torch.device(device)
        # float32 hızlıdır; Brian2 float64 kullanır. Eşiğe çok yakın durumlarda yuvarlama farkı tek
        # bir spike'ı değiştirebilir, bu yüzden birebir karşılaştırma float64 ile yapılır.
        self.dtype = dtype

        # Giden bağlantılar için CSR: spike atan nöronun tüm hedeflerine tek seferde erişim.
        order = np.argsort(pre, kind="stable")
        counts = np.bincount(pre, minlength=self.n)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        self._indptr = torch.as_tensor(indptr, dtype=torch.int64, device=self.device)
        self._post = torch.as_tensor(post[order], dtype=torch.int64, device=self.device)
        self._w = torch.as_tensor(
            signed_count[order] * params.w_syn, dtype=dtype, device=self.device
        )

    @classmethod
    def from_connectome(
        cls, connectome, params: LIFParams = DEFAULT_PARAMS, device="cpu", dtype=torch.float32
    ):
        return cls(
            connectome.n_neurons,
            connectome.pre,
            connectome.post,
            connectome.signed_count,
            params,
            device,
            dtype,
        )

    def weights_digest(self) -> str:
        digest = hashlib.sha256()
        for t in (self._indptr, self._post, self._w):
            digest.update(t.cpu().numpy().tobytes())
        return digest.hexdigest()

    @torch.inference_mode()
    def run(
        self,
        rates_hz: dict[int, float],
        duration_ms: float = 1000.0,
        n_trials: int = 30,
        seed: int = 0,
        silence: list[int] | None = None,
        record: list[int] | None = None,
    ) -> RunResult:
        """Simülasyonu çalıştırır.

        rates_hz: model indeksi → Poisson uyarım frekansı (Hz). Değer tek sayı (tüm denemeler) ya
            da n_trials uzunluğunda bir dizi olabilir; böylece farklı koşullar tek batch'te koşar.
            Anahtarlardaki tüm nöronlar Poisson hedefidir ve Shiu'daki PoissonInput gibi refrakter
            süreleri sıfırdır (frekansları 0 olsa bile).
        silence:  giden sinapsları kapatılan nöronlar (Shiu'daki susturma deneyi).
        record:   zamanları kaydedilecek nöronlar (raster için).
        """
        p, n, dev, b = self.params, self.n, self.device, n_trials
        n_steps = round(duration_ms / p.dt)
        delay = p.delay_steps
        gen = torch.Generator(device=dev).manual_seed(seed)

        decay_v = math.exp(-p.dt / p.t_mbr)
        decay_g = math.exp(-p.dt / p.tau)
        g_to_v = p.tau / (p.tau - p.t_mbr) * (decay_g - decay_v)
        v_offset = p.v_0 * (1.0 - decay_v)

        stim_idx = torch.tensor(list(rates_hz), dtype=torch.int64, device=dev)
        columns = [
            np.broadcast_to(np.asarray(r, dtype=np.float64), (b,)) for r in rates_hz.values()
        ]
        stim_hz = np.stack(columns, axis=1) if columns else np.zeros((b, 0))
        stim_p = torch.as_tensor(stim_hz * p.dt / 1000.0, dtype=torch.float32, device=dev)

        # Refrakter süresi olan nöronlar (Poisson hedefleri hariç)
        has_refractory = torch.ones(n, dtype=torch.bool, device=dev)
        has_refractory[stim_idx] = False
        silenced = torch.zeros(n, dtype=torch.bool, device=dev)
        if silence:
            silenced[torch.as_tensor(silence, device=dev)] = True
        rec_idx = torch.as_tensor(record or [], dtype=torch.int64, device=dev)

        v = torch.full((b, n), p.v_0, dtype=self.dtype, device=dev)
        g = torch.zeros((b, n), dtype=self.dtype, device=dev)
        vf, gf = v.view(-1), g.view(-1)
        counts = torch.zeros((b, n), dtype=torch.int32, device=dev)
        blocked = torch.zeros(b * n, dtype=torch.bool, device=dev)
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        queue = [empty] * delay  # yuva k: (adım % delay) == k olan adımda atılan spike'lar
        # Son (refrakter_adım - 1) adımda spike atanlar refrakterdir. Sadece onlar izlenir; tüm
        # dizi üzerinde maske tutmaktan çok daha ucuzdur (her adımda birkaç yüz nöron).
        refractory_ring = [empty] * max(p.refractory_steps - 1, 0)
        raster = []

        for step in range(n_steps):
            refractory = torch.cat(refractory_ring) if refractory_ring else empty

            # 1. durum güncellemesi: herkes güncellenir, refrakterlerin değerleri geri yazılır
            if refractory.numel():
                v_hold, g_hold = vf[refractory], gf[refractory]
            v.mul_(decay_v).add_(g, alpha=g_to_v).add_(v_offset)
            g.mul_(decay_g)
            if refractory.numel():
                vf[refractory], gf[refractory] = v_hold, g_hold

            # 2. eşik
            spk = v > p.v_th
            if refractory.numel():
                spk.view(-1)[refractory] = False
            flat = spk.view(-1).nonzero().squeeze(1)
            if flat.numel():
                counts.view(-1)[flat] += 1
                if rec_idx.numel():
                    hit = spk[:, rec_idx].nonzero()
                    if hit.numel():
                        raster.append(torch.cat([hit, torch.full_like(hit[:, :1], step)], 1))

            # 3. sinapslar: gecikmeli spike'lar ve Poisson girdileri
            slot = step % delay
            blocked[refractory] = True
            blocked[flat] = True
            self._propagate(queue[slot], g, blocked)
            blocked[refractory] = False
            blocked[flat] = False
            queue[slot] = flat[~silenced[flat % n]] if silence else flat
            if stim_idx.numel():
                fire = torch.rand((b, stim_idx.numel()), generator=gen, device=dev) < stim_p
                v[:, stim_idx] += fire.to(self.dtype) * p.w_poi

            # 4. reset
            if flat.numel():
                vf[flat] = p.v_rst
                gf[flat] = 0.0
            if refractory_ring:
                refractory_ring[step % len(refractory_ring)] = flat[has_refractory[flat % n]]

        raster_np = np.empty((0, 3))
        if raster:
            r = torch.cat(raster).cpu().numpy().astype(np.float64)
            r[:, 2] *= p.dt
            raster_np = r
        return RunResult(counts.cpu().numpy(), duration_ms, raster_np)

    def _propagate(self, flat: torch.Tensor, g: torch.Tensor, blocked: torch.Tensor) -> None:
        """`flat` içindeki spike'ların (deneme * N + nöron) giden sinapslarını g'ye ekler.

        `blocked` (refrakter veya bu adımda spike atmış) hedeflere yazılmaz.
        """
        if not flat.numel():
            return
        pre = flat % self.n
        trial = flat // self.n
        starts = self._indptr[pre]
        lengths = self._indptr[pre + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return
        owner = torch.repeat_interleave(torch.arange(len(pre), device=self.device), lengths)
        offset = torch.arange(total, device=self.device) - torch.repeat_interleave(
            torch.cumsum(lengths, 0) - lengths, lengths
        )
        syn = starts[owner] + offset
        target = trial[owner] * self.n + self._post[syn]
        keep = ~blocked[target]
        g.view(-1).index_add_(0, target[keep], self._w[syn[keep]])
