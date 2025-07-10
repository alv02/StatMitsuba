from typing import cast

import drjit as dr
import mitsuba as mi
import numpy as np
from drjit.auto.ad import Bool, Float, TensorXf

mi.set_variant("cuda_ad_rgb")


class StatisticalIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties, /) -> None:
        super().__init__(props)
        self.real_integrator: mi.SamplingIntegrator = cast(
            mi.SamplingIntegrator, props["_arg_0"]
        )
        print(self.real_integrator)

    def box_cox(self, samples, lam=0.5):
        return dr.log(samples) if lam == 0 else ((dr.power(samples, lam) - 1) / lam)

    def online_statistics(self, samples, size, spp):
        if mi.variant() != "scalar_rgb":
            samples_reshaped = dr.reshape(
                dtype=TensorXf, value=samples, shape=(3, size.y, size.x, spp), order="C"
            )
            samples_bc = self.box_cox(samples_reshaped)
            mu = dr.mean(samples_bc, axis=3)

            # Use dr.newaxis to expand the dimension of mu
            mu_expanded = mu[..., dr.newaxis]

            # Calculate deviations from the mean (centralization)
            delta = samples_bc - mu_expanded

            m3 = dr.mean(delta**3, axis=3)
            # Calculate variance (Bessel-corrected)
            mu_sq = dr.mean(samples_bc**2, axis=3)
            variance = (mu_sq - mu**2) * spp / (spp - 1)

            # When the variance is 0 there is no need for skewness correction
            estimands = dr.select(variance == 0, mu, mu + m3 / (6 * variance * spp))
            estimands_variance = variance / spp

            print(
                "Min Max m3: ",
                dr.min(m3),
                " ",
                dr.max(m3),
            )
            print(
                "Min Max estmiands: ",
                dr.min(estimands),
                " ",
                dr.max(estimands),
            )
            print(
                "Min Max estmiands variance: ",
                dr.min(estimands_variance),
                " ",
                dr.max(estimands_variance),
            )
            estimands_expanded = estimands[..., dr.newaxis]
            estimands_variance_expanded = estimands_variance[..., dr.newaxis]

            estimands_np = dr.detach(estimands_expanded)
            estimands_variance_np = dr.detach(estimands_variance_expanded)
            combined_statistics = np.concatenate(
                [estimands_np, estimands_variance_np], axis=3
            )
            spp_array = np.full_like(estimands, spp)
            spp_array = spp_array[..., np.newaxis]

            combined_with_spp = np.concatenate([combined_statistics, spp_array], axis=3)

            np.save(f"./io/steady/kitchen/stats_{spp}.npy", combined_with_spp)
            return combined_with_spp

    def should_stop(self) -> bool:
        return self.real_integrator.should_stop()

    def cancel(self) -> None:
        return self.real_integrator.cancel()

    def aov_names(self) -> list[str]:
        return self.real_integrator.aov_names()

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium | None = None,
        active: Bool = Bool(True),
    ) -> tuple[mi.Spectrum, Bool, list[Float]]:
        (spec, mask, aov) = self.real_integrator.sample(
            scene, sampler, ray, medium, active
        )
        film = scene.sensors()[0].film()
        size = film.size()
        spp = sampler.sample_count()
        self.online_statistics(spec, size, spp)
        return (spec, mask, aov)


mi.register_integrator(
    "statistical_integrator", lambda props: StatisticalIntegrator(props)
)

dr.set_flag(dr.JitFlag.Debug, True)
scene = mi.load_file("../scenes/kitchen/scene.xml")

sensor = scene.sensors()[0]
mi.render(scene)
mi.util.write_bitmap(f"./io/steady/kitchen/imagen.exr", sensor.film().bitmap())
