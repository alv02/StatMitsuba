from typing import cast

import drjit as dr
import mitsuba as mi
import numpy as np
from drjit.auto.ad import Bool, Float, TensorXf

mi.set_variant("llvm_ad_rgb")


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
            print(np.shape(samples))
            samples_box_cox = self.box_cox(samples_reshaped)
            np.save("samples.npy", samples_reshaped)

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
scene = mi.load_file("../scenes/cbox.xml")
sensor = scene.sensors()[0]
mi.render(scene)
mi.util.write_bitmap("aovs.exr", sensor.film().bitmap())
