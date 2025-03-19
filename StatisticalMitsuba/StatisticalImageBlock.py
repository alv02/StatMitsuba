import drjit as dr
import mitsuba as mi
from drjit.auto import TensorXf
from drjit.auto.ad import Bool, Float

mi.set_variant("scalar_rgb")


class StatisticalImageBlock(mi.ImageBlock):
    def __init__(
        self,
        size: mi.ScalarVector2u,
        offset: mi.ScalarPoint2i,
        channel_count: int,
        rfilter: (
            mi.ReconstructionFilter | None
        ) = None,  # Acepta None como valor por defecto
        border: bool = False,
        normalize: bool = False,
        coalesce: bool = False,
        compensate: bool = False,
        warn_negative: bool = False,
        warn_invalid: bool = False,
    ):
        super().__init__(
            size,
            offset,
            channel_count,
            None,
            border,
            normalize,
            coalesce,
            compensate,
            warn_negative,
            warn_invalid,
        )
