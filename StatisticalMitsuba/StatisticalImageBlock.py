from typing import override

import drjit as dr
import mitsuba as mi
from drjit.auto import TensorXf
from drjit.auto.ad import Bool, Float


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
            rfilter,
            border,
            normalize,
            coalesce,
            compensate,
            warn_negative,
            warn_invalid,
        )
        self.m_size = size
        self.m_offset = offset
        self.m_channel_count = channel_count
        self.sample_count = dr.zeros(TensorXf, shape=(size.x, size.y))
        self.sample_means = dr.zeros(TensorXf, shape=(size.x, size.y))

    def put_block(self, block: mi.ImageBlock) -> None:
        print("put_block")
        return super().put_block(block)

    def put(
        self,
        pos: mi.Point2f,
        wavelengths: mi.Spectrum,
        value: mi.Spectrum,
        alpha: Float = Float(1.0),
        weight: Float = Float(1.0),
        active: Bool = Bool(True),
    ) -> None:

        p: mi.Point2u = mi.Point2u(dr.floor(pos) - self.m_offset)
        index = dr.fma(p.y, self.size().x, p.x) * self.m_chanel_count
        print("Put")
        print(pos)
        print(value)
        super().put(pos, wavelengths, value, alpha, weight, active)

    def put(
        self,
        pos: mi.Point2f,
        value: mi.Spectrum,
        alpha: Float = Float(1.0),
        weight: Float = Float(1.0),
        active: Bool = Bool(True),
    ) -> None:
        p: mi.Point2u = mi.Point2u(dr.floor(pos) - self.m_offset)
        index = dr.fma(p.y, self.size().x, p.x) * self.m_chanel_count
        print("Put")
        print(pos)
        print(value)
        super().put(pos, value, alpha, weight, active)
