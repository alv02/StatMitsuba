from typing import Sequence

import drjit as dr
import mitsuba as mi
from drjit.auto import TensorXf
from drjit.auto.ad import Bool, Float
from StatisticalImageBlock import StatisticalImageBlock

"""

"""

mi.set_variant("scalar_rgb")


class MyFilm(mi.Film):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        hdrfilm_dict = {
            "type": "hdrfilm",
            "width": self.size().x,
            "height": self.size().y,
            "pixel_format": "luminance" if mi.is_monochromatic else "rgb",
            "crop_offset_x": self.crop_offset().x,
            "crop_offset_y": self.crop_offset().y,
            "crop_width": self.crop_size().x,
            "crop_height": self.crop_size().y,
            "sample_border": self.sample_border(),
            "rfilter": self.rfilter(),
        }

        self.hdrfilm: mi.Film = mi.load_dict(hdrfilm_dict)
        self.m_channels_count = self.hdrfilm.base_channels_count() + 1  # RGBA + W Creo

    def prepare(self, aovs: Sequence[str]) -> int:
        return self.hdrfilm.prepare(aovs)

    # TODO: Devolver StatisticalImageBlock
    def create_block(self, size, normalize: bool = False, borders: bool = False):
        block = StatisticalImageBlock(
            size=self.size(),
            offset=mi.ScalarPoint2i(self.crop_offset().x, self.crop_offset().y),
            channel_count=self.m_channels_count,
            rfilter=self.rfilter(),
        )
        print("create_block")
        print(block)
        print(block.channel_count())
        return block

    def develop(self, raw: bool = False) -> dr.auto.ad.TensorXf:
        image = self.hdrfilm.develop(raw=raw)
        return image

    def clear(self) -> None:
        self.hdrfilm.clear()
        return super().clear()

    def put_block(self, block: mi.ImageBlock) -> None:
        print("put_block")
        print(block)
        print(block.channel_count())
        return self.hdrfilm.put_block(block)


mi.register_film("myfilm", lambda props: MyFilm(props))
scene = mi.load_file("../scenes/cbox.xml")
image = mi.render(scene, spp=256)
mi.util.write_bitmap("my_first_render.exr", image)
